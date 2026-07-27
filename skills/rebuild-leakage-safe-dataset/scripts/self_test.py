#!/usr/bin/env python3
"""Self-test the leakage-safe manifest verifier."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from rebuild_leakage_safe import ContractError, load_manifest, main, verify_manifests


def digest(index: int) -> str:
    return f"{index:064x}"


def manifest(assignments: dict[str, list[str]], records: list[tuple[str, str, int]]) -> dict:
    return {
        "split_policy": {
            "source_groups": assignments,
            "target_sample_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "leakage": {
                "source_group_overlap_count": 0,
                "image_sha256_overlap_count": 0,
            },
        },
        "records": [
            {"source_group": group, "split": split, "image_sha256": digest(index)}
            for group, split, index in records
        ],
    }


class LeakageSafeRebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline_value = manifest(
            {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            [("train-a", "train", 1), ("val-a", "val", 2), ("test-a", "test", 3)],
        )
        self.baseline = self.write("baseline.json", self.baseline_value)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def verify(self, value: dict, mode: str = "append-isolated") -> dict:
        rebuilt = self.write("rebuilt.json", value)
        return verify_manifests(
            load_manifest(self.baseline),
            load_manifest(rebuilt),
            mode=mode,
            max_ratio_deviation=1.0,
        )

    def test_append_isolated_allows_new_groups_in_every_split(self) -> None:
        value = manifest(
            {
                "train": ["train-a", "train-b"],
                "val": ["val-a", "val-b"],
                "test": ["test-a", "test-b"],
            },
            [
                ("train-a", "train", 1),
                ("val-a", "val", 2),
                ("test-a", "test", 3),
                ("train-b", "train", 4),
                ("val-b", "val", 5),
                ("test-b", "test", 6),
            ],
        )
        result = self.verify(value)
        self.assertTrue(result["ok"])
        self.assertTrue(result["lineage"]["evaluation_expanded"])
        self.assertEqual(
            result["lineage"]["new_image_count_by_split"],
            {"train": 1, "val": 1, "test": 1},
        )

    def test_strict_eval_rejects_new_evaluation_group(self) -> None:
        value = manifest(
            {"train": ["train-a"], "val": ["val-a", "val-b"], "test": ["test-a"]},
            [
                ("train-a", "train", 1),
                ("val-a", "val", 2),
                ("test-a", "test", 3),
                ("val-b", "val", 4),
            ],
        )
        result = self.verify(value, mode="strict-eval")
        self.assertFalse(result["ok"])
        self.assertTrue(any("strict-eval" in error for error in result["errors"]))

    def test_rejects_existing_group_move(self) -> None:
        value = manifest(
            {"train": ["train-a", "val-a"], "val": [], "test": ["test-a"]},
            [("train-a", "train", 1), ("val-a", "train", 2), ("test-a", "test", 3)],
        )
        result = self.verify(value)
        self.assertFalse(result["ok"])
        self.assertTrue(result["lineage"]["moved_existing_groups"])

    def test_rejects_existing_image_source_group_change(self) -> None:
        value = manifest(
            {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            [("val-a", "val", 1), ("val-a", "val", 2), ("test-a", "test", 3)],
        )
        result = self.verify(value)
        self.assertFalse(result["ok"])
        self.assertEqual(result["lineage"]["regrouped_existing_hash_count"], 1)

    def test_rejects_missing_old_image(self) -> None:
        value = manifest(
            {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            [("train-a", "train", 1), ("val-a", "val", 2)],
        )
        result = self.verify(value)
        self.assertFalse(result["ok"])
        self.assertEqual(result["lineage"]["missing_existing_hash_count"], 1)

    def test_rejects_duplicate_hash(self) -> None:
        value = manifest(
            {"train": ["train-a"], "val": ["val-a"], "test": ["test-a"]},
            [
                ("train-a", "train", 1),
                ("val-a", "val", 1),
                ("test-a", "test", 3),
            ],
        )
        rebuilt = self.write("duplicate.json", value)
        with self.assertRaises(ContractError):
            load_manifest(rebuilt)

    def test_run_snapshots_invokes_builder_and_verifies(self) -> None:
        dataset = self.root / "dataset"
        dataset.mkdir()
        active = dataset / "manifest.json"
        active.write_text(json.dumps(self.baseline_value), encoding="utf-8")
        snapshot = self.root / "lineage" / "before.json"
        report = self.root / "lineage" / "report.json"
        builder = self.root / "builder.py"
        builder.write_text(
            "\n".join(
                (
                    "import json,sys",
                    "from pathlib import Path",
                    "active=Path(sys.argv[1])",
                    "baseline=Path(sys.argv[2])",
                    "value=json.loads(baseline.read_text(encoding='utf-8'))",
                    "value['split_policy']['source_groups']['train'].append('train-b')",
                    f"value['records'].append({{'source_group':'train-b','split':'train','image_sha256':'{digest(4)}'}})",
                    "active.write_text(json.dumps(value),encoding='utf-8')",
                )
            ),
            encoding="utf-8",
        )
        exit_code = main(
            [
                "run",
                "--manifest",
                str(active),
                "--snapshot-out",
                str(snapshot),
                "--report",
                str(report),
                "--max-ratio-deviation",
                "1",
                "--",
                sys.executable,
                str(builder),
                str(active),
                "{baseline_manifest}",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(snapshot.is_file())
        saved_report = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(saved_report["ok"])
        self.assertEqual(saved_report["lineage"]["new_image_count_by_split"]["train"], 1)

    def test_plan_preserves_old_splits_and_balances_new_groups(self) -> None:
        candidate_value = manifest(
            {
                "train": ["val-a", "new-a", "new-b", "new-c"],
                "val": ["train-a"],
                "test": ["test-a"],
            },
            [
                ("train-a", "val", 1),
                ("val-a", "train", 2),
                ("test-a", "test", 3),
                ("new-a", "train", 4),
                ("new-b", "train", 5),
                ("new-c", "train", 6),
            ],
        )
        candidate_value["split_policy"]["target_sample_ratios"] = {
            "train": 1 / 3,
            "val": 1 / 3,
            "test": 1 / 3,
        }
        candidate = self.write("candidate.json", candidate_value)
        output = self.root / "plan.json"
        exit_code = main(
            [
                "plan",
                "--baseline",
                str(self.baseline),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--max-ratio-deviation",
                "1",
            ]
        )
        self.assertEqual(exit_code, 0)
        planned = load_manifest(output)
        self.assertEqual(
            {group: planned.assignment[group] for group in self.baseline_value["split_policy"]["source_groups"]["train"]},
            {"train-a": "train"},
        )
        self.assertEqual(planned.assignment["val-a"], "val")
        self.assertEqual(planned.assignment["test-a"], "test")
        new_splits = [planned.assignment[group] for group in ("new-a", "new-b", "new-c")]
        self.assertCountEqual(new_splits, ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
