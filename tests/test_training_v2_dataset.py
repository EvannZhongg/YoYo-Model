import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common.files import sha256_file
from training_v2.prepare_dataset import SOURCE_POLICY, build_training_dataset, discover_annotation_sources
from training_v2.evaluate import _json_value
from training_v2.orientation_view import build_orientation_view


def _annotation(group: str, orientation: str, image_name: str, image_sha256: str) -> dict:
    return {
        "schema_version": "agent_yoyo_string_annotation_v4",
        "source_image": f"../../images/{group}/{image_name}",
        "image_sha256": image_sha256,
        "image_size": [64, 48],
        "source_group": group,
        "video_id": group,
        "visibility": "visible",
        "yoyo_bbox_pixel": [40, 20, 50, 32],
        "string_visibility": "visible",
        "string_polylines_pixel": [[[8, 10], [30, 20], [45, 25]]],
        "string_review_status": "approved",
        "trick_orientation": orientation,
        "quality": {
            "reviews": [
                {"decision": "approve", "review_scope": ["visible_geometry", "yoyo_bbox"]},
                {"decision": "approve", "review_scope": ["trick_orientation"]},
            ]
        },
    }


class FreshTrainingDatasetTests(unittest.TestCase):
    def test_json_value_preserves_array_shape(self):
        class ArrayValue:
            def tolist(self):
                return [[4, 1], [2, 3]]

        self.assertEqual(_json_value(ArrayValue()), [[4, 1], [2, 3]])

    def _write_source(self, root: Path, groups: list[str], source_tint: int = 120) -> None:
        for group_index, group in enumerate(groups):
            for sample_index, orientation in enumerate(("normal", "horizontal", "not_applicable")):
                name = f"sample_{sample_index}.jpg"
                image_path = root / "images" / group / name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (64, 48), (group_index * 20, sample_index * 40, source_tint))
                image.save(image_path)
                import hashlib

                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                label_path = root / "labels" / group / f"sample_{sample_index}.json"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    json.dumps(_annotation(group, orientation, name, digest)),
                    encoding="utf-8",
                )

    def test_builds_three_aligned_tasks_with_group_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            nypc = base / "annotations" / "NYPC1A"
            world = base / "annotations" / "world_final"
            self._write_source(nypc, ["a", "b", "c"], source_tint=80)
            self._write_source(world, ["d", "e", "f"], source_tint=180)
            first = build_training_dataset([nypc, world], base / "output-a", seed=7)
            second = build_training_dataset([world, nypc], base / "output-b", seed=7)

            self.assertEqual(first["dataset_id"], second["dataset_id"])
            self.assertEqual(first["sample_count"], 18)
            self.assertEqual(
                first["source_policy"],
                SOURCE_POLICY,
            )
            self.assertEqual(first["distributions"]["by_source_dataset"]["NYPC1A"]["samples"], 9)
            self.assertEqual(first["distributions"]["by_source_dataset"]["world_final"]["samples"], 9)
            self.assertEqual(first["source_inventory"]["NYPC1A"]["labels_discovered"], 9)
            self.assertEqual(first["source_inventory"]["world_final"]["samples_included"], 9)
            self.assertEqual(first["split_policy"]["leakage"]["source_group_overlap_count"], 0)
            group_sets = [set(first["split_policy"]["source_groups"][split]) for split in ("train", "val", "test")]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            for split in ("train", "val", "test"):
                self.assertGreater(first["counts"][split]["samples"], 0)
                for orientation in ("normal", "horizontal", "not_applicable"):
                    self.assertGreater(first["counts"][split][f"orientation:{orientation}"], 0)
            output = Path(first["output_dir"])
            self.assertEqual(len(list((output / "canonical" / "images").rglob("*.jpg"))), 18)
            self.assertEqual(len(list((output / "canonical" / "labels").rglob("*.json"))), 18)
            self.assertEqual(len((output / "canonical" / "index.jsonl").read_text(encoding="utf-8").splitlines()), 18)
            self.assertTrue((output / "detection" / "data.yaml").is_file())
            self.assertTrue((output / "string_segmentation" / "data.yaml").is_file())
            semantic_manifest = json.loads((output / "string_segmentation" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(semantic_manifest["dataset_id"], first["dataset_id"])
            self.assertEqual(semantic_manifest["counts"]["train"]["total"], first["counts"]["train"]["samples"])
            self.assertEqual(len(list((output / "detection" / "labels").rglob("*.txt"))), 18)
            self.assertEqual(len(list((output / "string_segmentation" / "labels").rglob("*.txt"))), 18)
            orientation = first["tasks"]["orientation"]
            self.assertEqual(orientation["train_balance"]["original_counts"], {"horizontal": 4, "normal": 4, "not_applicable": 4})
            self.assertEqual(orientation["train_balance"]["repeated_image_count"], 0)
            self.assertEqual(len(list((output / "orientation").rglob("*.jpg"))), 18)
            roi = build_orientation_view(output)
            self.assertEqual(roi["counts"]["test"]["total"], first["counts"]["test"]["samples"])
            self.assertTrue(all(Path(record["image"]).is_file() for record in roi["records"]))
            for split in ("train", "val", "test"):
                class_dirs = sorted(path.name for path in (output / "orientation" / split).iterdir() if path.is_dir())
                self.assertEqual(class_dirs, ["horizontal", "normal", "not_applicable"])
                self.assertTrue(all(not any(path.is_dir() for path in (output / "orientation" / split / name).iterdir()) for name in class_dirs))

    def test_canonical_rebuild_keeps_image_hash_suffix_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "annotations" / "source"
            self._write_source(source, ["a", "b", "c", "d", "e", "f"])
            first = build_training_dataset([source], base / "dataset-v1", seed=7)
            second = build_training_dataset([base / "dataset-v1" / "canonical"], base / "dataset-v2", seed=7)

            first_names = sorted(Path(record["canonical_image"]).name for record in first["records"])
            second_names = sorted(Path(record["canonical_image"]).name for record in second["records"])
            self.assertEqual(second_names, first_names)
            self.assertTrue(all(name.count("-") < 10 for name in second_names))

    def test_discovers_all_non_score_annotation_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations"
            for name in ("NYPC1A", "world_final", "score_annotations", "notes"):
                (annotations / name).mkdir(parents=True)
            (annotations / "NYPC1A" / "labels").mkdir()
            (annotations / "world_final" / "labels").mkdir()
            (annotations / "score_annotations" / "labels").mkdir()

            self.assertEqual(
                [path.name for path in discover_annotation_sources(annotations)],
                ["NYPC1A", "world_final"],
            )
            with self.assertRaisesRegex(ValueError, "Task-specific annotation stores"):
                build_training_dataset([annotations / "score_annotations"], Path(directory) / "output")

    def test_frozen_manifest_preserves_existing_groups_and_assigns_new_groups_to_train(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "annotations" / "world_final"
            self._write_source(source, ["a", "b", "c", "d", "e", "f"])
            original = build_training_dataset([source], base / "dataset-v1", seed=11)
            original_manifest = base / "dataset-v1" / "manifest.json"
            original_assignment = {
                group: split
                for split, groups in original["split_policy"]["source_groups"].items()
                for group in groups
            }

            self._write_source(source, ["new-g", "new-h"], source_tint=220)
            expanded = build_training_dataset(
                [source],
                base / "dataset-v2",
                seed=99,
                freeze_splits_from=original_manifest,
            )
            expanded_assignment = {
                group: split
                for split, groups in expanded["split_policy"]["source_groups"].items()
                for group in groups
            }

            self.assertEqual(
                {group: expanded_assignment[group] for group in original_assignment},
                original_assignment,
            )
            self.assertEqual(expanded_assignment["new-g"], "train")
            self.assertEqual(expanded_assignment["new-h"], "train")
            self.assertEqual(expanded["split_policy"]["strategy"], "frozen_source_groups_new_sources_train")
            self.assertEqual(expanded["split_policy"]["frozen_source_group_count"], 6)
            self.assertEqual(expanded["split_policy"]["new_train_source_groups"], ["new-g", "new-h"])
            self.assertEqual(expanded["split_policy"]["frozen_from_manifest_sha256"], sha256_file(original_manifest))
            self.assertEqual(
                expanded["split_policy"]["source_groups"]["val"],
                original["split_policy"]["source_groups"]["val"],
            )
            self.assertEqual(
                expanded["split_policy"]["source_groups"]["test"],
                original["split_policy"]["source_groups"]["test"],
            )

    def test_frozen_manifest_rejects_missing_or_duplicate_source_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            full_source = base / "annotations" / "full"
            partial_source = base / "annotations" / "partial"
            self._write_source(full_source, ["a", "b", "c", "d", "e", "f"])
            original = build_training_dataset([full_source], base / "dataset-v1", seed=13)
            original_manifest = base / "dataset-v1" / "manifest.json"
            self._write_source(partial_source, ["a", "b", "c", "d", "e"], source_tint=230)

            with self.assertRaisesRegex(ValueError, "missing from the current annotations"):
                build_training_dataset(
                    [partial_source],
                    base / "missing-output",
                    freeze_splits_from=original_manifest,
                )

            duplicate_manifest = base / "duplicate-manifest.json"
            duplicate = json.loads(json.dumps(original))
            repeated = duplicate["split_policy"]["source_groups"]["train"][0]
            duplicate["split_policy"]["source_groups"]["val"].append(repeated)
            duplicate_manifest.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more than once"):
                build_training_dataset(
                    [full_source],
                    base / "duplicate-output",
                    freeze_splits_from=duplicate_manifest,
                )

if __name__ == "__main__":
    unittest.main()
