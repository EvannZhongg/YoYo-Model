import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common.files import sha256_file
from training_v3.evaluate import (
    _artifact_suffix,
    _check_dataset_manifest,
    _detection_recall_from_confusion,
    _json_value,
)
from training_v3.prepare_dataset import ANNOTATION_SCHEMA_VERSION, SOURCE_POLICY, build_training_dataset, discover_annotation_sources
from training_v3.orientation_view import build_orientation_view


def _annotation(group: str, orientation: str, image_name: str, image_sha256: str) -> dict:
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
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
            self.assertEqual(first["annotation_schema_version"], ANNOTATION_SCHEMA_VERSION)
            self.assertEqual(first["distributions"]["by_source_dataset"]["NYPC1A"]["samples"], 9)
            self.assertEqual(first["distributions"]["by_source_dataset"]["world_final"]["samples"], 9)
            self.assertEqual(first["source_inventory"]["NYPC1A"]["labels_discovered"], 9)
            self.assertEqual(first["source_inventory"]["world_final"]["samples_included"], 9)
            self.assertEqual(first["split_policy"]["leakage"]["source_group_overlap_count"], 0)
            self.assertEqual(first["schema_version"], "yoyo_multitask_dataset_v6")
            self.assertTrue(all("string_visibility" in record for record in first["records"]))
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
            for label in (output / "canonical" / "labels").rglob("*.json"):
                annotation = json.loads(label.read_text(encoding="utf-8"))
                self.assertEqual(annotation["schema_version"], ANNOTATION_SCHEMA_VERSION)
                self.assertNotIn("dataset_management", annotation)
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
            self.assertFalse(roi["input_dependencies"]["string_geometry"])
            self.assertEqual(roi["counts"]["test"]["total"], first["counts"]["test"]["samples"])
            self.assertTrue(all(Path(record["image"]).is_file() for record in roi["records"]))
            for split in ("train", "val", "test"):
                class_dirs = sorted(path.name for path in (output / "orientation" / split).iterdir() if path.is_dir())
                self.assertEqual(class_dirs, ["horizontal", "normal", "not_applicable"])
                self.assertTrue(all(not any(path.is_dir() for path in (output / "orientation" / split / name).iterdir()) for name in class_dirs))

    def test_uncertain_yoyo_is_excluded_from_detection_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "annotations" / "source"
            self._write_source(source, ["a", "b", "c"])
            label = source / "labels" / "a" / "sample_0.json"
            annotation = json.loads(label.read_text(encoding="utf-8"))
            annotation["visibility"] = "uncertain"
            annotation["yoyo_bbox_pixel"] = None
            annotation["bbox_review_status"] = "needs_review"
            label.write_text(json.dumps(annotation), encoding="utf-8")

            manifest = build_training_dataset([source], base / "output", seed=7)
            record = next(item for item in manifest["records"] if item["source_group"] == "a" and item["trick_orientation"] == "normal")
            self.assertTrue(record["yoyo_ignored"])
            self.assertEqual(sum(split["yoyo_ignored"] for split in manifest["counts"].values()), 1)
            relative = Path(record["canonical_label"]).relative_to(base / "output" / "canonical" / "labels")
            split = record["split"]
            self.assertFalse((base / "output" / "detection" / "labels" / split / relative.with_suffix(".txt")).exists())
            self.assertFalse(any((base / "output" / "orientation" / split / "normal").glob("a__sample_0*")))

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


class TrainingEvaluationTests(unittest.TestCase):
    def test_background_predictions_count_as_false_negatives(self) -> None:
        recall, matrix = _detection_recall_from_confusion(
            [[10.0, 2.0], [5.0, 0.0]],
            ["yoyo"],
        )

        self.assertEqual(matrix, [[10.0, 2.0], [5.0, 0.0]])
        self.assertAlmostEqual(recall["yoyo"], 10.0 / 15.0)

    def test_native_manifest_has_no_artifact_suffix(self) -> None:
        matches, warning = _check_dataset_manifest("abc", "abc", False)

        self.assertTrue(matches)
        self.assertEqual(warning, "")
        self.assertEqual(_artifact_suffix(matches, "abc"), "")

    def test_external_manifest_requires_explicit_opt_in(self) -> None:
        with self.assertRaises(RuntimeError):
            _check_dataset_manifest("old", "new", False)

        matches, warning = _check_dataset_manifest("old", "1234567890abcdef", True)
        self.assertFalse(matches)
        self.assertIn("cross-model comparison", warning)
        self.assertEqual(_artifact_suffix(matches, "1234567890abcdef"), "_external_1234567890ab")


if __name__ == "__main__":
    unittest.main()
