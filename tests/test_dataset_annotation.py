import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from helpers import make_annotation_dataset, make_consecutive_dataset
from workbench.dataset_annotation import (
    REVIEW_MAP_FILENAME,
    list_annotation_datasets,
    load_annotation_sample,
    open_annotation_dataset,
    save_annotation_sample,
    set_all_annotation_samples_reviewed,
    set_annotation_sample_reviewed,
)


class DatasetAnnotationWorkbenchTests(unittest.TestCase):
    def test_lists_only_current_json_annotation_datasets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = make_annotation_dataset(root)
            derived = root / "derived"
            (derived / "labels").mkdir(parents=True)
            (derived / "images").mkdir()
            (derived / "labels" / "frame.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(list_annotation_datasets(), [{"name": "review_set", "path": str(dataset.resolve())}])

    def test_rejects_v4_annotation_schema(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            label_path = dataset / "canonical" / "labels" / key
            document = json.loads(label_path.read_text(encoding="utf-8"))
            document["schema_version"] = "agent_yoyo_string_annotation_v4"
            label_path.write_text(json.dumps(document), encoding="utf-8")
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(list_annotation_datasets(), [])
                with self.assertRaisesRegex(ValueError, "unsupported annotation schema"):
                    load_annotation_sample(str(dataset), key)

    def test_lists_every_annotation_dataset_under_datasets_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = make_annotation_dataset(root, "alpha")
            second, _ = make_annotation_dataset(root, "second")
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(
                    list_annotation_datasets(),
                    [
                        {"name": "alpha", "path": str(first.resolve())},
                        {"name": "second", "path": str(second.resolve())},
                    ],
                )

    def test_excludes_datasets_with_consecutive_group_mapping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            regular, _ = make_annotation_dataset(root, "regular")
            consecutive, _ = make_consecutive_dataset(root)
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(
                    list_annotation_datasets(),
                    [{"name": regular.name, "path": str(regular.resolve())}],
                )
            self.assertTrue((consecutive / "consecutive_groups.json").is_file())

    def test_open_and_load_expose_every_image_label_pair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.gr.set_static_paths"),
            ):
                opened = open_annotation_dataset(str(dataset))
                loaded = load_annotation_sample(str(dataset), key)

            self.assertEqual(opened["sample_count"], 1)
            self.assertEqual(opened["samples"][0]["key"], key)
            self.assertEqual(Path(loaded["image_path"]).name, "frame-001.jpg")

    def test_load_prefers_dataset_owned_image_over_external_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            label_path = dataset / "canonical" / "labels" / key
            canonical_image = dataset / "canonical" / "images" / "performer-01" / "frame-001.jpg"
            external_image = root / "annotation_archive" / "frame-001.jpg"
            external_image.parent.mkdir()
            Image.new("RGB", (400, 200), "black").save(external_image)
            document = json.loads(label_path.read_text(encoding="utf-8"))
            document["source_image"] = str(external_image)
            label_path.write_text(json.dumps(document), encoding="utf-8")

            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.gr.set_static_paths") as set_static_paths,
            ):
                loaded = load_annotation_sample(str(dataset), key)

            self.assertEqual(Path(loaded["image_path"]), canonical_image.resolve())
            set_static_paths.assert_called_once_with(paths=[dataset / "canonical" / "images"])

    def test_review_status_uses_separate_mapping_without_changing_label(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            review_path = root / REVIEW_MAP_FILENAME
            label_path = dataset / "canonical" / "labels" / key
            label_before = label_path.read_bytes()
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review_path),
            ):
                result = set_annotation_sample_reviewed(str(dataset), key, "reviewer-1")
                opened = open_annotation_dataset(str(dataset))

            self.assertTrue(result["reviewed"])
            self.assertTrue(opened["samples"][0]["reviewed"])
            self.assertEqual(opened["reviewed_count"], 1)
            self.assertEqual(label_path.read_bytes(), label_before)
            self.assertFalse((dataset / REVIEW_MAP_FILENAME).exists())
            review_map = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review_map["datasets"]["review_set"]["samples"][key]["reviewer"], "reviewer-1")

    def test_label_change_invalidates_separate_review_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            review_path = root / REVIEW_MAP_FILENAME
            label_path = dataset / "canonical" / "labels" / key
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review_path),
            ):
                set_annotation_sample_reviewed(str(dataset), key, "reviewer-1")
                document = json.loads(label_path.read_text(encoding="utf-8"))
                document["notes"] = "changed after review"
                label_path.write_text(json.dumps(document), encoding="utf-8")
                opened = open_annotation_dataset(str(dataset))

            self.assertFalse(opened["samples"][0]["reviewed"])

    def test_bulk_review_rebinds_current_label_revision_without_changing_label(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            review_path = root / REVIEW_MAP_FILENAME
            label_path = dataset / "canonical" / "labels" / key
            label_before = label_path.read_bytes()
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review_path),
            ):
                first = set_all_annotation_samples_reviewed(str(dataset), "bulk-reviewer")
                second = set_all_annotation_samples_reviewed(str(dataset), "bulk-reviewer")
                opened = open_annotation_dataset(str(dataset))

            self.assertEqual(first["updated_count"], 1)
            self.assertEqual(second["updated_count"], 0)
            self.assertEqual(first["reviewed_count"], 1)
            self.assertEqual(opened["reviewed_count"], 1)
            self.assertEqual(label_path.read_bytes(), label_before)

    def test_saving_annotation_removes_prior_review_mapping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            review_path = root / REVIEW_MAP_FILENAME
            edit = {
                "yoyo_visibility": "visible",
                "trick_orientation": "normal",
                "yoyo_bbox_pixel": [100, 50, 140, 90],
                "string_visibility": "partial",
                "string_polylines_pixel": [[[10, 20], [30, 40]]],
                "string_review_status": "approved",
                "notes": "updated",
            }
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review_path),
            ):
                set_annotation_sample_reviewed(str(dataset), key, "reviewer-1")
                save_annotation_sample(str(dataset), key, edit)

            review_map = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertNotIn("review_set", review_map["datasets"])

    def test_save_preserves_original_pixel_coordinates_and_updates_999_space(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            edit = {
                "yoyo_visibility": "visible",
                "trick_orientation": "horizontal",
                "yoyo_bbox_pixel": [200, 25, 300, 75],
                "string_visibility": "visible",
                "string_polylines_pixel": [[[40, 20], [200, 100], [360, 180]]],
                "string_review_status": "approved",
                "bbox_review_status": "reviewed",
                "reviewer": "tester",
                "notes": "coordinate correction",
            }
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                result = save_annotation_sample(str(dataset), key, edit)

            annotation = result["annotation"]
            self.assertEqual(annotation["yoyo_bbox_pixel"], [200.0, 25.0, 300.0, 75.0])
            self.assertEqual(annotation["trick_orientation"], "horizontal")
            self.assertEqual(annotation["yoyo_bbox_2d"], [499.5, 124.875, 749.25, 374.625])
            self.assertEqual(annotation["string_polylines_2d"][0], [[99.9, 99.9], [499.5, 499.5], [899.1, 899.1]])
            self.assertIsNone(annotation["string_mask_polygons_pixel"])
            self.assertEqual(annotation["string_path"]["paths"][0]["edges"][0]["evidence"], "observed")
            self.assertEqual(annotation["workbench_edits"][-1]["actor"], "tester")
            self.assertIn("trick_orientation", annotation["workbench_edits"][-1]["fields"])

    def test_save_rejects_invalid_trick_orientation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            edit = {
                "yoyo_visibility": "visible",
                "trick_orientation": "unknown",
                "yoyo_bbox_pixel": [100, 50, 140, 90],
                "string_visibility": "partial",
                "string_polylines_pixel": [[[10, 20], [30, 40]]],
                "string_review_status": "approved",
            }
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                with self.assertRaisesRegex(ValueError, "invalid trick orientation"):
                    save_annotation_sample(str(dataset), key, edit)

    def test_save_rejects_geometry_outside_original_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = make_annotation_dataset(root)
            edit = {
                "yoyo_visibility": "visible",
                "trick_orientation": "normal",
                "yoyo_bbox_pixel": [10, 10, 401, 100],
                "string_visibility": "not_visible",
                "string_polylines_pixel": [],
                "string_review_status": "reviewed",
            }
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                with self.assertRaisesRegex(ValueError, "outside the image"):
                    save_annotation_sample(str(dataset), key, edit)


if __name__ == "__main__":
    unittest.main()
