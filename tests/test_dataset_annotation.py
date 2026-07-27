import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from workbench.dataset_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    dataset_annotation_component_kwargs,
    list_annotation_datasets,
    load_annotation_sample,
    open_annotation_dataset,
    save_annotation_sample,
)


class DatasetAnnotationWorkbenchTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, str]:
        dataset = root / "review_set"
        group = "performer-01"
        image_path = dataset / "canonical" / "images" / group / "frame-001.jpg"
        label_path = dataset / "canonical" / "labels" / group / "frame-001.json"
        image_path.parent.mkdir(parents=True)
        label_path.parent.mkdir(parents=True)
        Image.new("RGB", (400, 200), "white").save(image_path)
        label_path.write_text(
            json.dumps(
                {
                    "schema_version": ANNOTATION_SCHEMA_VERSION,
                    "source_image": "../../images/performer-01/frame-001.jpg",
                    "image_size": [400, 200],
                    "source_group": group,
                    "frame_index": 12,
                    "visibility": "visible",
                    "yoyo_bbox_pixel": [100, 50, 140, 90],
                    "string_visibility": "partial",
                    "string_polylines_pixel": [[[10, 20], [30, 40]]],
                    "string_review_status": "unresolved",
                    "string_path": {"topology": "single_path", "paths": []},
                }
            ),
            encoding="utf-8",
        )
        return dataset, f"{group}/frame-001.json"

    def test_lists_only_current_json_annotation_datasets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self._dataset(root)
            derived = root / "derived"
            (derived / "labels").mkdir(parents=True)
            (derived / "images").mkdir()
            (derived / "labels" / "frame.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(list_annotation_datasets(), [{"name": "review_set", "path": str(dataset.resolve())}])

    def test_component_exposes_redraw_precision_and_packed_server_requests(self):
        component = dataset_annotation_component_kwargs()
        self.assertIn("重绘绳线", component["value"])
        self.assertIn('data-tool="box"', component["value"])
        self.assertIn("canvasPoint(event)", component["js_on_load"])
        self.assertIn("contextmenu", component["js_on_load"])
        self.assertIn("ui_save_annotation_sample({", component["js_on_load"])

    def test_open_and_load_expose_every_image_label_pair(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = self._dataset(root)
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.gr.set_static_paths"),
            ):
                opened = open_annotation_dataset(str(dataset))
                loaded = load_annotation_sample(str(dataset), key)

            self.assertEqual(opened["sample_count"], 1)
            self.assertEqual(opened["samples"][0]["key"], key)
            self.assertEqual(Path(loaded["image_path"]).name, "frame-001.jpg")

    def test_save_preserves_original_pixel_coordinates_and_updates_1000_space(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = self._dataset(root)
            edit = {
                "yoyo_visibility": "visible",
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
            self.assertEqual(annotation["yoyo_bbox_2d"], [500.0, 125.0, 750.0, 375.0])
            self.assertEqual(annotation["string_polylines_2d"][0], [[100.0, 100.0], [500.0, 500.0], [900.0, 900.0]])
            self.assertIsNone(annotation["string_mask_polygons_pixel"])
            self.assertEqual(annotation["workbench_edits"][-1]["actor"], "tester")

    def test_save_rejects_geometry_outside_original_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, key = self._dataset(root)
            edit = {
                "yoyo_visibility": "visible",
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
