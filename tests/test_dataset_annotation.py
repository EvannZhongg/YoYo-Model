import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from helpers import make_annotation_dataset, make_consecutive_dataset
from workbench.dataset_annotation import (
    REVIEW_MAP_FILENAME,
    dataset_annotation_component_kwargs,
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

    def test_component_exposes_redraw_precision_and_packed_server_requests(self):
        component = dataset_annotation_component_kwargs()
        self.assertIn("重绘绳线", component["value"])
        self.assertIn('data-tool="box"', component["value"])
        self.assertIn("canvasPoint(event)", component["js_on_load"])
        self.assertIn("contextmenu", component["js_on_load"])
        self.assertIn("function deleteSelectedPoint", component["js_on_load"])
        self.assertIn('event.key!=="Delete"', component["js_on_load"])
        self.assertIn("line.splice(selected.point,1)", component["js_on_load"])
        self.assertIn("已删除标注点并连接相邻点", component["js_on_load"])
        self.assertIn("ui_save_annotation_sample({", component["js_on_load"])
        self.assertIn('value="partially_visible">部分可见', component["value"])
        self.assertIn('value="out_of_frame">画面外', component["value"])
        self.assertIn('value="absent">不存在', component["value"])
        self.assertIn('id="yda-trick-orientation"', component["value"])
        self.assertNotIn('id="yda-item-name"', component["value"])
        self.assertNotIn('id="yda-item-group"', component["value"])
        self.assertIn('id="yda-dirty" role="status"', component["value"])
        self.assertIn('value="horizontal">水平（horizontal）', component["value"])
        self.assertIn('trick_orientation:$("#yda-trick-orientation").value', component["js_on_load"])
        self.assertIn('trickOrientation:$("#yda-trick-orientation").value', component["js_on_load"])
        self.assertIn('$("#yda-trick-orientation").value=annotation.trick_orientation', component["js_on_load"])
        self.assertIn("ui_set_annotation_sample_reviewed({", component["js_on_load"])
        self.assertIn('id="yda-toggle-annotations"', component["value"])
        self.assertIn('addEventListener("wheel"', component["js_on_load"])
        self.assertIn("toggleAnnotations()", component["js_on_load"])
        self.assertIn("await selectSample(state.samples[index+1].key)", component["js_on_load"])
        self.assertIn('id="yda-add-line"', component["value"])
        self.assertIn("function startNewLine", component["js_on_load"])
        self.assertIn("startNewLine(false)", component["js_on_load"])
        self.assertIn("event.stopPropagation()", component["js_on_load"])
        self.assertIn("capture:true", component["js_on_load"])
        self.assertIn('id="yda-reset"', component["value"])
        self.assertIn("function resetUnsavedChanges", component["js_on_load"])
        self.assertIn("state.baseline=editorSnapshot()", component["js_on_load"])
        self.assertIn("function refreshDatasetOptions", component["js_on_load"])
        self.assertIn("syncDatasetChoice(result.dataset_path)", component["js_on_load"])
        self.assertIn('["pointerenter","focus","pointerdown"]', component["js_on_load"])

    def test_canvas_zoom_is_isolated_from_fixed_sidebars(self):
        component = dataset_annotation_component_kwargs()
        html = component["value"]
        css = component["css_template"]
        javascript = component["js_on_load"]

        self.assertIn('class="yda__canvas-layer"', html)
        self.assertIn("grid-template-columns:240px minmax(0,1fr) 330px", css)
        self.assertIn("grid-template-columns:210px minmax(0,1fr) 300px", css)
        self.assertIn("height:clamp(700px,calc(100dvh - 100px),960px)", css)
        self.assertIn(".yda__editor-scroll { overflow:visible; padding-right:0; }", css)
        self.assertIn(".yda__canvas-layer", css)
        self.assertIn("contain:size layout paint", css)
        self.assertIn('$("#yda-viewport").addEventListener("wheel"', javascript)
        self.assertIn("event.preventDefault()", javascript)

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

    def test_bulk_review_rebinds_current_label_hash_without_changing_label(self):
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

    def test_save_preserves_original_pixel_coordinates_and_updates_1000_space(self):
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
            self.assertEqual(annotation["yoyo_bbox_2d"], [500.0, 125.0, 750.0, 375.0])
            self.assertEqual(annotation["string_polylines_2d"][0], [[100.0, 100.0], [500.0, 500.0], [900.0, 900.0]])
            self.assertIsNone(annotation["string_mask_polygons_pixel"])
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
