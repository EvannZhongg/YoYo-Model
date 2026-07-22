import json
import tempfile
import unittest
from pathlib import Path

from annotation.review import update_annotation_status, validate_review_gate


class ReviewGateTests(unittest.TestCase):
    def _label(self, root: Path, **overrides) -> Path:
        data = {
            "visibility": "visible",
            "yoyo_bbox_pixel": [10, 20, 40, 60],
            "bbox": [{"bbox_pixel": [10, 20, 40, 60]}],
            "string_visibility": "visible",
            "string_polylines_pixel": [[[12, 22], [30, 50]]],
            "bbox_review_status": "auto_labeled_needs_review",
            "string_review_status": "auto_labeled_needs_review",
            "review_status": "auto_labeled_needs_review",
            "bad_case": [],
        }
        data.update(overrides)
        path = root / "label.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_visible_yoyo_without_bbox_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._label(Path(tmp), yoyo_bbox_pixel=None, bbox=[])
            with self.assertRaisesRegex(ValueError, "requires a valid bbox"):
                update_annotation_status(path, "approved", component="bbox")
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["bbox_review_status"], "auto_labeled_needs_review")

    def test_visible_string_without_geometry_cannot_be_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._label(Path(tmp), string_polylines_pixel=None)
            with self.assertRaisesRegex(ValueError, "requires a reviewed stroke or mask"):
                update_annotation_status(path, "reviewed", component="string")

    def test_not_visible_clears_old_string_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._label(Path(tmp))
            saved = update_annotation_status(
                path,
                "approved",
                component="string",
                string_visibility="not_visible",
            )
        self.assertEqual(saved["string_visibility"], "not_visible")
        self.assertNotIn("string_polylines_pixel", saved)
        self.assertIn("string_not_visible", saved["bad_case"])

    def test_scene_label_is_persisted_with_auditable_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._label(Path(tmp))
            saved = update_annotation_status(
                path,
                "approved",
                component="bbox",
                yoyo_visibility="visible",
                scene_label="non_trick",
            )
        self.assertEqual(saved["scene_label"], "non_trick")
        self.assertIn("non_trick_scene", saved["bad_case"])
        self.assertEqual(validate_review_gate(saved, "bbox"), [])


if __name__ == "__main__":
    unittest.main()
