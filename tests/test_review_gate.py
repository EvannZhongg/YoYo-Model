import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.review import update_annotation_status, validate_review_gate
from annotation.video_frame_annotator import _visualization_review_labels


class ReviewGateTests(unittest.TestCase):
    def test_visualization_banner_distinguishes_reviewed_string_mask_from_proposal(self):
        reviewed = {
            "bbox_review_status": "auto_labeled_needs_review",
            "string_review_status": "reviewed",
        }
        pending = {
            "bbox_review_status": "auto_labeled_needs_review",
            "string_review_status": "auto_labeled_needs_review",
        }

        self.assertEqual(
            _visualization_review_labels(reviewed, True),
            ("COMPONENT REVIEW", "REVIEWED STRING MASK"),
        )
        self.assertEqual(
            _visualization_review_labels(pending, True),
            ("VLM REVIEW ONLY", "COLOR MASK PROPOSAL | STRING REVIEW REQUIRED"),
        )

    def test_review_update_refreshes_visualization_with_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            label = dataset / "annotations" / "labels" / "train" / "source" / "frame.json"
            image = dataset / "frame.jpg"
            label.parent.mkdir(parents=True)
            image.write_bytes(b"placeholder")
            label.write_text(
                json.dumps(
                    {
                        "source_image": str(image),
                        "review_status": "partially_reviewed",
                        "bbox_review_status": "auto_labeled_needs_review",
                        "string_review_status": "reviewed",
                        "visibility": "visible",
                        "yoyo_bbox_pixel": [1, 2, 10, 12],
                        "string_visibility": "not_visible",
                    }
                ),
                encoding="utf-8",
            )

            with patch("annotation.video_frame_annotator.draw_visualization") as draw:
                update_annotation_status(label, "approved", component="bbox")

            drawn_data = draw.call_args.args[1]
            output_path = draw.call_args.args[2]
            self.assertEqual(drawn_data["bbox_review_status"], "approved")
            self.assertEqual(drawn_data["review_status"], "reviewed")
            self.assertEqual(
                output_path,
                dataset / "annotations" / "visualizations" / "train" / "source" / "frame_vis.jpg",
            )

    def test_review_update_appends_dataset_audit_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            label = dataset / "annotations" / "labels" / "train" / "source" / "frame.json"
            label.parent.mkdir(parents=True)
            label.write_text(
                json.dumps(
                    {
                        "review_status": "auto_labeled_needs_review",
                        "bbox_review_status": "auto_labeled_needs_review",
                        "string_review_status": "auto_labeled_needs_review",
                        "visibility": "visible",
                        "yoyo_bbox_pixel": [1, 2, 10, 12],
                        "string_visibility": "visible",
                        "string_polylines_pixel": [[[1, 2], [3, 4]]],
                    }
                ),
                encoding="utf-8",
            )

            update_annotation_status(label, "reviewed", reviewer="tester", notes="checked", component="string")

            event = json.loads((dataset / "manual_review_log.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["component"], "string")
            self.assertEqual(event["status"], "reviewed")
            self.assertEqual(event["reviewer"], "tester")
            self.assertEqual(event["reason"], "checked")

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

    def test_unresolved_is_audited_without_becoming_training_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            path = dataset / "annotations" / "labels" / "train" / "source" / "frame.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "review_status": "auto_labeled_needs_review",
                "bbox_review_status": "reviewed",
                "string_review_status": "auto_labeled_needs_review",
                "visibility": "visible",
                "yoyo_bbox_pixel": [10, 20, 40, 60],
                "string_visibility": "uncertain",
            }), encoding="utf-8")
            saved = update_annotation_status(
                path,
                "unresolved",
                reviewer="geometry-critic",
                notes="Motion blur prevents a defensible centerline.",
                component="string",
            )
            event = json.loads((dataset / "manual_review_log.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(saved["string_review_status"], "unresolved")
        self.assertEqual(saved["review_status"], "partially_reviewed")
        self.assertEqual(event["status"], "unresolved")
        self.assertIn("Motion blur", event["reason"])


if __name__ == "__main__":
    unittest.main()
