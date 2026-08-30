import unittest
from unittest.mock import patch

import numpy as np

from workbench.preannotation import _draft_document


def _document() -> dict:
    return {
        "schema_version": "agent_yoyo_string_annotation_v5",
        "image_size": [200, 100],
        "active_yoyo": {
            "visibility": "uncertain",
            "not_visible_reason": None,
            "trick_orientation": "normal",
            "presentation_orientation": "frontal",
            "bbox_pixel": None,
            "bbox_2d": None,
            "bbox_review_status": "needs_review",
        },
        "backup_yoyos": [],
        "string_visibility": "uncertain",
        "reviewed_at_utc": "2026-01-01T00:00:00+00:00",
        "reviewer": "previous-reviewer",
        "string_polylines_pixel": None,
        "string_polylines_2d": None,
        "string_polyline_pixel": None,
        "string_polyline_2d": None,
        "string_mask_polygons_pixel": None,
        "string_path": {
            "topology": "uncertain",
            "reconstruction_status": "uncertain",
            "paths": [],
            "unresolved_gaps": [],
        },
        "quality": {"revision": 0, "min_model_approvals": 1, "history": [], "reviews": []},
        "workbench_edits": [],
    }


class PreannotationTests(unittest.TestCase):
    def test_draft_uses_current_fields_and_inferred_path_evidence(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = [{"class_name": "yoyo", "confidence": 0.9, "bbox": [20, 30, 60, 70]}]
        with patch(
            "workbench.preannotation._predict_string_model",
            return_value={"polylines": [[[5, 10], [15, 20]]]},
        ):
            result = _draft_document(_document(), image, detections, object(), None, "cpu")

        self.assertNotIn("preannotation", result)
        self.assertEqual(result["active_yoyo"]["bbox_pixel"], [20.0, 30.0, 60.0, 70.0])
        self.assertEqual(result["string_review_status"], "needs_review")
        self.assertIsNone(result["reviewed_at_utc"])
        self.assertIsNone(result["reviewer"])
        self.assertEqual(result["string_path"]["paths"][0]["path_id"], "model-line-1")
        self.assertEqual(result["string_path"]["paths"][0]["edges"][0]["evidence"], "inferred")
        self.assertEqual(result["workbench_edits"][-1]["actor"], "workbench-preannotator")


if __name__ == "__main__":
    unittest.main()
