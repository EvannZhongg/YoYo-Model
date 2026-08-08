import copy
import importlib.util
import unittest
from collections import Counter
from pathlib import Path


PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "yoyo-string-annotation-vlm-assisted"
    / "scripts"
    / "annotation_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("annotation_pipeline_v5_test", PIPELINE_PATH)
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class AnnotationUpgradeV5Tests(unittest.TestCase):
    def test_upgrade_removes_only_deprecated_fields_and_adds_nothing(self):
        before = {
            "schema_version": PIPELINE.V4_SCHEMA_VERSION,
            "image_size": [100, 50],
            "visibility": "not_visible",
            "yoyo_bbox_pixel": None,
            "yoyo_bbox_2d": None,
            "bbox": [],
            "string_polylines_pixel": [[[0, 0], [100, 50]]],
            "string_polylines_2d": [[[0, 0], [1000, 1000]]],
            "string_polyline_pixel": [[0, 0], [100, 50]],
            "string_polyline_2d": [[0, 0], [1000, 1000]],
            "string_path": {
                "topology": "not_visible",
                "reconstruction_status": "not_visible",
                "paths": [{
                    "start_anchor": "left_hand",
                    "end_anchor": "unknown",
                    "points_pixel": [[0, 0], [100, 50]],
                    "points_2d": [[0, 0], [1000, 1000]],
                    "edges": [{"from": 0, "to": 1, "evidence": "reviewed", "confidence": 1.0}],
                }],
            },
            "dataset_management": {"split": "train"},
            "quality": {"history": [{"hands_2d": {"left": [1, 2]}}]},
        }
        original_keys = set(before)
        removals = Counter()

        after = PIPELINE._upgrade_v5_document(copy.deepcopy(before), removals)

        self.assertLessEqual(set(after), original_keys)
        self.assertEqual(after["schema_version"], PIPELINE.SCHEMA_VERSION)
        self.assertNotIn("dataset_management", after)
        self.assertNotIn("hands_2d", after["quality"]["history"][0])
        self.assertEqual(after["visibility"], "uncertain")
        self.assertEqual(after["string_polylines_2d"][0][-1], [999.0, 999.0])
        path = after["string_path"]["paths"][0]
        self.assertEqual(path["start_anchor"], "unknown")
        self.assertEqual(path["edges"][0]["evidence"], "observed")
        self.assertEqual(removals, {"dataset_management": 1, "hands_2d": 1})


if __name__ == "__main__":
    unittest.main()
