import unittest
from types import SimpleNamespace

import numpy as np

from video_tracking.tracker import (
    _assign_tracker_ids,
    _carry_preferred_track_id,
    _pick_yoyo,
    _semantic_inference_parameters,
)


def detection(bbox, confidence=0.9):
    return {
        "bbox": bbox,
        "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        "confidence": confidence,
        "class_name": "yoyo",
        "class_id": 0,
    }


class YoyoSelectionTests(unittest.TestCase):
    def test_overlapping_predictions_are_one_yoyo(self):
        selected, flags = _pick_yoyo(
            [
                detection([100, 100, 200, 200], 0.95),
                detection([90, 90, 210, 210], 0.4),
            ]
        )
        self.assertEqual(selected["confidence"], 0.95)
        self.assertNotIn("multiple_yoyo", flags)

    def test_spatially_distinct_predictions_are_multiple_yoyo(self):
        _, flags = _pick_yoyo(
            [
                detection([100, 100, 180, 180]),
                detection([300, 250, 380, 330]),
            ]
        )
        self.assertIn("multiple_yoyo", flags)

    def test_tracker_ids_are_matched_by_geometry_when_output_order_changes(self):
        detections = [
            detection([10, 10, 30, 30]),
            detection([100, 100, 140, 140]),
        ]
        tracked = SimpleNamespace(
            xyxy=np.asarray([[100, 100, 140, 140], [10, 10, 30, 30]], dtype=np.float32),
            class_id=np.asarray([0, 0], dtype=np.int32),
            tracker_id=np.asarray([22, 11], dtype=np.int32),
        )

        _assign_tracker_ids(detections, tracked)

        self.assertEqual(detections[0]["track_id"], 11)
        self.assertEqual(detections[1]["track_id"], 22)

    def test_selection_prefers_the_existing_track_over_a_confidence_spike(self):
        stable = detection([100, 100, 180, 180], 0.65)
        stable["track_id"] = 7
        distractor = detection([300, 250, 380, 330], 0.98)
        distractor["track_id"] = 9

        selected, flags = _pick_yoyo([distractor, stable], preferred_track_id=7)

        self.assertEqual(selected["track_id"], 7)
        self.assertIn("multiple_yoyo", flags)

    def test_short_spatially_continuous_gap_carries_selected_track_id(self):
        candidate = detection([115, 110, 215, 210], 0.8)

        carried = _carry_preferred_track_id(
            candidate,
            preferred_track_id=7,
            previous_bbox=[100, 100, 200, 200],
            gap_frames=2,
            max_gap_frames=12,
            multiple_yoyo=False,
        )

        self.assertTrue(carried)
        self.assertEqual(candidate["track_id"], 7)
        self.assertEqual(candidate["track_id_source"], "temporal_carry")

    def test_track_id_carry_rejects_ambiguous_or_distant_detection(self):
        distant = detection([1000, 1000, 1100, 1100], 0.8)
        ambiguous = detection([110, 110, 210, 210], 0.8)

        self.assertFalse(
            _carry_preferred_track_id(distant, 7, [100, 100, 200, 200], 2, 12, multiple_yoyo=False)
        )
        self.assertFalse(
            _carry_preferred_track_id(ambiguous, 7, [100, 100, 200, 200], 2, 12, multiple_yoyo=True)
        )

    def test_semantic_inference_scale_preserves_stride_and_area_filter(self):
        width, height, area_scale = _semantic_inference_parameters(
            {"input_width": 960, "input_height": 544},
            2.0,
        )
        self.assertEqual((width, height), (1920, 1088))
        self.assertEqual(area_scale, 4.0)
        with self.assertRaisesRegex(ValueError, "between 0.5 and 2.0"):
            _semantic_inference_parameters({"input_width": 960, "input_height": 544}, 2.5)


if __name__ == "__main__":
    unittest.main()
