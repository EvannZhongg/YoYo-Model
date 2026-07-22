import unittest

from video_tracking.tracker import _is_trick_active, _pick_yoyo


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

    def test_activity_threshold_is_not_multiplied_by_fps(self):
        yoyo = detection([100, 100, 180, 180])
        # 0.08 * sqrt(3840^2 + 2160^2) is about 352 px/s.
        self.assertTrue(_is_trick_active(yoyo, 400.0, None, 3840, 2160, 0.08))
        self.assertFalse(_is_trick_active(yoyo, 200.0, None, 3840, 2160, 0.08))


if __name__ == "__main__":
    unittest.main()
