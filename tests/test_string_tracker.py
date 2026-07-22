import unittest

import cv2
import numpy as np

from video_tracking.string_tracker import _color_line_observation, estimate_string, propagate_optical_flow


class StringTrackerTemporalTests(unittest.TestCase):
    def _shifted_frames(self):
        height, width = 180, 240
        previous = np.zeros((height, width, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        cv2.line(previous, (35, 90), (145, 90), (255, 255, 255), 3)
        cv2.line(current, (41, 90), (151, 90), (255, 255, 255), 3)
        for x in range(35, 146, 10):
            cv2.circle(previous, (x, 90), 2, (180, 180, 180), -1)
        for x in range(41, 152, 10):
            cv2.circle(current, (x, 90), 2, (180, 180, 180), -1)
        return previous, current

    def test_optical_flow_has_forward_backward_gate(self):
        previous, current = self._shifted_frames()
        result = propagate_optical_flow(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            [[35, 90], [145, 90]],
            current.shape[1],
            current.shape[0],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "lucas_kanade_optical_flow")
        self.assertIn("flow_forward_backward_error", result)
        self.assertLess(result["flow_forward_backward_error"], 4.0)

    def test_observation_and_flow_are_fused(self):
        previous, current = self._shifted_frames()
        previous_string = {
            "points": [[35, 90], [145, 90]],
            "confidence": 0.62,
            "method": "color_hough_observation",
            "propagation_age_frames": 0,
        }
        result = estimate_string(
            current,
            {"center": [41, 90], "bbox": [30, 80, 52, 102]},
            [],
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            previous_string,
            observation={
                "points": [[41, 90], [151, 90]],
                "confidence": 0.80,
                "method": "yolo_segmentation",
                "needs_review": False,
            },
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "temporal_fusion")
        self.assertEqual(result["propagation_age_frames"], 0)
        self.assertIn("yolo_segmentation", result["source_methods"])

    def test_string_can_persist_without_yoyo_as_review_case(self):
        previous, current = self._shifted_frames()
        result = estimate_string(
            current,
            None,
            [],
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            {
                "points": [[35, 90], [145, 90]],
                "confidence": 0.62,
                "method": "color_hough_observation",
                "propagation_age_frames": 0,
            },
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "lucas_kanade_optical_flow")
        self.assertEqual(result["propagation_age_frames"], 1)

    def test_unanchored_semantic_string_is_suppressed_without_yoyo(self):
        result = estimate_string(
            np.zeros((180, 240, 3), dtype=np.uint8),
            None,
            [],
            None,
            None,
            observation={
                "points": [[20.0, 20.0], [100.0, 100.0]],
                "confidence": 0.9,
                "method": "semantic_segmentation",
            },
        )
        self.assertIsNone(result)

    def test_far_frame_edge_color_line_is_not_a_trusted_unknown_anchor(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.line(frame, (200, 1055), (1000, 1055), (0, 255, 0), 4)
        yoyo = {"center": [960, 540], "bbox": [920, 500, 1000, 580]}
        unknown = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=False,
            mark_far_ambiguous=True,
        )
        self.assertIsNotNone(unknown)
        self.assertTrue(unknown["spatially_ambiguous"])
        self.assertLessEqual(unknown["confidence"], 0.24)

        detached = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=False,
            mark_far_ambiguous=False,
        )
        self.assertIsNotNone(detached)
        self.assertFalse(detached["spatially_ambiguous"])


if __name__ == "__main__":
    unittest.main()
