from __future__ import annotations

import inspect
import unittest

import numpy as np

from training_v3.orientation_view import _crop_box
from video_tracking.rtmpose_backend import WholebodyPrediction, _rtmlib_device, hand_landmarks
from config import TRACKING_CONFIG
from video_tracking.tracker import _predict_pose, parse_args
from string_segmentation.evaluate_consecutive import evaluate_consecutive_checkpoint


class TrainingV3Tests(unittest.TestCase):
    def test_tracking_cli_pose_defaults_to_config_and_supports_explicit_disable(self):
        self.assertFalse(TRACKING_CONFIG.enable_pose)
        self.assertEqual(parse_args(["input.mp4"]).pose, TRACKING_CONFIG.enable_pose)
        self.assertTrue(parse_args(["input.mp4", "--pose"]).pose)
        self.assertFalse(parse_args(["input.mp4", "--no-pose"]).pose)
        self.assertTrue(TRACKING_CONFIG.string_color_semantic_prefilter)
        self.assertTrue(parse_args(["input.mp4"]).string_color_semantic_prefilter)
        self.assertEqual(TRACKING_CONFIG.string_max_components, 32)
        self.assertEqual(parse_args(["input.mp4"]).string_max_components, 32)
        self.assertTrue(TRACKING_CONFIG.string_cuda_graph)
        self.assertTrue(parse_args(["input.mp4"]).string_cuda_graph)
        self.assertAlmostEqual(TRACKING_CONFIG.string_inference_scale, 1.125)
        self.assertAlmostEqual(parse_args(["input.mp4"]).string_inference_scale, 1.125)
        self.assertFalse(parse_args(["input.mp4", "--no-string-cuda-graph"]).string_cuda_graph)
        self.assertEqual(
            parse_args(["input.mp4", "--string-max-components", "5"]).string_max_components,
            5,
        )
        self.assertEqual(
            inspect.signature(evaluate_consecutive_checkpoint)
            .parameters["max_components"].default,
            TRACKING_CONFIG.string_max_components,
        )
        self.assertFalse(
            parse_args(["input.mp4", "--no-string-color-semantic-prefilter"])
            .string_color_semantic_prefilter
        )
        self.assertTrue(parse_args(["input.mp4"]).orientation_direct_inference)
        self.assertFalse(
            parse_args(["input.mp4", "--no-orientation-direct-inference"])
            .orientation_direct_inference
        )

    def test_orientation_crop_uses_only_yoyo_box(self):
        base = {"yoyo_bbox_pixel": [400, 300, 440, 340]}
        with_extra_context = {
            **base,
            "hands_pixel": {"left": [10, 10], "right": [1900, 1000]},
            "string_polylines_pixel": [[[0, 0], [1919, 1079]]],
        }
        self.assertEqual(_crop_box(base, 1920, 1080), _crop_box(with_extra_context, 1920, 1080))

    def test_no_yoyo_orientation_crop_is_deterministic_center_negative(self):
        self.assertEqual(_crop_box({}, 1000, 800), (388, 288, 612, 512))

    def test_hand_landmarks_uses_coco_wholebody_ranges(self):
        points = np.arange(133 * 2, dtype=np.float32).reshape(133, 2)
        scores = np.ones(133, dtype=np.float32)
        left = hand_landmarks(points, scores, "left")
        right = hand_landmarks(points, scores, "right")
        self.assertEqual((left[0]["global_index"], left[-1]["global_index"]), (91, 111))
        self.assertEqual((right[0]["global_index"], right[-1]["global_index"]), (112, 132))

    def test_device_mapping_accepts_cli_cuda_forms(self):
        self.assertEqual(_rtmlib_device("0"), "cuda")
        self.assertEqual(_rtmlib_device("cuda:0"), "cuda")
        self.assertEqual(_rtmlib_device("cpu"), "cpu")

    def test_tracker_exports_body_pose_and_detailed_hand_landmarks(self):
        class FakeModel:
            backend_name = "rtmpose-m_wholebody_onnx"
            keypoint_schema = "coco_wholebody_133"

            def predict(self, frame):
                points = np.zeros((1, 133, 2), dtype=np.float32)
                scores = np.full((1, 133), 0.9, dtype=np.float32)
                points[0, :, 0] = np.arange(133)
                points[0, :, 1] = np.arange(133) + 10
                return WholebodyPrediction(points, scores, np.array([[0, 0, 200, 200]], dtype=np.float32))

        wrists, pose, metadata = _predict_pose(FakeModel(), np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["wholebody_keypoint_count"], 133)
        self.assertIsNone(metadata["box_confidence"])
        self.assertFalse(metadata["box_confidence_available"])
        self.assertEqual(len(pose), 17)
        self.assertEqual(len(wrists), 2)
        self.assertEqual(len(wrists[0]["landmarks"]), 21)

if __name__ == "__main__":
    unittest.main()
