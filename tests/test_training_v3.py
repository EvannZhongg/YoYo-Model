from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from training_v3.orientation_view import _crop_box
from training_v3.strip_pose_annotations import _digest_without_pose, strip_pose_fields
from video_tracking.rtmpose_backend import WholebodyPrediction, _rtmlib_device, hand_landmarks
from video_tracking.tracker import _predict_pose


class TrainingV3Tests(unittest.TestCase):
    def test_orientation_crop_ignores_hands_and_string_geometry(self):
        base = {"yoyo_bbox_pixel": [400, 300, 440, 340]}
        with_legacy_context = {
            **base,
            "hands_pixel": {"left": [10, 10], "right": [1900, 1000]},
            "string_polylines_pixel": [[[0, 0], [1919, 1079]]],
        }
        self.assertEqual(_crop_box(base, 1920, 1080), _crop_box(with_legacy_context, 1920, 1080))

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
        self.assertEqual(len(pose), 17)
        self.assertEqual(len(wrists), 2)
        self.assertEqual(len(wrists[0]["landmarks"]), 21)

    def test_pose_cleanup_changes_only_pose_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "labels" / "group" / "sample.json"
            target.parent.mkdir(parents=True)
            old = {"schema_version": "legacy", "yoyo_bbox_pixel": [1, 2, 3, 4], "string_visibility": "visible", "hands_pixel": {"left": [1, 2]}, "hands_2d": {"left": [3, 4]}, "pose": [{"x": 1}]}
            target.write_text(json.dumps(old), encoding="utf-8")
            before = _digest_without_pose(old)

            result = strip_pose_fields(root / "labels")
            current = json.loads(target.read_text(encoding="utf-8"))

            self.assertEqual(result["updated_label_count"], 1)
            self.assertEqual(_digest_without_pose(current), before)
            self.assertNotIn("hands_2d", current)
            self.assertNotIn("pose", current)
            self.assertEqual(current["yoyo_bbox_pixel"], [1, 2, 3, 4])
            self.assertEqual(current["string_visibility"], "visible")


if __name__ == "__main__":
    unittest.main()
