from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from training_v3.orientation_view import _crop_box
from training_v3.strip_pose_annotations import _content_digest, _digest_without_pose, strip_pose_fields
from video_tracking.rtmpose_backend import WholebodyPrediction, _rtmlib_device, hand_landmarks
from config import TRACKING_CONFIG
from video_tracking.tracker import _hash_run_inputs, _predict_pose, parse_args


class TrainingV3Tests(unittest.TestCase):
    def test_run_input_hashing_preserves_empty_optional_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"abc")

            result = _hash_run_inputs({"present": path, "disabled": None})

        self.assertEqual(
            result["present"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(result["disabled"], "")

    def test_tracking_cli_pose_defaults_to_config_and_supports_explicit_disable(self):
        self.assertFalse(TRACKING_CONFIG.enable_pose)
        self.assertEqual(parse_args(["input.mp4"]).pose, TRACKING_CONFIG.enable_pose)
        self.assertTrue(parse_args(["input.mp4", "--pose"]).pose)
        self.assertFalse(parse_args(["input.mp4", "--no-pose"]).pose)
        self.assertTrue(TRACKING_CONFIG.string_color_semantic_prefilter)
        self.assertTrue(parse_args(["input.mp4"]).string_color_semantic_prefilter)
        self.assertFalse(
            parse_args(["input.mp4", "--no-string-color-semantic-prefilter"])
            .string_color_semantic_prefilter
        )
        self.assertTrue(parse_args(["input.mp4"]).parallel_run_input_hashing)
        self.assertFalse(
            parse_args(["input.mp4", "--no-parallel-run-input-hashing"])
            .parallel_run_input_hashing
        )
        self.assertTrue(parse_args(["input.mp4"]).orientation_direct_inference)
        self.assertFalse(
            parse_args(["input.mp4", "--no-orientation-direct-inference"])
            .orientation_direct_inference
        )

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
        self.assertIsNone(metadata["box_confidence"])
        self.assertFalse(metadata["box_confidence_available"])
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

    def test_pose_cleanup_recurses_through_history_and_migrates_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "labels" / "group" / "sample.json"
            target.parent.mkdir(parents=True)
            previous = {
                "image_sha256": "a" * 64,
                "image_size": [1920, 1080],
                "source_group": "group",
                "visibility": "visible",
                "yoyo_bbox_pixel": [800, 700, 850, 750],
                "string_visibility": "visible",
                "string_polylines_pixel": [[[100, 200], [400, 500], [820, 710]]],
                "string_mask_polygons_pixel": None,
                "hands_pixel": {"left": [100, 200], "right": None},
                "yoyo_division": "1A",
                "scene_label": "trick",
                "trick_orientation": "normal",
                "string_path": {
                    "topology": "open",
                    "reconstruction_status": "complete",
                    "paths": [{
                        "path_id": "rope",
                        "start_anchor": "left_hand",
                        "end_anchor": "yoyo",
                        "points_pixel": [[100, 200], [400, 500], [820, 710]],
                        "edges": [],
                    }],
                    "unresolved_gaps": [],
                },
                "bad_case": [],
                "notes": "string remains unchanged",
            }
            current = json.loads(json.dumps(previous))
            current["hands_2d"] = {"left": [52, 185], "right": None}
            current["string_path"]["paths"][0]["start_anchor"] = "right_hand"
            before_digest = _content_digest(previous)
            current_digest = _content_digest(current)
            current.update({
                "schema_version": "agent_yoyo_string_annotation_v5",
                "quality": {
                    "history": [{
                        "before_sha256": before_digest,
                        "after_sha256": current_digest,
                        "previous_content": previous,
                    }],
                    "reviews": [{"decision": "approve", "content_sha256": current_digest}],
                },
            })
            target.write_text(json.dumps(current), encoding="utf-8")

            result = strip_pose_fields(root / "labels")
            cleaned = json.loads(target.read_text(encoding="utf-8"))
            historical = cleaned["quality"]["history"][0]["previous_content"]
            path = cleaned["string_path"]["paths"][0]

            self.assertNotIn("hands_pixel", historical)
            self.assertNotIn("hands_2d", cleaned)
            self.assertEqual(historical["string_path"]["paths"][0]["start_anchor"], "unknown")
            self.assertEqual(path["start_anchor"], "unknown")
            self.assertEqual(path["end_anchor"], "yoyo")
            self.assertEqual(cleaned["yoyo_bbox_pixel"], [800, 700, 850, 750])
            self.assertEqual(cleaned["string_polylines_pixel"], [[[100, 200], [400, 500], [820, 710]]])
            self.assertEqual(path["points_pixel"], [[100, 200], [400, 500], [820, 710]])
            self.assertEqual(cleaned["quality"]["reviews"][0]["content_sha256"], _content_digest(cleaned))
            self.assertEqual(cleaned["quality"]["history"][0]["after_sha256"], _content_digest(cleaned))
            self.assertEqual(result["removed_field_counts"]["hands_pixel"], 2)
            self.assertEqual(result["removed_field_counts"]["hands_2d"], 1)
            self.assertEqual(result["replaced_anchor_counts"], {"left_hand": 1, "right_hand": 1})


if __name__ == "__main__":
    unittest.main()
