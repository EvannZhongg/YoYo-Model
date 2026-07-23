import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_dataset.hard_negative_candidates import (
    _parse_offsets,
    build_neighbor_candidates,
    promote_reviewed_negatives,
)


class HardNegativeCandidateTests(unittest.TestCase):
    def test_extracts_only_untracked_train_neighbors_without_mutating_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"video placeholder")
            sources = {
                "current_action_group": "1A",
                "sources": [{
                    "video_id": "train-a",
                    "source_group": "train-a",
                    "split": "train",
                    "path": str(video),
                    "sha256": "abc",
                    "fps": 5.0,
                }],
            }
            (root / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
            existing = {"video_id": "train-a", "frame_index": 3}
            frames_path = root / "frames.jsonl"
            frames_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")
            queue = {
                "rows": [
                    {
                        "queue_rank": 1,
                        "video_id": "train-a",
                        "split": "train",
                        "frame_index": 4,
                        "yoyo_visibility": "absent",
                        "false_positive": True,
                        "model": {"predicted_pixels": 80},
                    },
                    {
                        "queue_rank": 2,
                        "video_id": "train-a",
                        "split": "train",
                        "frame_index": 8,
                        "yoyo_visibility": "visible",
                        "false_positive": True,
                        "model": {"predicted_pixels": 90},
                    },
                    {
                        "queue_rank": 3,
                        "video_id": "train-a",
                        "split": "train",
                        "frame_index": 12,
                        "yoyo_visibility": "absent",
                        "false_positive": False,
                        "model": {"predicted_pixels": 0},
                    },
                ]
            }
            queue_path = root / "queue.json"
            queue_path.write_text(json.dumps(queue), encoding="utf-8")
            before = frames_path.read_bytes()
            frame = np.full((32, 64, 3), 90, dtype=np.uint8)

            with patch("video_dataset.hard_negative_candidates._read_video_frame", return_value=frame):
                result = build_neighbor_candidates(
                    root,
                    queue_path,
                    offsets_seconds=[-0.2, 0.2],
                    output_name="neighbors",
                )
            payload = json.loads((root / "neighbors.json").read_text(encoding="utf-8"))
            after = frames_path.read_bytes()

        self.assertEqual(result["count"], 1)
        self.assertEqual(payload["rows"][0]["frame_index"], 5)
        self.assertEqual(payload["rows"][0]["split"], "train")
        self.assertIn("REVIEW ONLY", payload["policy"])
        self.assertTrue(payload["selection"]["require_false_positive"])
        self.assertEqual(after, before)

    def test_clean_negative_anchors_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"video placeholder")
            (root / "sources.json").write_text(json.dumps({
                "sources": [{
                    "video_id": "train-a",
                    "source_group": "train-a",
                    "split": "train",
                    "path": str(video),
                    "sha256": "abc",
                    "fps": 5.0,
                }],
            }), encoding="utf-8")
            (root / "frames.jsonl").write_text("", encoding="utf-8")
            queue_path = root / "queue.json"
            queue_path.write_text(json.dumps({"rows": [{
                "queue_rank": 1,
                "video_id": "train-a",
                "split": "train",
                "frame_index": 10,
                "yoyo_visibility": "absent",
                "false_positive": False,
                "model": {"predicted_pixels": 0},
            }]}), encoding="utf-8")
            frame = np.full((32, 64, 3), 90, dtype=np.uint8)

            with patch("video_dataset.hard_negative_candidates._read_video_frame", return_value=frame):
                default_result = build_neighbor_candidates(
                    root,
                    queue_path,
                    offsets_seconds=[0.2],
                    output_name="default_neighbors",
                )
                expanded_result = build_neighbor_candidates(
                    root,
                    queue_path,
                    offsets_seconds=[0.2],
                    output_name="expanded_neighbors",
                    require_false_positive=False,
                )
            expanded = json.loads((root / "expanded_neighbors.json").read_text(encoding="utf-8"))

        self.assertEqual(default_result["count"], 0)
        self.assertEqual(expanded_result["count"], 1)
        self.assertFalse(expanded["selection"]["require_false_positive"])

    def test_excluded_source_group_is_absent_from_neighbor_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"video placeholder")
            (root / "sources.json").write_text(json.dumps({"sources": [{
                "video_id": "ab03bb7118b0",
                "source_group": "ab03bb7118b0",
                "split": "train",
                "path": str(video),
                "sha256": "abc",
                "fps": 5.0,
            }]}), encoding="utf-8")
            (root / "frames.jsonl").write_text("", encoding="utf-8")
            queue_path = root / "queue.json"
            queue_path.write_text(json.dumps({"rows": [{
                "queue_rank": 1,
                "video_id": "ab03bb7118b0",
                "source_group": "ab03bb7118b0",
                "split": "train",
                "frame_index": 10,
                "yoyo_visibility": "absent",
                "false_positive": True,
                "model": {"predicted_pixels": 80},
            }]}), encoding="utf-8")
            with patch("video_dataset.hard_negative_candidates._read_video_frame") as reader:
                result = build_neighbor_candidates(
                    root, queue_path, offsets_seconds=[0.2], exclude_source_groups="ab03bb7118b0"
                )
        self.assertEqual(result["count"], 0)
        reader.assert_not_called()

    def test_offset_parser_rejects_only_zero(self):
        self.assertEqual(_parse_offsets("1,-0.5,1"), [-0.5, 1.0])
        with self.assertRaises(ValueError):
            _parse_offsets("0")

    def test_explicit_promotion_creates_reviewed_string_negative_and_audit_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            source_image = root / "candidate.jpg"
            import cv2

            cv2.imwrite(str(source_image), np.full((32, 64, 3), 90, dtype=np.uint8))
            (root / "sources.json").write_text(json.dumps({
                "current_action_group": "1A",
                "sources": [{
                    "video_id": "train-a",
                    "source_group": "train-a",
                    "split": "train",
                    "path": str(video),
                    "sha256": "abc",
                    "action_group": "1A",
                }],
            }), encoding="utf-8")
            (root / "frames.jsonl").write_text("", encoding="utf-8")
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps({"rows": [
                {
                    "candidate_image": str(source_image),
                    "source_video": str(video),
                    "source_video_sha256": "abc",
                    "video_id": "train-a",
                    "source_group": "train-a",
                    "split": "train",
                    "frame_index": 25,
                    "timestamp_s": 0.5,
                },
                {
                    "candidate_image": str(source_image),
                    "source_video": str(video),
                    "source_video_sha256": "abc",
                    "video_id": "train-a",
                    "source_group": "train-a",
                    "split": "train",
                    "frame_index": 30,
                    "timestamp_s": 0.6,
                },
            ]}), encoding="utf-8")

            result = promote_reviewed_negatives(
                root,
                candidates,
                {("train-a", 25), ("train-a", 30)},
                reviewer="tester",
                reason="Confirmed no visible string.",
            )
            label_path = root / "annotations" / "labels" / "train" / "train-a" / "frame_00000025.json"
            second_label = root / "annotations" / "labels" / "train" / "train-a" / "frame_00000030.json"
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            second_exists = second_label.exists()
            updated_candidates = json.loads(candidates.read_text(encoding="utf-8"))
            frame_rows = [json.loads(line) for line in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]
            review_log = (root / "manual_review_log.jsonl").read_text(encoding="utf-8")

        self.assertEqual(result["count"], 2)
        self.assertEqual(len({row["label_path"] for row in result["promoted"]}), 2)
        self.assertTrue(second_exists)
        self.assertEqual(annotation["string_review_status"], "reviewed")
        self.assertEqual(annotation["string_visibility"], "not_visible")
        self.assertEqual(annotation["review_status"], "partially_reviewed")
        self.assertEqual(len(frame_rows), 2)
        self.assertFalse(frame_rows[0]["candidate_only"])
        self.assertTrue(all(row["review_status"] == "approved_negative" for row in updated_candidates["rows"]))
        self.assertIn('"component": "string"', review_log)


if __name__ == "__main__":
    unittest.main()
