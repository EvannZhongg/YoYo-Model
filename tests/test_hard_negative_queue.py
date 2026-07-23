import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from video_dataset.hard_negative_queue import build_hard_negative_queue


class HardNegativeQueueTests(unittest.TestCase):
    def test_queue_filters_ranks_and_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels"
            source = root / "source.jpg"
            weights = root / "weights.pt"
            weights.write_bytes(b"test checkpoint")
            cv2.imwrite(str(source), np.full((32, 64, 3), 80, dtype=np.uint8))

            def write(split: str, group: str, frame: int, **updates):
                path = labels / split / group / f"frame_{frame:08d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "video_id": group,
                    "source_group": group,
                    "split": split,
                    "frame_index": frame,
                    "source_image": str(source),
                    "string_review_status": "reviewed",
                    "string_visibility": "not_visible",
                    "visibility": "absent",
                }
                data.update(updates)
                path.write_text(json.dumps(data), encoding="utf-8")
                return path

            included = [
                write("train", "a", 10),
                write("train", "b", 20, string_review_status="approved"),
                write("train", "c", 30),
            ]
            write("train", "pending", 99, string_review_status="auto_labeled_needs_review")
            write("train", "positive", 98, string_visibility="visible")
            write("val", "frozen", 97)
            before = {path: path.read_bytes() for path in labels.rglob("*.json")}

            signals = {
                10: (50, 1, 0.99),
                20: (50, 2, 0.80),
                30: (0, 0, 0.49),
            }

            def fake_signal(data, weights, device, cache, preview):
                pixels, components, maximum = signals[int(data["frame_index"])]
                preview.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview), np.full((32, 64, 3), 100, dtype=np.uint8))
                return 0.0, [], {
                    "status": "ok",
                    "predicted_pixels": pixels,
                    "predicted_fraction": pixels / 1000,
                    "components": components,
                    "max_probability": maximum,
                    "mean_probability": 0.1,
                    "prediction_preview": str(preview.resolve()),
                }

            with patch("video_dataset.hard_negative_queue._model_signal", side_effect=fake_signal):
                result = build_hard_negative_queue(root, weights)
            payload = json.loads((root / "string_hard_negative_queue.json").read_text(encoding="utf-8"))
            after = {path: path.read_bytes() for path in labels.rglob("*.json")}

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["false_positive_count"], 2)
        self.assertEqual([row["frame_index"] for row in payload["rows"]], [20, 10, 30])
        self.assertEqual([row["queue_rank"] for row in payload["rows"]], [1, 2, 3])
        self.assertEqual({Path(row["label_path"]) for row in payload["rows"]}, set(included))
        self.assertEqual(payload["split"], "train")
        self.assertIn("REVIEW ONLY", payload["policy"])
        self.assertEqual(before, after)

    def test_excluded_source_group_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels"
            source = root / "source.jpg"
            weights = root / "weights.pt"
            weights.write_bytes(b"test checkpoint")
            cv2.imwrite(str(source), np.full((32, 64, 3), 80, dtype=np.uint8))
            for group in ("ab03bb7118b0", "kept"):
                path = labels / "train" / group / "frame_00000000.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "video_id": group,
                    "source_group": group,
                    "split": "train",
                    "frame_index": 0,
                    "source_image": str(source),
                    "string_review_status": "reviewed",
                    "string_visibility": "not_visible",
                    "visibility": "absent",
                }), encoding="utf-8")
            with patch("video_dataset.hard_negative_queue._model_signal", return_value=(0.0, [], {"predicted_pixels": 0})) as signal:
                result = build_hard_negative_queue(root, weights, exclude_source_groups="ab03bb7118b0")
        self.assertEqual(result["count"], 1)
        self.assertEqual(signal.call_count, 1)


if __name__ == "__main__":
    unittest.main()
