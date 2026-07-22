import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from string_segmentation.semantic_model import LetterboxMeta
from video_dataset.string_review_queue import _save_model_preview, build_queue, load_prediction_polylines


class StringReviewQueueTests(unittest.TestCase):
    def test_semantic_preview_restores_letterbox_to_original_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preview.jpg"
            image = np.full((32, 64, 3), 80, dtype=np.uint8)
            probability = np.zeros((32, 32), dtype=np.float32)
            probability[12:16, 12:20] = 1.0
            meta = LetterboxMeta(
                original_width=64,
                original_height=32,
                target_width=32,
                target_height=32,
                resized_width=32,
                resized_height=16,
                pad_x=0,
                pad_y=8,
                scale=0.5,
            )
            _save_model_preview(image, probability, meta, 0.9, output)
            preview = cv2.imread(str(output))

        self.assertEqual(preview.shape[:2], (32, 64))
        self.assertGreater(int(np.abs(preview.astype(np.int16) - image.astype(np.int16)).sum()), 0)

    def test_queue_is_review_only_ranked_and_diverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels" / "train" / "video-a"
            labels.mkdir(parents=True)
            common = {
                "video_id": "video-a",
                "source_group": "video-a",
                "action_group": "1A",
                "split": "train",
                "string_review_status": "auto_labeled_needs_review",
                "string_visibility": "uncertain",
                "bad_case": ["motion_blur"],
                "qa": {"priority": "high", "warnings": ["visible_string_without_polyline"]},
            }
            (labels / "frame_00000000.json").write_text(json.dumps(common), encoding="utf-8")
            second = dict(common)
            second["video_id"] = "video-b"
            second["source_group"] = "video-b"
            (root / "annotations" / "labels" / "train" / "video-b").mkdir(parents=True)
            (root / "annotations" / "labels" / "train" / "video-b" / "frame_00000000.json").write_text(json.dumps(second), encoding="utf-8")
            reviewed = dict(common)
            reviewed["string_review_status"] = "reviewed"
            (labels / "frame_00000001.json").write_text(json.dumps(reviewed), encoding="utf-8")

            result = build_queue(root, limit=2)
            payload = json.loads((root / "string_review_queue.json").read_text(encoding="utf-8"))

        self.assertEqual(result["count"], 2)
        self.assertEqual(payload["action_group"], "1A")
        self.assertEqual([row["queue_rank"] for row in payload["rows"]], [1, 2])
        self.assertEqual({row["source_group"] for row in payload["rows"]}, {"video-a", "video-b"})
        self.assertTrue(all("reviewed" not in row["label_path"] for row in payload["rows"]))
        self.assertTrue(all(row["reasons"] for row in payload["rows"]))

    def test_prediction_geometry_loader_does_not_touch_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = root / "annotations" / "labels" / "train" / "v" / "f.json"
            label.parent.mkdir(parents=True)
            original = {"string_review_status": "auto_labeled_needs_review"}
            label.write_text(json.dumps(original), encoding="utf-8")
            queue = {
                "rows": [{
                    "label_path": str(label.resolve()),
                    "model": {"prediction_polylines": [[[1, 2], [3.5, 4.5]]]},
                }]
            }
            (root / "string_review_queue.json").write_text(json.dumps(queue), encoding="utf-8")
            strokes = load_prediction_polylines(root, label)
            after = json.loads(label.read_text(encoding="utf-8"))

        self.assertEqual(strokes, [[[1.0, 2.0], [3.5, 4.5]]])
        self.assertEqual(after, original)


if __name__ == "__main__":
    unittest.main()
