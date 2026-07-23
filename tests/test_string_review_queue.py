import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from string_segmentation.semantic_model import LetterboxMeta
from video_dataset.string_review_queue import (
    _annotation_agreement,
    _save_model_preview,
    build_queue,
    load_prediction_polylines,
)


class StringReviewQueueTests(unittest.TestCase):
    def test_annotation_agreement_distinguishes_match_from_mismatch(self):
        meta = LetterboxMeta(
            original_width=40,
            original_height=40,
            target_width=40,
            target_height=40,
            resized_width=40,
            resized_height=40,
            pad_x=0,
            pad_y=0,
            scale=1.0,
        )
        annotation = {"string_polylines_pixel": [[[4, 6], [35, 6]]]}
        matching = np.zeros((40, 40), dtype=np.uint8)
        cv2.polylines(matching, [np.asarray([[4, 6], [35, 6]], dtype=np.int32)], False, 1, 8, cv2.LINE_AA)
        mismatching = np.zeros((40, 40), dtype=np.uint8)
        cv2.polylines(mismatching, [np.asarray([[4, 34], [35, 34]], dtype=np.int32)], False, 1, 8, cv2.LINE_AA)

        perfect = _annotation_agreement(matching > 0, annotation, meta)
        mismatch = _annotation_agreement(mismatching > 0, annotation, meta)

        self.assertEqual(perfect["exact_dice"], 1.0)
        self.assertEqual(perfect["tolerant_f1"], 1.0)
        self.assertEqual(mismatch["exact_dice"], 0.0)
        self.assertEqual(mismatch["tolerant_f1"], 0.0)

    def test_agreement_strategy_requires_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires with_model"):
                build_queue(tmp, strategy="agreement")

    @patch("video_dataset.string_review_queue._model_signal")
    def test_agreement_strategy_ranks_higher_tolerant_f1_first(self, model_signal):
        def signal(data, *_args, **_kwargs):
            f1 = 0.9 if int(data["frame_index"]) == 2 else 0.4
            return 0.0, [], {
                "status": "ok",
                "components": 1,
                "prediction_confidence": 0.8,
                "annotation_agreement": {"tolerant_f1": f1},
            }

        model_signal.side_effect = signal
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels" / "train" / "video-a"
            labels.mkdir(parents=True)
            for frame_index in (1, 2):
                (labels / f"frame_{frame_index:08d}.json").write_text(json.dumps({
                    "video_id": "video-a",
                    "source_group": "video-a",
                    "split": "train",
                    "frame_index": frame_index,
                    "string_review_status": "auto_labeled_needs_review",
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[1, frame_index], [10, frame_index]]],
                    "qa": {"warnings": []},
                }), encoding="utf-8")

            result = build_queue(root, split="train", with_model=True, weights="unused.pt", strategy="agreement")
            payload = json.loads((root / "string_review_queue.json").read_text(encoding="utf-8"))

        self.assertEqual(result["strategy"], "agreement")
        self.assertEqual([row["frame_index"] for row in payload["rows"]], [2, 1])
        self.assertTrue(payload["policy"].startswith("annotation/model agreement"))

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
        self.assertEqual(payload["strategy"], "uncertainty")

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

    def test_excluded_source_group_is_absent_before_model_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels" / "train"
            for group in ("ab03bb7118b0", "kept"):
                path = labels / group / "frame_00000000.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "video_id": group,
                    "source_group": group,
                    "split": "train",
                    "string_review_status": "auto_labeled_needs_review",
                    "string_visibility": "uncertain",
                }), encoding="utf-8")
            result = build_queue(root, split="train", exclude_source_groups="ab03bb7118b0")
            payload = json.loads((root / "string_review_queue.json").read_text(encoding="utf-8"))

        self.assertEqual(result["count"], 1)
        self.assertEqual(payload["exclude_source_groups"], ["ab03bb7118b0"])
        self.assertEqual(payload["rows"][0]["source_group"], "kept")

    def test_unresolved_labels_are_terminal_and_absent_from_active_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "annotations" / "labels" / "train"
            for group, status in (("unresolved", "unresolved"), ("pending", "auto_labeled_needs_review")):
                path = labels / group / "frame.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "video_id": group,
                    "source_group": group,
                    "split": "train",
                    "string_review_status": status,
                    "string_visibility": "uncertain",
                }), encoding="utf-8")
            result = build_queue(root, split="train")
            payload = json.loads((root / "string_review_queue.json").read_text(encoding="utf-8"))

        self.assertEqual(result["count"], 1)
        self.assertEqual(payload["rows"][0]["source_group"], "pending")


if __name__ == "__main__":
    unittest.main()
