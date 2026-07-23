import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation import string_prelabel, video_frame_annotator
from video_dataset.select_candidates import select_candidates


HOLDOUT = "ab03bb7118b0"


class AnnotationHoldoutTests(unittest.TestCase):
    def test_candidate_selection_skips_excluded_sources_before_video_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = {name: root / f"{name}.mp4" for name in (HOLDOUT, "kept")}
            for path in videos.values():
                path.write_bytes(b"placeholder")
            sources = {
                "sources": [
                    {
                        "video_id": name,
                        "source_group": name,
                        "split": "train",
                        "path": str(path),
                        "sha256": name,
                        "fps": 30.0,
                    }
                    for name, path in videos.items()
                ]
            }
            (root / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
            captured_paths = []

            class EmptyCapture:
                def __init__(self, path):
                    captured_paths.append(str(path))

                def isOpened(self):
                    return True

                def read(self):
                    return False, None

                def release(self):
                    return None

            with patch("video_dataset.select_candidates.YOLO"), patch(
                "video_dataset.select_candidates.cv2.VideoCapture", EmptyCapture
            ):
                result = select_candidates(
                    root,
                    root / "weights.pt",
                    1.0,
                    0.2,
                    640,
                    "train",
                    0,
                    0,
                    HOLDOUT,
                )

        self.assertEqual(result["videos_processed"], 1)
        self.assertEqual(result["exclude_source_groups"], [HOLDOUT])
        self.assertEqual(captured_paths, [str(videos["kept"])])

    def test_vlm_selection_skips_excluded_source_before_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {
                    "frame_path": str(root / f"{group}.jpg"),
                    "source_group": group,
                    "video_id": group,
                    "split": "train",
                    "frame_index": 0,
                    "candidate_only": True,
                }
                for group in (HOLDOUT, "kept")
            ]
            (root / "frames.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )

            def fake_annotation(record, *_args):
                return {"label_path": record["video_id"], "review_status": "auto_labeled_needs_review"}

            argv = [
                "video_frame_annotator",
                "--dataset-dir",
                str(root),
                "--split",
                "train",
                "--candidates-only",
                "--exclude-source-groups",
                HOLDOUT,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                video_frame_annotator, "annotate_record", side_effect=fake_annotation
            ) as annotate:
                self.assertEqual(video_frame_annotator.main(), 0)

        self.assertEqual(annotate.call_count, 1)
        self.assertEqual(annotate.call_args.args[0]["source_group"], "kept")

    def test_color_prelabel_skips_excluded_source_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group in (HOLDOUT, "kept"):
                path = root / "annotations" / "labels" / "train" / group / "frame.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"source_group": group, "split": "train"}), encoding="utf-8"
                )
            argv = [
                "string_prelabel",
                "--dataset-dir",
                str(root),
                "--split",
                "train",
                "--exclude-source-groups",
                HOLDOUT,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                string_prelabel, "prelabel_annotation", return_value={"status": "updated"}
            ) as prelabel:
                self.assertEqual(string_prelabel.main(), 0)

        self.assertEqual(prelabel.call_count, 1)
        self.assertEqual(prelabel.call_args.args[0].parent.name, "kept")


if __name__ == "__main__":
    unittest.main()
