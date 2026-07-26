import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import create_demo, run_video_tracking
from workbench.commands import workbench_evaluate_v2v3, workbench_train_v2v3
from workbench.score_annotation import (
    MAJOR_PENALTIES,
    SCHEMA_VERSION,
    score_annotation_component_kwargs,
    validate_score_annotation,
)
from workbench.tracking import tracking_review_gallery


class UnifiedWorkbenchTests(unittest.TestCase):
    def test_ui_only_exposes_unified_training_and_frame_tracking(self):
        config = create_demo().get_config_file()
        props = [component.get("props", {}) for component in config.get("components", [])]
        labels = {str(item.get("label", "")) for item in props}
        values = {str(item.get("value", "")) for item in props}

        self.assertIn("Unified Dataset", labels)
        self.assertIn("Run Full Video Tracking", values)
        self.assertTrue(any("悠悠球计分标注" in value for value in values))
        self.assertNotIn("Named trick model", labels)
        self.assertNotIn("Trick Label", labels)
        self.assertNotIn("Segment Manifest", labels)
        self.assertNotIn("Candidate Clips", labels)
        self.assertFalse(any("legacy-video" in value for value in values))

    def test_score_annotation_component_has_local_resume_and_frame_controls(self):
        component = score_annotation_component_kwargs()
        html = component["value"]
        javascript = component["js_on_load"]

        self.assertIn('accept="video/*"', html)
        self.assertIn("Evidence interval", html)
        self.assertIn("动作名称", html)
        self.assertEqual(html.count('class="ysa__track-row"'), 3)
        self.assertIn('data-track="positive"', html)
        self.assertIn('data-track="negative"', html)
        self.assertIn('data-track="major_penalty"', html)
        self.assertIn("localStorage.setItem", javascript)
        self.assertIn("video.currentTime + 1 / fps()", javascript)
        self.assertIn("yoyo-score-annotation:v1:", javascript)
        self.assertIn("beginClipDrag", javascript)
        self.assertIn("beginTrackDraft", javascript)
        self.assertIn("syncSelectedFromEditor", javascript)

    def test_score_annotation_schema_accepts_complete_overlapping_intervals(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "events": [
                {
                    "label": {"family": "positive", "score_delta": 7},
                    "timing": {"evidence_start_s": 1.0, "anchor_s": 2.5, "evidence_end_s": 3.0},
                },
                {
                    "label": {"family": "major_penalty", "penalty_type": "disassembly", "score_delta": -5},
                    "timing": {"evidence_start_s": 2.8, "anchor_s": 3.2, "evidence_end_s": 4.0},
                },
            ],
        }

        validate_score_annotation(document)
        self.assertEqual(MAJOR_PENALTIES["restart"]["score_delta"], -1)
        self.assertEqual(MAJOR_PENALTIES["discard"]["score_delta"], -3)

    def test_score_annotation_schema_rejects_anchor_outside_evidence(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "competition": {"division": "5A"},
            "annotator": {"judge": "judge1"},
            "events": [{
                "label": {"family": "negative", "score_delta": -2},
                "timing": {"evidence_start_s": 3.0, "anchor_s": 2.0, "evidence_end_s": 4.0},
            }],
        }

        with self.assertRaisesRegex(ValueError, "evidence_start"):
            validate_score_annotation(document)

    @patch("app._tracking_review_gallery", return_value=[])
    @patch("app.track_video")
    def test_tracking_forwards_frame_models_without_segment_arguments(self, track_video, _gallery):
        track_video.return_value = {
            "frame_count": 12,
            "output_video": "tracked.mp4",
            "metadata_jsonl": "frames.jsonl",
            "run_manifest": "run.json",
            "review_sheet": "review.jpg",
            "run_dir": "run",
            "bad_case_counts": {},
            "string_geometry_counts": {"hand_supported_observation_frames": 4},
            "string_model": "semantic:model.pt",
            "string_inference_frame_count": 3,
            "orientation_model": "orientation.pt",
            "orientation_inference_frame_count": 2,
            "orientation_summary": {"label": "horizontal"},
            "tracking_loop_fps": 4.5,
            "output_width": 1920,
            "output_height": 1080,
            "weights": "detector.pt",
        }

        outputs = run_video_tracking(
            "input.mp4", "detector.pt", "runs/tracking", 0.25, 0.7, 1280, "cuda",
            False, "pose.pt", True, "string.pt", 0.2, 2.0, 10.0,
            "hand_and_yoyo_attached", True, "orientation.pt", 5.0, 1920,
        )

        self.assertEqual(len(outputs), 11)
        self.assertEqual(outputs[0], "tracked.mp4")
        self.assertEqual(outputs[1], "frames.jsonl")
        self.assertEqual(outputs[2], "run.json")
        self.assertIn("Semantic inference frames: 3", outputs[-1])
        kwargs = track_video.call_args.kwargs
        self.assertEqual(kwargs["string_inference_fps"], 10.0)
        self.assertEqual(kwargs["orientation_inference_fps"], 5.0)
        self.assertTrue(kwargs["enable_orientation_model"])
        self.assertEqual(kwargs["visualization_max_width"], 1920)
        self.assertFalse(any("trick" in key and key != "enable_orientation_model" for key in kwargs))
        self.assertFalse(any("segment" in key or "clip" in key or "activity" in key for key in kwargs))

    def test_tracking_without_video_returns_current_output_shape(self):
        outputs = run_video_tracking(
            None, "detector.pt", "runs/tracking", 0.25, 0.7, 1280, "cpu",
            False, "", False, "", 0.2, 1.0, 10.0,
            "unknown", False, "", 5.0, 1920,
        )

        self.assertEqual(len(outputs), 11)
        self.assertEqual(outputs[4], [])
        self.assertIn("No video provided", outputs[-1])

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_unified_training_uses_manifest_and_training_v2(self, run_command):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text('{"dataset_id":"dataset-123"}', encoding="utf-8")
            result = workbench_train_v2v3(str(dataset), str(Path(directory) / "runs"), "orientation", 3, "cpu")

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("training_v2.train", args)
        self.assertEqual(args[args.index("--task") + 1], "orientation")

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_unified_semantic_training_uses_canonical_string_view(self, run_command):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text('{"dataset_id":"dataset-123"}', encoding="utf-8")
            result = workbench_train_v2v3(str(dataset), str(Path(directory) / "runs"), "semantic_string", 5, "cpu")

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("string_segmentation.train_semantic", args)
        self.assertEqual(args[args.index("--name") + 1], "dataset-123_semantic_string")
        self.assertEqual(Path(args[args.index("--dataset-dir") + 1]).name, "string_segmentation")

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_evaluation_uses_training_v2_evaluator(self, run_command):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text("{}", encoding="utf-8")
            result = workbench_evaluate_v2v3(str(run), "cpu")

        self.assertEqual(result, "ok")
        self.assertEqual(run_command.call_args.args[0][:3], ["-m", "training_v2.evaluate", str(run)])

    def test_tracking_gallery_uses_frame_index_without_segments(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            review = run / "review_frames"
            review.mkdir()
            image = review / "frame_10.jpg"
            image.write_bytes(b"image")
            (run / "tracking_review_index.json").write_text(
                json.dumps([{"frame": {"frame_index": 10, "timestamp_s": 0.2}, "image": str(image)}]),
                encoding="utf-8",
            )

            gallery = tracking_review_gallery(run)

        self.assertEqual(len(gallery), 1)
        self.assertIn("f10", gallery[0][1])


if __name__ == "__main__":
    unittest.main()
