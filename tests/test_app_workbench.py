import fnmatch
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import create_demo, run_video_tracking
from config import SEMANTIC_STRING_CONFIG, STRING_TRACKING_CONFIG, TRACKING_CONFIG
from workbench.commands import workbench_evaluate_v2v3, workbench_train_v2v3
from workbench.score_annotation import (
    ANCHOR_SOURCES,
    EXCLUSION_REASONS,
    MAJOR_PENALTIES,
    SCENE_TYPES,
    SCHEMA_VERSION,
    delete_score_annotation,
    list_score_annotations,
    load_score_annotation,
    load_score_annotation_session,
    resolve_score_video_source,
    save_score_annotation,
    validate_score_annotation,
)
from workbench.tracking import tracking_review_gallery


class UnifiedWorkbenchTests(unittest.TestCase):
    def test_ui_exposes_unified_training_and_frame_tracking(self):
        config = create_demo().get_config_file()
        props = [component.get("props", {}) for component in config.get("components", [])]
        labels = {str(item.get("label", "")) for item in props}
        values = {str(item.get("value", "")) for item in props}

        self.assertIn("Unified Dataset", labels)
        self.assertIn("Temporal 组标注", labels)
        self.assertIn("Run Full Video Tracking", values)
        self.assertTrue(any("悠悠球计分标注" in value for value in values))

    def test_video_tracking_input_accepts_mp4_and_mov(self):
        config = create_demo().get_config_file()
        input_video = next(
            component["props"]
            for component in config.get("components", [])
            if component.get("props", {}).get("label") == "Input Video"
        )
        pattern = input_video["glob"]
        self.assertTrue(fnmatch.fnmatch("match.mp4", pattern))
        self.assertTrue(fnmatch.fnmatch("match.mov", pattern))
        self.assertTrue(fnmatch.fnmatch("match.MOV", pattern))
        self.assertFalse(fnmatch.fnmatch("match.avi", pattern))

    def test_score_annotation_disk_storage_round_trip_and_delete(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "annotation_id": "annotation-1",
            "revision": 1,
            "video": {
                "file_name": "contest.mp4",
                "browser_identity": "contest.mp4:12345:1700000000000",
                "source_path": "videos/contest.mp4",
            },
            "competition": {"division": "2A"},
            "annotator": {"judge": "judge2"},
            "scene_intervals": [{
                "scene_interval_id": "scene-1",
                "start_s": 0.0,
                "end_s": 2.5,
                "scene_type": "irrelevant_scene",
            }],
            "events": [],
            "excluded_intervals": [{
                "exclusion_id": "excluded-1",
                "start_s": 4.0,
                "end_s": 5.5,
                "reason": "defocus",
                "training_eligible": False,
            }],
            "created_at": "2026-07-26T00:00:00Z",
            "updated_at": "2026-07-26T00:00:01Z",
        }
        with TemporaryDirectory() as directory, patch(
            "workbench.score_annotation.SCORE_ANNOTATION_DIR", Path(directory) / "scores"
        ):
            saved = save_score_annotation(document)
            document["revision"] = 2
            document["updated_at"] = "2026-07-26T00:00:02Z"
            updated = save_score_annotation(json.dumps(document))
            loaded = load_score_annotation(document["video"]["browser_identity"])
            listed = list_score_annotations()

            self.assertEqual(saved["storage_key"], updated["storage_key"])
            self.assertEqual(loaded["document"]["revision"], 2)
            self.assertEqual(loaded["document"]["scene_intervals"][0]["scene_type"], "irrelevant_scene")
            self.assertFalse(loaded["document"]["excluded_intervals"][0]["training_eligible"])
            self.assertEqual(len(listed), 1)
            self.assertTrue(Path(updated["path"]).is_file())
            self.assertTrue(delete_score_annotation(updated["storage_key"]))
            self.assertEqual(list_score_annotations(), [])
            with self.assertRaisesRegex(ValueError, "invalid"):
                delete_score_annotation("../outside.json")

    def test_score_annotation_session_resolves_video_from_required_source_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            video_path = video_dir / "contest.mp4"
            video_path.write_bytes(b"score-video")
            modified_ms = int(video_path.stat().st_mtime * 1000)
            document = {
                "schema_version": SCHEMA_VERSION,
                "annotation_id": "annotation-source",
                "revision": 1,
                "video": {
                    "file_name": video_path.name,
                    "file_size_bytes": video_path.stat().st_size,
                    "last_modified_ms": modified_ms,
                    "browser_identity": f"{video_path.name}:{video_path.stat().st_size}:{modified_ms}",
                    "source_path": "videos/contest.mp4",
                },
                "competition": {"division": "1A"},
                "annotator": {"judge": "judge1"},
                "scene_intervals": [],
                "events": [],
                "excluded_intervals": [],
            }
            with (
                patch("workbench.score_annotation.BASE_DIR", root),
                patch("workbench.score_annotation.SCORE_VIDEO_DIRS", (video_dir,)),
                patch("workbench.score_annotation.SCORE_ANNOTATION_DIR", root / "scores"),
                patch("workbench.score_annotation.gr.set_static_paths") as set_static_paths,
            ):
                saved = save_score_annotation(document)
                self.assertEqual(
                    resolve_score_video_source(document["video"]),
                    "videos/contest.mp4",
                )
                session = load_score_annotation_session(saved["storage_key"])

            self.assertEqual(Path(session["video_path"]), video_path.resolve())
            set_static_paths.assert_called_once_with(paths=[video_path.resolve()])

    def test_score_annotation_requires_managed_video_source_path(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {},
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "scene_intervals": [],
            "events": [],
        }
        with self.assertRaisesRegex(ValueError, "video.source_path is required"):
            validate_score_annotation(document)
        document["video"]["source_path"] = "../outside.mp4"
        with self.assertRaisesRegex(ValueError, "invalid video.source_path"):
            validate_score_annotation(document)

    def test_score_annotation_schema_accepts_complete_overlapping_intervals(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {"source_path": "videos/contest.mp4"},
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "scene_intervals": [{
                "scene_interval_id": "scene-1",
                "start_s": 0.0,
                "end_s": 1.0,
                "scene_type": "player_entry_exit",
            }],
            "events": [
                {
                    "label": {"family": "positive", "score_delta": 7},
                    "timing": {"evidence_start_s": 1.0, "anchor_s": 2.5, "evidence_end_s": 3.0, "anchor_source": "manual"},
                },
                {
                    "label": {"family": "major_penalty", "penalty_type": "disassembly", "score_delta": -5},
                    "timing": {"evidence_start_s": 2.8, "anchor_s": 3.2, "evidence_end_s": 4.0, "anchor_source": "evidence_end_default"},
                },
            ],
        }

        validate_score_annotation(document)
        self.assertEqual(MAJOR_PENALTIES["restart"]["score_delta"], -1)
        self.assertEqual(MAJOR_PENALTIES["discard"]["score_delta"], -3)
        self.assertEqual(ANCHOR_SOURCES, ("evidence_end_default", "manual"))
        self.assertEqual(SCENE_TYPES, ("irrelevant_scene", "player_entry_exit"))
        self.assertEqual(EXCLUSION_REASONS, ("defocus", "occlusion", "corrupted_frames", "other"))

    def test_score_annotation_schema_validates_training_exclusions(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {"source_path": "videos/contest.mp4"},
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "scene_intervals": [],
            "events": [],
            "excluded_intervals": [{
                "exclusion_id": "excluded-1",
                "start_s": 1.0,
                "end_s": 2.0,
                "reason": "defocus",
                "training_eligible": False,
            }],
        }

        validate_score_annotation(document)
        document["excluded_intervals"][0]["training_eligible"] = True
        with self.assertRaisesRegex(ValueError, "training_eligible"):
            validate_score_annotation(document)
        document["excluded_intervals"][0]["training_eligible"] = False
        document["excluded_intervals"][0]["end_s"] = 1.0
        with self.assertRaisesRegex(ValueError, "start_s < end_s"):
            validate_score_annotation(document)

    def test_score_annotation_schema_validates_scene_intervals(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {"source_path": "videos/contest.mp4"},
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "events": [],
            "scene_intervals": [{
                "scene_interval_id": "scene-1",
                "start_s": 1.0,
                "end_s": 2.0,
                "scene_type": "player_entry_exit",
            }],
            "excluded_intervals": [],
        }

        validate_score_annotation(document)
        document["scene_intervals"][0]["scene_type"] = "sponsor_page"
        with self.assertRaisesRegex(ValueError, "scene_type"):
            validate_score_annotation(document)
        document["scene_intervals"][0]["scene_type"] = "irrelevant_scene"
        document["scene_intervals"][0]["end_s"] = 1.0
        with self.assertRaisesRegex(ValueError, "start_s < end_s"):
            validate_score_annotation(document)

    def test_score_annotation_schema_rejects_anchor_outside_evidence(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {"source_path": "videos/contest.mp4"},
            "competition": {"division": "5A"},
            "annotator": {"judge": "judge1"},
            "scene_intervals": [],
            "events": [{
                "label": {"family": "negative", "score_delta": -2},
                "timing": {"evidence_start_s": 3.0, "anchor_s": 2.0, "evidence_end_s": 4.0},
            }],
        }

        with self.assertRaisesRegex(ValueError, "evidence_start"):
            validate_score_annotation(document)

    def test_score_annotation_schema_rejects_invalid_anchor_source(self):
        document = {
            "schema_version": SCHEMA_VERSION,
            "video": {"source_path": "videos/contest.mp4"},
            "competition": {"division": "1A"},
            "annotator": {"judge": "judge1"},
            "scene_intervals": [],
            "events": [{
                "label": {"family": "positive", "score_delta": 2},
                "timing": {
                    "evidence_start_s": 1.0,
                    "anchor_s": 2.0,
                    "evidence_end_s": 2.0,
                    "anchor_source": "automatic",
                },
            }],
        }

        with self.assertRaisesRegex(ValueError, "anchor_source"):
            validate_score_annotation(document)

        del document["events"][0]["timing"]["anchor_source"]
        with self.assertRaisesRegex(ValueError, "anchor_source"):
            validate_score_annotation(document)

    @patch("app._tracking_review_gallery", return_value=[])
    @patch("app.track_video")
    def test_tracking_forwards_configured_frame_models(self, track_video, _gallery):
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
            "input.mov", "detector.pt", "runs/tracking", 0.25, 0.7, 1280, "cuda",
            False, "rtmpose.onnx", True, "string.pt", 0.2, 2.0, 10.0,
            "1A", True, "orientation.pt", 5.0, 1920,
        )

        self.assertEqual(len(outputs), 11)
        self.assertEqual(outputs[0], "tracked.mp4")
        self.assertEqual(outputs[1], "frames.jsonl")
        self.assertEqual(outputs[2], "run.json")
        self.assertIn("Semantic inference frames: 3", outputs[-1])
        kwargs = track_video.call_args.kwargs
        self.assertEqual(kwargs["source_video_path"], "input.mov")
        self.assertEqual(kwargs["string_inference_fps"], 10.0)
        self.assertEqual(kwargs["orientation_inference_fps"], 5.0)
        self.assertTrue(kwargs["enable_orientation_model"])
        self.assertEqual(kwargs["visualization_max_width"], 1920)

        track_video.reset_mock()
        run_video_tracking(
            "input.mp4", "detector.pt", "runs/tracking", 0.25, 0.7, 1280, "cuda",
            False, "rtmpose.onnx", True, str(STRING_TRACKING_CONFIG.weights_path), 0.2, 2.0, 10.0,
            "1A", True, "orientation.pt", 5.0, 1920,
        )
        default_kwargs = track_video.call_args.kwargs
        self.assertEqual(
            default_kwargs["string_color_semantic_prefilter"],
            TRACKING_CONFIG.string_color_semantic_prefilter,
        )
        self.assertEqual(TRACKING_CONFIG.string_color_probability_min_mean, 0.70)

    def test_tracking_without_video_returns_empty_output(self):
        outputs = run_video_tracking(
            None, "detector.pt", "runs/tracking", 0.25, 0.7, 1280, "cpu",
            False, "", False, "", 0.2, 1.0, 10.0,
            "unknown", False, "", 5.0, 1920,
        )

        self.assertEqual(len(outputs), 11)
        self.assertEqual(outputs[4], [])
        self.assertIn("No video provided", outputs[-1])

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_orientation_training_uses_independent_entry_point(self, run_command):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text('{"dataset_id":"dataset-123"}', encoding="utf-8")
            result = workbench_train_v2v3(str(dataset), str(Path(directory) / "runs"), "orientation", 3, "cpu")

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("yoyo_orientation.train", args)
        self.assertEqual(Path(args[args.index("--view-manifest") + 1]).parent.name, "orientation_roi")

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_string_training_uses_canonical_string_view(self, run_command):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text('{"dataset_id":"dataset-123"}', encoding="utf-8")
            result = workbench_train_v2v3(str(dataset), str(Path(directory) / "runs"), "string_tracking", 5, "cpu")

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("string_tracking.train", args)
        self.assertEqual(args[args.index("--name") + 1], "dataset-123_semantic_string")
        self.assertEqual(Path(args[args.index("--dataset-dir") + 1]).name, "string_segmentation")
        self.assertEqual(args[args.index("--architecture") + 1], SEMANTIC_STRING_CONFIG.architecture)
        self.assertEqual(
            float(args[args.index("--hard-negative-weight") + 1]),
            SEMANTIC_STRING_CONFIG.hard_negative_weight,
        )
        self.assertEqual(
            int(args[args.index("--freeze-backbone-epochs") + 1]),
            SEMANTIC_STRING_CONFIG.freeze_backbone_epochs,
        )
        self.assertIn("--pretrained-backbone", args)

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_evaluation_uses_manifest_task_evaluator(self, run_command):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text('{"task":"orientation"}', encoding="utf-8")
            result = workbench_evaluate_v2v3(str(run), "cpu")

        self.assertEqual(result, "ok")
        self.assertEqual(run_command.call_args.args[0][:3], ["-m", "yoyo_orientation.evaluate", str(run)])

    def test_tracking_gallery_uses_frame_index(self):
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
