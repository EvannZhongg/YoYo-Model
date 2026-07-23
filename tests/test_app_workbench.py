import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app import (
    run_video_tracking,
    workbench_refresh,
)
from workbench.commands import (
    _workbench_holdout_defaults,
    workbench_candidates,
    workbench_evaluate_pipeline,
    workbench_hard_negative_neighbors,
    workbench_hard_negative_queue,
    workbench_prelabel_strings,
    workbench_prepare_string,
    workbench_qa_export,
    workbench_string_review_queue,
    workbench_train_semantic,
    workbench_train_string,
    workbench_vlm,
)
from workbench.review import review_label_paths as _review_label_paths
from workbench.tracking import (
    tracking_review_caption as _tracking_review_caption,
    tracking_review_gallery as _tracking_review_gallery,
)


class WorkbenchSemanticTrainingTests(unittest.TestCase):
    @patch("app._workbench_stats", return_value="stats")
    @patch("app.refresh_review_queue", return_value=("queue", "image", "summary"))
    def test_workbench_refresh_combines_queue_and_stats(self, refresh_queue, workbench_stats):
        result = workbench_refresh("dataset", "pending", "train", "all")

        self.assertEqual(result, ("queue", "image", "summary", "stats"))
        refresh_queue.assert_called_once_with("dataset", "pending", "train", "all", "")
        workbench_stats.assert_called_once_with("dataset")

    @patch("app.track_video")
    def test_upload_tracking_forwards_cadence_and_preview_limit(self, track_video):
        track_video.return_value = {
            "frame_count": 12,
            "output_video": "tracked.mp4",
            "metadata_jsonl": "frames.jsonl",
            "segments_json": "segments.json",
            "run_manifest": "run.json",
            "review_sheet": "review.jpg",
            "trick_token_manifest": "tokens.json",
            "trick_token_count": 0,
            "segments": [],
            "bad_case_counts": {},
            "string_geometry_counts": {"hand_supported_observation_frames": 4},
            "string_model": "semantic:model.pt",
            "string_inference_frame_count": 3,
            "tracking_loop_fps": 4.5,
            "output_width": 1920,
            "output_height": 1080,
            "weights": "detector.pt",
        }

        outputs = run_video_tracking(
            "input.mp4", "detector.pt", "runs/tracking", 0.25, 0.7, 640, "cuda",
            False, "pose.pt", True, "string.pt", 0.2, 2.0, 10.0,
            "hand_and_yoyo_attached", False, 4.0, 180.0, 0.08, 120,
            1920,
        )

        self.assertEqual(outputs[0], "tracked.mp4")
        self.assertEqual(outputs[5], [])
        self.assertEqual(outputs[8], "frames.jsonl")
        self.assertIn("Semantic inference frames: 3", outputs[-1])
        self.assertIn("hand_supported_observation_frames", outputs[-1])
        kwargs = track_video.call_args.kwargs
        self.assertEqual(kwargs["string_inference_fps"], 10.0)
        self.assertEqual(kwargs["max_frames"], 120)
        self.assertEqual(kwargs["visualization_max_width"], 1920)

    def test_tracking_review_gallery_preserves_index_order_and_run_boundary(self):
        with TemporaryDirectory() as directory:
            temp_root = Path(directory)
            run_dir = temp_root / "run"
            frame_dir = run_dir / "review_frames"
            frame_dir.mkdir(parents=True)
            first = frame_dir / "frame_20.jpg"
            first_raw = frame_dir / "frame_20_raw.jpg"
            second = frame_dir / "frame_10.jpg"
            outside = temp_root / "outside.jpg"
            for path in (first, first_raw, second, outside):
                path.write_bytes(b"review")
            mismatch_record = {
                "frame_index": 20,
                "timestamp_s": 0.4,
                "yoyo": {"center": [100.0, 100.0]},
                "string": {
                    "method": "semantic_segmentation",
                    "confidence": 0.9839,
                    "component_count": 4,
                    "hand_anchor_status": "mismatch",
                    "distance_to_nearest_wrist_px": 273.15,
                    "hand_anchor_threshold_px": 110.15,
                },
                "pose_person": {
                    "status": "ok",
                    "needs_review": True,
                    "review_reasons": ["multiple_people_cold_start"],
                },
                "bad_case": ["string_hand_anchor_mismatch"],
            }
            entries = [
                {
                    "frame": mismatch_record,
                    "image": str(first.resolve()),
                    "overlay_image": str(first.resolve()),
                    "raw_image": str(first_raw.resolve()),
                },
                {
                    "frame": {"frame_index": 10, "timestamp_s": 0.2, "bad_case": []},
                    "image": "review_frames/frame_10.jpg",
                },
                {"frame": {"frame_index": 30}, "image": str(outside.resolve())},
                {"frame": {"frame_index": 40}, "image": "review_frames/missing.jpg"},
            ]
            (run_dir / "tracking_review_index.json").write_text(
                json.dumps(entries),
                encoding="utf-8",
            )

            gallery = _tracking_review_gallery(run_dir)

        self.assertEqual(
            [Path(item[0]).name for item in gallery],
            ["frame_20_raw.jpg", "frame_20.jpg", "frame_10.jpg"],
        )
        self.assertIn("f20 | 0.40s | yoyo=yes", gallery[0][1])
        self.assertIn("components=4", gallery[0][1])
        self.assertIn("hand=mismatch 273/110px", gallery[0][1])
        self.assertIn("pose=review:multiple_people_cold_start", gallery[0][1])
        self.assertIn("bad=string_hand_anchor_mismatch", gallery[0][1])
        self.assertIn("view=raw", gallery[0][1])
        self.assertIn("view=overlay", gallery[1][1])

    def test_tracking_review_gallery_handles_missing_or_malformed_index(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertEqual(_tracking_review_gallery(run_dir), [])
            (run_dir / "tracking_review_index.json").write_text("{", encoding="utf-8")
            self.assertEqual(_tracking_review_gallery(run_dir), [])

    def test_tracking_review_caption_handles_absent_observations(self):
        caption = _tracking_review_caption(
            {"frame_index": 3, "timestamp_s": 0.1, "bad_case": ["no_yoyo"]}
        )

        self.assertIn("yoyo=no", caption)
        self.assertIn("string=none:- components=0", caption)
        self.assertIn("hand=-", caption)
        self.assertIn("pose=missing", caption)

    def test_tracking_without_video_returns_all_gallery_outputs(self):
        outputs = run_video_tracking(
            None, "detector.pt", "runs/tracking", 0.25, 0.7, 640, "cpu",
            False, "", False, "", 0.2, 1.0, 10.0,
            "unknown", False, 0.0, 180.0, 0.08, 0, 1920,
        )

        self.assertEqual(len(outputs), 14)
        self.assertEqual(outputs[5], [])
        self.assertEqual(outputs[8:13], (None, {}, {}, None, ""))
        self.assertIn("No video provided", outputs[-1])

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_annotation_refinement_commands_forward_holdout_exclusion(self, run_command):
        workbench_candidates("datasets/video_v1", "detector.pt", 1.0, 0.2, 5, "ab03bb7118b0")
        self.assertIn("--exclude-source-groups", run_command.call_args.args[0])
        run_command.reset_mock()
        workbench_vlm("datasets/video_v1", "train", 4, 2, True, "ab03bb7118b0")
        self.assertIn("--exclude-source-groups", run_command.call_args.args[0])
        run_command.reset_mock()
        workbench_prelabel_strings("datasets/video_v1", "train", 4, "ab03bb7118b0")
        self.assertIn("--exclude-source-groups", run_command.call_args.args[0])

    def test_generic_review_paths_exclude_holdout_source_groups(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for group in ("ab03bb7118b0", "kept"):
                path = root / "annotations" / "labels" / "train" / group / "frame.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    '{"source_group": "%s", "split": "train", "review_status": "auto_labeled_needs_review"}' % group,
                    encoding="utf-8",
                )
            paths = _review_label_paths(str(root), "auto_labeled_needs_review", "train", "all", "ab03bb7118b0")
        self.assertEqual([path.parent.name for path in paths], ["kept"])

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_string_review_queue_forwards_holdout_exclusion(self, run_command):
        log, sheet = workbench_string_review_queue(
            "datasets/video_v1",
            "train",
            16,
            True,
            "runs/semantic/current/weights/best.pt",
            "cuda",
            "ab03bb7118b0",
            "agreement",
        )
        self.assertEqual(log, "ok")
        self.assertTrue(str(sheet).endswith("string_review_queue.jpg"))
        args = run_command.call_args.args[0]
        self.assertEqual(args[args.index("--split") + 1], "train")
        self.assertEqual(args[args.index("--exclude-source-groups") + 1], "ab03bb7118b0")
        self.assertEqual(args[args.index("--strategy") + 1], "agreement")

    @patch("workbench.commands._run_workbench_command")
    def test_agreement_review_queue_requires_model(self, run_command):
        log, sheet = workbench_string_review_queue(
            "datasets/video_v1", "train", 16, False, "unused.pt", "cpu", "", "agreement"
        )

        self.assertIn("requires the current semantic model", log)
        self.assertIsNone(sheet)
        run_command.assert_not_called()

    def test_workbench_reads_current_derived_holdout_policy(self):
        groups, exclude_original_test = _workbench_holdout_defaults()

        self.assertIn("ab03bb7118b0", groups)
        self.assertTrue(exclude_original_test)

    @patch("workbench.commands._run_workbench_command")
    def test_existing_dataset_versions_are_immutable_by_default(self, run_command):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "string_seg_v1"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

            result = workbench_prepare_string("datasets/video_v1", str(output_dir))

        self.assertIn("immutable by default", result)
        run_command.assert_not_called()

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_string_training_reuses_manifest_without_rebuilding_dataset(self, run_command):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "string_seg_candidate"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

            result = workbench_train_string(
                "datasets/video_v1",
                str(output_dir),
                2,
                "cpu",
                "workbench_string_unique_test",
            )

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("--no-prepare", args)
        self.assertNotIn("--clear-dataset", args)
        self.assertEqual(args[args.index("--name") + 1], "workbench_string_unique_test")

    @patch("workbench.commands._run_workbench_command")
    def test_string_training_refuses_missing_manifest(self, run_command):
        with TemporaryDirectory() as temp_dir:
            result = workbench_train_string(
                "datasets/video_v1",
                str(Path(temp_dir) / "missing_dataset"),
                2,
                "cpu",
                "workbench_string_missing_manifest",
            )

        self.assertIn("manifest is missing", result)
        run_command.assert_not_called()

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_exports_forward_derived_holdout_policy(self, run_command):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "string_seg_candidate"
            result = workbench_prepare_string(
                "datasets/video_v1",
                str(output_dir),
                False,
                "holdout-a,holdout-b",
                True,
            )

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertEqual(args[args.index("--holdout-source-groups") + 1], "holdout-a,holdout-b")
        self.assertIn("--exclude-original-test", args)

    @patch("workbench.commands._run_workbench_command")
    def test_yolo_export_requires_explicit_replacement(self, run_command):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "yolo_v1"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

            result = workbench_qa_export("datasets/video_v1", str(output_dir))

        self.assertIn("immutable by default", result)
        run_command.assert_not_called()

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_pipeline_evaluation_defaults_to_explicit_reviewed_inputs(self, run_command):
        log, sheet = workbench_evaluate_pipeline(
            "datasets/video_v1",
            "datasets/video_v1/string_seg_v15",
            "runs/yolo/v6/best.pt",
            "runs/semantic/v3/best.pt",
            2.0,
            "val",
            "runs/pipeline_eval/trial",
            "0",
            False,
        )

        self.assertEqual(log, "ok")
        self.assertIsNone(sheet)
        args = run_command.call_args.args[0]
        self.assertEqual(args[args.index("--split") + 1], "val")
        self.assertEqual(args[args.index("--detector-weights") + 1], "runs/yolo/v6/best.pt")
        self.assertEqual(args[args.index("--string-weights") + 1], "runs/semantic/v3/best.pt")
        self.assertEqual(args[args.index("--string-inference-scale") + 1], "2.0")
        self.assertEqual(args[args.index("--annotations-dir") + 1], str(Path("datasets/video_v1") / "annotations"))

    @patch("workbench.commands._run_workbench_command", return_value="should not run")
    def test_pipeline_test_requires_explicit_confirmation(self, run_command):
        log, sheet = workbench_evaluate_pipeline(
            "datasets/video_v1",
            "datasets/video_v1/string_seg_v15",
            "detector.pt",
            "string.pt",
            2.0,
            "test",
            "runs/pipeline_eval/trial",
            "0",
            False,
        )

        self.assertIn("requires explicit", log)
        self.assertIsNone(sheet)
        run_command.assert_not_called()

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_semantic_training_exposes_transfer_and_negative_controls(self, run_command):
        result = workbench_train_semantic(
            "datasets/video_v1/string_seg_v8",
            "runs/semantic",
            "trial",
            12,
            "cuda",
            "lraspp_mobilenet_v3",
            True,
            "runs/semantic/base/weights/best.pt",
            0.0002,
            0.005,
            8,
            12,
        )

        self.assertEqual(result, "ok")
        args = run_command.call_args.args[0]
        self.assertIn("--pretrained-backbone", args)
        self.assertEqual(args[args.index("--architecture") + 1], "lraspp_mobilenet_v3")
        self.assertEqual(args[args.index("--initial-weights") + 1], "runs/semantic/base/weights/best.pt")
        self.assertEqual(args[args.index("--hard-negative-weight") + 1], "0.005")
        self.assertEqual(args[args.index("--early-stopping-patience") + 1], "8")
        self.assertEqual(args[args.index("--early-stopping-min-epochs") + 1], "12")

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_hard_negative_queue_uses_explicit_weights_and_name(self, run_command):
        log, sheet, queue = workbench_hard_negative_queue(
            "datasets/video_v1",
            "runs/semantic/base/weights/best.pt",
            "cuda",
            "queue_trial",
            "ab03bb7118b0",
        )

        self.assertEqual(log, "ok")
        self.assertIsNone(sheet)
        self.assertEqual(queue, str(Path("datasets/video_v1/queue_trial.json")))
        args = run_command.call_args.args[0]
        self.assertEqual(args[args.index("--weights") + 1], "runs/semantic/base/weights/best.pt")
        self.assertEqual(args[args.index("--output-name") + 1], "queue_trial")
        self.assertEqual(args[args.index("--exclude-source-groups") + 1], "ab03bb7118b0")

    @patch("workbench.commands._run_workbench_command", return_value="ok")
    def test_neighbor_candidates_expose_review_only_expansion_controls(self, run_command):
        log, sheet, candidates = workbench_hard_negative_neighbors(
            "datasets/video_v1",
            "datasets/video_v1/queue.json",
            "-0.5,0.5",
            12,
            48,
            True,
            True,
            "neighbor_trial",
            "ab03bb7118b0",
        )

        self.assertEqual(log, "ok")
        self.assertIsNone(sheet)
        self.assertEqual(candidates, str(Path("datasets/video_v1/neighbor_trial.json")))
        args = run_command.call_args.args[0]
        self.assertIn("--include-yoyo-visible", args)
        self.assertIn("--include-clean-anchors", args)
        self.assertEqual(args[args.index("--offset-seconds") + 1], "-0.5,0.5")
        self.assertEqual(args[args.index("--exclude-source-groups") + 1], "ab03bb7118b0")


if __name__ == "__main__":
    unittest.main()
