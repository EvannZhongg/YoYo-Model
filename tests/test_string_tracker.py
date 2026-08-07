import unittest
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from string_segmentation.semantic_model import PreparedCalibratedEnsemblePredictor
from video_tracking.string_tracker import (
    _color_line_observation,
    _resample_polyline,
    _saturated_line_mask,
    estimate_string,
    propagate_optical_flow,
    update_adaptive_string_domain_gate,
)
from video_tracking.review_sheet import (
    _pose_caption,
    _select_rows,
    _string_caption,
    make_tracking_review_sheet,
)
from video_tracking.tracker import (
    _AsyncVideoWriter,
    _augment_semantic_color_observation,
    _draw_frame,
    _can_seed_previous_string,
    _inference_interval_frames,
    _load_string_model,
    _prepare_semantic_letterbox,
    _predict_string_model,
    _select_pose_person,
    _should_reacquire_string,
    _should_run_scheduled_string_model,
    _visualization_size,
)


class StringTrackerTemporalTests(unittest.TestCase):
    def test_scheduled_string_model_requires_a_consumable_anchor(self):
        self.assertFalse(
            _should_run_scheduled_string_model(True, None, None, None, 20, 8)
        )
        self.assertTrue(
            _should_run_scheduled_string_model(
                True, {"center": [1.0, 1.0]}, None, None, 20, 8,
            )
        )
        self.assertTrue(
            _should_run_scheduled_string_model(
                True, None, {"points": [[0.0, 0.0], [1.0, 1.0]]}, None, 20, 8,
            )
        )
        self.assertTrue(
            _should_run_scheduled_string_model(True, None, None, 13, 20, 8)
        )
        self.assertFalse(
            _should_run_scheduled_string_model(True, None, None, 11, 20, 8)
        )
        self.assertFalse(
            _should_run_scheduled_string_model(
                False, {"center": [1.0, 1.0]}, None, None, 20, 8,
            )
        )

    def test_async_video_writer_preserves_order_and_releases_backend(self):
        class FakeWriter:
            def __init__(self):
                self.values = []
                self.released = False

            def write(self, frame):
                self.values.append(int(frame[0, 0]))

            def release(self):
                self.released = True

        backend = FakeWriter()
        writer = _AsyncVideoWriter(backend, queue_size=1)
        for value in (3, 1, 4, 1, 5):
            writer.write(np.full((1, 1), value, dtype=np.uint8))
        writer.release()

        self.assertEqual(backend.values, [3, 1, 4, 1, 5])
        self.assertTrue(backend.released)

    def test_async_video_writer_surfaces_background_failure(self):
        class FailingWriter:
            def write(self, _):
                raise OSError("encode failed")

            def release(self):
                pass

        writer = _AsyncVideoWriter(FailingWriter(), queue_size=1)
        writer.write(np.zeros((1, 1), dtype=np.uint8))

        with self.assertRaisesRegex(RuntimeError, "Background video write failed"):
            writer.release()

    def test_color_observation_resamples_temporal_reference_once_per_frame(self):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        yoyo = {"center": [120.0, 80.0], "bbox": [110.0, 70.0, 130.0, 90.0]}
        lines = np.asarray(
            [[[10, 10, 80, 20]], [[20, 30, 100, 40]], [[30, 50, 120, 60]]],
            dtype=np.int32,
        )

        with (
            patch("video_tracking.string_tracker.cv2.HoughLinesP", return_value=lines),
            patch(
                "video_tracking.string_tracker._resample_polyline",
                wraps=_resample_polyline,
            ) as resample,
        ):
            result = _color_line_observation(
                frame,
                yoyo,
                require_yoyo_proximity=False,
                reference_points=[[100.0, 80.0], [140.0, 80.0]],
            )

        self.assertIsNotNone(result)
        self.assertEqual(resample.call_count, 1)

    def test_color_observation_prefilter_requires_semantic_support(self):
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.line(frame, (120, 90), (220, 30), (0, 255, 0), 4)
        yoyo = {"center": [120.0, 90.0], "bbox": [108.0, 78.0, 132.0, 102.0]}
        meta = SimpleNamespace(scale=1.0, pad_x=0, pad_y=0)

        self.assertIsNone(
            _color_line_observation(
                frame, yoyo, require_yoyo_proximity=False,
                semantic_probability=np.zeros((180, 240), dtype=np.float32),
                semantic_meta=meta,
            )
        )
        self.assertIsNotNone(
            _color_line_observation(
                frame, yoyo, require_yoyo_proximity=False,
                semantic_probability=np.ones((180, 240), dtype=np.float32),
                semantic_meta=meta,
            )
        )

    def test_semantic_color_mask_crop_matches_full_roi_processing(self):
        rng = np.random.default_rng(42)
        roi = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        supports = []
        for bounds in ((24, 20, 70, 60), (0, 0, 35, 28), (90, 65, 128, 96)):
            support = np.zeros(roi.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bounds
            support[y1:y2, x1:x2] = 1
            supports.append(support)
        disjoint = np.zeros(roi.shape[:2], dtype=np.uint8)
        disjoint[15:25, 10:30] = 1
        disjoint[70:85, 100:120] = 1
        supports.append(disjoint)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        full = cv2.inRange(
            hsv,
            np.asarray([35, 70, 55], dtype=np.uint8),
            np.asarray([179, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, kernel, iterations=1)
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel, iterations=1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_ridge = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        bright = cv2.bitwise_and(
            cv2.inRange(
                hsv,
                np.asarray([0, 0, 120], dtype=np.uint8),
                np.asarray([179, 69, 255], dtype=np.uint8),
            ),
            cv2.inRange(bright_ridge, 12, 255),
        )
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)
        full = cv2.bitwise_or(full, bright)

        for support in supports:
            np.testing.assert_array_equal(
                _saturated_line_mask(roi, support, include_bright_lines=True),
                cv2.bitwise_and(full, support),
            )

    def test_estimate_string_reuses_precomputed_current_gray(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        current_gray = np.zeros((80, 120), dtype=np.uint8)
        observation = {
            "points": [[20.0, 20.0], [80.0, 60.0]],
            "confidence": 0.8,
            "method": "semantic_segmentation",
        }

        previous_string = {
            "points": [[18.0, 20.0], [78.0, 60.0]],
            "confidence": 0.7,
            "method": "semantic_segmentation",
        }
        with (
            patch("video_tracking.string_tracker.cv2.cvtColor") as convert,
            patch("video_tracking.string_tracker._propagate_string_geometry") as propagate,
        ):
            result = estimate_string(
                frame,
                None,
                [],
                current_gray,
                previous_string,
                observation=observation,
                allow_unanchored_semantic=True,
                current_gray=current_gray,
            )

        self.assertIsNotNone(result)
        convert.assert_not_called()
        propagate.assert_not_called()

    def test_estimate_string_defers_gray_conversion_until_flow_gap(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        previous_frame = np.full_like(frame, 7)
        previous_string = {
            "points": [[18.0, 20.0], [78.0, 60.0]],
            "confidence": 0.7,
            "method": "semantic_segmentation",
        }
        with (
            patch(
                "video_tracking.string_tracker.cv2.cvtColor",
                side_effect=lambda image, _: image[:, :, 0],
            ) as convert,
            patch("video_tracking.string_tracker._propagate_string_geometry", return_value=None) as propagate,
        ):
            result = estimate_string(
                frame,
                None,
                [],
                None,
                previous_string,
                allow_color_fallback=False,
                previous_frame=previous_frame,
            )

        self.assertIsNone(result)
        self.assertEqual(convert.call_count, 2)
        self.assertEqual(propagate.call_count, 1)
        np.testing.assert_array_equal(propagate.call_args.args[0], previous_frame[:, :, 0])
        np.testing.assert_array_equal(propagate.call_args.args[1], frame[:, :, 0])

    def test_adaptive_domain_gate_rejects_strong_or_near_observations(self):
        for confidence, distance in ((0.90, 100.0), (0.75, 20.0)):
            history = []
            for _ in range(12):
                history, triggered, _ = update_adaptive_string_domain_gate(
                    history,
                    {
                        "method": "semantic_segmentation",
                        "confidence": confidence,
                        "distance_to_yoyo_px": distance,
                    },
                    3840,
                    2160,
                    12,
                    0,
                    0.82,
                    0.018,
                )
            self.assertFalse(triggered)

    def test_adaptive_domain_gate_requires_persistent_joint_evidence(self):
        history = []
        triggered = False
        for _ in range(12):
            history, triggered, metrics = update_adaptive_string_domain_gate(
                history,
                {
                    "method": "semantic_segmentation",
                    "confidence": 0.75,
                    "distance_to_yoyo_px": 100.0,
                },
                3840,
                2160,
                12,
                0,
                0.82,
                0.018,
            )

        self.assertTrue(triggered)
        self.assertEqual(metrics["color_accepts"], 0)
        self.assertLess(metrics["mean_confidence"], 0.82)
        self.assertGreater(metrics["mean_distance_ratio"], 0.018)

        _, color_triggered, _ = update_adaptive_string_domain_gate(
            history[:-1],
            {
                "method": "semantic_color_probability_union",
                "confidence": 0.75,
                "distance_to_yoyo_px": 100.0,
            },
            3840,
            2160,
            12,
            0,
            0.82,
            0.018,
        )
        self.assertFalse(color_triggered)

        _, bright_triggered, bright_metrics = update_adaptive_string_domain_gate(
            history[:-1],
            {
                "method": "semantic_color_probability_union",
                "color_line_features": "saturated_and_bright",
                "confidence": 0.75,
                "distance_to_yoyo_px": 100.0,
            },
            3840,
            2160,
            12,
            0,
            0.82,
            0.018,
        )
        self.assertTrue(bright_triggered)
        self.assertEqual(bright_metrics["color_accepts"], 0)

    def test_semantic_ensemble_fuses_probabilities_before_geometry(self):
        meta = SimpleNamespace(
            original_width=16,
            original_height=16,
            target_width=16,
            target_height=16,
            resized_width=16,
            resized_height=16,
            pad_x=0,
            pad_y=0,
            scale=1.0,
        )
        model = {
            "kind": "semantic_ensemble",
            "model": object(),
            "checkpoint": {
                "threshold": 0.4,
                "model_config": {"input_width": 16, "input_height": 16},
            },
            "ensemble_model": object(),
            "ensemble_alpha": 0.3,
            "ensemble_candidate_threshold": 0.5,
            "device": "cpu",
        }
        observation = {"points": [[1.0, 1.0], [4.0, 4.0]], "polylines": []}

        with (
            patch(
                "video_tracking.tracker.prepare_letterboxed_input",
                return_value=(object(), meta),
            ) as prepare,
            patch(
                "video_tracking.tracker.predict_prepared_calibrated_ensemble",
                return_value=np.full((16, 16), 0.5, dtype=np.float32),
            ) as predict,
            patch(
                "video_tracking.tracker.semantic_mask_observation",
                return_value=observation,
            ) as geometry,
        ):
            result = _predict_string_model(
                model,
                np.zeros((16, 16, 3), dtype=np.uint8),
                yoyo=None,
                confidence=0.2,
                imgsz=16,
                device="cpu",
                yoyo_division="1A",
            )

        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(predict.call_count, 1)
        self.assertIs(predict.call_args.args[0], model["model"])
        self.assertIs(predict.call_args.args[1], model["ensemble_model"])
        self.assertEqual(predict.call_args.args[3:], (0.3, 0.4, 0.5))
        self.assertAlmostEqual(float(geometry.call_args.args[0][0, 0]), 0.5, places=5)
        self.assertEqual(geometry.call_args.kwargs["threshold"], 0.5)
        self.assertEqual(result["semantic_probability_ensemble"]["alpha"], 0.3)

    def test_prepared_semantic_letterbox_skips_duplicate_resize(self):
        meta = SimpleNamespace(
            original_width=32,
            original_height=32,
            target_width=32,
            target_height=32,
            resized_width=32,
            resized_height=32,
            pad_x=0,
            pad_y=0,
            scale=1.0,
        )
        model = {
            "kind": "semantic_ensemble",
            "model": object(),
            "checkpoint": {
                "threshold": 0.4,
                "model_config": {"input_width": 32, "input_height": 32},
            },
            "ensemble_model": object(),
            "ensemble_alpha": 0.3,
            "ensemble_candidate_threshold": 0.5,
            "device": "cpu",
        }
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        tensor = object()
        observation = {"points": [[1.0, 1.0], [4.0, 4.0]], "polylines": []}

        with (
            patch("video_tracking.tracker.prepare_letterboxed_input") as prepare,
            patch(
                "video_tracking.tracker.normalize_image_for_inference",
                return_value=tensor,
            ) as normalize,
            patch(
                "video_tracking.tracker.predict_prepared_calibrated_ensemble",
                return_value=np.full((32, 32), 0.5, dtype=np.float32),
            ) as predict,
            patch(
                "video_tracking.tracker.semantic_mask_observation",
                return_value=observation,
            ),
        ):
            result = _predict_string_model(
                model,
                image,
                None,
                0.2,
                32,
                "cpu",
                "1A",
                prepared_letterbox=(image, None, meta),
            )

        prepare.assert_not_called()
        normalize.assert_called_once_with(image, "cpu")
        self.assertIs(predict.call_args.args[2], tensor)
        self.assertEqual(result["points"], observation["points"])

    def test_scaled_semantic_inference_keeps_fixed_component_filter(self):
        meta = SimpleNamespace(
            original_width=1920, original_height=1080,
            target_width=1440, target_height=816,
            resized_width=1440, resized_height=810,
            pad_x=0, pad_y=3, scale=0.75,
        )
        model = {
            "kind": "semantic",
            "model": object(),
            "checkpoint": {
                "threshold": 0.4,
                "model_config": {"input_width": 960, "input_height": 544},
            },
            "device": "cpu",
        }

        with (
            patch(
                "video_tracking.tracker.prepare_letterboxed_input",
                return_value=(object(), meta),
            ) as prepare,
            patch(
                "video_tracking.tracker.predict_prepared_probability",
                return_value=np.full((816, 1440), 0.5, dtype=np.float32),
            ),
            patch(
                "video_tracking.tracker.semantic_mask_observation",
                return_value={"points": [[1.0, 1.0], [4.0, 4.0]], "polylines": []},
            ) as geometry,
        ):
            _predict_string_model(
                model,
                np.zeros((1080, 1920, 3), dtype=np.uint8),
                None,
                0.2,
                1024,
                "cpu",
                "1A",
                semantic_inference_scale=1.5,
            )

        self.assertEqual(prepare.call_args.args[1:3], (1440, 816))
        self.assertEqual(geometry.call_args.kwargs["min_component_pixels"], 8)

    def test_parallel_semantic_letterbox_uses_active_model_size(self):
        model = {
            "kind": "semantic_adaptive_ensemble",
            "checkpoint": {"model_config": {"input_width": 64, "input_height": 32}},
            "adaptive_checkpoint": {
                "model_config": {"input_width": 96, "input_height": 64},
            },
            "adaptive_enabled": True,
        }
        expected = (np.zeros((64, 96, 3), dtype=np.uint8), None, object())

        with patch("video_tracking.tracker.letterbox", return_value=expected) as resize:
            actual = _prepare_semantic_letterbox(
                model,
                np.zeros((20, 40, 3), dtype=np.uint8),
                1.0,
            )

        self.assertIs(actual, expected)
        self.assertEqual(resize.call_args.args[1:], (96, 64))

    def test_adaptive_ensemble_selects_primary_and_alpha_from_state(self):
        meta = SimpleNamespace(
            original_width=16, original_height=16, target_width=16, target_height=16,
            resized_width=16, resized_height=16, pad_x=0, pad_y=0, scale=1.0,
        )
        primary, adaptive, secondary = object(), object(), object()
        model = {
            "kind": "semantic_adaptive_ensemble",
            "model": primary,
            "checkpoint": {
                "threshold": 0.4,
                "model_config": {"input_width": 16, "input_height": 16},
            },
            "adaptive_model": adaptive,
            "adaptive_checkpoint": {
                "threshold": 0.45,
                "model_config": {"input_width": 16, "input_height": 16},
            },
            "ensemble_model": secondary,
            "ensemble_alpha": 0.3,
            "adaptive_ensemble_alpha": 0.5,
            "ensemble_candidate_threshold": 0.5,
            "adaptive_enabled": False,
            "device": "cpu",
        }
        observation = {"points": [[1.0, 1.0], [4.0, 4.0]], "polylines": []}
        with (
            patch(
                "video_tracking.tracker.prepare_letterboxed_input",
                return_value=(object(), meta),
            ) as prepare,
            patch(
                "video_tracking.tracker.predict_prepared_calibrated_ensemble",
                return_value=np.full((16, 16), 0.5, dtype=np.float32),
            ) as predict,
            patch("video_tracking.tracker.semantic_mask_observation", return_value=observation),
        ):
            before = _predict_string_model(
                model, np.zeros((16, 16, 3), dtype=np.uint8), None, 0.2, 16, "cpu", "1A",
            )
            before_ensemble = deepcopy(before["semantic_probability_ensemble"])
            model["adaptive_enabled"] = True
            after = _predict_string_model(
                model, np.zeros((16, 16, 3), dtype=np.uint8), None, 0.2, 16, "cpu", "1A",
            )

        self.assertEqual(prepare.call_count, 2)
        self.assertIs(predict.call_args_list[0].args[0], primary)
        self.assertIs(predict.call_args_list[0].args[1], secondary)
        self.assertEqual(predict.call_args_list[0].args[3:], (0.3, 0.4, 0.5))
        self.assertIs(predict.call_args_list[1].args[0], adaptive)
        self.assertIs(predict.call_args_list[1].args[1], secondary)
        self.assertEqual(predict.call_args_list[1].args[3:], (0.5, 0.45, 0.5))
        self.assertEqual(before_ensemble["alpha"], 0.3)
        self.assertFalse(before_ensemble["adaptive_primary"])
        self.assertEqual(after["semantic_probability_ensemble"]["alpha"], 0.5)
        self.assertTrue(after["semantic_probability_ensemble"]["adaptive_primary"])

    def test_load_string_model_builds_adaptive_ensemble(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("primary.pt", "secondary.pt", "adaptive.pt")]
            for path in paths:
                path.touch()
            checkpoints = [
                {"model_config": {"input_width": 16, "input_height": 16}},
                {"model_config": {"input_width": 16, "input_height": 16}},
                {"model_config": {"input_width": 16, "input_height": 16}},
            ]
            with (
                patch("video_tracking.tracker.is_semantic_checkpoint", return_value=True),
                patch("video_tracking.tracker.load_semantic_checkpoint", side_effect=[
                    ("primary", checkpoints[0]),
                    ("secondary", checkpoints[1]),
                    ("adaptive", checkpoints[2]),
                ]),
            ):
                model, status = _load_string_model(
                    paths[0], True, "cpu", paths[1], 0.3, 0.5, paths[2], 0.5,
                )

        self.assertEqual(model["kind"], "semantic_adaptive_ensemble")
        self.assertEqual(model["adaptive_model"], "adaptive")
        self.assertFalse(model["adaptive_enabled"])
        self.assertEqual(model["adaptive_ensemble_alpha"], 0.5)
        self.assertIsInstance(
            model["ensemble_predictor"], PreparedCalibratedEnsemblePredictor,
        )
        self.assertIsInstance(
            model["adaptive_ensemble_predictor"], PreparedCalibratedEnsemblePredictor,
        )
        self.assertTrue(status.startswith("semantic_adaptive_ensemble:"))

    def test_semantic_probability_gate_controls_color_augmentation(self):
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.line(frame, (120, 90), (200, 40), (0, 255, 0), 4)
        yoyo = {"center": [120.0, 90.0], "bbox": [108.0, 78.0, 132.0, 102.0]}
        observation = {
            "points": [[120.0, 90.0], [150.0, 70.0]],
            "polylines": [[[120.0, 90.0], [150.0, 70.0]]],
            "confidence": 0.8,
            "method": "semantic_segmentation",
        }
        meta = SimpleNamespace(scale=1.0, pad_x=0, pad_y=0)

        accepted = _augment_semantic_color_observation(
            frame,
            yoyo,
            observation,
            np.ones((180, 240), dtype=np.float32),
            meta,
            threshold=0.4,
            min_mean=0.4,
            min_fraction_at_0_10=0.5,
        )
        rejected = _augment_semantic_color_observation(
            frame,
            yoyo,
            observation,
            np.zeros((180, 240), dtype=np.float32),
            meta,
            threshold=0.4,
            min_mean=0.4,
            min_fraction_at_0_10=0.5,
        )

        self.assertEqual(accepted["method"], "semantic_color_probability_union")
        self.assertEqual(len(accepted["polylines"]), 2)
        self.assertEqual(rejected, observation)

    def test_bright_ridge_fallback_recovers_low_saturation_string(self):
        frame = np.full((180, 240, 3), (130, 160, 190), dtype=np.uint8)
        cv2.line(frame, (120, 90), (205, 38), (235, 235, 235), 4)
        yoyo = {"center": [120.0, 90.0], "bbox": [108.0, 78.0, 132.0, 102.0]}
        observation = {
            "points": [[120.0, 90.0], [150.0, 70.0]],
            "polylines": [[[120.0, 90.0], [150.0, 70.0]]],
            "confidence": 0.8,
            "method": "semantic_segmentation",
        }
        probability = np.full((180, 240), 0.5, dtype=np.float32)
        meta = SimpleNamespace(scale=1.0, pad_x=0, pad_y=0)

        saturated_only = _augment_semantic_color_observation(
            frame, yoyo, observation, probability, meta, 0.4, 0.4, 0.5,
        )
        with_bright_ridge = _augment_semantic_color_observation(
            frame, yoyo, observation, probability, meta, 0.4, 0.4, 0.5,
            include_bright_lines=True,
            bright_line_min_mean=0.4,
        )
        strict_bright_ridge = _augment_semantic_color_observation(
            frame, yoyo, observation, probability, meta, 0.4, 0.4, 0.5,
            include_bright_lines=True,
            bright_line_min_mean=0.7,
        )

        self.assertEqual(saturated_only, observation)
        self.assertEqual(with_bright_ridge["method"], "semantic_color_probability_union")
        self.assertEqual(with_bright_ridge["color_line_features"], "saturated_and_bright")
        self.assertEqual(strict_bright_ridge, observation)

    def test_pose_person_selection_prefers_visible_wrists_near_yoyo(self):
        points = np.zeros((2, 17, 2), dtype=np.float32)
        confidence = np.zeros((2, 17), dtype=np.float32)
        confidence[0, :5] = 0.9
        points[1, 9] = [490.0, 300.0]
        points[1, 10] = [520.0, 310.0]
        confidence[1, :] = 0.8
        boxes = np.asarray([[0, 400, 300, 700], [300, 50, 700, 700]], dtype=np.float32)
        selection = _select_pose_person(
            points, confidence, boxes, {"center": [500.0, 320.0]}, 1280, 720,
        )

        self.assertIsNotNone(selection)
        index, metadata = selection
        self.assertEqual(index, 1)
        self.assertEqual(metadata["visible_wrist_count"], 2)
        self.assertLess(metadata["nearest_wrist_to_yoyo_px"], 40.0)
        self.assertTrue(metadata["needs_review"])
        self.assertIn("multiple_people_cold_start", metadata["review_reasons"])

    def test_pose_person_selection_keeps_temporal_performer_without_yoyo_or_wrists(self):
        points = np.zeros((2, 17, 2), dtype=np.float32)
        confidence = np.full((2, 17), 0.99, dtype=np.float32)
        confidence[1, 9:11] = 0.05
        boxes = np.asarray([
            [20.0, 100.0, 260.0, 700.0],
            [405.0, 55.0, 805.0, 705.0],
        ], dtype=np.float32)

        selection = _select_pose_person(
            points,
            confidence,
            boxes,
            None,
            1280,
            720,
            [400.0, 50.0, 800.0, 700.0],
        )

        self.assertIsNotNone(selection)
        index, metadata = selection
        self.assertEqual(index, 1)
        self.assertEqual(metadata["visible_wrist_count"], 0)
        self.assertTrue(metadata["temporal_reference_used"])
        self.assertGreater(metadata["temporal_bbox_iou"], 0.95)
        self.assertFalse(metadata["needs_review"])

    def test_pose_person_selection_discards_distant_temporal_reference(self):
        points = np.zeros((2, 17, 2), dtype=np.float32)
        confidence = np.full((2, 17), 0.8, dtype=np.float32)
        points[0, 9:11] = [[140.0, 300.0], [160.0, 300.0]]
        points[1, 9:11] = [[590.0, 300.0], [610.0, 300.0]]
        boxes = np.asarray([
            [20.0, 50.0, 280.0, 700.0],
            [450.0, 50.0, 750.0, 700.0],
        ], dtype=np.float32)

        selection = _select_pose_person(
            points,
            confidence,
            boxes,
            {"center": [600.0, 320.0]},
            1920,
            1080,
            [1450.0, 100.0, 1850.0, 1000.0],
        )

        self.assertIsNotNone(selection)
        index, metadata = selection
        self.assertEqual(index, 1)
        self.assertTrue(metadata["temporal_reference_available"])
        self.assertFalse(metadata["temporal_reference_used"])
        self.assertTrue(metadata["needs_review"])
        self.assertIn("temporal_reference_rejected", metadata["review_reasons"])
        self.assertEqual(
            metadata["selection_method"],
            "person_extent_pose_quality_then_yoyo_proximity",
        )

    def test_tracking_review_sampler_keeps_events_and_full_video_context(self):
        rows = []
        for index in range(100):
            pose_person = {
                "status": "ok",
                "person_index": 0,
                "person_count": 2,
                "needs_review": index == 70,
            }
            rows.append({
                "frame_index": index,
                "yoyo": {"center": [100.0, 100.0]},
                "string": {"method": "semantic_segmentation"},
                "pose_person": pose_person,
                "bad_case": ["pose_identity_needs_review"] if index == 70 else [],
            })

        selected = _select_rows(rows, 8)
        selected_frames = [row["frame_index"] for row in selected]

        self.assertEqual(len(selected), 8)
        self.assertIn(0, selected_frames)
        self.assertIn(70, selected_frames)
        self.assertIn(99, selected_frames)
        self.assertGreater(sum(frame > 50 for frame in selected_frames), 2)

    def test_tracking_review_index_pairs_source_raw_and_overlay_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            source_path = root / "source.mp4"
            tracked_path = run_dir / "tracked.mp4"

            source_writer = cv2.VideoWriter(
                str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
            )
            tracked_writer = cv2.VideoWriter(
                str(tracked_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
            )
            self.assertTrue(source_writer.isOpened())
            self.assertTrue(tracked_writer.isOpened())
            source_colors = [(230, 0, 0), (0, 230, 0), (0, 0, 230), (180, 180, 180)]
            for color in source_colors:
                source_writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
            for color in ((40, 40, 40), (90, 90, 90)):
                tracked_writer.write(np.full((24, 32, 3), color, dtype=np.uint8))
            source_writer.release()
            tracked_writer.release()

            records = [
                {"frame_index": 1, "timestamp_s": 0.1, "bad_case": []},
                {"frame_index": 2, "timestamp_s": 0.2, "bad_case": []},
            ]
            (run_dir / "frames.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            make_tracking_review_sheet(
                run_dir,
                max_samples=2,
                columns=2,
                thumb_width=64,
                source_video_path=source_path,
            )
            index = json.loads(
                (run_dir / "tracking_review_index.json").read_text(encoding="utf-8")
            )
            first_raw = cv2.imread(index[0]["raw_image"], cv2.IMREAD_COLOR)

            self.assertEqual([item["source_frame_index"] for item in index], [1, 2])
            self.assertEqual(index[0]["raw_image_size"], [64, 48])
            self.assertEqual(index[0]["overlay_image_size"], [32, 24])
            self.assertNotEqual(index[0]["raw_image"], index[0]["overlay_image"])
            self.assertTrue(Path(index[0]["raw_image"]).is_file())
            self.assertTrue(Path(index[0]["overlay_image"]).is_file())
            self.assertIsNotNone(first_raw)
            self.assertGreater(float(first_raw[:, :, 1].mean()), float(first_raw[:, :, 0].mean()))
            self.assertGreater(float(first_raw[:, :, 1].mean()), float(first_raw[:, :, 2].mean()))

    def test_pose_review_caption_exposes_identity_review(self):
        record = {
            "frame_index": 7,
            "timestamp_s": 0.14,
            "pose_person": {
                "status": "ok",
                "person_index": 1,
                "person_count": 3,
                "temporal_reference_used": True,
                "temporal_bbox_iou": 0.0876,
                "review_reasons": ["low_temporal_iou"],
            },
            "bad_case": ["pose_identity_needs_review"],
        }

        caption = _pose_caption(record)
        self.assertIn("pose=p1/3 temporal iou=0.088", caption)
        self.assertIn("low_temporal_iou", caption)

    def test_tracking_review_sampler_guarantees_hand_anchor_mismatch(self):
        rows = []
        for index in range(100):
            bad_case = [f"event_{index}"]
            string = {"method": "semantic_segmentation"}
            if index == 53:
                bad_case.append("string_hand_anchor_mismatch")
                string.update(
                    {
                        "hand_anchor_mismatch": True,
                        "distance_to_nearest_wrist_px": 273.15,
                        "hand_anchor_threshold_px": 110.15,
                    }
                )
            rows.append({"frame_index": index, "string": string, "bad_case": bad_case})

        selected = _select_rows(rows, 8)
        mismatch_only = _select_rows(rows, 1)

        self.assertIn(53, [row["frame_index"] for row in selected])
        self.assertEqual([row["frame_index"] for row in mismatch_only], [53])

    def test_string_review_caption_exposes_anchor_distance(self):
        caption = _string_caption(
            {
                "string": {
                    "method": "semantic_segmentation",
                    "confidence": 0.9839,
                    "hand_anchor_status": "mismatch",
                    "distance_to_nearest_wrist_px": 273.15,
                    "hand_anchor_threshold_px": 110.15,
                }
            }
        )

        self.assertEqual(
            caption,
            "string=semantic_segmentation conf=0.9839 hand=mismatch 273/110px",
        )

    def test_visualization_size_caps_only_large_sources(self):
        self.assertEqual(_visualization_size(3840, 2160, 1920), (1920, 1080))
        self.assertEqual(_visualization_size(1280, 720, 1920), (1280, 720))
        self.assertEqual(_visualization_size(3840, 2160, 0), (3840, 2160))

    def test_scaled_visualization_preserves_source_coordinate_metadata(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        detections = [{
            "bbox": [200.0, 120.0, 260.0, 180.0],
            "center": [230.0, 150.0],
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "yoyo",
            "track_id": 1,
        }]
        string = {
            "points": [[120.0, 80.0], [230.0, 150.0]],
            "confidence": 0.8,
            "method": "semantic_segmentation",
        }
        original_detections = deepcopy(detections)
        original_string = deepcopy(string)

        rendered = _draw_frame(
            frame, detections, [], string, {}, {}, 30, 2, 0.6, (320, 180)
        )

        self.assertEqual(rendered.shape[:2], (180, 320))
        self.assertEqual(detections, original_detections)
        self.assertEqual(string, original_string)
        self.assertGreater(int(rendered.sum()), 0)

    def test_semantic_inference_interval_adapts_to_video_fps(self):
        self.assertEqual(_inference_interval_frames(50.0, 10.0), 5)
        self.assertEqual(_inference_interval_frames(30.0, 10.0), 3)
        self.assertEqual(_inference_interval_frames(8.0, 10.0), 1)
        self.assertEqual(_inference_interval_frames(50.0, 0.0), 1)

    def test_failed_cadence_propagation_requests_model_reacquisition(self):
        yoyo = {"center": [120.0, 90.0]}
        previous = {"points": [[20.0, 20.0], [120.0, 90.0]]}

        self.assertTrue(_should_reacquire_string(False, True, yoyo, previous, None))
        self.assertFalse(_should_reacquire_string(True, True, yoyo, previous, None))
        self.assertFalse(_should_reacquire_string(False, True, None, previous, None))
        self.assertTrue(_should_reacquire_string(False, True, yoyo, None, None))
        self.assertFalse(_should_reacquire_string(False, True, yoyo, previous, {"points": []}))

    def _shifted_frames(self):
        height, width = 180, 240
        previous = np.zeros((height, width, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        cv2.line(previous, (35, 90), (145, 90), (255, 255, 255), 3)
        cv2.line(current, (41, 90), (151, 90), (255, 255, 255), 3)
        for x in range(35, 146, 10):
            cv2.circle(previous, (x, 90), 2, (180, 180, 180), -1)
        for x in range(41, 152, 10):
            cv2.circle(current, (x, 90), 2, (180, 180, 180), -1)
        return previous, current

    def test_optical_flow_has_forward_backward_gate(self):
        previous, current = self._shifted_frames()
        result = propagate_optical_flow(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            [[35, 90], [145, 90]],
            current.shape[1],
            current.shape[0],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "lucas_kanade_optical_flow")
        self.assertIn("flow_forward_backward_error", result)
        self.assertLess(result["flow_forward_backward_error"], 4.0)

    def test_optical_flow_uses_local_roi_on_large_frames(self):
        previous = np.zeros((720, 1280, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        cv2.line(previous, (420, 360), (620, 360), (255, 255, 255), 3)
        cv2.line(current, (426, 360), (626, 360), (255, 255, 255), 3)
        for x in range(420, 621, 20):
            cv2.circle(previous, (x, 360), 2, (180, 180, 180), -1)
            cv2.circle(current, (x + 6, 360), 2, (180, 180, 180), -1)

        result = propagate_optical_flow(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            [[420, 360], [620, 360]],
            current.shape[1],
            current.shape[0],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["flow_region"], "roi")
        self.assertLess(result["flow_region_fraction"], 0.5)

    def test_fresh_observation_keeps_geometry_when_flow_agrees(self):
        previous, current = self._shifted_frames()
        previous_string = {
            "points": [[35, 90], [145, 90]],
            "confidence": 0.62,
            "method": "color_hough_observation",
            "propagation_age_frames": 0,
        }
        result = estimate_string(
            current,
            {"center": [41, 90], "bbox": [30, 80, 52, 102]},
            [],
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            previous_string,
            observation={
                "points": [[41, 90], [151, 90]],
                "confidence": 0.80,
                "method": "yolo_segmentation",
                "needs_review": False,
            },
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "yolo_segmentation")
        self.assertEqual(result["propagation_age_frames"], 0)
        self.assertNotIn("temporal_consistent", result)
        self.assertEqual(result["points"], [[41, 90], [151, 90]])

    def test_string_can_persist_without_yoyo_as_review_case(self):
        previous, current = self._shifted_frames()
        result = estimate_string(
            current,
            None,
            [],
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            {
                "points": [[35, 90], [145, 90]],
                "confidence": 0.62,
                "method": "color_hough_observation",
                "propagation_age_frames": 0,
            },
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "lucas_kanade_optical_flow")
        self.assertEqual(result["propagation_age_frames"], 1)

    def test_optical_flow_propagates_all_reviewed_string_components(self):
        previous = np.zeros((360, 640, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        previous_lines = [
            [[300.0, 120.0], [420.0, 120.0]],
            [[80.0, 250.0], [200.0, 250.0]],
        ]
        for start, end in previous_lines:
            cv2.line(previous, tuple(map(int, start)), tuple(map(int, end)), (255, 255, 255), 3)
            cv2.line(
                current,
                (int(start[0] + 4), int(start[1])),
                (int(end[0] + 4), int(end[1])),
                (255, 255, 255),
                3,
            )
            for x in range(int(start[0]), int(end[0]) + 1, 12):
                cv2.circle(previous, (x, int(start[1])), 2, (180, 180, 180), -1)
                cv2.circle(current, (x + 4, int(start[1])), 2, (180, 180, 180), -1)

        result = estimate_string(
            current,
            {"center": [424.0, 120.0], "bbox": [414.0, 110.0, 434.0, 130.0]},
            [{"name": "right_wrist", "x": 84.0, "y": 250.0, "confidence": 0.95}],
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            {
                "points": previous_lines[0],
                "polylines": previous_lines,
                "confidence": 0.7,
                "method": "semantic_segmentation",
                "propagation_age_frames": 0,
            },
            yoyo_division="1A",
            allow_color_fallback=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "lucas_kanade_optical_flow")
        self.assertEqual(result["flow_component_count"], 2)
        self.assertEqual(result["flow_source_component_count"], 2)
        self.assertFalse(result["flow_partial_component_loss"])
        self.assertEqual(len(result["polylines"]), 2)
        self.assertEqual(result["hand_anchor_status"], "not_applicable")

    def test_unanchored_semantic_string_is_suppressed_without_yoyo(self):
        result = estimate_string(
            np.zeros((180, 240, 3), dtype=np.uint8),
            None,
            [],
            None,
            None,
            observation={
                "points": [[20.0, 20.0], [100.0, 100.0]],
                "confidence": 0.9,
                "method": "semantic_segmentation",
            },
        )
        self.assertIsNone(result)

    def test_recent_yoyo_context_can_allow_unanchored_semantic_string(self):
        observation = {
            "points": [[20.0, 20.0], [100.0, 100.0]],
            "confidence": 0.9,
            "method": "semantic_segmentation",
        }

        result = estimate_string(
            np.zeros((180, 240, 3), dtype=np.uint8),
            None,
            [],
            None,
            None,
            observation=observation,
            allow_unanchored_semantic=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "semantic_segmentation")

    def test_wrist_and_yoyo_alone_do_not_create_a_string(self):
        result = estimate_string(
            np.zeros((180, 240, 3), dtype=np.uint8),
            {"center": [120.0, 90.0], "bbox": [108.0, 78.0, 132.0, 102.0]},
            [{"name": "right_wrist", "x": 60.0, "y": 90.0, "confidence": 0.95}],
            None,
            None,
            yoyo_division="1A",
        )
        self.assertIsNone(result)

    def test_division_metadata_does_not_apply_hand_anchor_gate(self):
        result = estimate_string(
            np.zeros((2160, 3840, 3), dtype=np.uint8),
            {"center": [2378.0, 882.0], "bbox": [2340.0, 844.0, 2416.0, 920.0]},
            [{"name": "left_wrist", "x": 1900.0, "y": 1219.0, "confidence": 0.9}],
            None,
            None,
            yoyo_division="1A",
            observation={
                "points": [[2136.0, 956.0], [2323.0, 1083.0]],
                "polygons": [[[2136.0, 956.0], [2323.0, 956.0], [2323.0, 1083.0], [2136.0, 1083.0]]],
                "confidence": 0.8,
                "method": "semantic_segmentation",
                "needs_review": False,
            },
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["hand_anchor_status"], "not_applicable")
        self.assertFalse(result["hand_anchor_mismatch"])
        self.assertTrue(_can_seed_previous_string(result))

    def test_division_metadata_leaves_hand_anchor_not_applicable(self):
        result = estimate_string(
            np.zeros((360, 640, 3), dtype=np.uint8),
            {"center": [240.0, 160.0], "bbox": [230.0, 150.0, 250.0, 170.0]},
            [{"name": "right_wrist", "x": 80.0, "y": 90.0, "confidence": 0.95}],
            None,
            None,
            yoyo_division="1A",
            observation={
                "points": [[70.0, 90.0], [240.0, 160.0]],
                "confidence": 0.8,
                "method": "semantic_segmentation",
                "needs_review": False,
            },
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["hand_anchor_status"], "not_applicable")
        self.assertFalse(result["hand_anchor_mismatch"])
        self.assertIsNone(result["distance_to_nearest_wrist_px"])
        self.assertTrue(_can_seed_previous_string(result))

    def test_learned_model_absence_suppresses_color_fallback(self):
        frame = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.line(frame, (120, 90), (200, 40), (0, 255, 0), 4)
        yoyo = {"center": [120.0, 90.0], "bbox": [108.0, 78.0, 132.0, 102.0]}

        fallback = estimate_string(frame, yoyo, [], None, None)
        suppressed = estimate_string(
            frame,
            yoyo,
            [],
            None,
            None,
            allow_color_fallback=False,
        )

        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["method"], "color_hough_observation")
        self.assertIsNone(suppressed)

    def test_far_frame_edge_color_line_is_not_a_trusted_unknown_anchor(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.line(frame, (200, 1055), (1000, 1055), (0, 255, 0), 4)
        yoyo = {"center": [960, 540], "bbox": [920, 500, 1000, 580]}
        unknown = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=False,
            mark_far_ambiguous=True,
        )
        self.assertIsNotNone(unknown)
        self.assertTrue(unknown["spatially_ambiguous"])
        self.assertLessEqual(unknown["confidence"], 0.24)

        detached = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=False,
            mark_far_ambiguous=False,
        )
        self.assertIsNotNone(detached)
        self.assertFalse(detached["spatially_ambiguous"])


if __name__ == "__main__":
    unittest.main()
