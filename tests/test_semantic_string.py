import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from string_segmentation.evaluate_semantic import _artifact_suffix, _check_dataset_manifest
from string_segmentation.semantic_metrics import balanced_validation_key, metrics_at_threshold
from string_segmentation.semantic_model import (
    LetterboxMeta,
    ReviewedStringDataset,
    TinyUNet,
    build_string_model,
    focal_dice_loss,
    fuse_calibrated_probabilities,
    letterbox,
    load_checkpoint,
    normalize_image,
    normalize_image_for_inference,
    predict_prepared_probability,
    prepare_letterboxed_input,
    polyline_probability_support,
    render_yolo_segmentation,
    save_checkpoint,
    semantic_mask_observation,
)
from string_segmentation.model_soup import interpolate_state_dicts
from string_segmentation.train_semantic import _initialization_lineage, _reviewed_sample_weights


class SemanticStringTests(unittest.TestCase):
    def test_calibrated_probability_fusion_aligns_model_thresholds(self):
        primary = np.asarray([[0.3985, 0.8]], dtype=np.float32)
        secondary = np.asarray([[0.5, 0.8]], dtype=np.float32)

        fused = fuse_calibrated_probabilities(
            primary,
            secondary,
            alpha=0.3,
            primary_threshold=0.3985,
            secondary_threshold=0.5,
        )

        self.assertAlmostEqual(float(fused[0, 0]), 0.5, places=5)
        self.assertGreater(float(fused[0, 1]), 0.8)

    def test_calibrated_probability_fusion_rejects_invalid_inputs(self):
        probability = np.full((2, 2), 0.5, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            fuse_calibrated_probabilities(probability, probability[:1], 0.3, 0.4, 0.5)
        with self.assertRaisesRegex(ValueError, "alpha"):
            fuse_calibrated_probabilities(probability, probability, 1.1, 0.4, 0.5)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            fuse_calibrated_probabilities(probability, probability, 0.3, 0.0, 0.5)

    def test_polyline_probability_support_samples_source_space_geometry(self):
        probability = np.zeros((10, 10), dtype=np.float32)
        probability[3:8, :] = 0.8
        meta = LetterboxMeta(10, 10, 10, 10, 10, 10, 0, 0, 1.0)

        support = polyline_probability_support(
            probability,
            meta,
            [[2.0, 5.0], [7.0, 5.0]],
            threshold=0.5,
        )

        self.assertGreater(support["mean"], 0.79)
        self.assertEqual(support["fraction_at_0_10"], 1.0)
        self.assertEqual(support["fraction_at_threshold"], 1.0)

    def test_model_soup_interpolates_floats_and_keeps_integer_buffers_discrete(self):
        baseline = {
            "weight": torch.tensor([0.0, 2.0]),
            "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
        }
        candidate = {
            "weight": torch.tensor([4.0, 6.0]),
            "num_batches_tracked": torch.tensor(9, dtype=torch.int64),
        }

        quarter = interpolate_state_dicts(baseline, candidate, 0.25)
        three_quarters = interpolate_state_dicts(baseline, candidate, 0.75)

        self.assertTrue(torch.equal(quarter["weight"], torch.tensor([1.0, 3.0])))
        self.assertEqual(int(quarter["num_batches_tracked"]), 3)
        self.assertEqual(int(three_quarters["num_batches_tracked"]), 9)

    def test_model_soup_rejects_incompatible_or_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            interpolate_state_dicts({"weight": torch.zeros(1)}, {"weight": torch.ones(1)}, 1.1)
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            interpolate_state_dicts({"left": torch.zeros(1)}, {"right": torch.ones(1)}, 0.5)
        with self.assertRaisesRegex(ValueError, "tensor mismatch"):
            interpolate_state_dicts({"weight": torch.zeros(1)}, {"weight": torch.ones(2)}, 0.5)

    def test_semantic_warm_start_lineage_rejects_evaluation_source_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "parent"
            weights = run_dir / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.touch()
            (run_dir / "run_manifest.json").write_text(
                '{"source_groups":{"train":["train-a","leaked-val"]}}',
                encoding="utf-8",
            )

            lineage = _initialization_lineage(
                weights,
                {"train": {"train-b"}, "val": {"leaked-val"}, "test": {"test-a"}},
            )

        self.assertEqual(lineage["kind"], "versioned_run")
        self.assertEqual(lineage["evaluation_source_overlap"]["val"], ["leaked-val"])
        self.assertFalse(lineage["promotion_eligible"])

    def test_semantic_warm_start_lineage_accepts_disjoint_evaluation_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "parent"
            weights = run_dir / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.touch()
            (run_dir / "run_manifest.json").write_text(
                '{"source_groups":{"train":["train-a"]}}',
                encoding="utf-8",
            )

            lineage = _initialization_lineage(
                weights,
                {"train": {"train-a", "train-b"}, "val": {"val-a"}, "test": {"test-a"}},
            )

        self.assertEqual(lineage["evaluation_source_overlap"], {"val": [], "test": []})
        self.assertTrue(lineage["promotion_eligible"])

    def test_reviewed_negative_sampling_weights_only_empty_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            positive = Path(directory) / "positive.txt"
            negative = Path(directory) / "negative.txt"
            positive.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
            negative.write_text("", encoding="utf-8")
            dataset = SimpleNamespace(pairs=[(None, positive), (None, negative)])

            weights = _reviewed_sample_weights(dataset, 4.0)

        self.assertEqual(weights, [1.0, 4.0])

    def test_inference_normalization_matches_training_normalization(self):
        image = np.random.default_rng(42).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)

        expected = normalize_image(image).unsqueeze(0)
        actual = normalize_image_for_inference(image, "cpu")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_shared_preprocessing_matches_independent_predictions(self):
        image = np.random.default_rng(7).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        primary = TinyUNet(base_channels=4).eval()
        secondary = TinyUNet(base_channels=4).eval()

        primary_image, _, independent_meta = letterbox(image, 96, 64)
        primary_tensor = normalize_image_for_inference(primary_image, "cpu")
        independent_primary = predict_prepared_probability(primary, primary_tensor)
        secondary_image, _, secondary_meta = letterbox(image, 96, 64)
        secondary_tensor = normalize_image_for_inference(secondary_image, "cpu")
        independent_secondary = predict_prepared_probability(secondary, secondary_tensor)
        tensor, shared_meta = prepare_letterboxed_input(image, 96, 64, "cpu")
        shared_primary = predict_prepared_probability(primary, tensor)
        shared_secondary = predict_prepared_probability(secondary, tensor)

        self.assertEqual(independent_meta, shared_meta)
        self.assertEqual(secondary_meta, shared_meta)
        np.testing.assert_array_equal(shared_primary, independent_primary)
        np.testing.assert_array_equal(shared_secondary, independent_secondary)

    def test_reviewed_dataset_reads_unicode_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "字符串"
            image_root = root / "images" / "train" / "中文组"
            label_root = root / "labels" / "train" / "中文组"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)
            image_path = image_root / "frame.jpg"
            label_path = label_root / "frame.txt"
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            image[10:14, 8:40] = (0, 255, 0)
            encoded, buffer = cv2.imencode(".jpg", image)
            self.assertTrue(encoded)
            buffer.tofile(image_path)
            label_path.write_text(
                "0 0.15 0.30 0.85 0.30 0.85 0.45 0.15 0.45\n",
                encoding="utf-8",
            )
            dataset = ReviewedStringDataset(root, "train", 48, 32, 1, False)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 48))
            self.assertGreater(float(sample["mask"].sum()), 0.0)

    def test_validation_selection_balances_string_quality_and_presence(self):
        reliable = {
            "tolerant": {"f1": 0.55},
            "image_presence": {"f1": 1.0},
            "negative_mean_false_positive_pixels": 0.0,
            "pixel": {"dice": 0.20},
        }
        false_positive_prone = {
            "tolerant": {"f1": 0.65},
            "image_presence": {"f1": 0.50},
            "negative_mean_false_positive_pixels": 1000.0,
            "pixel": {"dice": 0.25},
        }
        self.assertGreater(
            balanced_validation_key(reliable),
            balanced_validation_key(false_positive_prone),
        )

    def test_validation_selection_breaks_balanced_ties_with_negative_pixels(self):
        clean = {
            "tolerant": {"f1": 0.60},
            "image_presence": {"f1": 0.80},
            "negative_mean_false_positive_pixels": 0.0,
            "pixel": {"dice": 0.20},
        }
        noisy = {
            "tolerant": {"f1": 0.60},
            "image_presence": {"f1": 0.80},
            "negative_mean_false_positive_pixels": 50.0,
            "pixel": {"dice": 0.20},
        }
        self.assertGreater(balanced_validation_key(clean), balanced_validation_key(noisy))

    def test_dataset_mismatch_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "allow-dataset-mismatch"):
            _check_dataset_manifest("training-hash", "evaluation-hash", False)

    def test_allowed_dataset_mismatch_is_recorded(self):
        matches, warning = _check_dataset_manifest("training-hash", "evaluation-hash", True)
        self.assertFalse(matches)
        self.assertIn("explicit cross-model comparison", warning)

    def test_evaluation_artifacts_are_versioned_for_noncanonical_runs(self):
        self.assertEqual(_artifact_suffix(True, "abcdef", None), "")
        self.assertEqual(_artifact_suffix(True, "abcdef", 0.8956), "_threshold_0p8956")
        self.assertEqual(
            _artifact_suffix(False, "abcdef0123456789", 0.5),
            "_external_abcdef012345_threshold_0p5",
        )

    def test_unet_preserves_spatial_shape(self):
        model = TinyUNet(base_channels=4)
        output = model(torch.zeros((1, 3, 64, 96), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 96))

    def test_lraspp_preserves_spatial_shape_without_downloading_weights(self):
        model = build_string_model("lraspp_mobilenet_v3", pretrained_backbone=False)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, 64, 96), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 96))

    def test_yolo_polygon_renders_binary_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "mask.txt"
            label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
            mask = render_yolo_segmentation(label, 100, 80)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertGreater(int(mask.sum()), 1500)
            self.assertEqual(int(mask[40, 50]), 1)

    def test_perfect_prediction_has_perfect_metrics(self):
        target = np.zeros((32, 48), dtype=np.uint8)
        target[10:14, 8:40] = 1
        metrics = metrics_at_threshold(
            [{"probability": target.astype(np.float32), "target": target, "image_path": "sample.jpg"}],
            threshold=0.5,
            min_component_pixels=1,
        )
        self.assertEqual(metrics["pixel"]["dice"], 1.0)
        self.assertEqual(metrics["tolerant"]["f1"], 1.0)
        self.assertEqual(metrics["image_presence"]["f1"], 1.0)

    def test_loss_penalizes_hard_background_responses(self):
        target = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        clean_logits = torch.full_like(target, -8.0)
        false_string_logits = clean_logits.clone()
        false_string_logits[:, :, 15:17, 6:26] = 8.0
        clean_loss, clean_parts = focal_dice_loss(clean_logits, target)
        false_loss, false_parts = focal_dice_loss(false_string_logits, target)
        self.assertGreater(float(false_loss), float(clean_loss))
        self.assertGreater(false_parts["hard_negative"], clean_parts["hard_negative"])

    def test_hard_negative_term_does_not_suppress_positive_images(self):
        target = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        target[:, :, 15:17, 6:26] = 1.0
        _, parts = focal_dice_loss(torch.zeros_like(target), target)
        self.assertEqual(parts["hard_negative"], 0.0)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model = TinyUNet(base_channels=4)
            config = {"base_channels": 4, "input_width": 96, "input_height": 64}
            save_checkpoint(path, model, config, 0.4, 2, {"pixel": {"dice": 0.2}}, "abc")
            restored, checkpoint = load_checkpoint(path)
            self.assertIsInstance(restored, TinyUNet)
            self.assertEqual(checkpoint["format"], "yoyo_string_semantic_unet_v1")
            self.assertEqual(checkpoint["threshold"], 0.4)

    def test_semantic_mask_exports_reviewable_geometry(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[30:33, 20:76] = 0.9
        meta = LetterboxMeta(192, 128, 96, 64, 96, 64, 0, 0, 0.5)
        result = semantic_mask_observation(probability, meta, threshold=0.8)
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "semantic_segmentation")
        self.assertTrue(result["needs_review"])
        self.assertGreaterEqual(len(result["points"]), 2)
        self.assertGreaterEqual(len(result["polygon"]), 3)

    def test_division_does_not_reject_component_far_from_yoyo(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[5:8, 5:35] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [80, 52], "bbox": [76, 48, 84, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)

    def test_division_does_not_reject_component_inside_yoyo_body(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[43:53, 72:84] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [78, 48], "bbox": [70, 40, 86, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)

    def test_division_keeps_string_extending_outside_yoyo_body(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[47:50, 20:81] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [78, 48], "bbox": [70, 40, 86, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)
        self.assertLess(result["yoyo_body_overlap_fraction"], 0.60)

    def test_division_does_not_change_component_selection(self):
        probability = np.zeros((600, 1000), dtype=np.float32)
        probability[296:305, 820:901] = 0.95  # yoyo-side primary
        probability[296:305, 100:181] = 0.95  # direct wrist support
        probability[296:305, 210:401] = 0.95  # one-hop continuation
        probability[100:109, 100:181] = 0.95  # unrelated background
        meta = LetterboxMeta(1000, 600, 1000, 600, 1000, 600, 0, 0, 1.0)
        yoyo = {"center": [900, 300], "bbox": [880, 280, 920, 320]}

        without_hands = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
        )
        with_hands = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
            hand_points=[[100.0, 300.0]],
        )
        capped = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
            max_components=1,
            hand_points=[[100.0, 300.0]],
        )

        self.assertIsNotNone(without_hands)
        self.assertIsNotNone(with_hands)
        self.assertEqual(without_hands["component_count"], 4)
        self.assertEqual(with_hands["component_selection"], "confidence")
        self.assertEqual(with_hands["component_count"], 4)
        self.assertEqual(with_hands["hand_supported_component_count"], 0)
        self.assertIsNotNone(capped)
        self.assertEqual(capped["component_selection"], "confidence")
        self.assertEqual(capped["hand_supported_component_count"], 0)

    def test_far_component_is_kept_as_ambiguous(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[5:8, 5:35] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [80, 52], "bbox": [76, 48, 84, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            yoyo_division="1A",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["spatially_ambiguous"])
        self.assertFalse(result["anchored_to_yoyo"])


if __name__ == "__main__":
    unittest.main()
