import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from string_segmentation.evaluate_semantic import _artifact_suffix, _check_dataset_manifest
from string_segmentation.semantic_metrics import balanced_validation_key, metrics_at_threshold
from string_segmentation.semantic_model import (
    LetterboxMeta,
    TinyUNet,
    build_string_model,
    focal_dice_loss,
    load_checkpoint,
    normalize_image,
    normalize_image_for_inference,
    render_yolo_segmentation,
    save_checkpoint,
    semantic_mask_observation,
)


class SemanticStringTests(unittest.TestCase):
    def test_inference_normalization_matches_training_normalization(self):
        image = np.random.default_rng(42).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)

        expected = normalize_image(image).unsqueeze(0)
        actual = normalize_image_for_inference(image, "cpu")

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

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

    def test_attached_mode_rejects_component_far_from_yoyo(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[5:8, 5:35] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [80, 52], "bbox": [76, 48, 84, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
        )
        self.assertIsNone(result)

    def test_attached_mode_rejects_component_inside_yoyo_body(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[43:53, 72:84] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [78, 48], "bbox": [70, 40, 86, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
        )
        self.assertIsNone(result)

    def test_attached_mode_keeps_string_extending_outside_yoyo_body(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[47:50, 20:81] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [78, 48], "bbox": [70, 40, 86, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)
        self.assertLess(result["yoyo_body_overlap_fraction"], 0.60)

    def test_attached_mode_retains_observed_hand_supported_components(self):
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
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
        )
        with_hands = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
            hand_points=[[100.0, 300.0]],
        )
        capped = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="hand_and_yoyo_attached",
            min_component_pixels=1,
            max_components=1,
            hand_points=[[100.0, 300.0]],
        )

        self.assertIsNotNone(without_hands)
        self.assertIsNotNone(with_hands)
        self.assertEqual(without_hands["component_count"], 1)
        self.assertEqual(with_hands["component_selection"], "yoyo_and_hand_anchors")
        self.assertEqual(with_hands["component_count"], 3)
        self.assertEqual(with_hands["hand_supported_component_count"], 2)
        self.assertTrue(all(min(point[1] for point in polygon) > 250 for polygon in with_hands["polygons"]))
        self.assertIsNotNone(capped)
        self.assertEqual(capped["component_selection"], "yoyo_anchor")
        self.assertEqual(capped["hand_supported_component_count"], 0)

    def test_unknown_mode_keeps_far_component_as_ambiguous(self):
        probability = np.zeros((64, 96), dtype=np.float32)
        probability[5:8, 5:35] = 0.95
        meta = LetterboxMeta(96, 64, 96, 64, 96, 64, 0, 0, 1.0)
        yoyo = {"center": [80, 52], "bbox": [76, 48, 84, 56]}
        result = semantic_mask_observation(
            probability,
            meta,
            threshold=0.8,
            yoyo=yoyo,
            attachment_class="unknown",
            min_component_pixels=1,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["spatially_ambiguous"])
        self.assertFalse(result["anchored_to_yoyo"])


if __name__ == "__main__":
    unittest.main()
