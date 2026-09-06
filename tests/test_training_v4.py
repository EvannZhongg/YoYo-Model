import unittest

import cv2
import numpy as np
import torch

from training_v4.data import centerline_targets
from training_v4.evaluate import decode_centerline
from training_v4.model import build_model
from training_v4.train import fuse_geometry, geometry_loss


class TrainingV4Tests(unittest.TestCase):
    def test_targets_include_soft_heatmap_and_tangent_context(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (4, 8), (58, 52), 1, 5)
        heat, tangent, context, valid = centerline_targets(mask)
        self.assertEqual(heat.shape, (64, 64))
        self.assertEqual(tangent.shape, (2, 64, 64))
        self.assertEqual(context.shape, (64, 64))
        self.assertEqual(valid.shape, (64, 64))
        self.assertGreater(float(heat.max()), 0.9)
        self.assertGreater(int(context.sum()), int(mask.sum()))
        self.assertGreater(float(valid.sum()), 0.0)

    def test_decoder_connects_soft_support_to_high_seed(self):
        fused = np.zeros((32, 32), dtype=np.float32)
        fused[16, 8] = 0.9
        fused[16, 9:16] = 0.5
        tangent = np.zeros((2, 32, 32), dtype=np.float32)
        tangent[0, 16, 8:16] = 1.0
        decoded = decode_centerline(fused, tangent, 0.7)
        self.assertGreaterEqual(int(decoded[16, 8]), 1)
        self.assertGreater(int(decoded.sum()), 1)

    def test_multitask_model_and_loss_contract(self):
        model = build_model("tiny_unet", base_channels=4)
        images = torch.randn(2, 3, 64, 64)
        target = torch.rand(2, 4, 64, 64)
        context = torch.ones(2, 1, 64, 64)
        valid = torch.ones(2, 1, 64, 64)
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 4, 64, 64))
        loss, parts = geometry_loss(output, target, context, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("tangent", parts)
        self.assertEqual(tuple(fuse_geometry(output).shape), (2, 1, 64, 64))

    def test_double_angle_direction_is_pi_periodic(self):
        theta = 0.37
        target = torch.tensor([np.cos(2.0 * theta), np.sin(2.0 * theta)])
        equivalent = torch.tensor([np.cos(2.0 * (theta + np.pi)), np.sin(2.0 * (theta + np.pi))])
        perpendicular = torch.tensor([np.cos(2.0 * (theta + np.pi / 2.0)), np.sin(2.0 * (theta + np.pi / 2.0))])
        self.assertTrue(torch.allclose(target, equivalent, atol=1e-6))
        self.assertAlmostEqual(float(torch.dot(target, perpendicular)), -1.0, places=6)
