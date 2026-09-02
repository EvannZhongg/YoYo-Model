import unittest

import cv2
import numpy as np
import torch

from training_v4.data import centerline_targets
from training_v4.evaluate import decode_centerline
from training_v4.model import build_model


class TrainingV4Tests(unittest.TestCase):
    def test_targets_preserve_thin_centerline_and_direction(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (4, 8), (58, 52), 1, 1)
        heat, direction, context = centerline_targets(mask)
        self.assertEqual(heat.shape, (64, 64))
        self.assertEqual(direction.shape, (2, 64, 64))
        self.assertGreater(float(heat.max()), 0.9)
        self.assertGreater(int(context.sum()), int(mask.sum()))

    def test_direction_decoder_connects_context_to_seed(self):
        heat = np.zeros((32, 32), dtype=np.float32)
        heat[16, 8] = 1.0
        heat[16, 9:16] = 0.9
        direction = np.zeros((2, 32, 32), dtype=np.float32)
        decoded = decode_centerline(heat, direction, 0.7)
        self.assertGreaterEqual(int(decoded[16, 8]), 1)
        self.assertGreater(int(decoded.sum()), 1)

    def test_tiny_model_has_heatmap_and_direction_outputs(self):
        model = build_model("tiny_unet", base_channels=4)
        output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
