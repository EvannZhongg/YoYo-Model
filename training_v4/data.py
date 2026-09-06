from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from string_segmentation.semantic_model import (
    _skeletonize,
    letterbox,
    normalize_image,
    render_yolo_segmentation,
)


def centerline_targets(
    mask: np.ndarray,
    radius: float = 6.0,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create soft centerline, tangent and context targets from a mask."""
    skeleton = (_skeletonize(mask) > 0).astype(np.uint8)
    if not np.any(skeleton):
        shape = mask.shape
        return (
            np.zeros(shape, np.float32),
            np.zeros((2, *shape), np.float32),
            np.zeros(shape, np.float32),
            np.zeros(shape, np.float32),
        )
    distance = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    heat = np.exp(-(distance * distance) / (2.0 * max(float(sigma), 1e-3) ** 2)).astype(np.float32)
    # The structure-tensor principal direction is the ridge normal; rotate it
    # by 90 degrees to obtain the local line tangent.  Encoding twice the
    # angle removes the equivalent theta/theta+pi representation.
    smooth = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=2.0, sigmaY=2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=2.0, sigmaY=2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=2.0, sigmaY=2.0)
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tangent_angle = normal_angle + (np.pi / 2.0)
    tx = np.cos(2.0 * tangent_angle).astype(np.float32)
    ty = np.sin(2.0 * tangent_angle).astype(np.float32)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    valid_tangent = (skeleton > 0).astype(np.float32) * (coherence > 1e-5).astype(np.float32)
    context = (distance <= float(radius)).astype(np.float32)
    context[skeleton > 0] = 1.0
    valid_tangent *= context
    tangent = np.stack((tx, ty), axis=0).astype(np.float32)
    tangent[:, valid_tangent <= 0] = 0.0
    return heat.astype(np.float32), tangent, context.astype(np.float32), valid_tangent


class CenterlineDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        input_width: int,
        input_height: int,
        augment: bool = False,
        radius: float = 6.0,
        sigma: float = 2.0,
    ):
        self.root = Path(dataset_dir)
        self.split = str(split)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.augment = bool(augment)
        self.radius = float(radius)
        self.sigma = float(sigma)
        image_root = self.root / "images" / self.split
        label_root = self.root / "labels" / self.split
        self.pairs = [
            (path, label_root / path.relative_to(image_root).with_suffix(".txt"))
            for path in sorted(image_root.rglob("*"))
            if path.is_file() and (label_root / path.relative_to(image_root).with_suffix(".txt")).exists()
        ]
        if not self.pairs:
            raise RuntimeError(f"No centerline samples for split={self.split}: {self.root}")
        self._target_cache: dict[int, tuple[np.ndarray, ...]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.pairs[index]
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        mask = render_yolo_segmentation(label_path, image.shape[1], image.shape[0])
        image, mask, meta = letterbox(image, self.input_width, self.input_height, mask)
        assert mask is not None
        cached = self._target_cache.get(index)
        if cached is None:
            radius = max(1.0, self.radius * float(meta.scale))
            sigma = max(0.5, self.sigma * float(meta.scale))
            cached = tuple(np.asarray(value, dtype=np.float16) for value in centerline_targets(mask, radius, sigma))
            self._target_cache[index] = cached
        heat, tangent, context, tangent_valid = (np.asarray(value, dtype=np.float32).copy() for value in cached)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            heat = np.ascontiguousarray(heat[:, ::-1])
            tangent = np.ascontiguousarray(tangent[:, :, ::-1])
            context = np.ascontiguousarray(context[:, ::-1])
            tangent_valid = np.ascontiguousarray(tangent_valid[:, ::-1])
            tangent[1] *= -1.0
        if self.augment:
            image = np.clip(
                image.astype(np.float32) * random.uniform(0.88, 1.12) + random.uniform(-10.0, 10.0),
                0,
                255,
            ).astype(np.uint8)
        target = np.concatenate((mask[None].astype(np.float32), heat[None], tangent), axis=0)
        return {
            "image": normalize_image(image),
            "target": torch.from_numpy(target),
            "context": torch.from_numpy(context[None]),
            "tangent_valid": torch.from_numpy(tangent_valid[None]),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "positive": bool(np.any(mask)),
        }
