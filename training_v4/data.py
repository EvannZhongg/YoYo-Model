from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from string_segmentation.semantic_model import _skeletonize, letterbox, normalize_image, render_yolo_segmentation


def centerline_targets(mask: np.ndarray, radius: float = 6.0, sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert the reviewed polygon/buffer to a one-pixel geometric centerline.
    # Targets are cached per sample, so this iterative operation is paid once.
    skeleton = _skeletonize(mask)
    heat = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    if heat.max() > 0:
        heat /= heat.max()
    distance = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(distance.astype(np.float32))
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    direction = np.stack((-gx / norm, -gy / norm), axis=0).astype(np.float32)
    context = (distance <= float(radius)).astype(np.float32)
    context[skeleton > 0] = 1.0
    direction[:, context <= 0] = 0.0
    return heat.astype(np.float32), direction, context


class CenterlineDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str, input_width: int, input_height: int, augment: bool = False, radius: float = 6.0):
        self.root = Path(dataset_dir); self.split = split; self.input_width = int(input_width); self.input_height = int(input_height); self.augment = bool(augment); self.radius = float(radius)
        self._target_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        image_root, label_root = self.root / "images" / split, self.root / "labels" / split
        self.pairs = [(p, label_root / p.relative_to(image_root).with_suffix(".txt")) for p in sorted(image_root.rglob("*")) if p.is_file() and (label_root / p.relative_to(image_root).with_suffix(".txt")).exists()]
        if not self.pairs: raise RuntimeError(f"No centerline samples for split={split}: {self.root}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.pairs[index]
        encoded = np.fromfile(image_path, dtype=np.uint8); image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is None: raise RuntimeError(f"Could not read image: {image_path}")
        mask = render_yolo_segmentation(label_path, image.shape[1], image.shape[0])
        image, mask, _ = letterbox(image, self.input_width, self.input_height, mask); assert mask is not None
        cached_target = self._target_cache.get(index)
        if cached_target is None:
            cached_target = centerline_targets(mask, self.radius)
            self._target_cache[index] = tuple(np.asarray(value, dtype=np.float16) for value in cached_target)
        heat, direction, context = (np.asarray(value, dtype=np.float32).copy() for value in cached_target)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1]); mask = np.ascontiguousarray(mask[:, ::-1])
            heat = np.ascontiguousarray(heat[:, ::-1]); direction = np.ascontiguousarray(direction[:, :, ::-1]); context = np.ascontiguousarray(context[:, ::-1])
            direction[0] *= -1.0
        if self.augment:
            image = np.clip(image.astype(np.float32) * random.uniform(0.88, 1.12) + random.uniform(-10, 10), 0, 255).astype(np.uint8)
        return {"image": normalize_image(image), "target": torch.from_numpy(np.concatenate((heat[None], direction), axis=0)), "context": torch.from_numpy(context[None]), "image_path": str(image_path), "source_shape": (int(mask.shape[0]), int(mask.shape[1]))}
