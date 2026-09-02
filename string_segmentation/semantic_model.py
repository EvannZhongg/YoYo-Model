"""Lightweight semantic model and reviewed YOLO-mask dataset for thin strings."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


CHECKPOINT_FORMAT = "yoyo_string_semantic_unet_v1"
TRANSFER_CHECKPOINT_FORMAT = "yoyo_string_semantic_transfer_v1"
CHECKPOINT_FORMATS = {CHECKPOINT_FORMAT, TRANSFER_CHECKPOINT_FORMAT}
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_INFERENCE_NORMALIZATION_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TinyUNet(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        channels = [base_channels * multiplier for multiplier in (1, 2, 4, 8, 16)]
        self.enc1 = ConvBlock(3, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.enc4 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)
        self.dec4 = ConvBlock(channels[4] + channels[3], channels[3])
        self.dec3 = ConvBlock(channels[3] + channels[2], channels[2])
        self.dec2 = ConvBlock(channels[2] + channels[1], channels[1])
        self.dec1 = ConvBlock(channels[1] + channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(value)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.dec4(torch.cat((self._up(bottleneck, enc4), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self._up(dec4, enc3), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self._up(dec3, enc2), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self._up(dec2, enc1), enc1), dim=1))
        return self.output(dec1)


class LRASPPString(nn.Module):
    """Binary LR-ASPP head on an ImageNet-pretrained MobileNetV3 backbone."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)["out"]


class _FPNRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class MobileNetV3FPNString(nn.Module):
    """High-resolution FPN decoder on an ImageNet-pretrained MobileNetV3 encoder."""

    _FEATURE_INDICES = (1, 3, 6, 12, 16)
    _FEATURE_CHANNELS = (16, 24, 40, 112, 960)

    def __init__(self, decoder_channels: int = 32, pretrained_backbone: bool = False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.encoder = mobilenet_v3_large(weights=backbone_weights).features
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(input_channels, decoder_channels, 1, bias=False),
                nn.GroupNorm(_group_count(decoder_channels), decoder_channels),
            )
            for input_channels in self._FEATURE_CHANNELS
        )
        self.refine = nn.ModuleList(_FPNRefine(decoder_channels) for _ in range(4))
        self.classifier = nn.Sequential(
            _FPNRefine(decoder_channels),
            nn.Conv2d(decoder_channels, 1, 1),
        )

    def train(self, mode: bool = True) -> MobileNetV3FPNString:
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_size = value.shape[-2:]
        features = []
        selected = set(self._FEATURE_INDICES)
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in selected:
                features.append(value)

        pyramid = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            pyramid = nn.functional.interpolate(
                pyramid,
                size=features[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid = self.refine[level](pyramid + self.lateral[level](features[level]))
        logits = self.classifier(pyramid)
        return nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_string_model(
    architecture: str = "tiny_unet",
    base_channels: int = 16,
    pretrained_backbone: bool = False,
) -> nn.Module:
    architecture = str(architecture or "tiny_unet").strip().lower()
    if architecture == "tiny_unet":
        return TinyUNet(base_channels=int(base_channels))
    if architecture == "lraspp_mobilenet_v3":
        return LRASPPString(pretrained_backbone=bool(pretrained_backbone))
    if architecture == "mobilenet_v3_fpn":
        return MobileNetV3FPNString(
            decoder_channels=max(16, int(base_channels) * 2),
            pretrained_backbone=bool(pretrained_backbone),
        )
    raise ValueError(f"Unsupported semantic string architecture: {architecture}")


@dataclass(frozen=True)
class LetterboxMeta:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int
    scale: float


def letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, LetterboxMeta]:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Cannot letterbox an empty image")
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized_image
    mask_canvas = None
    if mask is not None:
        resized_mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        mask_canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized_mask
    meta = LetterboxMeta(
        original_width=width,
        original_height=height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_x=pad_x,
        pad_y=pad_y,
        scale=scale,
    )
    return canvas, mask_canvas, meta


def normalize_image(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()


def normalize_image_for_inference(
    image_bgr: np.ndarray,
    device: str | torch.device,
) -> torch.Tensor:
    """Transfer uint8 pixels first, then normalize without large CPU float copies."""
    target_device = torch.device(device)
    array = np.ascontiguousarray(image_bgr)
    tensor = torch.from_numpy(array).to(target_device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0]].float().div_(255.0)
    cache_key = str(target_device)
    constants = _INFERENCE_NORMALIZATION_CACHE.get(cache_key)
    if constants is None:
        constants = (
            torch.as_tensor(IMAGE_MEAN, dtype=torch.float32, device=target_device).view(1, 3, 1, 1),
            torch.as_tensor(IMAGE_STD, dtype=torch.float32, device=target_device).view(1, 3, 1, 1),
        )
        _INFERENCE_NORMALIZATION_CACHE[cache_key] = constants
    mean, std = constants
    return tensor.sub_(mean).div_(std)


def render_yolo_segmentation(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        raise FileNotFoundError(f"Semantic label not found: {label_path}")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 7:
            continue
        coordinates = [float(value) for value in values[1:]]
        if len(coordinates) % 2:
            continue
        points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
        points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points.round().astype(np.int32)], 1)
    return mask


def image_label_pairs(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    image_root = dataset_dir / "images" / split
    label_root = dataset_dir / "labels" / split
    if not image_root.exists() or not label_root.exists():
        return []
    pairs = []
    for image_path in sorted(path for path in image_root.rglob("*") if path.is_file()):
        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


class ReviewedStringDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        input_width: int,
        input_height: int,
        min_mask_width_px: int = 1,
        augment: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.min_mask_width_px = max(1, int(min_mask_width_px))
        self.augment = bool(augment)
        self.pairs = image_label_pairs(self.dataset_dir, split)
        if not self.pairs:
            raise RuntimeError(f"No reviewed semantic samples found for split={split}: {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, label_path = self.pairs[index]
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is None:
            raise RuntimeError(f"Could not read semantic training image: {image_path}")
        mask = render_yolo_segmentation(label_path, image.shape[1], image.shape[0])
        image, mask, _ = letterbox(image, self.input_width, self.input_height, mask)
        assert mask is not None
        if self.min_mask_width_px > 1 and np.any(mask):
            kernel = np.ones((self.min_mask_width_px, self.min_mask_width_px), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment:
            gain = random.uniform(0.88, 1.12)
            bias = random.uniform(-10.0, 10.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return {
            "image": normalize_image(image),
            "mask": torch.from_numpy(mask.astype(np.float32)[None, ...]),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "positive": bool(np.any(mask)),
        }


def focal_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    hard_negative_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    probability = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha = 0.85 * target + 0.15 * (1.0 - target)
    focal = (alpha * (1.0 - pt).pow(2.0) * bce).mean()
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    hard_negative_rows = []
    for sample_bce, sample_target in zip(bce, target):
        if torch.any(sample_target >= 0.5):
            continue
        background_loss = sample_bce[sample_target < 0.5]
        if background_loss.numel():
            count = min(background_loss.numel(), max(64, int(background_loss.numel() * 0.001)))
            hard_negative_rows.append(torch.topk(background_loss, count).values.mean())
    hard_negative = torch.stack(hard_negative_rows).mean() if hard_negative_rows else logits.new_zeros(())
    hard_negative_weight = max(0.0, float(hard_negative_weight))
    loss = focal + dice + hard_negative_weight * hard_negative
    return loss, {
        "focal": float(focal.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "hard_negative": float(hard_negative.detach().cpu()),
        "hard_negative_weight": hard_negative_weight,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    threshold: float,
    epoch: int,
    validation_metrics: dict[str, Any],
    dataset_manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": (
                TRANSFER_CHECKPOINT_FORMAT
                if str(model_config.get("architecture", "tiny_unet")) != "tiny_unet"
                else CHECKPOINT_FORMAT
            ),
            "model_config": model_config,
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "epoch": int(epoch),
            "validation_metrics": validation_metrics,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError(f"Not a supported semantic string checkpoint: {path}")
    config = dict(checkpoint.get("model_config") or {})
    model = build_string_model(
        architecture=str(config.get("architecture", "tiny_unet")),
        base_channels=int(config.get("base_channels", 16)),
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def is_semantic_checkpoint(path: str | Path) -> bool:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(checkpoint, dict) and checkpoint.get("format") in CHECKPOINT_FORMATS


@torch.inference_mode()
def prepare_letterboxed_input(
    frame_bgr: np.ndarray,
    input_width: int,
    input_height: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, LetterboxMeta]:
    """Letterbox and normalize one frame for one or more compatible models."""
    image, _, meta = letterbox(frame_bgr, input_width, input_height)
    tensor = normalize_image_for_inference(image, device)
    return tensor, meta


@torch.inference_mode()
def predict_prepared_probability(
    model: nn.Module,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Run a semantic model on a prepared NCHW tensor."""
    probability = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    return probability


@torch.inference_mode()
def predict_letterboxed(
    model: nn.Module,
    frame_bgr: np.ndarray,
    input_width: int,
    input_height: int,
    device: str | torch.device,
) -> tuple[np.ndarray, LetterboxMeta]:
    tensor, meta = prepare_letterboxed_input(
        frame_bgr,
        input_width,
        input_height,
        device,
    )
    probability = predict_prepared_probability(model, tensor)
    return probability, meta


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Reviewed string dataset manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def restore_coordinates(points: np.ndarray, meta: LetterboxMeta) -> list[list[float]]:
    """Map target-space points back to the original video frame."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if not len(values):
        return []
    values[:, 0] = (values[:, 0] - meta.pad_x) / max(meta.scale, 1e-6)
    values[:, 1] = (values[:, 1] - meta.pad_y) / max(meta.scale, 1e-6)
    values[:, 0] = np.clip(values[:, 0], 0, meta.original_width - 1)
    values[:, 1] = np.clip(values[:, 1], 0, meta.original_height - 1)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in values]


def polyline_probability_support(
    probability: np.ndarray,
    meta: LetterboxMeta,
    points: list[list[float]],
    threshold: float,
    thickness: int = 3,
) -> dict[str, float]:
    """Summarize semantic probability around a source-space polyline."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if len(values) < 2:
        return {}
    values[:, 0] = values[:, 0] * float(meta.scale) + int(meta.pad_x)
    values[:, 1] = values[:, 1] * float(meta.scale) + int(meta.pad_y)
    values[:, 0] = np.clip(values[:, 0], 0, probability.shape[1] - 1)
    values[:, 1] = np.clip(values[:, 1], 0, probability.shape[0] - 1)
    mask = np.zeros(probability.shape, dtype=np.uint8)
    line_points = values.round().astype(np.int32)
    for start, end in zip(line_points[:-1], line_points[1:]):
        cv2.line(mask, tuple(start), tuple(end), 1, thickness=max(1, int(thickness)))
    samples = probability[mask > 0]
    if not len(samples):
        return {}
    return {
        "mean": round(float(np.mean(samples)), 6),
        "p50": round(float(np.percentile(samples, 50)), 6),
        "p90": round(float(np.percentile(samples, 90)), 6),
        "fraction_at_0_10": round(float(np.mean(samples >= 0.10)), 6),
        "fraction_at_0_20": round(float(np.mean(samples >= 0.20)), 6),
        "fraction_at_threshold": round(float(np.mean(samples >= float(threshold))), 6),
    }


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Morphological skeletonization without relying on opencv-contrib."""
    working = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(working)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(working):
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
        working = eroded
    return (skeleton > 0).astype(np.uint8)


def _longest_skeleton_path(skeleton: np.ndarray, max_points: int = 64) -> np.ndarray:
    coordinates = np.argwhere(skeleton > 0)
    if len(coordinates) < 2:
        return np.empty((0, 2), dtype=np.float32)
    nodes = {(int(row), int(column)): index for index, (row, column) in enumerate(coordinates)}
    adjacency: list[list[int]] = [[] for _ in range(len(coordinates))]
    for index, (row, column) in enumerate(coordinates):
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                neighbor = nodes.get((int(row + row_delta), int(column + column_delta)))
                if neighbor is not None:
                    adjacency[index].append(neighbor)

    def breadth_first(start: int) -> tuple[int, dict[int, int], dict[int, int]]:
        distances = {start: 0}
        parents = {start: -1}
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    parents[neighbor] = node
                    queue.append(neighbor)
        farthest = max(distances, key=distances.get)
        return farthest, distances, parents

    first, _, _ = breadth_first(0)
    second, _, parents = breadth_first(first)
    path_indices = []
    current = second
    while current >= 0:
        path_indices.append(current)
        current = parents.get(current, -1)
    path_indices.reverse()
    path = coordinates[path_indices][:, [1, 0]].astype(np.float32)
    if len(path) > max_points:
        indices = np.linspace(0, len(path) - 1, max_points, dtype=np.int32)
        path = path[indices]
    return path


def _skeleton_cover_paths(
    skeleton: np.ndarray,
    max_paths: int,
    max_points: int,
) -> list[np.ndarray]:
    """Cover a branched skeleton with a bounded set of non-overlapping paths."""
    coordinates = np.argwhere(skeleton > 0)
    if len(coordinates) < 2:
        return []
    node_ids = np.full(skeleton.shape, -1, dtype=np.int32)
    node_ids[coordinates[:, 0], coordinates[:, 1]] = np.arange(len(coordinates), dtype=np.int32)
    adjacency: list[list[int]] = [[] for _ in range(len(coordinates))]
    height, width = skeleton.shape
    for index, (row, column) in enumerate(coordinates):
        row = int(row)
        column = int(column)
        for neighbor_row in range(max(0, row - 1), min(height, row + 2)):
            for neighbor_column in range(max(0, column - 1), min(width, column + 2)):
                neighbor = int(node_ids[neighbor_row, neighbor_column])
                if neighbor >= 0 and neighbor != index:
                    adjacency[index].append(neighbor)

    active = np.ones(len(coordinates), dtype=np.bool_)

    def traverse(start: int, parents: np.ndarray | None = None) -> tuple[int, list[int]]:
        visited = np.zeros(len(coordinates), dtype=np.bool_)
        visited[start] = True
        if parents is not None:
            parents[start] = -1
        queue = [start]
        distances = [0]
        farthest = start
        farthest_distance = 0
        for position, node in enumerate(queue):
            distance = distances[position]
            for neighbor in adjacency[node]:
                if not active[neighbor] or visited[neighbor]:
                    continue
                visited[neighbor] = True
                if parents is not None:
                    parents[neighbor] = node
                next_distance = distance + 1
                queue.append(neighbor)
                distances.append(next_distance)
                if next_distance > farthest_distance:
                    farthest = neighbor
                    farthest_distance = next_distance
        return farthest, queue

    paths: list[np.ndarray] = []
    while len(paths) < max(1, int(max_paths)):
        largest_component: list[int] = []
        first = -1
        unseen = active.copy()
        for start in np.flatnonzero(unseen):
            if not unseen[start]:
                continue
            component_first, component = traverse(int(start))
            unseen[component] = False
            if len(component) > len(largest_component):
                largest_component = component
                first = component_first
        if len(largest_component) < 2:
            break

        parents = np.full(len(coordinates), -1, dtype=np.int32)
        second, _ = traverse(first, parents)
        path_indices = []
        current = second
        while current >= 0:
            path_indices.append(current)
            current = int(parents[current])
        path_indices.reverse()
        active[path_indices] = False
        path = coordinates[path_indices][:, [1, 0]].astype(np.float32)
        if len(path) > max(2, int(max_points)):
            indices = np.linspace(0, len(path) - 1, max(2, int(max_points)), dtype=np.int32)
            path = path[indices]
        paths.append(path)
    return paths


def hysteresis_mask(
    probability: np.ndarray,
    high_threshold: float,
    low_threshold: float | None = None,
) -> np.ndarray:
    """Threshold probabilities, optionally growing weak pixels from high seeds."""
    high = float(high_threshold)
    if low_threshold is None:
        return (np.asarray(probability) >= high).astype(np.uint8)
    low = float(low_threshold)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("low_threshold must be in [0, threshold]")
    values = np.asarray(probability)
    low_mask = (values >= low).astype(np.uint8)
    seeds = (values >= high).astype(np.uint8)
    if not np.any(seeds):
        return np.zeros_like(low_mask)
    _, labels = cv2.connectedComponents(low_mask, connectivity=8)
    seed_labels = np.unique(labels[seeds > 0])
    seed_labels = seed_labels[seed_labels > 0]
    return np.isin(labels, seed_labels).astype(np.uint8)


def semantic_mask_observation(
    probability: np.ndarray,
    meta: LetterboxMeta,
    threshold: float,
    yoyo: dict[str, Any] | None = None,
    yoyo_division: str = "1A",
    min_component_pixels: int = 12,
    max_components: int = 8,
    max_polyline_points: int = 64,
    hand_points: list[list[float]] | None = None,
    low_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Turn a low-resolution semantic mask into review-only string geometry."""
    high = float(threshold)
    binary = hysteresis_mask(probability, high, low_threshold)
    if not np.any(binary):
        return None
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    yoyo_center = np.asarray((yoyo or {}).get("center", [0, 0]), dtype=np.float32)
    yoyo_target_center = yoyo_center * float(meta.scale) + np.asarray([meta.pad_x, meta.pad_y], dtype=np.float32)
    hand_target_points = [
        np.asarray(
            [
                float(point[0]) * float(meta.scale) + float(meta.pad_x),
                float(point[1]) * float(meta.scale) + float(meta.pad_y),
            ],
            dtype=np.float32,
        )
        for point in (hand_points or [])
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]
    yoyo_anchor_limit = None
    yoyo_target_bbox: tuple[int, int, int, int] | None = None
    if yoyo is not None:
        bbox = yoyo.get("bbox") or []
        if len(bbox) == 4:
            target_w = max(1.0, (float(bbox[2]) - float(bbox[0])) * float(meta.scale))
            target_h = max(1.0, (float(bbox[3]) - float(bbox[1])) * float(meta.scale))
            # The mask may stop at the visible yoyo edge, so allow several
            # yoyo radii for string-to-body contact.  This remains much
            # tighter than accepting any background component in the frame.
            yoyo_anchor_limit = max(18.0, 3.0 * float(np.hypot(target_w, target_h) * 0.5))
            yoyo_target_bbox = (
                max(0, int(np.floor(float(bbox[0]) * meta.scale + meta.pad_x))),
                max(0, int(np.floor(float(bbox[1]) * meta.scale + meta.pad_y))),
                min(meta.target_width, int(np.ceil(float(bbox[2]) * meta.scale + meta.pad_x))),
                min(meta.target_height, int(np.ceil(float(bbox[3]) * meta.scale + meta.pad_y))),
            )
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        component_x = int(stats[index, cv2.CC_STAT_LEFT])
        component_y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = (
            labels[
                component_y : component_y + component_height,
                component_x : component_x + component_width,
            ]
            == index
        ).astype(np.uint8)
        # Preserve the zero background that surrounded the component in the
        # full target canvas while avoiding full-frame contour/skeleton work.
        padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        offset = np.asarray([component_x - 1, component_y - 1], dtype=np.float32)
        polygon = contour.reshape(-1, 2).astype(np.float32) + offset
        if len(polygon) < 3:
            continue
        skeleton = _skeletonize(padded)
        paths = _skeleton_cover_paths(
            skeleton,
            max_paths=max(1, int(max_components)),
            max_points=max(2, int(max_polyline_points)),
        )
        if paths:
            paths = [path + offset for path in paths]
        else:
            centered = polygon - polygon.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ vh[0]
            paths = [polygon[[int(np.argmin(projection)), int(np.argmax(projection))]]]
        probability_crop = probability[
            component_y : component_y + component_height,
            component_x : component_x + component_width,
        ]
        mean_probability = float(probability_crop[component > 0].mean())
        component_center = np.asarray(centroids[index], dtype=np.float32)
        if yoyo:
            point_distance = float(np.min(np.linalg.norm(polygon - yoyo_target_center, axis=1)))
            distance = min(float(np.linalg.norm(component_center - yoyo_target_center)), point_distance)
        else:
            distance = 0.0
        yoyo_body_overlap_fraction = 0.0
        if yoyo_target_bbox is not None and area > 0:
            x1, y1, x2, y2 = yoyo_target_bbox
            overlap_x1 = max(x1, component_x)
            overlap_y1 = max(y1, component_y)
            overlap_x2 = min(x2, component_x + component_width)
            overlap_y2 = min(y2, component_y + component_height)
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                yoyo_body_overlap_fraction = float(
                    component[
                        overlap_y1 - component_y : overlap_y2 - component_y,
                        overlap_x1 - component_x : overlap_x2 - component_x,
                    ].sum()
                    / area
                )
        candidates.append(
            {
                "area": area,
                "mean_probability": mean_probability,
                "distance_to_yoyo_target_px": distance,
                "yoyo_body_overlap_fraction": yoyo_body_overlap_fraction,
                "polygon": polygon,
                "paths": paths,
            }
        )
    if not candidates:
        return None
    selection_mode = "confidence"
    hand_supported_ids: set[int] = set()
    if selection_mode == "confidence":
        if yoyo is not None:
            candidates.sort(key=lambda item: (item["distance_to_yoyo_target_px"], -item["mean_probability"], -item["area"]))
        else:
            candidates.sort(key=lambda item: (-item["mean_probability"], -item["area"]))
    component_limit = max(1, int(max_components))
    selected = []
    selected_polygons = []
    selected_polylines = []
    remaining_paths: list[list[list[float]]] = []
    for item in candidates[:component_limit]:
        polygon = restore_coordinates(item["polygon"], meta)
        paths = [
            restore_coordinates(path, meta)
            for path in item["paths"]
            if len(path) >= 2
        ]
        if len(polygon) < 3 or not paths:
            continue
        selected.append(item)
        selected_polygons.append(polygon)
        selected_polylines.append(paths[0])
        remaining_paths.append(paths[1:])
    if not selected:
        return None
    while len(selected_polylines) < component_limit:
        added = False
        for paths in remaining_paths:
            if not paths or len(selected_polylines) >= component_limit:
                continue
            selected_polylines.append(paths.pop(0))
            added = True
        if not added:
            break
    hand_supported_component_count = sum(id(item) in hand_supported_ids for item in selected)
    if selection_mode == "yoyo_and_hand_anchors" and hand_supported_component_count == 0:
        selection_mode = "yoyo_anchor"
    primary = selected[0]
    primary_distance_px = float(primary["distance_to_yoyo_target_px"] / max(float(meta.scale), 1e-6)) if yoyo else None
    spatially_ambiguous = bool(
        yoyo_anchor_limit is not None
        and yoyo is not None
        and primary["distance_to_yoyo_target_px"] > yoyo_anchor_limit
    )
    return {
        "points": selected_polylines[0],
        "polygon": selected_polygons[0],
        "polylines": selected_polylines,
        "polygons": selected_polygons,
        "confidence": round(float(primary["mean_probability"]), 4),
        "method": "semantic_segmentation",
        "needs_review": True,
        "probability_threshold": round(high, 4),
        "low_probability_threshold": round(float(low_threshold), 4) if low_threshold is not None else None,
        "mask_area_target_px": int(sum(item["area"] for item in selected)),
        "component_count": len(selected),
        "polyline_count": len(selected_polylines),
        "component_selection": selection_mode,
        "hand_supported_component_count": hand_supported_component_count,
        "distance_to_yoyo_px": round(primary_distance_px, 2) if primary_distance_px is not None else None,
        "yoyo_body_overlap_fraction": round(float(primary["yoyo_body_overlap_fraction"]), 4),
        "anchored_to_yoyo": bool(yoyo is not None and not spatially_ambiguous),
        "spatially_ambiguous": spatially_ambiguous,
    }
