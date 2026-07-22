"""Evaluate a reviewed yoyo string segmentation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_counts(data_yaml: Path, split: str) -> tuple[int, int, int, int]:
    """Return image, positive-image, negative-image, and instance counts."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = Path(str(config.get("path") or data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    split_value = config.get(split)
    if not split_value:
        raise ValueError(f"Split {split!r} is missing from {data_yaml}")
    image_root = Path(str(split_value))
    if not image_root.is_absolute():
        image_root = dataset_root / image_root
    images = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Split {split!r} has no images under {image_root}")
    label_root = dataset_root / "labels" / split
    labels_by_stem = {
        path.relative_to(label_root).with_suffix(""): path
        for path in label_root.rglob("*.txt")
    } if label_root.exists() else {}
    positive = 0
    instances = 0
    for image in images:
        relative = image.relative_to(image_root).with_suffix("")
        label = labels_by_stem.get(relative)
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()] if label else []
        if lines:
            positive += 1
            instances += len(lines)
    return len(images), positive, len(images) - positive, instances


def _metric_block(metrics, attr: str) -> dict[str, float | None]:
    block = getattr(metrics, attr, None)
    if block is None:
        return {"precision": None, "recall": None, "map50": None, "map50_95": None}
    def value(name: str) -> float | None:
        item = getattr(block, name, None)
        try:
            return float(item)
        except (TypeError, ValueError):
            return None
    return {
        "precision": value("mp"),
        "recall": value("mr"),
        "map50": value("map50"),
        "map50_95": value("map"),
    }


def evaluate(
    weights: Path,
    data: Path,
    split: str = "test",
    imgsz: int = 960,
    batch: int = 4,
    device: str = "",
    output: Path | None = None,
) -> dict:
    model = YOLO(str(weights))
    image_count, positive_count, negative_count, instance_count = split_counts(data, split)
    kwargs = {
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "workers": 0,
        "verbose": False,
        "plots": False,
    }
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "segment",
        "split": split,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "images": image_count,
        "positive_images": positive_count,
        "negative_images": negative_count,
        "ground_truth_instances": instance_count,
        "box": _metric_block(metrics, "box"),
        "seg": _metric_block(metrics, "seg"),
        "fitness": float(metrics.fitness) if getattr(metrics, "fitness", None) is not None else None,
        "limitations": [
            "Metrics are based on the current small reviewed split and are not production-level evidence.",
            "Thin-string masks are sensitive to image size and annotation buffer width.",
        ],
    }
    destination = output or weights.parent.parent / f"{split}_segmentation_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO string segmentation weights.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(
        Path(args.weights),
        Path(args.data),
        args.split,
        args.imgsz,
        args.batch,
        args.device,
        Path(args.output) if args.output else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
