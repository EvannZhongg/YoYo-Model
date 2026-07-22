"""Evaluate a trained YOLO detector and save versioned metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_counts(data_yaml: Path, split: str) -> tuple[int, int, int]:
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
    images = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"Split {split!r} has no images under {image_root}")
    label_root = dataset_root / "labels" / split
    labels = list(label_root.rglob("*.txt")) if label_root.exists() else []
    positives = sum(bool(path.read_text(encoding="utf-8").strip()) for path in labels)
    return len(images), positives, max(0, len(images) - positives)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO weights on val or test split.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    weights = Path(args.weights)
    data = Path(args.data)
    model = YOLO(str(weights))
    image_count, positive_count, negative_count = split_counts(data, args.split)
    kwargs = {"data": str(data), "split": args.split, "imgsz": args.imgsz, "batch": args.batch, "workers": 0, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)
    box = metrics.box
    result = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "data_yaml": str(data.resolve()),
        "data_yaml_sha256": sha256_file(data),
        "images": image_count,
        "positive_images": positive_count,
        "negative_images": negative_count,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "fitness": float(metrics.fitness),
        "limitations": ["Metrics are based on the current small reviewed split and are not production-level evidence."],
    }
    output = Path(args.output) if args.output else weights.parent.parent / f"{args.split}_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
