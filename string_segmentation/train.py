"""Train and version a YOLO segmentation model for reviewed yoyo strings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import STRING_SEGMENTATION_CONFIG, YOLO_CONFIG
from string_segmentation.prepare_dataset import prepare_string_dataset
from yolo_training.download_model import download_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(value: str):
    return value if value == "auto" else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reviewed yoyo string segmentation model.")
    parser.add_argument("--annotations-dir", default=str(STRING_SEGMENTATION_CONFIG.annotations_dir))
    parser.add_argument("--dataset-dir", default=str(STRING_SEGMENTATION_CONFIG.dataset_dir))
    parser.add_argument("--weights", default=str(STRING_SEGMENTATION_CONFIG.weights_path))
    parser.add_argument("--epochs", type=int, default=STRING_SEGMENTATION_CONFIG.epochs)
    parser.add_argument("--imgsz", type=int, default=STRING_SEGMENTATION_CONFIG.imgsz)
    parser.add_argument("--batch", default=STRING_SEGMENTATION_CONFIG.batch)
    parser.add_argument("--workers", type=int, default=STRING_SEGMENTATION_CONFIG.workers)
    parser.add_argument("--device", default=STRING_SEGMENTATION_CONFIG.device)
    parser.add_argument("--mask-ratio", type=int, default=STRING_SEGMENTATION_CONFIG.mask_ratio)
    parser.add_argument("--translate", type=float, default=STRING_SEGMENTATION_CONFIG.translate)
    parser.add_argument("--scale", type=float, default=STRING_SEGMENTATION_CONFIG.scale)
    parser.add_argument("--mosaic", type=float, default=STRING_SEGMENTATION_CONFIG.mosaic)
    parser.add_argument("--project", default=str(STRING_SEGMENTATION_CONFIG.project))
    parser.add_argument("--name", default=STRING_SEGMENTATION_CONFIG.run_name)
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--clear-dataset", action="store_true")
    parser.add_argument("--auto-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be at least 1")
    for name in ("translate", "scale", "mosaic"):
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name} must not be negative")
    dataset_dir = Path(args.dataset_dir)
    if not args.no_prepare:
        manifest = prepare_string_dataset(
            Path(args.annotations_dir), dataset_dir, STRING_SEGMENTATION_CONFIG.line_width_px, args.clear_dataset
        )
    else:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    train_positive = int(manifest.get("counts", {}).get("train", {}).get("positive", 0))
    val_positive = int(manifest.get("counts", {}).get("val", {}).get("positive", 0))
    if train_positive < 1 or val_positive < 1:
        raise RuntimeError(
            "String training requires at least one reviewed visible/partial string in both train and val; "
            f"found train={train_positive}, val={val_positive}. Review string annotations in Video Workbench first."
        )
    weights = Path(args.weights)
    if not weights.exists():
        if not args.auto_download:
            raise FileNotFoundError(f"Segmentation weights not found: {weights}. Use --auto-download or download it manually.")
        weights = download_model(
            STRING_SEGMENTATION_CONFIG.model_name,
            weights.parent,
            YOLO_CONFIG.model_url_template.format(model_name=STRING_SEGMENTATION_CONFIG.model_name),
        )
    from ultralytics import YOLO

    kwargs = {
        "data": str(dataset_dir / "data.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": _batch(str(args.batch)),
        "workers": args.workers,
        "mask_ratio": args.mask_ratio,
        "translate": args.translate,
        "scale": args.scale,
        "mosaic": args.mosaic,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        kwargs["device"] = args.device
    model = YOLO(str(weights))
    model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir)
    run = {
        "schema_version": "1.0",
        "task": "segment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(save_dir.resolve()),
        "source_weights": str(weights.resolve()),
        "source_weights_sha256": _sha256(weights),
        "dataset_manifest": str((dataset_dir / "manifest.json").resolve()),
        "dataset_manifest_sha256": _sha256(dataset_dir / "manifest.json"),
        "parameters": kwargs,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "artifacts": {
            "best": str((save_dir / "weights" / "best.pt").resolve()),
            "last": str((save_dir / "weights" / "last.pt").resolve()),
        },
    }
    try:
        import torch
        import ultralytics
        run["environment"].update({"torch": torch.__version__, "ultralytics": ultralytics.__version__})
    except Exception:
        pass
    (save_dir / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
