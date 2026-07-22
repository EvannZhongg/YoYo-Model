import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR, DATASET_CONFIG, YOLO_CONFIG
from yolo_training.download_model import download_model
from yolo_training.prepare_dataset import prepare_yolo_dataset


LOG_FILE = BASE_DIR / "train_yolo.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO dataset and train YOLO11 on yoyo annotations.")
    parser.add_argument("--annotations-dir", default=str(DATASET_CONFIG.annotation_output_dir), help="Annotation root directory.")
    parser.add_argument("--dataset-dir", default=str(YOLO_CONFIG.dataset_dir), help="YOLO dataset output directory.")
    parser.add_argument("--weights", default=str(YOLO_CONFIG.weights_path), help="YOLO model weights path.")
    parser.add_argument("--epochs", type=int, default=YOLO_CONFIG.epochs, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=YOLO_CONFIG.imgsz, help="Training image size.")
    parser.add_argument("--batch", default=YOLO_CONFIG.batch, help="Batch size, or 'auto'.")
    parser.add_argument("--workers", type=int, default=YOLO_CONFIG.workers, help="Data loader workers.")
    parser.add_argument("--device", default=YOLO_CONFIG.device, help="Device, e.g. 0, cpu, or empty for auto.")
    parser.add_argument("--project", default=str(YOLO_CONFIG.project), help="Training project output directory.")
    parser.add_argument("--name", default=YOLO_CONFIG.run_name, help="Training run name.")
    parser.add_argument("--no-prepare", action="store_true", help="Skip YOLO dataset generation and use existing data.yaml.")
    parser.add_argument("--clear-dataset", action="store_true", help="Clear YOLO dataset directory before preparation.")
    parser.add_argument("--auto-download", action="store_true", help="Download weights if they do not exist.")
    return parser.parse_args()


def parse_batch(value: str):
    return value if value == "auto" else int(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    data_yaml_path = dataset_dir / "data.yaml"
    weights_path = Path(args.weights)

    if not args.no_prepare:
        manifest = prepare_yolo_dataset(
            annotations_dir=Path(args.annotations_dir),
            output_dir=dataset_dir,
            train_split=YOLO_CONFIG.train_split,
            seed=YOLO_CONFIG.seed,
            clear=args.clear_dataset,
        )
        data_yaml_path = Path(manifest["data_yaml"])

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}. Run prepare_yolo_dataset.py first.")

    if not weights_path.exists():
        if args.auto_download:
            weights_path = download_model(YOLO_CONFIG.model_name, weights_path.parent)
        else:
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}. "
                "Run download_yolo_model.py first, or pass --auto-download."
            )

    from ultralytics import YOLO

    logger.info("Training YOLO11")
    logger.info("Weights: %s", weights_path)
    logger.info("Data: %s", data_yaml_path)

    model = YOLO(str(weights_path))
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (BASE_DIR / project_path).resolve()
    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": parse_batch(str(args.batch)),
        "workers": args.workers,
        "project": str(project_path),
        "name": args.name,
        "exist_ok": YOLO_CONFIG.exist_ok,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / args.name))
    run_manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.name,
        "run_dir": str(save_dir.resolve()),
        "source_weights": str(weights_path.resolve()),
        "source_weights_sha256": sha256_file(weights_path),
        "dataset_yaml": str(data_yaml_path.resolve()),
        "dataset_manifest": str((dataset_dir / "manifest.json").resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json") if (dataset_dir / "manifest.json").exists() else "",
        "parameters": train_kwargs,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "artifacts": {
            "best": str((save_dir / "weights" / "best.pt").resolve()),
            "last": str((save_dir / "weights" / "last.pt").resolve()),
        },
    }
    try:
        import torch
        import ultralytics

        run_manifest["environment"].update({"torch": torch.__version__, "ultralytics": ultralytics.__version__})
    except Exception:
        pass
    (save_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Run manifest: %s", save_dir / "run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
