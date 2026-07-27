"""Train and version the three models produced by the unified annotation dataset."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.files import sha256_file
from config import BASE_DIR
from training_v2.prepare_dataset import SOURCE_POLICY
from yolo_training.download_model import download_model


TASKS = ("detection", "string_segmentation", "orientation")
DEFAULT_WEIGHTS = {
    "detection": "yolo11n.pt",
    "string_segmentation": "yolo11n-seg.pt",
    "orientation": "yolo11n-cls.pt",
}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _weights_for(
    task: str,
    models_dir: Path,
    auto_download: bool,
    override: str | Path | None = None,
) -> Path:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Initial weights not found: {path}")
        return path.resolve()
    name = DEFAULT_WEIGHTS[task]
    path = models_dir / name
    if path.exists():
        return path.resolve()
    if not auto_download:
        raise FileNotFoundError(f"Initial weights not found: {path}. Re-run with --auto-download.")
    return download_model(name, models_dir).resolve()


def _task_data(manifest: dict[str, Any], task: str) -> str:
    return str(manifest["tasks"][task]["data"])


def _initialization_lineage(initial_weights: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    parent_manifest_path = initial_weights.parent.parent / "run_manifest.json"
    lineage: dict[str, Any] = {
        "kind": "versioned_run" if parent_manifest_path.exists() else "foundation_pretrained",
        "parent_run_manifest": str(parent_manifest_path.resolve()) if parent_manifest_path.exists() else "",
        "evaluation_source_overlap": {"val": [], "test": []},
        "promotion_eligible": True,
    }
    if not parent_manifest_path.exists():
        return lineage
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_train = set((parent.get("source_groups") or {}).get("train", []))
    current_groups = (manifest.get("split_policy") or {}).get("source_groups", {})
    overlap = {
        split: sorted(parent_train & set(current_groups.get(split, [])))
        for split in ("val", "test")
    }
    lineage["evaluation_source_overlap"] = overlap
    lineage["promotion_eligible"] = not any(overlap.values())
    return lineage


def train_task(
    task: str,
    dataset_dir: Path,
    project_dir: Path,
    models_dir: Path,
    epochs: int,
    imgsz: int,
    batch: str,
    workers: int,
    device: str,
    seed: int,
    auto_download: bool,
    initial_weights_override: str | Path | None = None,
    run_tag: str = "",
    patience: int = 20,
    optimizer: str = "auto",
    learning_rate: float | None = None,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    dataset_manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_policy") != SOURCE_POLICY:
        raise RuntimeError("Refusing to train from a dataset that does not prove the unified annotation source policy")
    initial_weights = _weights_for(task, models_dir, auto_download, initial_weights_override)
    initialization_lineage = _initialization_lineage(initial_weights, manifest)
    from ultralytics import YOLO
    import torch
    import ultralytics

    model = YOLO(str(initial_weights))
    model_token = initial_weights.stem.replace("_", "-")
    suffix = f"_{run_tag.strip()}" if run_tag.strip() else ""
    run_name = f"{manifest['dataset_id']}_{task}_{model_token}{suffix}"
    train_kwargs: dict[str, Any] = {
        "data": _task_data(manifest, task),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": "auto" if str(batch) == "auto" else int(batch),
        "workers": workers,
        "project": str(project_dir.resolve()),
        "name": run_name,
        "exist_ok": False,
        "seed": seed,
        "patience": max(0, int(patience)),
        "cos_lr": True,
        "optimizer": optimizer,
    }
    if learning_rate is not None:
        train_kwargs["lr0"] = float(learning_rate)
    if task in {"detection", "string_segmentation"}:
        train_kwargs.update({"mosaic": 0.0, "scale": 0.15, "translate": 0.05})
    if task == "string_segmentation":
        train_kwargs.update({"mask_ratio": 1, "close_mosaic": 0})
    if task == "orientation":
        train_kwargs["dropout"] = 0.2
    if device:
        train_kwargs["device"] = device
    results = model.train(**train_kwargs)
    save_dir = Path(model.trainer.save_dir).resolve()
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    results_dict = _json_value(getattr(results, "results_dict", {}))
    run_manifest = {
        "schema_version": "yoyo_training_run_v2",
        "task": task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "run_dir": str(save_dir),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source_policy": manifest["source_policy"],
        "source_annotation_sha256": manifest["source_annotation_sha256"],
        "source_groups": manifest["split_policy"]["source_groups"],
        "dataset_counts": manifest["counts"],
        "initial_weights": str(initial_weights),
        "initial_weights_sha256": sha256_file(initial_weights),
        "initialization_lineage": initialization_lineage,
        "parameters": train_kwargs,
        "metrics": results_dict,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "artifacts": {
            "best": str(best_path),
            "best_sha256": sha256_file(best_path) if best_path.exists() else "",
            "last": str(last_path),
            "last_sha256": sha256_file(last_path) if last_path.exists() else "",
        },
        "promotion": {
            "status": "candidate" if initialization_lineage["promotion_eligible"] else "ineligible_source_overlap",
            "rule": (
                "Promote only after task-specific evaluation on the untouched test split."
                if initialization_lineage["promotion_eligible"]
                else "Do not promote: the initialization run trained on sources assigned to this dataset's val/test split."
            ),
        },
    }
    (save_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train yoyo models from the unified NYPC1A dataset with versioned run manifests.")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "yoyo_dataset"))
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--project-dir", default=str(BASE_DIR / "runs" / "v2v3"))
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--detection-weights", default="", help="Optional pretrained detector checkpoint.")
    parser.add_argument("--string-weights", default="", help="Optional pretrained segmenter checkpoint.")
    parser.add_argument("--orientation-weights", default="", help="Optional pretrained classifier checkpoint.")
    parser.add_argument("--run-tag", default="", help="Short candidate label included in the versioned run name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--detection-imgsz", type=int, default=1280)
    parser.add_argument("--string-imgsz", type=int, default=1280)
    parser.add_argument("--orientation-imgsz", type=int, default=320)
    parser.add_argument("--detection-batch", default="2")
    parser.add_argument("--string-batch", default="2")
    parser.add_argument("--orientation-batch", default="16")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--auto-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = TASKS if args.task == "all" else (args.task,)
    runs = []
    for task in selected:
        task_imgsz = {
            "detection": args.detection_imgsz,
            "string_segmentation": args.string_imgsz,
            "orientation": args.orientation_imgsz,
        }[task]
        task_batch = {
            "detection": args.detection_batch,
            "string_segmentation": args.string_batch,
            "orientation": args.orientation_batch,
        }[task]
        task_weights = {
            "detection": args.detection_weights,
            "string_segmentation": args.string_weights,
            "orientation": args.orientation_weights,
        }[task]
        runs.append(
            train_task(
                task=task,
                dataset_dir=Path(args.dataset_dir),
                project_dir=Path(args.project_dir),
                models_dir=Path(args.models_dir),
                epochs=args.epochs,
                imgsz=task_imgsz,
                batch=str(task_batch),
                workers=args.workers,
                device=str(args.device),
                seed=args.seed,
                auto_download=args.auto_download,
                initial_weights_override=task_weights or None,
                run_tag=args.run_tag,
                patience=args.patience,
                optimizer=args.optimizer,
                learning_rate=args.lr0,
            )
        )
    task_suffix = "all" if args.task == "all" else args.task
    suite_path = Path(args.project_dir) / f"{runs[0]['dataset_id']}_{task_suffix}_suite.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": "yoyo_training_suite_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": runs[0]["dataset_id"],
                "tasks": [{"task": run["task"], "run_manifest": str(Path(run["run_dir"]) / "run_manifest.json")} for run in runs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"dataset_id": runs[0]["dataset_id"], "runs": [run["run_dir"] for run in runs], "suite": str(suite_path.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
