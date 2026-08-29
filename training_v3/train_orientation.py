"""Train a versioned classifier from the ROI orientation view."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from common.files import sha256_file
from config import BASE_DIR
from training_v3.train import _initialization_lineage, _json_value, _weights_for


def train_orientation(
    view_manifest_path: Path,
    project_dir: Path,
    epochs: int,
    device: str,
    initial_weights: str | Path | None = None,
    run_tag: str = "",
    optimizer: str = "auto",
    learning_rate: float | None = None,
    imgsz: int = 320,
    batch: int = 16,
    patience: int = 20,
    dropout: float = 0.2,
    freeze: int = 0,
    seed: int = 20260726,
) -> dict:
    view_manifest_path = view_manifest_path.resolve()
    view = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    parent_path = Path(view["parent_manifest"])
    if sha256_file(parent_path) != view["parent_manifest_sha256"]:
        raise RuntimeError("Canonical dataset manifest changed after ROI view creation")
    weights = _weights_for(
        "orientation",
        BASE_DIR / "models",
        auto_download=True,
        override=initial_weights,
    )
    lineage_manifest = {"split_policy": {"source_groups": view["source_groups"]}}
    initialization_lineage = _initialization_lineage(weights, lineage_manifest)
    from ultralytics import YOLO
    import torch
    import ultralytics

    model_token = weights.stem.replace("_", "-")
    suffix = f"_{run_tag.strip()}" if run_tag.strip() else ""
    name = f"{view['dataset_id']}_{view['view_id']}_{model_token}{suffix}"
    kwargs = {
        "data": view["data"],
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": 0,
        "project": str(project_dir.resolve()),
        "name": name,
        "exist_ok": False,
        "seed": int(seed),
        "patience": max(0, int(patience)),
        "cos_lr": True,
        "dropout": float(dropout),
        "optimizer": optimizer,
    }
    if freeze > 0:
        kwargs["freeze"] = int(freeze)
    if learning_rate is not None:
        kwargs["lr0"] = float(learning_rate)
    if device:
        kwargs["device"] = device
    model = YOLO(str(weights))
    results = model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    manifest = {
        "schema_version": "yoyo_training_run_v2",
        "task": "orientation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": name,
        "run_dir": str(save_dir),
        "dataset_id": view["dataset_id"],
        "dataset_view_id": view["view_id"],
        "dataset_manifest": str(view_manifest_path),
        "dataset_manifest_sha256": sha256_file(view_manifest_path),
        "source_policy": view["source_policy"],
        "source_groups": view["source_groups"],
        "dataset_counts": view["counts"],
        "orientation_classes": list(view.get("classes") or []),
        "orientation_label_field": str(view.get("label_field") or "trick_orientation"),
        "coarse_mapping": view.get("coarse_mapping") or {},
        "initial_weights": str(weights),
        "initial_weights_sha256": sha256_file(weights),
        "initialization_lineage": initialization_lineage,
        "parameters": kwargs,
        "metrics": _json_value(getattr(results, "results_dict", {})),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "artifacts": {
            "best": str(best),
            "best_sha256": sha256_file(best),
            "last": str(last),
            "last_sha256": sha256_file(last),
        },
        "promotion": {
            "status": "candidate" if initialization_lineage["promotion_eligible"] else "ineligible_source_overlap",
            "rule": (
                "Evaluate once on the untouched ROI test split."
                if initialization_lineage["promotion_eligible"]
                else "Do not promote: initialization training sources overlap this view's evaluation sources."
            ),
        },
    }
    (save_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ROI trick-orientation classifier.")
    parser.add_argument("--view-manifest", default=str(BASE_DIR / "datasets" / "1Ayoyo_dataset" / "orientation_roi" / "manifest.json"))
    parser.add_argument("--project-dir", default=str(BASE_DIR / "runs" / "v2v3"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--initial-weights", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers for transfer learning.")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    result = train_orientation(
        Path(args.view_manifest),
        Path(args.project_dir),
        args.epochs,
        args.device,
        args.initial_weights or None,
        args.run_tag,
        args.optimizer,
        args.lr0,
        args.imgsz,
        args.batch,
        args.patience,
        args.dropout,
        args.freeze,
        args.seed,
    )
    print(json.dumps({"run_dir": result["run_dir"], "view_id": result["dataset_view_id"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
