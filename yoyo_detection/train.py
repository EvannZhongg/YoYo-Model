"""Detector-only training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import BASE_DIR
from yoyo_detection._training import train_task


def train_detection(
    dataset_dir: str | Path,
    project_dir: str | Path,
    models_dir: str | Path,
    epochs: int = 100,
    imgsz: int = 1280,
    batch: str = "2",
    workers: int = 0,
    device: str = "0",
    seed: int = 20260726,
    initial_weights: str | Path | None = None,
    run_tag: str = "",
) -> dict:
    return train_task(
        task="detection", dataset_dir=Path(dataset_dir), project_dir=Path(project_dir), models_dir=Path(models_dir),
        epochs=int(epochs), imgsz=int(imgsz), batch=str(batch), workers=int(workers), device=str(device), seed=int(seed),
        auto_download=True, initial_weights_override=initial_weights, run_tag=run_tag,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the yoyo detector.")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "1Ayoyo_dataset"))
    parser.add_argument("--project-dir", default=str(BASE_DIR / "runs" / "detection"))
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--initial-weights", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", default="2")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    values = vars(args).copy()
    values["initial_weights"] = values.get("initial_weights") or None
    result = train_detection(**values)
    print({"run_dir": result["run_dir"], "task": result["task"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
