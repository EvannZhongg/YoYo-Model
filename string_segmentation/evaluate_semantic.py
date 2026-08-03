"""Evaluate one semantic string checkpoint on an untouched reviewed split."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.files import sha256_file
from config import SEMANTIC_STRING_CONFIG
from string_segmentation.device import resolve_device
from string_segmentation.semantic_metrics import collect_probabilities, metrics_at_threshold, remove_small_components
from string_segmentation.semantic_model import (
    ReviewedStringDataset,
    fuse_calibrated_probabilities,
    letterbox,
    load_checkpoint,
)


def _check_dataset_manifest(
    checkpoint_manifest_hash: str,
    current_manifest_hash: str,
    allow_dataset_mismatch: bool,
) -> tuple[bool, str]:
    matches = not checkpoint_manifest_hash or checkpoint_manifest_hash == current_manifest_hash
    if matches:
        return True, ""
    warning = (
        "Dataset manifest differs from the checkpoint training manifest. Metrics are valid only as an "
        "explicit cross-model comparison on this exact evaluation dataset, not as the checkpoint's native test result."
    )
    if not allow_dataset_mismatch:
        raise RuntimeError(
            "Dataset manifest differs from the checkpoint training manifest; refusing evaluation unless "
            "--allow-dataset-mismatch is explicitly provided"
        )
    return False, warning


def _artifact_suffix(
    dataset_matches_checkpoint: bool,
    current_manifest_hash: str,
    threshold_override: float | None,
) -> str:
    parts: list[str] = []
    if not dataset_matches_checkpoint:
        parts.append(f"external_{current_manifest_hash[:12]}")
    if threshold_override is not None:
        threshold_token = f"{float(threshold_override):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        parts.append(f"threshold_{threshold_token}")
    return f"_{'_'.join(parts)}" if parts else ""


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Could not encode semantic evaluation sheet: {path}")
    encoded.tofile(str(path))


def _prediction_sheet(samples: list[dict[str, Any]], threshold: float, output: Path) -> Path:
    cell_width, image_height, text_height = 480, 272, 58
    columns = 2
    rows = max(1, (len(samples) + columns - 1) // columns)
    canvas = np.full((rows * (image_height + text_height), columns * cell_width, 3), 255, dtype=np.uint8)
    for index, sample in enumerate(samples):
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_height, target_width = sample["target"].shape
        image, _, _ = letterbox(image, target_width, target_height)
        target = (sample["target"] > 0).astype(np.uint8)
        prediction = remove_small_components(sample["probability"] >= threshold, 8)
        overlay = image.copy()
        overlay[target > 0] = (40, 210, 40)
        prediction_only = np.logical_and(prediction > 0, target == 0)
        overlap = np.logical_and(prediction > 0, target > 0)
        overlay[prediction_only] = (210, 50, 210)
        overlay[overlap] = (40, 220, 240)
        display = cv2.resize(overlay, (cell_width, image_height), interpolation=cv2.INTER_AREA)
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + text_height)
        canvas[y : y + image_height, x : x + cell_width] = display
        name = Path(sample["image_path"]).parent.name + "/" + Path(sample["image_path"]).name
        cv2.putText(canvas, name[:70], (x + 5, y + image_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"target_px={int(target.sum())} pred_px={int(prediction.sum())} threshold={threshold:.2f}",
            (x + 5, y + image_height + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    _write_image(output, canvas)
    return output


def evaluate(
    weights: str | Path,
    dataset_dir: str | Path,
    split: str,
    device_value: str,
    threshold_override: float | None = None,
    allow_dataset_mismatch: bool = False,
    ensemble_weights: str | Path | None = None,
    ensemble_alpha: float = 0.0,
    ensemble_candidate_threshold: float = 0.5,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    weights = Path(weights)
    dataset_dir = Path(dataset_dir)
    device = resolve_device(device_value)
    model, checkpoint = load_checkpoint(weights, device)
    config = checkpoint["model_config"]
    dataset = ReviewedStringDataset(
        dataset_dir,
        split,
        int(config["input_width"]),
        int(config["input_height"]),
        int(config.get("min_mask_width_px", 2)),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    primary_threshold = float(
        checkpoint.get("threshold", 0.5) if threshold_override is None else threshold_override
    )
    ensemble_path = Path(ensemble_weights) if ensemble_weights else None
    if not 0.0 <= float(ensemble_alpha) <= 1.0:
        raise ValueError("ensemble_alpha must be between 0 and 1")
    if not 0.0 < float(ensemble_candidate_threshold) < 1.0:
        raise ValueError("ensemble_candidate_threshold must be between 0 and 1")
    if ensemble_path is not None and float(ensemble_alpha) > 0.0:
        ensemble_model, ensemble_checkpoint = load_checkpoint(ensemble_path, device)
        if ensemble_checkpoint.get("model_config") != checkpoint.get("model_config"):
            raise ValueError("Semantic ensemble checkpoints use incompatible model configurations")
        samples = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                primary_probability = torch.sigmoid(model(images)).detach().cpu().numpy()[:, 0]
                secondary_probability = torch.sigmoid(ensemble_model(images)).detach().cpu().numpy()[:, 0]
                targets = batch["mask"].numpy()[:, 0]
                for primary, secondary, target, path in zip(
                    primary_probability,
                    secondary_probability,
                    targets,
                    batch["image_path"],
                ):
                    samples.append({
                        "probability": fuse_calibrated_probabilities(
                            primary,
                            secondary,
                            alpha=float(ensemble_alpha),
                            primary_threshold=primary_threshold,
                            secondary_threshold=float(ensemble_candidate_threshold),
                        ),
                        "target": (target > 0.5).astype(np.uint8),
                        "image_path": str(path),
                    })
        threshold = 0.5
    else:
        samples = collect_probabilities(model, loader, device)
        threshold = primary_threshold
    metrics = metrics_at_threshold(samples, threshold, tolerance_px=3, min_component_pixels=8)
    manifest_path = dataset_dir / "manifest.json"
    current_manifest_hash = sha256_file(manifest_path)
    checkpoint_manifest_hash = str(checkpoint.get("dataset_manifest_sha256", ""))
    dataset_matches_checkpoint, mismatch_warning = _check_dataset_manifest(
        checkpoint_manifest_hash,
        current_manifest_hash,
        allow_dataset_mismatch,
    )
    if mismatch_warning:
        warnings.warn(mismatch_warning, RuntimeWarning, stacklevel=2)
    run_dir = Path(output_dir) if output_dir is not None else weights.parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = _artifact_suffix(dataset_matches_checkpoint, current_manifest_hash, threshold_override)
    sheet_path = run_dir / f"{split}_semantic_predictions{suffix}.jpg"
    _prediction_sheet(samples, threshold, sheet_path)
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "binary_semantic_segmentation",
        "split": split,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "ensemble_weights": str(ensemble_path.resolve()) if ensemble_path is not None else "",
        "ensemble_weights_sha256": (
            sha256_file(ensemble_path) if ensemble_path is not None else ""
        ),
        "ensemble_alpha": float(ensemble_alpha),
        "ensemble_candidate_threshold": float(ensemble_candidate_threshold),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": current_manifest_hash,
        "checkpoint_dataset_manifest_sha256": checkpoint_manifest_hash,
        "dataset_matches_checkpoint": dataset_matches_checkpoint,
        "warnings": [mismatch_warning] if mismatch_warning else [],
        "threshold_source": (
            "calibrated_probability_ensemble"
            if ensemble_path is not None and float(ensemble_alpha) > 0.0
            else ("checkpoint_validation_selection" if threshold_override is None else "explicit_override")
        ),
        "metrics": metrics,
        "prediction_sheet": str(sheet_path.resolve()),
        "limitations": [
            "The independent split is very small and does not establish production reliability.",
            "Exact pixel metrics are strict for thin strings; tolerant metrics use a 3-pixel radius and are reported separately.",
        ],
    }
    output = run_dir / f"{split}_semantic_metrics{suffix}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_path"] = str(output.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic string segmentation on a reviewed split.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default=str(SEMANTIC_STRING_CONFIG.dataset_dir))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=SEMANTIC_STRING_CONFIG.device)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Evaluate on a different manifest and record the mismatch for controlled same-set comparisons.",
    )
    parser.add_argument("--ensemble-weights", default="")
    parser.add_argument("--ensemble-alpha", type=float, default=0.0)
    parser.add_argument("--ensemble-candidate-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(
        args.weights,
        args.dataset_dir,
        args.split,
        args.device,
        args.threshold,
        args.allow_dataset_mismatch,
        args.ensemble_weights or None,
        args.ensemble_alpha,
        args.ensemble_candidate_threshold,
        args.output_dir or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
