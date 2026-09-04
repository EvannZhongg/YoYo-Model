"""Evaluate one semantic string checkpoint on an untouched reviewed split."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import DataLoader

from common.files import sha256_file
from config import SEMANTIC_STRING_CONFIG, TRACKING_CONFIG
from string_segmentation.device import resolve_device
from string_segmentation.semantic_metrics import collect_probabilities, metrics_at_threshold, remove_small_components
from string_segmentation.semantic_model import (
    ReviewedStringDataset,
    hysteresis_mask,
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
    low_threshold: float | None = None,
) -> str:
    parts: list[str] = []
    if not dataset_matches_checkpoint:
        parts.append(f"external_{current_manifest_hash[:12]}")
    if threshold_override is not None:
        threshold_token = f"{float(threshold_override):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        parts.append(f"threshold_{threshold_token}")
    if low_threshold is not None:
        low_token = f"{float(low_threshold):.4f}".rstrip("0").rstrip(".").replace(".", "p")
        parts.append(f"low_{low_token}")
    return f"_{'_'.join(parts)}" if parts else ""


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Could not encode semantic evaluation sheet: {path}")
    encoded.tofile(str(path))


def _read_image(path: str | Path) -> np.ndarray | None:
    """Read image bytes through NumPy so Unicode Windows paths work reliably."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _prediction_sheet(
    samples: list[dict[str, Any]],
    threshold: float,
    output: Path,
    low_threshold: float | None = None,
) -> Path:
    cell_width, image_height, text_height = 480, 272, 58
    columns = 2
    rows = max(1, (len(samples) + columns - 1) // columns)
    canvas = np.full((rows * (image_height + text_height), columns * cell_width, 3), 255, dtype=np.uint8)
    for index, sample in enumerate(samples):
        image = _read_image(sample["image_path"])
        if image is None:
            continue
        target_height, target_width = sample["target"].shape
        image, _, _ = letterbox(image, target_width, target_height)
        target = (sample["target"] > 0).astype(np.uint8)
        prediction = remove_small_components(
            hysteresis_mask(sample["probability"], threshold, low_threshold), 8
        )
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
    output_dir: str | Path | None = None,
    min_component_pixels: int = 8,
    low_threshold: float | None = None,
    max_components: int = TRACKING_CONFIG.string_max_components,
    inference_scale: float = TRACKING_CONFIG.string_inference_scale,
) -> dict[str, Any]:
    weights = Path(weights)
    dataset_dir = Path(dataset_dir)
    device = resolve_device(device_value)
    model, checkpoint = load_checkpoint(weights, device)
    config = checkpoint["model_config"]
    inference_scale = float(inference_scale)
    if not 0.5 <= inference_scale <= 2.0:
        raise ValueError("inference_scale must be between 0.5 and 2.0")
    input_width = max(32, int(round(int(config["input_width"]) * inference_scale / 16.0)) * 16)
    input_height = max(32, int(round(int(config["input_height"]) * inference_scale / 16.0)) * 16)
    dataset = ReviewedStringDataset(
        dataset_dir,
        split,
        input_width,
        input_height,
        int(config.get("min_mask_width_px", 1)),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    threshold = float(
        checkpoint.get("threshold", 0.5) if threshold_override is None else threshold_override
    )
    samples = collect_probabilities(model, loader, device)
    metrics = metrics_at_threshold(
        samples,
        threshold,
        tolerance_px=3,
        min_component_pixels=max(1, int(min_component_pixels)),
        low_threshold=low_threshold,
        max_components=max(1, int(max_components)),
    )
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
    suffix = _artifact_suffix(dataset_matches_checkpoint, current_manifest_hash, threshold_override, low_threshold)
    sheet_path = run_dir / f"{split}_semantic_predictions{suffix}.jpg"
    _prediction_sheet(samples, threshold, sheet_path, low_threshold)
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "binary_semantic_segmentation",
        "split": split,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "min_component_pixels": int(min_component_pixels),
        "max_components": max(1, int(max_components)),
        "inference_scale": inference_scale,
        "input_size": [input_width, input_height],
        "low_threshold": low_threshold,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": current_manifest_hash,
        "checkpoint_dataset_manifest_sha256": checkpoint_manifest_hash,
        "dataset_matches_checkpoint": dataset_matches_checkpoint,
        "warnings": [mismatch_warning] if mismatch_warning else [],
        "threshold_source": "checkpoint_validation_selection" if threshold_override is None else "explicit_override",
        "metrics": metrics,
        "prediction_sheet": str(sheet_path.resolve()),
        "limitations": [
            "The independent split is very small and does not establish production reliability.",
            "Exact pixel metrics are strict for thin strings; tolerant metrics use a 3-pixel radius and are reported separately.",
            "Canonical validation ranking uses pooled centerline F1 at an 8 source-pixel tolerance after mask skeletonization; consecutive evaluation additionally includes final tracking post-processing.",
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
    parser.add_argument("--inference-scale", type=float, default=TRACKING_CONFIG.string_inference_scale)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Evaluate on a different manifest and record the mismatch for controlled same-set comparisons.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=TRACKING_CONFIG.string_max_components)
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
        args.output_dir or None,
        args.min_component_pixels,
        args.low_threshold,
        args.max_components,
        args.inference_scale,
    )
    document = json.dumps(result, ensure_ascii=False, indent=2)
    encoding = sys.stdout.encoding or "utf-8"
    print(document.encode(encoding, errors="backslashreplace").decode(encoding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
