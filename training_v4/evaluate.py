from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common.files import sha256_file
from string_segmentation.device import resolve_device
from string_segmentation.semantic_model import _skeleton_cover_paths, _skeletonize, letterbox, normalize_image, render_yolo_segmentation, restore_coordinates
from video_tracking.sequence_metrics import centerline_pair_metrics
from .model import build_model
from .train import CHECKPOINT_FORMAT, fuse_geometry


def load_model(weights: str | Path, device: torch.device):
    path = Path(weights).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported centerline-fusion checkpoint: {path}")
    config = checkpoint.get("model_config") or {}
    model = build_model(str(config.get("architecture", "mobilenet_v3_fpn")), int(config.get("base_channels", 16)), False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def decode_centerline(fused_probability: np.ndarray, tangent: np.ndarray | None, threshold: float = 0.25) -> np.ndarray:
    """Turn fused geometry confidence into a thin mask for polyline extraction."""
    score = np.asarray(fused_probability, dtype=np.float32)
    high = score >= float(threshold)
    if not np.any(high):
        return np.zeros_like(high, dtype=np.uint8)
    support = score >= max(0.05, float(threshold) * 0.45)
    if tangent is not None:
        magnitude = np.linalg.norm(np.asarray(tangent, dtype=np.float32), axis=0)
        support &= magnitude >= 0.05
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    selected = np.unique(labels[high])
    selected = selected[selected > 0]
    return np.isin(labels, selected).astype(np.uint8)


def _paths(binary: np.ndarray, meta) -> list[list[list[float]]]:
    skeleton = _skeletonize(binary)
    return [restore_coordinates(path, meta) for path in _skeleton_cover_paths(skeleton, 8, 256) if len(path) >= 2]


@torch.inference_mode()
def evaluate(weights: str | Path, dataset_dir: str | Path, split: str = "test", device_name: str = "cuda", thresholds: list[float] | None = None, max_samples: int | None = None) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    device = resolve_device(device_name)
    model, checkpoint = load_model(weights, device)
    config = checkpoint["model_config"]
    width, height = int(config["input_width"]), int(config["input_height"])
    image_root, label_root = root / "images" / split, root / "labels" / split
    pairs = [(path, label_root / path.relative_to(image_root).with_suffix(".txt")) for path in sorted(image_root.rglob("*")) if path.is_file() and (label_root / path.relative_to(image_root).with_suffix(".txt")).exists()]
    if max_samples is not None:
        pairs = pairs[: max(0, int(max_samples))]
    values = [float(checkpoint.get("threshold", 0.25))] if thresholds is None else [float(value) for value in thresholds]
    aggregate = {str(value): {"target_samples": 0, "prediction_samples": 0, "target_hits": 0, "prediction_hits": 0} for value in values}
    rows = []
    for image_path, label_path in pairs:
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is None:
            continue
        boxed, mask, meta = letterbox(image, width, height, render_yolo_segmentation(label_path, image.shape[1], image.shape[0]))
        assert mask is not None
        output = model(normalize_image(boxed).unsqueeze(0).to(device))
        fused = fuse_geometry(output)[0, 0].cpu().numpy()
        tangent = torch.tanh(output[0, 2:]).cpu().numpy()
        target_lines = _paths(mask, meta)
        for threshold in values:
            predicted_lines = _paths(decode_centerline(fused, tangent, threshold), meta)
            metrics = centerline_pair_metrics(target_lines, predicted_lines, tolerance_px=(8.0,), spacing_px=2.0)
            tol = metrics["tolerances"]["8"]
            slot = aggregate[str(threshold)]
            for key in ("target_samples", "prediction_samples"):
                slot[key] += int(metrics[key])
            slot["target_hits"] += int(tol["target_hits"])
            slot["prediction_hits"] += int(tol["prediction_hits"])
        rows.append({"image": str(image_path), "target_present": bool(target_lines), "prediction_present": bool(_paths(decode_centerline(fused, tangent, values[0]), meta))})
    summary = {}
    for key, slot in aggregate.items():
        precision = slot["prediction_hits"] / max(1, slot["prediction_samples"])
        recall = slot["target_hits"] / max(1, slot["target_samples"])
        summary[key] = {"metric": "pooled_centerline_f1_at_8_source_px", "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), **slot}
    return {"schema_version": "yoyo_training_v4_eval_v3", "task": "mask_centerline_tangent_2theta_fusion", "weights": str(Path(weights).resolve()), "weights_sha256": sha256_file(Path(weights)), "dataset_dir": str(root), "split": split, "dataset_manifest_sha256": sha256_file(root / "manifest.json"), "summary": summary, "samples": len(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate centerline-fusion checkpoint on a reviewed split.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default="datasets/1Ayoyo_dataset/string_segmentation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()] if args.thresholds else None
    result = evaluate(args.weights, args.dataset_dir, args.split, args.device, thresholds, args.max_samples)
    output = Path(args.output) if args.output else Path(args.weights).resolve().parent.parent / f"{args.split}_centerline_fusion_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
