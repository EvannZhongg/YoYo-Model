from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common.files import sha256_file
from string_segmentation.device import resolve_device
from string_segmentation.semantic_model import letterbox, normalize_image, render_yolo_segmentation, restore_coordinates, _skeleton_cover_paths, _skeletonize
from video_tracking.sequence_metrics import centerline_pair_metrics
from training_v4.model import build_model


def load_model(weights: Path, device: torch.device):
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    config = checkpoint["model_config"]; model = build_model(config["architecture"], int(config.get("base_channels", 16)), False).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval(); return model, checkpoint


def decode_centerline(heat: np.ndarray, direction: np.ndarray, threshold: float) -> np.ndarray:
    """Use direction magnitude to connect weak context around strong seeds."""
    magnitude = np.linalg.norm(direction, axis=0)
    score = np.asarray(heat, dtype=np.float32) * (0.9 + 0.1 * np.clip(magnitude, 0.0, 1.0))
    high = score >= float(threshold)
    context = score >= float(threshold) * 0.65
    if not np.any(high):
        return np.zeros_like(high, dtype=np.uint8)
    # Guard the graph decoder against pathological all-background activations
    # from an untrained or corrupted checkpoint.
    if int(context.sum()) > int(context.size * 0.25):
        context = cv2.dilate(high.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1).astype(bool)
    count, labels = cv2.connectedComponents(context.astype(np.uint8), connectivity=8)
    seeds = np.unique(labels[high]); seeds = seeds[seeds > 0]
    return np.isin(labels, seeds).astype(np.uint8)


@torch.inference_mode()
def evaluate(weights: str | Path, dataset_dir: str | Path, split: str = "test", device_name: str = "cuda", thresholds: list[float] | None = None, max_samples: int | None = None) -> dict[str, Any]:
    weights, dataset_dir = Path(weights).resolve(), Path(dataset_dir).resolve(); device = resolve_device(device_name); model, checkpoint = load_model(weights, device); cfg = checkpoint["model_config"]; width, height = int(cfg["input_width"]), int(cfg["input_height"])
    image_root, label_root = dataset_dir / "images" / split, dataset_dir / "labels" / split
    pairs = [(p, label_root / p.relative_to(image_root).with_suffix(".txt")) for p in sorted(image_root.rglob("*")) if p.is_file() and (label_root / p.relative_to(image_root).with_suffix(".txt")).exists()]
    rows = []; thresholds = thresholds or [float(checkpoint.get("threshold", 0.5))]
    aggregate = {str(t): {"target_samples": 0, "prediction_samples": 0, "target_hits": 0, "prediction_hits": 0} for t in thresholds}
    if max_samples is not None:
        pairs = pairs[: max(0, int(max_samples))]
    for image_path, label_path in pairs:
        encoded = np.fromfile(image_path, dtype=np.uint8); image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None: continue
        boxed, mask, meta = letterbox(image, width, height, render_yolo_segmentation(label_path, image.shape[1], image.shape[0])); assert mask is not None
        tensor = normalize_image(boxed).unsqueeze(0).to(device); output = model(tensor); heat = torch.sigmoid(output[:, 0]).cpu().numpy()[0]; direction = torch.tanh(output[:, 1:]).cpu().numpy()[0]
        target_skel = _skeletonize(mask); target_paths = [restore_coordinates(path, meta) for path in _skeleton_cover_paths(target_skel, 8, 256)]
        for threshold in thresholds:
            decoded = decode_centerline(heat, direction, float(threshold))
            pred_skel = _skeletonize(decoded); pred_paths = [restore_coordinates(path, meta) for path in _skeleton_cover_paths(pred_skel, 8, 256)]
            metrics = centerline_pair_metrics(target_paths, pred_paths, tolerance_px=(8.0,), spacing_px=2.0); tol = metrics["tolerances"]["8"]
            slot = aggregate[str(threshold)]; slot["target_samples"] += metrics["target_samples"]; slot["prediction_samples"] += metrics["prediction_samples"]; slot["target_hits"] += tol["target_hits"]; slot["prediction_hits"] += tol["prediction_hits"]
            rows.append({"image": str(image_path), "threshold": threshold, "metrics": metrics})
    summary = {}
    for threshold, slot in aggregate.items():
        precision = slot["prediction_hits"] / max(1, slot["prediction_samples"]); recall = slot["target_hits"] / max(1, slot["target_samples"]); summary[threshold] = {"metric": "pooled_centerline_f1_at_8_source_px", "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), **slot}
    return {"schema_version": "yoyo_training_v4_eval_v1", "task": "centerline_heatmap_direction", "decoder": "component", "weights": str(weights), "weights_sha256": sha256_file(weights), "dataset_dir": str(dataset_dir), "split": split, "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"), "summary": summary, "samples": len(pairs), "rows": rows}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--dataset-dir", default="datasets/1Ayoyo_dataset/string_segmentation"); p.add_argument("--split", choices=["train", "val", "test"], default="test"); p.add_argument("--device", default="cuda"); p.add_argument("--thresholds", default="0.3,0.5,0.7"); p.add_argument("--max-samples", type=int, default=None); p.add_argument("--output", default=""); a = p.parse_args(); result = evaluate(a.weights, a.dataset_dir, a.split, a.device, [float(v) for v in a.thresholds.split(",") if v.strip()], a.max_samples); out = Path(a.output) if a.output else Path(a.weights).parent.parent / f"{a.split}_centerline_v4_metrics.json"; out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result["summary"], ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
