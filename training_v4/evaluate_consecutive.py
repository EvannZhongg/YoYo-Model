from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from string_segmentation.device import resolve_device
from string_segmentation.semantic_model import _skeleton_cover_paths, _skeletonize, letterbox, normalize_image, restore_coordinates
from video_tracking.sequence_metrics import _annotation_polylines, centerline_pair_metrics
from .evaluate import decode_centerline, load_model
from .train import fuse_geometry


def _presence_stats(records: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(target and prediction for target, prediction in records)
    fp = sum((not target) and prediction for target, prediction in records)
    fn = sum(target and (not prediction) for target, prediction in records)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    longest = current = 0
    for target, prediction in records:
        if target and not prediction:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), "tp": tp, "fp": fp, "fn": fn, "longest_missing_segment": longest, "max_recovery_delay": longest}


def _prediction_paths(binary: np.ndarray, meta) -> list[list[list[float]]]:
    skeleton = _skeletonize(binary)
    return [restore_coordinates(path, meta) for path in _skeleton_cover_paths(skeleton, 8, 256) if len(path) >= 2]


@torch.inference_mode()
def evaluate(weights: str | Path, dataset_dir: str | Path = "datasets/1Ayoyo_consecutive", device_name: str = "cuda", threshold: float | None = None, max_frames: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(dataset_dir).resolve()
    device = resolve_device(device_name)
    model, checkpoint = load_model(weights, device)
    config = checkpoint["model_config"]
    width, height = int(config["input_width"]), int(config["input_height"])
    selected_threshold = float(checkpoint.get("threshold", 0.25) if threshold is None else threshold)
    document = json.loads((root / "consecutive_groups.json").read_text(encoding="utf-8"))
    groups: list[dict[str, Any]] = []
    all_presence: list[tuple[bool, bool]] = []
    pooled_target = pooled_prediction = pooled_target_hits = pooled_prediction_hits = 0
    pooled_chamfer: list[float] = []
    pooled_hd95: list[float] = []
    seen = 0
    for group in document.get("groups") or []:
        if max_frames is not None and seen >= max_frames:
            break
        presence: list[tuple[bool, bool]] = []
        target_samples = prediction_samples = target_hits = prediction_hits = 0
        chamfer: list[float] = []
        hd95: list[float] = []
        for frame in group.get("frames") or []:
            if max_frames is not None and seen >= max_frames:
                break
            image_path = root / str(frame["image"])
            label_path = root / "canonical" / "labels" / Path(str(frame["sample_key"]))
            encoded = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
            if image is None or not label_path.exists():
                continue
            boxed, _, meta = letterbox(image, width, height)
            output = model(normalize_image(boxed).unsqueeze(0).to(device))
            fused = fuse_geometry(output)[0, 0].cpu().numpy()
            tangent = torch.tanh(output[0, 2:]).cpu().numpy()
            prediction_lines = _prediction_paths(decode_centerline(fused, tangent, selected_threshold), meta)
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            target_lines = _annotation_polylines(annotation)
            metrics = centerline_pair_metrics(target_lines, prediction_lines, tolerance_px=(8.0,), spacing_px=2.0)
            tolerance = metrics["tolerances"]["8"]
            target_samples += int(metrics["target_samples"])
            prediction_samples += int(metrics["prediction_samples"])
            target_hits += int(tolerance["target_hits"])
            prediction_hits += int(tolerance["prediction_hits"])
            if metrics.get("chamfer_mean_px") is not None:
                chamfer.append(float(metrics["chamfer_mean_px"]))
                pooled_chamfer.append(float(metrics["chamfer_mean_px"]))
            if metrics.get("hd95_px") is not None:
                hd95.append(float(metrics["hd95_px"]))
                pooled_hd95.append(float(metrics["hd95_px"]))
            presence.append((bool(target_lines), bool(prediction_lines)))
            all_presence.append((bool(target_lines), bool(prediction_lines)))
            seen += 1
        pooled_target += target_samples
        pooled_prediction += prediction_samples
        pooled_target_hits += target_hits
        pooled_prediction_hits += prediction_hits
        precision = prediction_hits / max(1, prediction_samples)
        recall = target_hits / max(1, target_samples)
        groups.append({"group_id": group.get("group_id"), "source_group": group.get("source_group"), "frames": len(presence), "centerline_f1_at_8": 2 * precision * recall / max(1e-9, precision + recall), "precision": precision, "recall": recall, "presence": _presence_stats(presence), "chamfer_mean_px": float(np.mean(chamfer)) if chamfer else None, "hd95_mean_px": float(np.mean(hd95)) if hd95 else None})
    pooled_precision = pooled_prediction_hits / max(1, pooled_prediction)
    pooled_recall = pooled_target_hits / max(1, pooled_target)
    return {
        "schema_version": "yoyo_training_v4_consecutive_eval_v2",
        "task": "mask_centerline_tangent_2theta_fusion",
        "weights": str(Path(weights).resolve()),
        "dataset_dir": str(root),
        "threshold": selected_threshold,
        "frames": seen,
        "fps": seen / max(1e-6, time.perf_counter() - started),
        "pooled": {"metric": "pooled_centerline_f1_at_8_source_px", "precision": pooled_precision, "recall": pooled_recall, "f1": 2 * pooled_precision * pooled_recall / max(1e-9, pooled_precision + pooled_recall), "target_samples": pooled_target, "prediction_samples": pooled_prediction, "presence": _presence_stats(all_presence), "chamfer_mean_px": float(np.mean(pooled_chamfer)) if pooled_chamfer else None, "hd95_mean_px": float(np.mean(pooled_hd95)) if pooled_hd95 else None},
        "weakest_source_group": min(groups, key=lambda row: row["centerline_f1_at_8"]) if groups else None,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate centerline-fusion tracking on consecutive frames.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default="datasets/1Ayoyo_consecutive")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate(args.weights, args.dataset_dir, args.device, args.threshold, args.max_frames)
    output = Path(args.output) if args.output else Path(args.weights).resolve().parent.parent / "consecutive_centerline_fusion_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    document = json.dumps({"pooled": result["pooled"], "weakest_source_group": result["weakest_source_group"], "fps": result["fps"]}, ensure_ascii=False, indent=2)
    encoding = __import__("sys").stdout.encoding or "utf-8"
    print(document.encode(encoding, errors="backslashreplace").decode(encoding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
