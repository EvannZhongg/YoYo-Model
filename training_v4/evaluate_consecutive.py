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
from training_v4.evaluate import decode_centerline, load_model
from video_tracking.sequence_metrics import _annotation_polylines, centerline_pair_metrics


def _presence_stats(records: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(target and predicted for target, predicted in records)
    fp = sum((not target) and predicted for target, predicted in records)
    fn = sum(target and (not predicted) for target, predicted in records)
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    longest = current = 0
    for target, predicted in records:
        if target and not predicted:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), "tp": tp, "fp": fp, "fn": fn, "longest_missing_segment": longest, "max_recovery_delay": longest}


@torch.inference_mode()
def evaluate(weights: str | Path, dataset_dir: str | Path, device_name: str = "cuda", threshold: float = 0.5, max_frames: int | None = None) -> dict[str, Any]:
    started = time.perf_counter(); root = Path(dataset_dir).resolve(); device = resolve_device(device_name); model, checkpoint = load_model(Path(weights).resolve(), device); config = checkpoint["model_config"]; width, height = int(config["input_width"]), int(config["input_height"])
    document = json.loads((root / "consecutive_groups.json").read_text(encoding="utf-8")); groups = []; total_target = total_prediction = total_target_hits = total_prediction_hits = 0; total_chamfer = []; total_hd95 = []; all_presence = []; seen = 0
    for group in document.get("groups") or []:
        group_target = group_prediction = group_target_hits = group_prediction_hits = 0; group_chamfer = []; group_hd95 = []; presence = []
        for frame in group.get("frames") or []:
            if max_frames is not None and seen >= int(max_frames): break
            image_path = root / str(frame["image"]); label_path = root / "canonical" / "labels" / Path(str(frame["sample_key"])); encoded = np.fromfile(image_path, dtype=np.uint8); image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or not label_path.exists(): continue
            boxed, _, meta = letterbox(image, width, height); output = model(normalize_image(boxed).unsqueeze(0).to(device)); heat = torch.sigmoid(output[:, 0]).cpu().numpy()[0]; direction = torch.tanh(output[:, 1:]).cpu().numpy()[0]; predicted = _skeletonize(decode_centerline(heat, direction, threshold)); prediction_lines = [restore_coordinates(path, meta) for path in _skeleton_cover_paths(predicted, 8, 256)]; target_lines = _annotation_polylines(json.loads(label_path.read_text(encoding="utf-8"))); metrics = centerline_pair_metrics(target_lines, prediction_lines, tolerance_px=(8.0,), spacing_px=2.0); tolerance = metrics["tolerances"]["8"]
            group_target += metrics["target_samples"]; group_prediction += metrics["prediction_samples"]; group_target_hits += tolerance["target_hits"]; group_prediction_hits += tolerance["prediction_hits"]; presence.append((bool(target_lines), bool(prediction_lines))); all_presence.append((bool(target_lines), bool(prediction_lines))); seen += 1
            if metrics.get("chamfer_mean_px") is not None: group_chamfer.append(float(metrics["chamfer_mean_px"])); total_chamfer.append(float(metrics["chamfer_mean_px"]))
            if metrics.get("hd95_px") is not None: group_hd95.append(float(metrics["hd95_px"])); total_hd95.append(float(metrics["hd95_px"]))
        total_target += group_target; total_prediction += group_prediction; total_target_hits += group_target_hits; total_prediction_hits += group_prediction_hits; precision = group_prediction_hits / max(1, group_prediction); recall = group_target_hits / max(1, group_target)
        groups.append({"group_id": group.get("group_id"), "source_group": group.get("source_group"), "frames": len(presence), "centerline_f1_at_8": 2 * precision * recall / max(1e-9, precision + recall), "precision": precision, "recall": recall, "presence": _presence_stats(presence), "chamfer_mean_px": float(np.mean(group_chamfer)) if group_chamfer else None, "hd95_mean_px": float(np.mean(group_hd95)) if group_hd95 else None})
        if max_frames is not None and seen >= int(max_frames): break
    precision = total_prediction_hits / max(1, total_prediction); recall = total_target_hits / max(1, total_target)
    return {"schema_version": "yoyo_training_v4_consecutive_eval_v1", "task": "centerline_heatmap_direction", "weights": str(Path(weights).resolve()), "dataset_dir": str(root), "threshold": threshold, "frames": seen, "fps": seen / max(1e-6, time.perf_counter() - started), "pooled": {"metric": "pooled_centerline_f1_at_8_source_px", "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-9, precision + recall), "target_samples": total_target, "prediction_samples": total_prediction, "presence": _presence_stats(all_presence), "chamfer_mean_px": float(np.mean(total_chamfer)) if total_chamfer else None, "hd95_mean_px": float(np.mean(total_hd95)) if total_hd95 else None}, "weakest_source_group": min(groups, key=lambda row: row["centerline_f1_at_8"]) if groups else None, "groups": groups}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--weights", required=True); parser.add_argument("--dataset-dir", default="datasets/1Ayoyo_consecutive"); parser.add_argument("--device", default="cuda"); parser.add_argument("--threshold", type=float, default=0.5); parser.add_argument("--max-frames", type=int, default=None); parser.add_argument("--output", default=""); args = parser.parse_args(); result = evaluate(args.weights, args.dataset_dir, args.device, args.threshold, args.max_frames); output = Path(args.output) if args.output else Path(args.weights).parent.parent / "consecutive_centerline_v4_metrics.json"; output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); document = json.dumps(result["pooled"], ensure_ascii=False, indent=2); encoding = __import__("sys").stdout.encoding or "utf-8"; print(document.encode(encoding, errors="backslashreplace").decode(encoding)); return 0


if __name__ == "__main__": raise SystemExit(main())
