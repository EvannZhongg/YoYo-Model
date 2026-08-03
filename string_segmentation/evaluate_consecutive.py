"""Evaluate semantic string checkpoints on reviewed consecutive frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common.files import sha256_file
from config import TRACKING_CONFIG
from string_segmentation.device import resolve_device
from string_segmentation.semantic_model import (
    load_checkpoint,
    polyline_probability_support,
    predict_letterboxed,
    semantic_mask_observation,
)
from video_tracking.sequence_metrics import evaluate_sequence
from video_tracking.string_tracker import _color_line_observation, estimate_string


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip("-_")


def _yoyo(annotation: dict[str, Any]) -> dict[str, Any] | None:
    bbox = annotation.get("yoyo_bbox_pixel") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [float(value) for value in bbox]
    return {
        "bbox": values,
        "center": [(values[0] + values[2]) * 0.5, (values[1] + values[3]) * 0.5],
        "track_id": 1,
    }


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise RuntimeError(f"Could not read consecutive frame: {path}")
    return image


@torch.inference_mode()
def evaluate_consecutive_checkpoint(
    weights: Path,
    dataset_dir: Path,
    output_dir: Path,
    device_name: str = "cuda",
    threshold: float | None = None,
    groups: list[str] | None = None,
    color_augment: bool = False,
    color_probability_min_mean: float | None = None,
    color_probability_min_fraction: float = 0.5,
    temporal: bool = False,
    max_propagation_frames: int = TRACKING_CONFIG.string_max_propagation_frames,
    max_forward_backward_error: float = TRACKING_CONFIG.string_flow_fb_max_error,
    fusion_distance_px: float = TRACKING_CONFIG.string_fusion_distance_px,
    unanchored_semantic_grace_frames: int = 12,
) -> dict[str, Any]:
    weights = weights.resolve()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(weights, device)
    config = checkpoint.get("model_config") or {}
    input_width = int(config.get("input_width", 960))
    input_height = int(config.get("input_height", 544))
    selected_threshold = float(checkpoint.get("threshold", 0.5) if threshold is None else threshold)
    document = json.loads((dataset_dir / "consecutive_groups.json").read_text(encoding="utf-8"))
    selected = set(groups or [])
    results: list[dict[str, Any]] = []

    for group in document.get("groups") or []:
        source_group = str(group.get("source_group") or group.get("group_id") or "")
        group_id = str(group.get("group_id") or source_group)
        if selected and source_group not in selected and group_id not in selected:
            continue
        records = []
        predicted_components = []
        target_components = []
        accepted_color_candidates = 0
        rejected_color_candidates = 0
        previous_gray: np.ndarray | None = None
        previous_string: dict[str, Any] | None = None
        last_yoyo_frame: int | None = None
        for frame in group.get("frames") or []:
            relative = Path(str(frame["sample_key"]))
            annotation_path = dataset_dir / "canonical" / "labels" / relative
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = dataset_dir / str(frame["image"])
            image = _read_image(image_path)
            probability, meta = predict_letterboxed(model, image, input_width, input_height, device)
            yoyo = _yoyo(annotation)
            observation = semantic_mask_observation(
                probability, meta, selected_threshold, yoyo=yoyo,
                yoyo_division=str(annotation.get("yoyo_division") or "1A"),
            )
            final_string = observation
            if color_augment and yoyo is not None:
                color = _color_line_observation(
                    image, yoyo, require_yoyo_proximity=False,
                    mark_far_ambiguous=True,
                    reference_points=(observation or {}).get("points"),
                )
                if color is not None:
                    color_support = polyline_probability_support(
                        probability, meta, color["points"], selected_threshold,
                    )
                    probability_gate_passed = bool(
                        color_probability_min_mean is None
                        or (
                            float(color_support.get("mean", 0.0)) >= color_probability_min_mean
                            and float(color_support.get("fraction_at_0_10", 0.0))
                            >= color_probability_min_fraction
                        )
                    )
                    if not probability_gate_passed:
                        rejected_color_candidates += 1
                    elif final_string is None:
                        accepted_color_candidates += 1
                        final_string = dict(color)
                        final_string["color_points"] = color["points"]
                        final_string["color_probability_support"] = color_support
                    else:
                        accepted_color_candidates += 1
                        final_string = dict(final_string)
                        polylines = list(final_string.get("polylines") or [final_string["points"]])
                        polylines.append(color["points"])
                        final_string.update({
                            "polylines": polylines,
                            "component_count": len(polylines),
                            "method": (
                                "semantic_color_probability_union"
                                if color_probability_min_mean is not None
                                else "semantic_color_union"
                            ),
                            "color_confidence": color.get("confidence"),
                            "color_distance_to_yoyo_px": color.get("distance_to_yoyo_px"),
                            "color_spatially_ambiguous": color.get("spatially_ambiguous"),
                            "color_points": color["points"],
                            "color_probability_support": color_support,
                        })
            if temporal:
                frame_index = int(frame["frame_index"])
                allow_unanchored_semantic = bool(
                    yoyo is None
                    and last_yoyo_frame is not None
                    and frame_index - last_yoyo_frame
                    <= max(0, int(unanchored_semantic_grace_frames))
                )
                final_string = estimate_string(
                    image,
                    yoyo,
                    [],
                    previous_gray,
                    previous_string,
                    yoyo_division=str(annotation.get("yoyo_division") or "1A"),
                    observation=final_string,
                    max_propagation_frames=max(0, int(max_propagation_frames)),
                    max_forward_backward_error=max_forward_backward_error,
                    fusion_distance_px=fusion_distance_px,
                    allow_color_fallback=False,
                    allow_unanchored_semantic=allow_unanchored_semantic,
                )
                previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                previous_string = (
                    final_string
                    if final_string is not None
                    and not final_string.get("spatially_ambiguous")
                    and not final_string.get("hand_anchor_mismatch")
                    else None
                )
                if yoyo is not None:
                    last_yoyo_frame = frame_index
            predicted_components.append(
                len((final_string or {}).get("polylines") or [])
                or int((final_string or {}).get("component_count", 0))
            )
            target_components.append(len(annotation.get("string_polylines_pixel") or []))
            records.append({
                "frame_index": int(frame["frame_index"]),
                "source_group": source_group,
                "yoyo": yoyo,
                "string": final_string,
                "bad_case": [],
            })
        stem = _safe_name(source_group)
        predictions_path = output_dir / f"{stem}.frames.jsonl"
        predictions_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        metrics = evaluate_sequence(
            dataset_dir, predictions_path, group_id=group_id, include_frames=True,
        )
        metrics_path = output_dir / f"{stem}.metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({
            "group_id": group_id,
            "source_group": source_group,
            "frame_count": len(records),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "mean_prediction_components": round(float(np.mean(predicted_components)), 4),
            "mean_target_components": round(float(np.mean(target_components)), 4),
            "zero_prediction_frames": int(sum(value == 0 for value in predicted_components)),
            "accepted_color_candidates": accepted_color_candidates,
            "rejected_color_candidates": rejected_color_candidates,
            "string": metrics["string"],
        })
    if not results:
        raise ValueError("No consecutive groups matched the requested selection")
    summary = {
        "schema_version": "yoyo_semantic_consecutive_evaluation_v1",
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "threshold": selected_threshold,
        "input_size": [input_width, input_height],
        "dataset_dir": str(dataset_dir),
        "color_augment": bool(color_augment),
        "color_probability_min_mean": color_probability_min_mean,
        "color_probability_min_fraction_at_0_10": color_probability_min_fraction,
        "temporal": bool(temporal),
        "max_propagation_frames": int(max_propagation_frames),
        "max_forward_backward_error": float(max_forward_backward_error),
        "fusion_distance_px": float(fusion_distance_px),
        "unanchored_semantic_grace_frames": int(unanchored_semantic_grace_frames),
        "groups": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a semantic checkpoint on consecutive string labels.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", default="datasets/1Ayoyo_consecutive")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--color-augment", action="store_true")
    parser.add_argument("--color-probability-min-mean", type=float, default=None)
    parser.add_argument("--color-probability-min-fraction", type=float, default=0.5)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument(
        "--max-propagation-frames",
        type=int,
        default=TRACKING_CONFIG.string_max_propagation_frames,
    )
    parser.add_argument(
        "--max-forward-backward-error",
        type=float,
        default=TRACKING_CONFIG.string_flow_fb_max_error,
    )
    parser.add_argument(
        "--fusion-distance-px",
        type=float,
        default=TRACKING_CONFIG.string_fusion_distance_px,
    )
    parser.add_argument("--unanchored-semantic-grace-frames", type=int, default=12)
    args = parser.parse_args()
    result = evaluate_consecutive_checkpoint(
        weights=Path(args.weights),
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        device_name=args.device,
        threshold=args.threshold,
        groups=args.group or None,
        color_augment=args.color_augment,
        color_probability_min_mean=args.color_probability_min_mean,
        color_probability_min_fraction=args.color_probability_min_fraction,
        temporal=args.temporal,
        max_propagation_frames=args.max_propagation_frames,
        max_forward_backward_error=args.max_forward_backward_error,
        fusion_distance_px=args.fusion_distance_px,
        unanchored_semantic_grace_frames=args.unanchored_semantic_grace_frames,
    )
    compact = [{
        "source_group": item["source_group"],
        "frame_count": item["frame_count"],
        "f1_at_8": item["string"]["centerline"]["tolerances"]["8"]["f1"],
        "recall_at_8": item["string"]["centerline"]["tolerances"]["8"]["recall"],
        "chamfer_mean_px": item["string"]["centerline"]["chamfer_mean_px"],
        "mean_prediction_components": item["mean_prediction_components"],
    } for item in result["groups"]]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
