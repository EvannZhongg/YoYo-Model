"""Evaluate semantic string checkpoints on reviewed consecutive frames."""

from __future__ import annotations

import argparse
import hashlib
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
    PreparedCalibratedEnsemblePredictor,
    load_checkpoint,
    polyline_probability_support,
    predict_prepared_calibrated_ensemble,
    predict_prepared_probability,
    prepare_letterboxed_input,
    semantic_mask_observation,
)
from video_tracking.sequence_metrics import evaluate_sequence
from video_tracking.string_tracker import (
    _color_line_observation,
    estimate_string,
    update_adaptive_string_domain_gate,
)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip("-_")


def _group_artifact_stem(group_id: str) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8]
    return f"{_safe_name(group_id)}-{digest}"


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
    color_semantic_prefilter: bool = TRACKING_CONFIG.string_color_semantic_prefilter,
    bright_line_augment: bool = TRACKING_CONFIG.string_bright_line_augment,
    bright_line_min_mean: float = TRACKING_CONFIG.string_bright_line_min_mean,
    temporal: bool = False,
    max_propagation_frames: int = TRACKING_CONFIG.string_max_propagation_frames,
    max_forward_backward_error: float = TRACKING_CONFIG.string_flow_fb_max_error,
    unanchored_semantic_grace_frames: int = 12,
    ensemble_weights: Path | None = None,
    ensemble_alpha: float = 0.0,
    ensemble_candidate_threshold: float = 0.5,
    adaptive_weights: Path | None = None,
    adaptive_ensemble_alpha: float = 0.0,
    adaptive_inference_scale: float = TRACKING_CONFIG.string_adaptive_inference_scale,
    adaptive_warmup_frames: int = 0,
    adaptive_max_color_accepts: int = 0,
    adaptive_max_mean_confidence: float = 1.0,
    adaptive_min_mean_distance_ratio: float = 0.0,
    adaptive_single_max_mean_confidence: float = 0.0,
    adaptive_single_threshold: float | None = None,
    adaptive_single_max_components: int = 0,
    cuda_graph: bool = True,
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
    primary_calibration_threshold = float(
        checkpoint.get("threshold", 0.5) if threshold is None else threshold
    )
    selected_threshold = primary_calibration_threshold
    ensemble_model = None
    ensemble_checkpoint = None
    if not 0.0 <= float(ensemble_alpha) <= 1.0:
        raise ValueError("ensemble_alpha must be between 0 and 1")
    if not 0.0 < float(ensemble_candidate_threshold) < 1.0:
        raise ValueError("ensemble_candidate_threshold must be between 0 and 1")
    if ensemble_weights is not None and float(ensemble_alpha) > 0.0:
        ensemble_weights = ensemble_weights.resolve()
        ensemble_model, ensemble_checkpoint = load_checkpoint(ensemble_weights, device)
        if ensemble_checkpoint.get("model_config") != checkpoint.get("model_config"):
            raise ValueError("Semantic ensemble checkpoints use incompatible model configurations")
        selected_threshold = 0.5
    adaptive_model = None
    adaptive_checkpoint = None
    if adaptive_weights is not None:
        if ensemble_model is None:
            raise ValueError("Adaptive semantic evaluation requires an ensemble model")
        if adaptive_warmup_frames < 1 or adaptive_max_color_accepts < 0:
            raise ValueError("Adaptive warmup must be positive and maximum color accepts non-negative")
        if not 0.0 <= float(adaptive_max_mean_confidence) <= 1.0:
            raise ValueError("Adaptive maximum mean confidence must be between 0 and 1")
        if float(adaptive_min_mean_distance_ratio) < 0.0:
            raise ValueError("Adaptive minimum mean distance ratio must be non-negative")
        if not 0.0 <= float(adaptive_single_max_mean_confidence) <= 1.0:
            raise ValueError("Adaptive single-model confidence gate must be between 0 and 1")
        if adaptive_single_threshold is not None and not 0.0 < float(adaptive_single_threshold) < 1.0:
            raise ValueError("Adaptive single-model threshold must be between 0 and 1")
        if int(adaptive_single_max_components) < 0:
            raise ValueError("Adaptive single-model component limit must be non-negative")
        if not 0.0 <= float(adaptive_ensemble_alpha) <= 1.0:
            raise ValueError("adaptive_ensemble_alpha must be between 0 and 1")
        if not 0.5 <= float(adaptive_inference_scale) <= 2.0:
            raise ValueError("adaptive_inference_scale must be between 0.5 and 2.0")
        adaptive_weights = adaptive_weights.resolve()
        adaptive_model, adaptive_checkpoint = load_checkpoint(adaptive_weights, device)
        if adaptive_checkpoint.get("model_config") != checkpoint.get("model_config"):
            raise ValueError("Adaptive semantic checkpoint uses an incompatible model configuration")
    ensemble_predictor = (
        PreparedCalibratedEnsemblePredictor(
            model,
            ensemble_model,
            float(ensemble_alpha),
            float(checkpoint.get("threshold", 0.5)),
            float(ensemble_candidate_threshold),
            cuda_graph,
        )
        if ensemble_model is not None
        else None
    )
    adaptive_ensemble_predictor = (
        PreparedCalibratedEnsemblePredictor(
            adaptive_model,
            ensemble_model,
            float(adaptive_ensemble_alpha),
            float(adaptive_checkpoint.get("threshold", 0.5)),
            float(ensemble_candidate_threshold),
            cuda_graph,
        )
        if adaptive_model is not None
        and adaptive_checkpoint is not None
        and ensemble_model is not None
        else None
    )
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
        adaptive_enabled = False
        adaptive_single_enabled = False
        adaptive_window: list[tuple[bool, float, float]] = []
        adaptive_gate_metrics: dict[str, float | int] = {}
        previous_frame: np.ndarray | None = None
        previous_string: dict[str, Any] | None = None
        last_yoyo_frame: int | None = None
        for frame in group.get("frames") or []:
            relative = Path(str(frame["sample_key"]))
            annotation_path = dataset_dir / "canonical" / "labels" / relative
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_path = dataset_dir / str(frame["image"])
            image = _read_image(image_path)
            active_model = adaptive_model if adaptive_enabled else model
            active_checkpoint = adaptive_checkpoint if adaptive_enabled else checkpoint
            active_primary_threshold = float(active_checkpoint.get("threshold", 0.5))
            active_ensemble_alpha = (
                float(adaptive_ensemble_alpha) if adaptive_enabled else float(ensemble_alpha)
            )
            active_scale = float(adaptive_inference_scale) if adaptive_enabled else 1.0
            active_input_width = max(32, int(round(input_width * active_scale / 16.0)) * 16)
            active_input_height = max(32, int(round(input_height * active_scale / 16.0)) * 16)
            tensor, meta = prepare_letterboxed_input(
                image, active_input_width, active_input_height, device,
            )
            use_ensemble = bool(
                ensemble_model is not None
                and not (adaptive_enabled and adaptive_single_enabled)
            )
            if use_ensemble:
                active_predictor = (
                    adaptive_ensemble_predictor if adaptive_enabled else ensemble_predictor
                )
                probability = (
                    active_predictor.predict(tensor)
                    if active_predictor is not None
                    else predict_prepared_calibrated_ensemble(
                        active_model,
                        ensemble_model,
                        tensor,
                        active_ensemble_alpha,
                        active_primary_threshold,
                        float(ensemble_candidate_threshold),
                    )
                )
            else:
                probability = predict_prepared_probability(active_model, tensor)
            active_selected_threshold = (
                float(
                    active_primary_threshold
                    if adaptive_single_threshold is None
                    else adaptive_single_threshold
                )
                if adaptive_enabled and adaptive_single_enabled
                else selected_threshold
            )
            yoyo = _yoyo(annotation)
            observation = semantic_mask_observation(
                probability,
                meta,
                active_selected_threshold,
                yoyo=yoyo,
                yoyo_division=str(annotation.get("yoyo_division") or "1A"),
                min_component_pixels=8,
                max_components=(
                    int(adaptive_single_max_components)
                    if adaptive_enabled
                    and adaptive_single_enabled
                    and int(adaptive_single_max_components) > 0
                    else 8
                ),
            )
            final_string = observation
            if color_augment and yoyo is not None:
                color = None
                color_support: dict[str, float] = {}
                tried_points: set[tuple[tuple[float, float], ...]] = set()
                had_color_candidate = False
                color_search_cache: dict[str, Any] = {}
                for use_bright_lines in ((False, True) if bright_line_augment else (False,)):
                    candidate = _color_line_observation(
                        image, yoyo, require_yoyo_proximity=False,
                        mark_far_ambiguous=True,
                        reference_points=(observation or {}).get("points"),
                        semantic_probability=probability if color_semantic_prefilter else None,
                        semantic_meta=meta if color_semantic_prefilter else None,
                        include_bright_lines=use_bright_lines,
                        search_cache=color_search_cache,
                    )
                    if candidate is None:
                        continue
                    point_key = tuple(
                        tuple(float(value) for value in point) for point in candidate["points"]
                    )
                    if point_key in tried_points:
                        continue
                    tried_points.add(point_key)
                    had_color_candidate = True
                    candidate_support = polyline_probability_support(
                        probability, meta, candidate["points"], active_selected_threshold,
                    )
                    candidate_min_mean = (
                        color_probability_min_mean
                        if adaptive_enabled
                        else max(
                            float(color_probability_min_mean or 0.0),
                            float(bright_line_min_mean),
                        )
                    ) if use_bright_lines else color_probability_min_mean
                    if bool(
                        color_probability_min_mean is None
                        or (
                            float(candidate_support.get("mean", 0.0))
                            >= float(candidate_min_mean)
                            and float(candidate_support.get("fraction_at_0_10", 0.0))
                            >= color_probability_min_fraction
                        )
                    ):
                        color = candidate
                        color_support = candidate_support
                        break
                if color is None:
                    rejected_color_candidates += int(had_color_candidate)
                elif final_string is None:
                    accepted_color_candidates += 1
                    final_string = dict(color)
                    final_string["color_points"] = color["points"]
                    final_string["color_probability_support"] = color_support
                    if color.get("line_features"):
                        final_string["color_line_features"] = color["line_features"]
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
                    if color.get("line_features"):
                        final_string["color_line_features"] = color["line_features"]
            if adaptive_model is not None and not adaptive_enabled:
                adaptive_window, triggered, adaptive_gate_metrics = (
                    update_adaptive_string_domain_gate(
                        adaptive_window,
                        final_string,
                        image.shape[1],
                        image.shape[0],
                        adaptive_warmup_frames,
                        adaptive_max_color_accepts,
                        adaptive_max_mean_confidence,
                        adaptive_min_mean_distance_ratio,
                    )
                )
                if triggered:
                    adaptive_enabled = True
                    adaptive_single_enabled = bool(
                        float(adaptive_single_max_mean_confidence) > 0.0
                        and float(adaptive_gate_metrics.get("mean_confidence", 1.0))
                        < float(adaptive_single_max_mean_confidence)
                    )
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
                    None,
                    previous_string,
                    yoyo_division=str(annotation.get("yoyo_division") or "1A"),
                    observation=final_string,
                    max_propagation_frames=max(0, int(max_propagation_frames)),
                    max_forward_backward_error=max_forward_backward_error,
                    allow_color_fallback=False,
                    allow_unanchored_semantic=allow_unanchored_semantic,
                    previous_frame=previous_frame,
                )
                previous_string = (
                    final_string
                    if final_string is not None
                    and not final_string.get("spatially_ambiguous")
                    and not final_string.get("hand_anchor_mismatch")
                    else None
                )
                previous_frame = image if previous_string is not None else None
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
        stem = _group_artifact_stem(group_id)
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
            "adaptive_enabled": adaptive_enabled,
            "adaptive_single_enabled": adaptive_single_enabled,
            "adaptive_gate_metrics": adaptive_gate_metrics,
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
        "color_semantic_prefilter": bool(color_semantic_prefilter),
        "bright_line_augment": bool(bright_line_augment),
        "bright_line_min_mean": float(bright_line_min_mean),
        "temporal": bool(temporal),
        "max_propagation_frames": int(max_propagation_frames),
        "max_forward_backward_error": float(max_forward_backward_error),
        "unanchored_semantic_grace_frames": int(unanchored_semantic_grace_frames),
        "ensemble_weights": str(ensemble_weights) if ensemble_weights is not None else "",
        "ensemble_weights_sha256": (
            sha256_file(ensemble_weights) if ensemble_weights is not None else ""
        ),
        "ensemble_alpha": float(ensemble_alpha),
        "ensemble_candidate_threshold": float(ensemble_candidate_threshold),
        "adaptive_weights": str(adaptive_weights) if adaptive_weights is not None else "",
        "adaptive_weights_sha256": sha256_file(adaptive_weights) if adaptive_weights is not None else "",
        "adaptive_ensemble_alpha": float(adaptive_ensemble_alpha),
        "adaptive_inference_scale": float(adaptive_inference_scale),
        "adaptive_warmup_frames": int(adaptive_warmup_frames),
        "adaptive_max_color_accepts": int(adaptive_max_color_accepts),
        "adaptive_max_mean_confidence": float(adaptive_max_mean_confidence),
        "adaptive_min_mean_distance_ratio": float(adaptive_min_mean_distance_ratio),
        "adaptive_single_max_mean_confidence": float(
            adaptive_single_max_mean_confidence
        ),
        "adaptive_single_threshold": (
            float(adaptive_single_threshold)
            if adaptive_single_threshold is not None
            else None
        ),
        "adaptive_single_max_components": int(adaptive_single_max_components),
        "cuda_graph_requested": bool(cuda_graph),
        "cuda_graph_primary": bool(
            ensemble_predictor is not None and ensemble_predictor.uses_cuda_graph
        ),
        "cuda_graph_adaptive": bool(
            adaptive_ensemble_predictor is not None
            and adaptive_ensemble_predictor.uses_cuda_graph
        ),
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
    parser.add_argument(
        "--color-semantic-prefilter",
        action=argparse.BooleanOptionalAction,
        default=TRACKING_CONFIG.string_color_semantic_prefilter,
    )
    parser.add_argument(
        "--bright-line-augment",
        action=argparse.BooleanOptionalAction,
        default=TRACKING_CONFIG.string_bright_line_augment,
    )
    parser.add_argument(
        "--bright-line-min-mean",
        type=float,
        default=TRACKING_CONFIG.string_bright_line_min_mean,
    )
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
    parser.add_argument("--unanchored-semantic-grace-frames", type=int, default=12)
    parser.add_argument("--ensemble-weights", default="")
    parser.add_argument("--ensemble-alpha", type=float, default=0.0)
    parser.add_argument("--ensemble-candidate-threshold", type=float, default=0.5)
    parser.add_argument("--adaptive-weights", default="")
    parser.add_argument("--adaptive-ensemble-alpha", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-inference-scale",
        type=float,
        default=TRACKING_CONFIG.string_adaptive_inference_scale,
    )
    parser.add_argument("--adaptive-warmup-frames", type=int, default=0)
    parser.add_argument("--adaptive-max-color-accepts", type=int, default=0)
    parser.add_argument("--adaptive-max-mean-confidence", type=float, default=1.0)
    parser.add_argument("--adaptive-min-mean-distance-ratio", type=float, default=0.0)
    parser.add_argument("--adaptive-single-max-mean-confidence", type=float, default=0.0)
    parser.add_argument("--adaptive-single-threshold", type=float, default=None)
    parser.add_argument("--adaptive-single-max-components", type=int, default=0)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture fixed-shape CUDA ensemble inference; CPU always uses eager inference.",
    )
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
        color_semantic_prefilter=args.color_semantic_prefilter,
        bright_line_augment=args.bright_line_augment,
        bright_line_min_mean=args.bright_line_min_mean,
        temporal=args.temporal,
        max_propagation_frames=args.max_propagation_frames,
        max_forward_backward_error=args.max_forward_backward_error,
        unanchored_semantic_grace_frames=args.unanchored_semantic_grace_frames,
        ensemble_weights=Path(args.ensemble_weights) if str(args.ensemble_weights).strip() else None,
        ensemble_alpha=args.ensemble_alpha,
        ensemble_candidate_threshold=args.ensemble_candidate_threshold,
        adaptive_weights=Path(args.adaptive_weights) if str(args.adaptive_weights).strip() else None,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
        adaptive_inference_scale=args.adaptive_inference_scale,
        adaptive_warmup_frames=args.adaptive_warmup_frames,
        adaptive_max_color_accepts=args.adaptive_max_color_accepts,
        adaptive_max_mean_confidence=args.adaptive_max_mean_confidence,
        adaptive_min_mean_distance_ratio=args.adaptive_min_mean_distance_ratio,
        adaptive_single_max_mean_confidence=args.adaptive_single_max_mean_confidence,
        adaptive_single_threshold=args.adaptive_single_threshold,
        adaptive_single_max_components=args.adaptive_single_max_components,
        cuda_graph=args.cuda_graph,
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
