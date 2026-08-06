"""YOYO video tracking and frame-level metadata export.

The detector is intentionally kept separate from the annotation protocol.  A
tracking run always produces machine-readable per-frame records, even when a
pose model is unavailable.  This makes failed/ambiguous frames reviewable and
allows a later string segmentation model to consume the same data.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.files import sha256_file
from config import BASE_DIR, TRACKING_CONFIG
from string_segmentation.semantic_model import (
    PreparedCalibratedEnsemblePredictor,
    is_semantic_checkpoint,
    load_checkpoint as load_semantic_checkpoint,
    polyline_probability_support,
    predict_prepared_calibrated_ensemble,
    predict_prepared_probability,
    prepare_letterboxed_input,
    semantic_mask_observation,
)
from video_tracking.review_sheet import make_tracking_review_sheet
from video_tracking.orientation import (
    OrientationTemporalFilter,
    carry_orientation,
    load_orientation_model,
    orientation_observation_is_unstable,
    predict_orientation,
)
from video_tracking.rtmpose_backend import (
    COCO_BODY_KEYPOINT_COUNT,
    DEFAULT_DETECTOR_PATH,
    DEFAULT_POSE_PATH,
    RTMPoseWholebody,
    hand_landmarks,
)
from video_tracking.string_tracker import (
    _color_line_observation,
    estimate_string,
    update_adaptive_string_domain_gate,
)


LOG_FILE = BASE_DIR / "track_video.log"
YOYO_TEMPORAL_TRUST_CONFIDENCE = 0.50
POSE_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (1, 3), (0, 2), (2, 4),
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rtmpose_model(
    weights_path: str | Path | None,
    device: str = "",
    detector_path: str | Path | None = None,
):
    requested = Path(weights_path or DEFAULT_POSE_PATH)
    detector = Path(detector_path or DEFAULT_DETECTOR_PATH)
    try:
        model = RTMPoseWholebody(requested, detector, device)
    except Exception as exc:
        logger.warning(
            "RTMPose model unavailable (%s): %s. Use the project RTMPose download command.",
            requested,
            exc,
        )
        return None, str(exc)
    return model, str(requested)


def _load_string_model(
    weights_path: str | Path | None,
    enabled: bool,
    device: str = "",
    ensemble_weights_path: str | Path | None = None,
    ensemble_alpha: float = 0.0,
    ensemble_candidate_threshold: float = 0.5,
    adaptive_weights_path: str | Path | None = None,
    adaptive_ensemble_alpha: float = 0.0,
):
    if not enabled:
        return None, "disabled"
    path = Path(weights_path) if weights_path else TRACKING_CONFIG.string_weights_path
    if not path.exists():
        logger.warning("String segmentation model not found; using review-only visual fallback: %s", path)
        return None, f"missing: {path}"
    if is_semantic_checkpoint(path):
        try:
            import torch

            requested_device = str(device).strip()
            if requested_device.isdigit():
                requested_device = f"cuda:{requested_device}"
            semantic_device = torch.device(
                requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model, checkpoint = load_semantic_checkpoint(path, semantic_device)
            bundle = {
                "kind": "semantic",
                "model": model,
                "checkpoint": checkpoint,
                "device": semantic_device,
                "path": str(path),
            }
            secondary_path = Path(ensemble_weights_path) if ensemble_weights_path else None
            if float(ensemble_alpha) > 0.0 and secondary_path is not None:
                if not secondary_path.is_file():
                    logger.warning("Semantic ensemble weights not found; using primary only: %s", secondary_path)
                elif not is_semantic_checkpoint(secondary_path):
                    logger.warning("Semantic ensemble weights are not a semantic checkpoint: %s", secondary_path)
                else:
                    secondary_model, secondary_checkpoint = load_semantic_checkpoint(
                        secondary_path, semantic_device,
                    )
                    if secondary_checkpoint.get("model_config") != checkpoint.get("model_config"):
                        logger.warning(
                            "Semantic ensemble model config differs from primary; using primary only: %s",
                            secondary_path,
                        )
                    else:
                        bundle.update({
                            "kind": "semantic_ensemble",
                            "ensemble_model": secondary_model,
                            "ensemble_checkpoint": secondary_checkpoint,
                            "ensemble_path": str(secondary_path),
                            "ensemble_alpha": float(ensemble_alpha),
                            "ensemble_candidate_threshold": float(ensemble_candidate_threshold),
                        })
            adaptive_path = Path(adaptive_weights_path) if adaptive_weights_path else None
            if adaptive_path is not None:
                if bundle["kind"] != "semantic_ensemble":
                    logger.warning("Adaptive semantic weights require a compatible ensemble; ignoring: %s", adaptive_path)
                elif not adaptive_path.is_file() or not is_semantic_checkpoint(adaptive_path):
                    logger.warning("Adaptive semantic weights unavailable; ignoring: %s", adaptive_path)
                else:
                    adaptive_model, adaptive_checkpoint = load_semantic_checkpoint(adaptive_path, semantic_device)
                    if adaptive_checkpoint.get("model_config") != checkpoint.get("model_config"):
                        logger.warning("Adaptive semantic model config differs from primary; ignoring: %s", adaptive_path)
                    else:
                        bundle.update({
                            "kind": "semantic_adaptive_ensemble",
                            "adaptive_model": adaptive_model,
                            "adaptive_checkpoint": adaptive_checkpoint,
                            "adaptive_path": str(adaptive_path),
                            "adaptive_ensemble_alpha": float(adaptive_ensemble_alpha),
                            "adaptive_enabled": False,
                        })
            if bundle["kind"] in {"semantic_ensemble", "semantic_adaptive_ensemble"}:
                bundle["ensemble_predictor"] = PreparedCalibratedEnsemblePredictor(
                    bundle["model"],
                    bundle["ensemble_model"],
                    bundle["ensemble_alpha"],
                    float(bundle["checkpoint"].get("threshold", 0.5)),
                    bundle["ensemble_candidate_threshold"],
                )
                if bundle["kind"] == "semantic_adaptive_ensemble":
                    bundle["adaptive_ensemble_predictor"] = PreparedCalibratedEnsemblePredictor(
                        bundle["adaptive_model"],
                        bundle["ensemble_model"],
                        bundle["adaptive_ensemble_alpha"],
                        float(bundle["adaptive_checkpoint"].get("threshold", 0.5)),
                        bundle["ensemble_candidate_threshold"],
                    )
            status = (
                f"semantic_adaptive_ensemble:{path}+{bundle['ensemble_path']}+{bundle['adaptive_path']}"
                if bundle["kind"] == "semantic_adaptive_ensemble"
                else f"semantic_ensemble:{path}+{bundle['ensemble_path']}"
                if bundle["kind"] == "semantic_ensemble"
                else f"semantic:{path}"
            )
            return bundle, status
        except Exception as exc:
            logger.warning("Semantic string weights unavailable (%s): %s", path, exc)
            return None, f"semantic_error: {exc}"
    from ultralytics import YOLO

    try:
        model = YOLO(str(path))
        names = {int(key): str(value).lower() for key, value in dict(getattr(model, "names", {}) or {}).items()}
        if names and not any(name in {"string", "yoyo_string", "yoyo-string", "string_segment"} for name in names.values()):
            logger.warning("String weights do not expose a string class; ignoring checkpoint: %s (%s)", path, names)
            return None, f"incompatible_classes: {names}"
        return model, str(path)
    except Exception as exc:
        logger.warning("String segmentation model unavailable (%s): %s", path, exc)
        return None, str(exc)


def _polygon_centerline(polygon: np.ndarray) -> list[list[float]]:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return []
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    endpoints = [points[int(np.argmin(projection))].tolist(), points[int(np.argmax(projection))].tolist()]
    return [[float(value) for value in point] for point in endpoints]


def _semantic_inference_parameters(
    model_config: dict[str, Any],
    scale_value: float,
) -> tuple[int, int, float]:
    scale = float(scale_value)
    if not 0.5 <= scale <= 2.0:
        raise ValueError("semantic_inference_scale must be between 0.5 and 2.0")
    base_width = int(model_config["input_width"])
    base_height = int(model_config["input_height"])
    input_width = max(32, int(round(base_width * scale / 16.0)) * 16)
    input_height = max(32, int(round(base_height * scale / 16.0)) * 16)
    component_area_scale = (input_width * input_height) / max(1.0, float(base_width * base_height))
    return input_width, input_height, component_area_scale


def _inference_interval_frames(video_fps: float, target_fps: float) -> int:
    """Return an adaptive frame interval; zero target means every frame."""
    if float(target_fps) <= 0.0 or float(video_fps) <= 0.0:
        return 1
    return max(1, int(round(float(video_fps) / float(target_fps))))


def _should_reacquire_string(
    scheduled_inference: bool,
    model_loaded: bool,
    yoyo: dict[str, Any] | None,
    previous_string: dict[str, Any] | None,
    current_string: dict[str, Any] | None,
) -> bool:
    """Re-run the model when an anchored track cannot cross a cadence gap."""
    return bool(
        not scheduled_inference
        and model_loaded
        and yoyo is not None
        and current_string is None
    )


def _can_seed_previous_string(string: dict[str, Any] | None) -> bool:
    return bool(
        string is not None
        and not string.get("spatially_ambiguous")
        and not string.get("hand_anchor_mismatch")
    )


def _augment_semantic_color_observation(
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    probability: np.ndarray,
    meta: Any,
    threshold: float,
    min_mean: float,
    min_fraction_at_0_10: float,
    semantic_prefilter: bool = False,
) -> dict[str, Any] | None:
    """Add a color line only when the semantic map independently supports it."""
    if observation is None or yoyo is None:
        return observation
    color = _color_line_observation(
        frame,
        yoyo,
        require_yoyo_proximity=False,
        mark_far_ambiguous=True,
        reference_points=observation.get("points"),
        semantic_probability=probability if semantic_prefilter else None,
        semantic_meta=meta if semantic_prefilter else None,
    )
    if color is None:
        return observation
    support = polyline_probability_support(
        probability,
        meta,
        color["points"],
        threshold,
    )
    if (
        float(support.get("mean", 0.0)) < float(min_mean)
        or float(support.get("fraction_at_0_10", 0.0)) < float(min_fraction_at_0_10)
    ):
        return observation
    result = dict(observation)
    polylines = list(result.get("polylines") or [result["points"]])
    polylines.append(color["points"])
    result.update(
        {
            "polylines": polylines,
            "component_count": len(polylines),
            "method": "semantic_color_probability_union",
            "needs_review": True,
            "color_points": color["points"],
            "color_confidence": color.get("confidence"),
            "color_distance_to_yoyo_px": color.get("distance_to_yoyo_px"),
            "color_spatially_ambiguous": color.get("spatially_ambiguous"),
            "color_probability_support": support,
        }
    )
    return result


def _predict_string_model(
    model,
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    confidence: float,
    imgsz: int,
    device: str,
    yoyo_division: str,
    semantic_inference_scale: float = 1.0,
    wrists: list[dict[str, Any]] | None = None,
    color_probability_augment: bool = False,
    color_probability_min_mean: float = 0.40,
    color_probability_min_fraction: float = 0.50,
    color_semantic_prefilter: bool = False,
) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, dict) and model.get("kind") in {
        "semantic", "semantic_ensemble", "semantic_adaptive_ensemble",
    }:
        adaptive_enabled = bool(model.get("adaptive_enabled"))
        checkpoint = model["adaptive_checkpoint"] if adaptive_enabled else model["checkpoint"]
        primary_model = model["adaptive_model"] if adaptive_enabled else model["model"]
        model_device = model["device"]
        model_config = checkpoint["model_config"]
        scale = float(semantic_inference_scale)
        input_width, input_height, component_area_scale = _semantic_inference_parameters(model_config, scale)
        tensor, meta = prepare_letterboxed_input(
            frame,
            input_width,
            input_height,
            model_device,
        )
        primary_threshold = float(checkpoint.get("threshold", 0.5))
        threshold = max(primary_threshold, float(confidence))
        ensemble_metadata = None
        if model.get("kind") in {"semantic_ensemble", "semantic_adaptive_ensemble"}:
            active_alpha = float(
                model["adaptive_ensemble_alpha"] if adaptive_enabled else model["ensemble_alpha"]
            )
            secondary_threshold = float(model["ensemble_candidate_threshold"])
            predictor = model.get(
                "adaptive_ensemble_predictor" if adaptive_enabled else "ensemble_predictor"
            )
            probability = (
                predictor.predict(tensor)
                if predictor is not None
                else predict_prepared_calibrated_ensemble(
                    primary_model,
                    model["ensemble_model"],
                    tensor,
                    active_alpha,
                    primary_threshold,
                    secondary_threshold,
                )
            )
            threshold = max(0.5, float(confidence))
            ensemble_metadata = {
                "alpha": round(active_alpha, 4),
                "primary_threshold": round(primary_threshold, 4),
                "secondary_threshold": round(secondary_threshold, 4),
                "fused_threshold": round(threshold, 4),
                "adaptive_primary": adaptive_enabled,
            }
        else:
            probability = predict_prepared_probability(primary_model, tensor)
        observation = semantic_mask_observation(
            probability,
            meta,
            threshold=threshold,
            yoyo=yoyo,
            min_component_pixels=max(1, int(round(8 * component_area_scale))),
            hand_points=[
                [float(wrist["x"]), float(wrist["y"])]
                for wrist in (wrists or [])
                if "x" in wrist and "y" in wrist
            ],
        )
        if color_probability_augment:
            observation = _augment_semantic_color_observation(
                frame,
                yoyo,
                observation,
                probability,
                meta,
                threshold,
                color_probability_min_mean,
                color_probability_min_fraction,
                color_semantic_prefilter,
            )
        if observation is not None:
            if ensemble_metadata is not None:
                observation["semantic_probability_ensemble"] = ensemble_metadata
            observation["inference_scale"] = round(scale, 4)
            observation["inference_size"] = [input_width, input_height]
        return observation
    kwargs: dict[str, Any] = {"source": frame, "conf": confidence, "imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    polygons = list(getattr(masks, "xy", []) or []) if masks is not None else []
    if not polygons:
        return None
    confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else [1.0] * len(polygons)
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist() if boxes is not None and boxes.cls is not None else [0] * len(polygons)
    names = {int(key): str(value).lower() for key, value in dict(getattr(model, "names", {}) or {}).items()}
    allowed_ids = {key for key, value in names.items() if value in {"string", "yoyo_string", "yoyo-string", "string_segment"}}
    if allowed_ids:
        filtered = [(polygon, score, class_id) for polygon, score, class_id in zip(polygons, confidences, class_ids) if class_id in allowed_ids]
        polygons = [item[0] for item in filtered]
        confidences = [item[1] for item in filtered]
    elif names:
        return None
    candidates = []
    for polygon, score in zip(polygons, confidences):
        array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(array) < 3:
            continue
        distance = 0.0
        if yoyo is not None:
            distance = float(np.min(np.linalg.norm(array - np.asarray(yoyo["center"], dtype=np.float32), axis=1)))
        candidates.append((distance, -float(score), array))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    _, negative_score, polygon = candidates[0]
    selected = [item[2] for item in candidates]
    return {
        "points": _polygon_centerline(polygon),
        "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
        "polylines": [_polygon_centerline(item) for item in selected],
        "polygons": [
            [[round(float(x), 2), round(float(y), 2)] for x, y in item]
            for item in selected
        ],
        "confidence": round(-negative_score, 4),
        "method": "yolo_segmentation",
        "needs_review": False,
    }


def _select_pose_person(
    points: np.ndarray,
    confidence: np.ndarray,
    boxes: np.ndarray,
    yoyo: dict[str, Any] | None,
    width: int,
    height: int,
    previous_person_bbox: list[float] | None = None,
) -> tuple[int, dict[str, Any]] | None:
    if len(points) == 0:
        return None
    yoyo_center = np.asarray((yoyo or {}).get("center", []), dtype=np.float32)
    has_yoyo = yoyo_center.shape == (2,)
    diagonal = max(1.0, float(np.hypot(width, height)))
    previous_box = np.asarray(previous_person_bbox or [], dtype=np.float32)
    has_previous = bool(
        previous_box.shape == (4,)
        and previous_box[2] > previous_box[0]
        and previous_box[3] > previous_box[1]
    )
    candidates = []
    for index, person_points in enumerate(points):
        person_confidence = confidence[index] if index < len(confidence) else np.ones(len(person_points), dtype=np.float32)
        visible = person_confidence >= 0.20
        visible_wrists = [wrist for wrist in (9, 10) if wrist < len(person_points) and visible[wrist]]
        wrist_distance = (
            min(float(np.linalg.norm(person_points[wrist] - yoyo_center)) for wrist in visible_wrists)
            if has_yoyo and visible_wrists
            else diagonal
        )
        body_confidence = person_confidence[:COCO_BODY_KEYPOINT_COUNT]
        body_visible = body_confidence >= 0.20
        mean_visible_confidence = float(body_confidence[body_visible].mean()) if np.any(body_visible) else 0.0
        box = boxes[index] if index < len(boxes) else np.zeros(4, dtype=np.float32)
        box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        temporal_iou = _bbox_iou(
            [float(value) for value in box],
            [float(value) for value in previous_box],
        ) if has_previous else 0.0
        temporal_center_distance = (
            float(np.linalg.norm(
                np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
                - np.asarray([(previous_box[0] + previous_box[2]) / 2.0, (previous_box[1] + previous_box[3]) / 2.0])
            ))
            if has_previous else diagonal
        )
        candidates.append({
            "index": index,
            "wrist_distance": wrist_distance,
            "visible_count": int(body_visible.sum()),
            "visible_wrist_count": len(visible_wrists),
            "box": box,
            "box_area": box_area,
            "box_height": max(0.0, float(box[3] - box[1])),
            "mean_body_confidence": mean_visible_confidence,
            "temporal_iou": temporal_iou,
            "temporal_center_distance": temporal_center_distance,
        })
    max_area = max((item["box_area"] for item in candidates), default=1.0)
    max_height = max((item["box_height"] for item in candidates), default=1.0)
    for item in candidates:
        proximity = max(0.0, 1.0 - float(item["wrist_distance"]) / diagonal) if has_yoyo else 0.0
        item["cold_start_score"] = (
            0.45 * item["box_area"] / max(1.0, max_area)
            + 0.25 * min(1.0, item["mean_body_confidence"])
            + 0.20 * item["box_height"] / max(1.0, max_height)
            + 0.10 * proximity
        )
        item["temporal_score"] = (
            item["temporal_iou"],
            -item["temporal_center_distance"] / diagonal,
            item["cold_start_score"],
        )
    temporal_candidates = [
        item for item in candidates
        if item["temporal_iou"] >= 0.05
        or item["temporal_center_distance"] / diagonal <= 0.12
    ] if has_previous else []
    temporal_reference_used = bool(temporal_candidates)
    candidate_pool = temporal_candidates if temporal_reference_used else candidates
    score_field = "temporal_score" if temporal_reference_used else "cold_start_score"
    chosen = max(candidate_pool, key=lambda item: item[score_field])
    selected = int(chosen["index"])
    wrist_distance = float(chosen["wrist_distance"])
    visible_count = int(chosen["visible_count"])
    visible_wrist_count = int(chosen["visible_wrist_count"])
    selected_box = chosen["box"]
    review_reasons: list[str] = []
    if len(points) > 1 and not temporal_reference_used:
        review_reasons.append("multiple_people_cold_start")
        if not has_yoyo:
            review_reasons.append("yoyo_absent_without_temporal_reference")
    if has_previous and not temporal_reference_used:
        review_reasons.append("temporal_reference_rejected")
    if temporal_reference_used and float(chosen["temporal_iou"]) < 0.10:
        review_reasons.append("low_temporal_iou")
    return selected, {
        "selection_method": (
            "temporal_continuity_then_person_extent_pose_quality_yoyo_proximity"
            if temporal_reference_used
            else "person_extent_pose_quality_then_yoyo_proximity"
        ),
        "person_index": int(selected),
        "person_count": int(len(points)),
        "bbox": [round(float(value), 2) for value in selected_box],
        "visible_keypoint_count": int(visible_count),
        "visible_wrist_count": int(visible_wrist_count),
        "nearest_wrist_to_yoyo_px": round(float(wrist_distance), 2) if has_yoyo and visible_wrist_count else None,
        "temporal_reference_available": bool(has_previous),
        "temporal_reference_used": temporal_reference_used,
        "temporal_bbox_iou": round(float(chosen["temporal_iou"]), 4) if has_previous else None,
        "temporal_center_distance_px": (
            round(float(chosen["temporal_center_distance"]), 2) if has_previous else None
        ),
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def _predict_pose(
    model,
    frame: np.ndarray,
    yoyo: dict[str, Any] | None = None,
    imgsz: int = 640,
    device: str = "",
    previous_person_bbox: list[float] | None = None,
    temporal_reference_age_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if model is None:
        return [], [], {"status": "disabled_or_unavailable"}
    try:
        result = model.predict(frame)
        all_points = result.keypoints
        all_confidence = result.scores
        boxes = result.boxes
        if not len(all_points):
            return [], [], {"status": "no_person"}
        selection = _select_pose_person(
            all_points, all_confidence, boxes, yoyo,
            frame.shape[1], frame.shape[0], previous_person_bbox,
        )
        if selection is None:
            return [], [], {"status": "no_person"}
        selected, metadata = selection
        points = all_points[selected]
        confidence = all_confidence[selected]
        # COCO WholeBody retains body wrist indexes 9/10 and adds 21 detailed
        # landmarks per hand at indexes 91:112 and 112:133.
        wrists: list[dict[str, Any]] = []
        for index, name, side in ((9, "left_wrist", "left"), (10, "right_wrist", "right")):
            if index >= len(points):
                continue
            conf = float(confidence[index]) if index < len(confidence) else 1.0
            if conf < 0.20:
                continue
            wrists.append({
                "name": name,
                "x": float(points[index][0]),
                "y": float(points[index][1]),
                "confidence": conf,
                "landmarks": hand_landmarks(points, confidence, side),
                "source": model.backend_name,
            })
        pose = []
        for index, point in enumerate(points[:COCO_BODY_KEYPOINT_COUNT]):
            conf = float(confidence[index]) if index < len(confidence) else 1.0
            pose.append({"index": index, "x": float(point[0]), "y": float(point[1]), "confidence": conf})
        metadata.update({
            "status": "ok",
            # RTMLib's detector returns coordinates only; do not fabricate a
            # zero score that downstream consumers could treat as evidence.
            "box_confidence": None,
            "box_confidence_available": False,
            "backend": model.backend_name,
            "keypoint_schema": model.keypoint_schema,
            "wholebody_keypoint_count": int(len(points)),
            "temporal_reference_age_frames": (
                int(temporal_reference_age_frames)
                if metadata.get("temporal_reference_available")
                and temporal_reference_age_frames is not None
                else None
            ),
        })
        return wrists, pose, metadata
    except Exception as exc:
        logger.debug("Pose inference failed: %s", exc)
        return [], [], {"status": "error", "error_type": type(exc).__name__}


def _extract_detections(result, class_names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    detections = []
    for bbox, confidence, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
            }
        )
    return detections


def _draw_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    wrists: list[dict[str, Any]],
    string: dict[str, Any] | None,
    traces: dict[int, list[tuple[int, int]]],
    class_colors: dict[int, tuple[int, int, int]],
    trace_length: int,
    line_thickness: int,
    text_scale: float,
    output_size: tuple[int, int] | None = None,
    pose: list[dict[str, Any]] | None = None,
    trick_orientation: dict[str, Any] | None = None,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    output_width, output_height = output_size or (source_width, source_height)
    scale_x = float(output_width) / max(1.0, float(source_width))
    scale_y = float(output_height) / max(1.0, float(source_height))
    canvas = (
        frame.copy()
        if (output_width, output_height) == (source_width, source_height)
        else cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
    )

    def scaled_points(values: Any) -> np.ndarray:
        points = np.asarray(values, dtype=np.float32).reshape(-1, 2).copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        return points.round().astype(np.int32)

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in scaled_points(np.asarray(detection["bbox"]).reshape(2, 2)).reshape(-1)]
        class_id = detection["class_id"]
        color = class_colors.setdefault(class_id, ((37 * (class_id + 1)) % 255, 180, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if detection.get("track_id") is not None:
            label = f"#{detection['track_id']} {label}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 2, cv2.LINE_AA)
        track_id = detection.get("track_id")
        if track_id is not None:
            trace = traces.setdefault(int(track_id), [])
            trace.append((int(detection["center"][0]), int(detection["center"][1])))
            del trace[:-trace_length]
            if len(trace) > 1:
                cv2.polylines(canvas, [scaled_points(trace)], False, color, line_thickness)
    pose_by_index = {int(point.get("index", -1)): point for point in (pose or [])}
    for start_index, end_index in POSE_EDGES:
        start, end = pose_by_index.get(start_index), pose_by_index.get(end_index)
        if not start or not end or float(start.get("confidence", 0.0)) < 0.20 or float(end.get("confidence", 0.0)) < 0.20:
            continue
        edge = scaled_points([[start["x"], start["y"]], [end["x"], end["y"]]])
        cv2.line(canvas, tuple(edge[0]), tuple(edge[1]), (70, 190, 90), 1, cv2.LINE_AA)
    for point_data in pose_by_index.values():
        if float(point_data.get("confidence", 0.0)) < 0.20:
            continue
        point = tuple(scaled_points([[point_data["x"], point_data["y"]]])[0])
        cv2.circle(canvas, point, 2, (70, 190, 90), -1, cv2.LINE_AA)
    for wrist in wrists:
        point = tuple(scaled_points([[wrist["x"], wrist["y"]]])[0])
        cv2.circle(canvas, point, 6, (0, 220, 255), -1)
        cv2.putText(canvas, wrist["name"], (point[0] + 7, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    if string is not None:
        polygons = string.get("polygons") or ([string["polygon"]] if string.get("polygon") else [])
        if polygons:
            mask_layer = canvas.copy()
            polygon_arrays = [scaled_points(polygon) for polygon in polygons]
            cv2.fillPoly(mask_layer, polygon_arrays, (255, 80, 30))
            canvas = cv2.addWeighted(mask_layer, 0.28, canvas, 0.72, 0)
            cv2.polylines(canvas, polygon_arrays, True, (255, 80, 30), 1)
        polylines = string.get("polylines") or ([string["points"]] if string.get("points") else [])
        point_arrays = [scaled_points(points) for points in polylines if len(points) >= 2]
        for points in point_arrays:
            cv2.polylines(canvas, [points], False, (255, 80, 30), 2)
        label = f"string {string.get('method', 'estimate')} {float(string.get('confidence', 0.0)):.2f} / review"
        label_point = tuple(point_arrays[0][0]) if point_arrays else (12, 42)
        cv2.putText(canvas, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 30), 1, cv2.LINE_AA)
    if trick_orientation is not None:
        orientation_label = (
            f"orientation {trick_orientation.get('label', 'unknown')} "
            f"{float(trick_orientation.get('confidence', 0.0)):.2f}"
        )
        cv2.putText(
            canvas,
            orientation_label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 230, 240),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _visualization_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Return even preview dimensions without changing source-space metadata."""
    target_width = int(max_width)
    if target_width <= 0 or width <= target_width:
        return int(width), int(height)
    output_width = max(2, target_width - target_width % 2)
    output_height = max(2, int(round(float(height) * output_width / max(1, width))))
    output_height -= output_height % 2
    return output_width, max(2, output_height)


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _assign_tracker_ids(detections: list[dict[str, Any]], tracked: Any) -> None:
    """Match ByteTrack outputs back to detector rows without assuming order."""
    tracked_ids = getattr(tracked, "tracker_id", None)
    tracked_boxes = getattr(tracked, "xyxy", None)
    if tracked_ids is None or tracked_boxes is None:
        return
    tracked_classes = getattr(tracked, "class_id", None)
    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for tracked_index, tracked_box in enumerate(tracked_boxes):
            if (
                tracked_classes is not None
                and tracked_index < len(tracked_classes)
                and int(tracked_classes[tracked_index]) != int(detection["class_id"])
            ):
                continue
            overlap = _bbox_iou(detection["bbox"], [float(value) for value in tracked_box])
            if overlap >= 0.1:
                candidates.append((overlap, detection_index, tracked_index))
    used_detections: set[int] = set()
    used_tracks: set[int] = set()
    for _, detection_index, tracked_index in sorted(candidates, reverse=True):
        if detection_index in used_detections or tracked_index in used_tracks:
            continue
        detections[detection_index]["track_id"] = int(tracked_ids[tracked_index])
        used_detections.add(detection_index)
        used_tracks.add(tracked_index)


def _carry_preferred_track_id(
    detection: dict[str, Any] | None,
    preferred_track_id: int | None,
    previous_bbox: list[float] | None,
    gap_frames: int,
    max_gap_frames: int,
    multiple_yoyo: bool,
) -> bool:
    """Bridge short ByteTrack ID gaps or switches for one spatially continuous yoyo."""
    if (
        detection is None
        or preferred_track_id is None
        or previous_bbox is None
        or multiple_yoyo
        or gap_frames < 1
        or gap_frames > max_gap_frames
    ):
        return False
    current_bbox = detection["bbox"]
    previous_center = ((previous_bbox[0] + previous_bbox[2]) / 2, (previous_bbox[1] + previous_bbox[3]) / 2)
    current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
    distance = math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])
    previous_diagonal = math.hypot(previous_bbox[2] - previous_bbox[0], previous_bbox[3] - previous_bbox[1])
    current_diagonal = math.hypot(current_bbox[2] - current_bbox[0], current_bbox[3] - current_bbox[1])
    if distance > 2.0 * max(1.0, previous_diagonal, current_diagonal):
        return False
    tracker_track_id = detection.get("track_id")
    if tracker_track_id is not None and int(tracker_track_id) == int(preferred_track_id):
        return False
    detection["track_id"] = int(preferred_track_id)
    detection["track_id_source"] = "temporal_carry"
    if tracker_track_id is not None:
        detection["tracker_track_id"] = int(tracker_track_id)
    return True


def _pick_yoyo(
    detections: list[dict[str, Any]],
    preferred_track_id: int | None = None,
    previous_bbox: list[float] | None = None,
    temporal_reference_trusted: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    yoyos = [item for item in detections if item["class_name"].lower() in {"yoyo", "yo-yo", "yoyo_body"}]
    if not yoyos:
        return None, ["no_yoyo"]
    yoyos.sort(key=lambda item: item["confidence"], reverse=True)
    distinct: list[dict[str, Any]] = []
    for candidate in yoyos:
        if any(_bbox_iou(candidate["bbox"], accepted["bbox"]) >= 0.35 for accepted in distinct):
            continue
        distinct.append(candidate)
    flags = ["multiple_yoyo"] if len(distinct) > 1 else []
    if preferred_track_id is not None:
        stable = next((item for item in distinct if item.get("track_id") == preferred_track_id), None)
        if stable is not None:
            stable["selection_source"] = "preferred_track"
            return stable, flags
    if len(distinct) > 1 and previous_bbox is not None and temporal_reference_trusted:
        previous_center = (
            (previous_bbox[0] + previous_bbox[2]) / 2.0,
            (previous_bbox[1] + previous_bbox[3]) / 2.0,
        )
        previous_diagonal = math.hypot(
            previous_bbox[2] - previous_bbox[0],
            previous_bbox[3] - previous_bbox[1],
        )
        temporal_candidates: list[tuple[float, float, dict[str, Any]]] = []
        for candidate in distinct:
            bbox = candidate["bbox"]
            center = candidate["center"]
            diagonal = max(
                1.0,
                previous_diagonal,
                math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            )
            normalized_distance = math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            ) / diagonal
            if normalized_distance <= 2.0:
                score = math.log(max(float(candidate["confidence"]), 1e-6)) - 1.5 * normalized_distance
                temporal_candidates.append((score, normalized_distance, candidate))
        if temporal_candidates:
            _, normalized_distance, selected = max(temporal_candidates, key=lambda item: item[0])
            selected["selection_source"] = "temporal_motion"
            selected["temporal_normalized_distance"] = round(float(normalized_distance), 4)
            flags.append("temporal_yoyo_selection")
            return selected, flags
    distinct[0]["selection_source"] = "confidence"
    return distinct[0], flags


def _orientation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [record["trick_orientation"] for record in records if record.get("trick_orientation")]
    if not predictions:
        return {"label": "unknown", "observed_frames": 0, "label_counts": {}, "mean_confidence": 0.0}
    counts = Counter(str(value.get("label", "unknown")) for value in predictions)
    label = counts.most_common(1)[0][0]
    selected = [value for value in predictions if value.get("label") == label]
    labels = [str(value.get("label", "unknown")) for value in predictions]
    raw_labels = [str(value.get("raw_label", value.get("label", "unknown"))) for value in predictions]
    return {
        "label": label,
        "observed_frames": len(predictions),
        "label_counts": dict(sorted(counts.items())),
        "switch_count": sum(left != right for left, right in zip(labels, labels[1:])),
        "raw_switch_count": sum(left != right for left, right in zip(raw_labels, raw_labels[1:])),
        "temporal_filter_enabled": any("temporal_filter" in value for value in predictions),
        "mean_confidence": round(
            sum(float(value.get("confidence", 0.0)) for value in selected) / max(1, len(selected)),
            6,
        ),
    }


def track_video(
    source_video_path: str | Path,
    weights_path: str | Path = TRACKING_CONFIG.weights_path,
    output_dir: str | Path = TRACKING_CONFIG.output_dir,
    confidence: float = TRACKING_CONFIG.confidence,
    iou: float = TRACKING_CONFIG.iou,
    imgsz: int = TRACKING_CONFIG.imgsz,
    device: str = TRACKING_CONFIG.device,
    trace_length: int = TRACKING_CONFIG.trace_length,
    line_thickness: int = TRACKING_CONFIG.line_thickness,
    text_scale: float = TRACKING_CONFIG.text_scale,
    visualization_max_width: int = TRACKING_CONFIG.visualization_max_width,
    pose_weights_path: str | Path | None = None,
    pose_detector_path: str | Path | None = None,
    enable_pose: bool = TRACKING_CONFIG.enable_pose,
    string_weights_path: str | Path | None = None,
    string_ensemble_weights_path: str | Path | None = TRACKING_CONFIG.string_ensemble_weights_path,
    string_ensemble_alpha: float = TRACKING_CONFIG.string_ensemble_alpha,
    string_ensemble_candidate_threshold: float = TRACKING_CONFIG.string_ensemble_candidate_threshold,
    string_adaptive_weights_path: str | Path | None = TRACKING_CONFIG.string_adaptive_weights_path,
    string_adaptive_ensemble_alpha: float = TRACKING_CONFIG.string_adaptive_ensemble_alpha,
    string_adaptive_window_frames: int = TRACKING_CONFIG.string_adaptive_window_frames,
    string_adaptive_max_color_accepts: int = TRACKING_CONFIG.string_adaptive_max_color_accepts,
    string_adaptive_max_mean_confidence: float = TRACKING_CONFIG.string_adaptive_max_mean_confidence,
    string_adaptive_min_mean_distance_ratio: float = TRACKING_CONFIG.string_adaptive_min_mean_distance_ratio,
    enable_string_model: bool = TRACKING_CONFIG.enable_string_model,
    string_confidence: float = TRACKING_CONFIG.string_confidence,
    string_inference_scale: float = TRACKING_CONFIG.string_inference_scale,
    string_inference_fps: float = TRACKING_CONFIG.string_inference_fps,
    string_color_probability_augment: bool = TRACKING_CONFIG.string_color_probability_augment,
    string_color_semantic_prefilter: bool = TRACKING_CONFIG.string_color_semantic_prefilter,
    string_color_probability_min_mean: float = TRACKING_CONFIG.string_color_probability_min_mean,
    string_color_probability_min_fraction: float = TRACKING_CONFIG.string_color_probability_min_fraction,
    string_max_propagation_frames: int = TRACKING_CONFIG.string_max_propagation_frames,
    string_flow_fb_max_error: float = TRACKING_CONFIG.string_flow_fb_max_error,
    yoyo_division: str = TRACKING_CONFIG.yoyo_division,
    orientation_weights_path: str | Path | None = None,
    enable_orientation_model: bool = TRACKING_CONFIG.enable_orientation_model,
    orientation_imgsz: int = TRACKING_CONFIG.orientation_imgsz,
    orientation_inference_fps: float = TRACKING_CONFIG.orientation_inference_fps,
    orientation_adaptive_inference: bool = TRACKING_CONFIG.orientation_adaptive_inference,
    orientation_burst_inference_fps: float = TRACKING_CONFIG.orientation_burst_inference_fps,
    orientation_adaptive_min_confidence: float = TRACKING_CONFIG.orientation_adaptive_min_confidence,
    orientation_adaptive_stable_observations: int = TRACKING_CONFIG.orientation_adaptive_stable_observations,
    orientation_temporal_filter: bool = TRACKING_CONFIG.orientation_temporal_filter,
    orientation_ema_alpha: float = TRACKING_CONFIG.orientation_ema_alpha,
    orientation_switch_margin: float = TRACKING_CONFIG.orientation_switch_margin,
    orientation_switch_confirmations: int = TRACKING_CONFIG.orientation_switch_confirmations,
    orientation_strong_switch_confidence: float = TRACKING_CONFIG.orientation_strong_switch_confidence,
    orientation_strong_switch_margin: float = TRACKING_CONFIG.orientation_strong_switch_margin,
    export_json: bool = True,
    start_seconds: float = 0.0,
    max_frames: int = 0,
) -> dict[str, Any]:
    if str(yoyo_division) not in {"1A", "2A", "3A", "4A", "5A"}:
        raise ValueError(f"Unsupported yoyo division: {yoyo_division}")
    if not 0.5 <= float(string_inference_scale) <= 2.0:
        raise ValueError("string_inference_scale must be between 0.5 and 2.0")
    if float(string_inference_fps) < 0.0:
        raise ValueError("string_inference_fps must be non-negative")
    if not 0.0 <= float(string_ensemble_alpha) <= 1.0:
        raise ValueError("string_ensemble_alpha must be between 0 and 1")
    if not 0.0 < float(string_ensemble_candidate_threshold) < 1.0:
        raise ValueError("string_ensemble_candidate_threshold must be between 0 and 1")
    if not 0.0 <= float(string_adaptive_ensemble_alpha) <= 1.0:
        raise ValueError("string_adaptive_ensemble_alpha must be between 0 and 1")
    if int(string_adaptive_window_frames) < 1 or int(string_adaptive_max_color_accepts) < 0:
        raise ValueError("adaptive window must be positive and maximum color accepts non-negative")
    if not 0.0 <= float(string_adaptive_max_mean_confidence) <= 1.0:
        raise ValueError("adaptive maximum mean confidence must be between 0 and 1")
    if float(string_adaptive_min_mean_distance_ratio) < 0.0:
        raise ValueError("adaptive minimum mean distance ratio must be non-negative")
    if not 0.0 <= float(string_color_probability_min_mean) <= 1.0:
        raise ValueError("string_color_probability_min_mean must be between 0 and 1")
    if not 0.0 <= float(string_color_probability_min_fraction) <= 1.0:
        raise ValueError("string_color_probability_min_fraction must be between 0 and 1")
    if float(orientation_inference_fps) < 0.0:
        raise ValueError("orientation_inference_fps must be non-negative")
    if float(orientation_burst_inference_fps) < 0.0:
        raise ValueError("orientation_burst_inference_fps must be non-negative")
    if not 0.0 <= float(orientation_adaptive_min_confidence) <= 1.0:
        raise ValueError("orientation_adaptive_min_confidence must be between 0 and 1")
    if int(orientation_adaptive_stable_observations) < 1:
        raise ValueError("orientation_adaptive_stable_observations must be positive")
    source_video_path, weights_path, output_dir = Path(source_video_path), Path(weights_path), Path(output_dir)
    if not source_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    class_names = {int(key): str(value) for key, value in dict(getattr(model, "names", {}) or {}).items()}
    pose_model, pose_error = (
        _load_rtmpose_model(pose_weights_path, device, pose_detector_path)
        if enable_pose else (None, None)
    )
    string_model, string_model_status = _load_string_model(
        string_weights_path,
        enable_string_model,
        device,
        string_ensemble_weights_path,
        string_ensemble_alpha,
        string_ensemble_candidate_threshold,
        string_adaptive_weights_path,
        string_adaptive_ensemble_alpha,
    )
    resolved_orientation_weights = Path(orientation_weights_path or TRACKING_CONFIG.orientation_weights_path)
    orientation_model, orientation_model_status = load_orientation_model(
        resolved_orientation_weights,
        enable_orientation_model,
    )
    orientation_filter_state = (
        OrientationTemporalFilter(
            ema_alpha=orientation_ema_alpha,
            switch_margin=orientation_switch_margin,
            switch_confirmations=orientation_switch_confirmations,
            strong_switch_confidence=orientation_strong_switch_confidence,
            strong_switch_margin=orientation_strong_switch_margin,
        )
        if orientation_model is not None and orientation_temporal_filter
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / f"{source_video_path.stem}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "tracked.mp4"
    json_path = run_dir / "frames.jsonl"
    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source_video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    string_inference_interval = _inference_interval_frames(fps, string_inference_fps)
    unanchored_semantic_grace_frames = max(2, int(round(fps * 0.25)))
    orientation_inference_interval = _inference_interval_frames(fps, orientation_inference_fps)
    orientation_burst_interval = _inference_interval_frames(fps, orientation_burst_inference_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_width, output_height = _visualization_size(width, height, visualization_max_width)
    start_frame = max(0, int(round(float(start_seconds) * fps)))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    try:
        import supervision as sv
        tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
    except Exception as exc:
        raise RuntimeError("supervision is required for stable track IDs") from exc

    records: list[dict[str, Any]] = []
    previous_center: tuple[float, float] | None = None
    last_seen_frame: int | None = None
    last_seen_edge_clipped = False
    missing_streak = 0
    previous_frame: np.ndarray | None = None
    previous_string: dict[str, Any] | None = None
    selected_track_id: int | None = None
    selected_track_bbox: list[float] | None = None
    selected_track_frame: int | None = None
    selected_yoyo_bbox: list[float] | None = None
    selected_yoyo_trusted = False
    selected_pose_bbox: list[float] | None = None
    selected_pose_frame: int | None = None
    traces: dict[int, list[tuple[int, int]]] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    frame_index = start_frame
    processed_frames = 0
    string_inference_frames = 0
    string_adaptive_history: list[tuple[bool, float, float]] = []
    string_adaptive_metrics: dict[str, float | int] = {}
    string_adaptive_trigger_frame: int | None = None
    string_adaptive_activation_frame: int | None = None
    string_adaptive_pending = False
    orientation_inference_frames = 0
    last_orientation: dict[str, Any] | None = None
    last_orientation_frame: int | None = None
    next_orientation_processed_frame = 0
    orientation_stable_observation_count = 0
    orientation_current_interval = orientation_burst_interval
    orientation_burst_inference_frames = 0
    metadata_file = open(json_path, "w", encoding="utf-8") if export_json else None
    loop_started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and processed_frames >= max_frames):
            break
        if string_adaptive_pending and isinstance(string_model, dict):
            string_model["adaptive_enabled"] = True
            string_adaptive_activation_frame = frame_index
            string_adaptive_pending = False
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": confidence,
            "iou": iou,
            "imgsz": imgsz,
            "augment": False,
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]
        detections_raw = _extract_detections(result, class_names)
        ultralytics_detections = sv.Detections.from_ultralytics(result)
        tracked = tracker.update_with_detections(ultralytics_detections)
        _assign_tracker_ids(detections_raw, tracked)
        yoyo, flags = _pick_yoyo(
            detections_raw,
            selected_track_id,
            previous_bbox=selected_yoyo_bbox,
            temporal_reference_trusted=selected_yoyo_trusted,
        )
        _carry_preferred_track_id(
            yoyo,
            selected_track_id,
            selected_track_bbox,
            frame_index - selected_track_frame if selected_track_frame is not None else 0,
            max(2, int(round(fps * 0.25))),
            "multiple_yoyo" in flags and "temporal_yoyo_selection" not in flags,
        )
        if yoyo is not None:
            selected_yoyo_bbox = [float(value) for value in yoyo["bbox"]]
            selected_yoyo_trusted = bool(
                float(yoyo["confidence"]) >= YOYO_TEMPORAL_TRUST_CONFIDENCE
                or yoyo.get("track_id") is not None
            )
        if yoyo is not None and yoyo.get("track_id") is not None:
            selected_track_id = int(yoyo["track_id"])
            selected_track_bbox = [float(value) for value in yoyo["bbox"]]
            selected_track_frame = frame_index
        center = tuple(yoyo["center"]) if yoyo else None
        speed = 0.0 if center is None or previous_center is None else math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) * fps
        pose_reference_age = (
            frame_index - selected_pose_frame if selected_pose_frame is not None else None
        )
        pose_reference_bbox = (
            selected_pose_bbox
            if pose_reference_age is not None
            and pose_reference_age <= max(2, int(round(fps * 0.25)))
            else None
        )
        wrists, pose, pose_person = _predict_pose(
            pose_model,
            frame,
            yoyo,
            imgsz,
            device,
            previous_person_bbox=pose_reference_bbox,
            temporal_reference_age_frames=pose_reference_age,
        )
        if pose_person.get("status") == "ok" and len(pose_person.get("bbox", [])) == 4:
            selected_pose_bbox = [float(value) for value in pose_person["bbox"]]
            selected_pose_frame = frame_index
        distance_to_hand = None
        if center and wrists:
            distance_to_hand = min(math.hypot(item["x"] - center[0], item["y"] - center[1]) for item in wrists)
        scheduled_string_inference = bool(
            string_model is not None and processed_frames % string_inference_interval == 0
        )
        model_string = None
        if scheduled_string_inference:
            model_string = _predict_string_model(
                string_model,
                frame,
                yoyo,
                string_confidence,
                imgsz,
                device,
                yoyo_division,
                string_inference_scale,
                wrists,
                string_color_probability_augment,
                string_color_probability_min_mean,
                string_color_probability_min_fraction,
                string_color_semantic_prefilter,
            )
            string_inference_frames += 1
            if (
                isinstance(string_model, dict)
                and string_model.get("kind") == "semantic_adaptive_ensemble"
                and not string_model.get("adaptive_enabled")
            ):
                string_adaptive_history, triggered, string_adaptive_metrics = (
                    update_adaptive_string_domain_gate(
                        string_adaptive_history,
                        model_string,
                        frame.shape[1],
                        frame.shape[0],
                        string_adaptive_window_frames,
                        string_adaptive_max_color_accepts,
                        string_adaptive_max_mean_confidence,
                        string_adaptive_min_mean_distance_ratio,
                    )
                )
                if triggered:
                    string_adaptive_trigger_frame = frame_index
                    string_adaptive_pending = True
        allow_unanchored_semantic = bool(
            yoyo is None
            and last_seen_frame is not None
            and frame_index - last_seen_frame <= unanchored_semantic_grace_frames
        )
        string = estimate_string(
            frame,
            yoyo,
            wrists,
            None,
            previous_string,
            yoyo_division,
            observation=model_string,
            max_propagation_frames=string_max_propagation_frames,
            max_forward_backward_error=string_flow_fb_max_error,
            # A loaded segmentation model returning no component is negative
            # evidence. Do not replace it with a weaker HSV/Hough proposal.
            allow_color_fallback=string_model is None,
            allow_unanchored_semantic=allow_unanchored_semantic,
            previous_frame=previous_frame,
        )
        reacquired_string = _should_reacquire_string(
            scheduled_string_inference,
            string_model is not None,
            yoyo,
            previous_string,
            string,
        )
        if reacquired_string:
            model_string = _predict_string_model(
                string_model,
                frame,
                yoyo,
                string_confidence,
                imgsz,
                device,
                yoyo_division,
                string_inference_scale,
                wrists,
                string_color_probability_augment,
                string_color_probability_min_mean,
                string_color_probability_min_fraction,
                string_color_semantic_prefilter,
            )
            string_inference_frames += 1
            string = estimate_string(
                frame,
                yoyo,
                wrists,
                None,
                previous_string,
                yoyo_division,
                observation=model_string,
                max_propagation_frames=string_max_propagation_frames,
                max_forward_backward_error=string_flow_fb_max_error,
                allow_color_fallback=False,
                allow_unanchored_semantic=allow_unanchored_semantic,
                previous_frame=previous_frame,
            )
        scheduled_orientation_inference = bool(
            orientation_model is not None and processed_frames >= next_orientation_processed_frame
        )
        orientation_inference_error = None
        if scheduled_orientation_inference:
            try:
                raw_orientation = predict_orientation(
                    orientation_model,
                    frame,
                    yoyo,
                    orientation_imgsz,
                    device,
                )
                trick_orientation = (
                    orientation_filter_state.update(raw_orientation)
                    if raw_orientation is not None and orientation_filter_state is not None
                    else raw_orientation
                )
            except Exception as exc:
                logger.warning("Orientation inference failed at frame %s: %s", frame_index, exc)
                orientation_inference_error = type(exc).__name__
                age = frame_index - last_orientation_frame if last_orientation_frame is not None else 0
                trick_orientation = carry_orientation(last_orientation, age)
            orientation_inference_frames += 1
            adaptive_unstable = orientation_observation_is_unstable(
                trick_orientation,
                orientation_adaptive_min_confidence,
                inference_error=orientation_inference_error is not None,
            )
            if orientation_adaptive_inference:
                orientation_stable_observation_count = (
                    0 if adaptive_unstable else orientation_stable_observation_count + 1
                )
                use_burst = orientation_stable_observation_count < int(
                    orientation_adaptive_stable_observations
                )
                orientation_current_interval = (
                    orientation_burst_interval if use_burst else orientation_inference_interval
                )
                orientation_burst_inference_frames += int(use_burst)
            else:
                orientation_current_interval = orientation_inference_interval
            next_orientation_processed_frame = processed_frames + orientation_current_interval
            if trick_orientation is not None:
                last_orientation = trick_orientation
                last_orientation_frame = frame_index
        else:
            age = frame_index - last_orientation_frame if last_orientation_frame is not None else 0
            trick_orientation = carry_orientation(last_orientation, age)
        if yoyo and yoyo["confidence"] < 0.35:
            flags.append("low_confidence")
        edge_clipped = bool(yoyo and (yoyo["bbox"][0] <= 1 or yoyo["bbox"][1] <= 1 or yoyo["bbox"][2] >= width - 1 or yoyo["bbox"][3] >= height - 1))
        if edge_clipped:
            flags.append("edge_clipped")
        if enable_pose and not wrists:
            flags.append("pose_missing")
        if enable_pose and pose_person.get("needs_review"):
            flags.append("pose_identity_needs_review")
        if string is not None:
            if string.get("needs_review", True):
                flags.append("string_needs_review")
            if float(string.get("confidence", 0.0)) < 0.35:
                flags.append("string_low_confidence")
            if not yoyo:
                flags.append("string_without_yoyo")
            if string.get("spatially_ambiguous"):
                flags.append("string_spatially_ambiguous")
            if string.get("hand_anchor_mismatch"):
                flags.append("string_hand_anchor_mismatch")
        elif yoyo:
            flags.append("string_not_observed")
            if previous_string is not None:
                flags.append("string_tracking_lost")
        elif previous_string is not None:
            flags.append("string_tracking_lost")
        if yoyo:
            visibility_state = "edge_clipped" if edge_clipped else "visible"
            missing_streak = 0
            last_seen_frame = frame_index
            last_seen_edge_clipped = edge_clipped
        else:
            missing_streak += 1
            visibility_state = "likely_out_of_frame" if last_seen_edge_clipped else "not_visible_or_occluded"
            flags.append(visibility_state)
        record = {
            "schema_version": "1.2",
            "frame_index": frame_index,
            "timestamp_s": frame_index / fps,
            "detections": detections_raw,
            "yoyo": yoyo,
            "hands": wrists,
            "pose": pose,
            "pose_person": pose_person,
            "string": string,
            "trick_orientation": trick_orientation,
            "orientation_model_inference": {
                "status": (
                    "error_carried" if orientation_inference_error and trick_orientation is not None
                    else "error" if orientation_inference_error
                    else "ran"
                    if scheduled_orientation_inference
                    else "carried"
                    if trick_orientation is not None
                    else "disabled_or_unavailable"
                ),
                "target_fps": float(orientation_inference_fps),
                "burst_fps": float(orientation_burst_inference_fps),
                "adaptive": bool(orientation_adaptive_inference),
                "adaptive_burst": bool(
                    orientation_adaptive_inference
                    and orientation_current_interval == orientation_burst_interval
                ),
                "interval_frames": int(orientation_current_interval),
                "stable_observation_count": int(orientation_stable_observation_count),
                "error_type": orientation_inference_error,
            },
            "yoyo_model_inference": {
                "status": "regular",
            },
            "string_model_inference": {
                "status": (
                    "ran"
                    if scheduled_string_inference or reacquired_string
                    else "skipped_interval"
                    if string_model is not None
                    else "disabled_or_unavailable"
                ),
                "target_fps": float(string_inference_fps),
                "interval_frames": int(string_inference_interval),
                "reason": (
                    "scheduled"
                    if scheduled_string_inference
                    else "flow_reacquire"
                    if reacquired_string
                    else "interval"
                    if string_model is not None
                    else "model_unavailable"
                ),
                "adaptive_primary": bool(
                    isinstance(string_model, dict) and string_model.get("adaptive_enabled")
                ),
            },
            "yoyo_division": yoyo_division,
            "visibility": {
                "state": visibility_state,
                "missing_streak_frames": missing_streak,
                "last_seen_frame": last_seen_frame,
            },
            "motion_speed_px_s": speed,
            "distance_to_hand_px": distance_to_hand,
            "bad_case": sorted(set(flags)),
            "quality": "review" if flags else "ok",
        }
        records.append(record)
        if metadata_file:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        writer.write(
            _draw_frame(
                frame,
                detections_raw,
                wrists,
                string,
                traces,
                colors,
                trace_length,
                line_thickness,
                text_scale,
                (output_width, output_height),
                pose,
                trick_orientation,
            )
        )
        previous_center = center
        # A visible string can persist briefly while the yoyo is occluded or
        # outside the frame; its record remains explicitly review-only.
        previous_string = (
            string
            if _can_seed_previous_string(string)
            else None
        )
        previous_frame = frame if previous_string is not None else None
        frame_index += 1
        processed_frames += 1
    capture.release()
    writer.release()
    if metadata_file:
        metadata_file.close()
    loop_seconds = max(0.0, time.perf_counter() - loop_started)
    loop_fps = float(processed_frames / loop_seconds) if loop_seconds > 0.0 else 0.0
    try:
        review_sheet_path = make_tracking_review_sheet(
            run_dir,
            source_video_path=source_video_path,
        )
    except Exception as exc:
        logger.warning("Could not create tracking review sheet: %s", exc)
        review_sheet_path = None
    bad_case_counts = Counter(flag for record in records for flag in record["bad_case"])
    component_selection_counts = Counter(
        str(record["string"].get("component_selection"))
        for record in records
        if record.get("string") and record["string"].get("component_selection")
    )
    string_geometry_counts = {
        "component_selection_counts": dict(sorted(component_selection_counts.items())),
        "hand_supported_observation_frames": sum(
            int((record.get("string") or {}).get("hand_supported_component_count", 0) > 0)
            for record in records
        ),
        "multi_component_observation_frames": sum(
            int(len((record.get("string") or {}).get("polylines") or []) > 1)
            for record in records
        ),
        "multi_component_flow_frames": sum(
            int((record.get("string") or {}).get("flow_component_count", 0) > 1)
            for record in records
        ),
        "partial_component_flow_loss_frames": sum(
            int(bool((record.get("string") or {}).get("flow_partial_component_loss")))
            for record in records
        ),
    }
    run_manifest = {
        "schema_version": "1.2",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source_video_path.resolve()),
        "source_video_sha256": sha256_file(source_video_path),
        "weights": str(weights_path.resolve()),
        "weights_sha256": sha256_file(weights_path),
        "string_model_kind": (
            string_model.get("kind", "yolo_segmentation") if isinstance(string_model, dict) else "yolo_segmentation"
        ) if string_model is not None else "disabled_or_unavailable",
        "string_weights_sha256": (
            sha256_file(Path(string_weights_path or TRACKING_CONFIG.string_weights_path))
            if string_model is not None else ""
        ),
        "string_ensemble_weights_sha256": (
            sha256_file(Path(string_model["ensemble_path"]))
            if isinstance(string_model, dict)
            and string_model.get("kind") in {"semantic_ensemble", "semantic_adaptive_ensemble"}
            else ""
        ),
        "string_adaptive_weights_sha256": (
            sha256_file(Path(string_model["adaptive_path"]))
            if isinstance(string_model, dict)
            and string_model.get("kind") == "semantic_adaptive_ensemble"
            else ""
        ),
        "pose_weights_sha256": (
            sha256_file(pose_model.pose_path)
            if pose_model is not None
            else ""
        ),
        "pose_detector_sha256": sha256_file(pose_model.detector_path) if pose_model is not None else "",
        "orientation_weights_sha256": (
            sha256_file(resolved_orientation_weights)
            if orientation_model is not None and resolved_orientation_weights.is_file()
            else ""
        ),
        "parameters": {
            "confidence": confidence,
            "iou": iou,
            "imgsz": imgsz,
            "device": device,
            "yoyo_temporal_selection": {
                "enabled": True,
                "score": "log_confidence_minus_1.5_normalized_center_distance",
                "trust_confidence": YOYO_TEMPORAL_TRUST_CONFIDENCE,
                "max_normalized_center_distance": 2.0,
            },
            "yoyo_track_id_carry": {
                "enabled": True,
                "max_gap_frames": max(2, int(round(fps * 0.25))),
                "spatial_gate": "2x_previous_or_current_bbox_diagonal",
                "requires_single_yoyo": True,
            },
            "visualization_max_width": int(visualization_max_width),
            "pose_enabled": enable_pose,
            "pose_backend": pose_model.backend_name if pose_model is not None else "unavailable",
            "pose_weights": str(pose_model.pose_path) if pose_model is not None else str(pose_weights_path or ""),
            "pose_detector": str(pose_model.detector_path) if pose_model is not None else str(pose_detector_path or ""),
            "string_model_enabled": bool(enable_string_model),
            "string_weights": str(string_weights_path or TRACKING_CONFIG.string_weights_path),
            "string_ensemble_weights": str(string_ensemble_weights_path or ""),
            "string_ensemble_alpha": float(string_ensemble_alpha),
            "string_ensemble_candidate_threshold": float(string_ensemble_candidate_threshold),
            "string_adaptive_weights": str(string_adaptive_weights_path or ""),
            "string_adaptive_ensemble_alpha": float(string_adaptive_ensemble_alpha),
            "string_adaptive_window_frames": int(string_adaptive_window_frames),
            "string_adaptive_max_color_accepts": int(string_adaptive_max_color_accepts),
            "string_adaptive_max_mean_confidence": float(string_adaptive_max_mean_confidence),
            "string_adaptive_min_mean_distance_ratio": float(string_adaptive_min_mean_distance_ratio),
            "string_cuda_graph": {
                "primary": bool(
                    isinstance(string_model, dict)
                    and getattr(string_model.get("ensemble_predictor"), "uses_cuda_graph", False)
                ),
                "adaptive": bool(
                    isinstance(string_model, dict)
                    and getattr(
                        string_model.get("adaptive_ensemble_predictor"),
                        "uses_cuda_graph",
                        False,
                    )
                ),
            },
            "string_confidence": string_confidence,
            "string_inference_scale": float(string_inference_scale),
            "string_inference_fps": float(string_inference_fps),
            "string_inference_interval_frames": int(string_inference_interval),
            "string_color_probability_augment": bool(string_color_probability_augment),
            "string_color_semantic_prefilter": bool(string_color_semantic_prefilter),
            "string_color_probability_min_mean": float(string_color_probability_min_mean),
            "string_color_probability_min_fraction": float(string_color_probability_min_fraction),
            "string_max_propagation_frames": int(string_max_propagation_frames),
            "string_flow_fb_max_error": float(string_flow_fb_max_error),
            "string_unanchored_semantic_grace_frames": int(unanchored_semantic_grace_frames),
            "yoyo_division": yoyo_division,
            "orientation_model_enabled": bool(enable_orientation_model),
            "orientation_weights": str(resolved_orientation_weights),
            "orientation_imgsz": int(orientation_imgsz),
            "orientation_inference_fps": float(orientation_inference_fps),
            "orientation_inference_interval_frames": int(orientation_inference_interval),
            "orientation_adaptive_inference": bool(orientation_adaptive_inference),
            "orientation_burst_inference_fps": float(orientation_burst_inference_fps),
            "orientation_burst_inference_interval_frames": int(orientation_burst_interval),
            "orientation_adaptive_min_confidence": float(orientation_adaptive_min_confidence),
            "orientation_adaptive_stable_observations": int(orientation_adaptive_stable_observations),
            "orientation_temporal_filter": bool(orientation_temporal_filter),
            "orientation_ema_alpha": float(orientation_ema_alpha),
            "orientation_switch_margin": float(orientation_switch_margin),
            "orientation_switch_confirmations": int(orientation_switch_confirmations),
            "orientation_strong_switch_confidence": float(orientation_strong_switch_confidence),
            "orientation_strong_switch_margin": float(orientation_strong_switch_margin),
            "start_seconds": start_seconds,
            "max_frames": max_frames,
        },
        "frame_count": processed_frames,
        "string_inference_frame_count": string_inference_frames,
        "string_adaptive_trigger_frame": string_adaptive_trigger_frame,
        "string_adaptive_activation_frame": string_adaptive_activation_frame,
        "string_adaptive_gate_metrics": string_adaptive_metrics,
        "orientation_inference_frame_count": orientation_inference_frames,
        "orientation_burst_inference_frame_count": orientation_burst_inference_frames,
        "orientation_summary": _orientation_summary(records),
        "performance": {
            "tracking_loop_seconds": round(loop_seconds, 4),
            "tracking_loop_fps": round(loop_fps, 4),
        },
        "fps": fps,
        "width": width,
        "height": height,
        "output_width": output_width,
        "output_height": output_height,
        "bad_case_counts": dict(sorted(bad_case_counts.items())),
        "string_geometry_counts": string_geometry_counts,
        "outputs": {
            "tracked_video": str(output_path),
            "frames_jsonl": str(json_path) if export_json else "",
            "review_sheet": str(review_sheet_path or ""),
            "review_index": str(run_dir / "tracking_review_index.json") if review_sheet_path else "",
        },
        "limitations": [
            "String observations are review-only; fresh model/color geometry is checked against forward/backward optical flow without being deformed, and flow-only propagation is capped by string_max_propagation_frames.",
            "Division metadata is recorded for provenance and does not impose attachment-specific geometric filtering.",
            "string_without_yoyo marks frames where a visible string estimate persists while the yoyo is out of frame or occluded; these frames require manual review.",
            "not_visible_or_occluded does not distinguish occlusion from an off-camera yoyo without manual review.",
            "trick_orientation is the supported coarse three-way frame classification.",
        ],
    }
    run_manifest_path = run_dir / "run.json"
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Tracking complete: frames=%s video=%s metadata=%s", processed_frames, output_path, json_path if export_json else "disabled")
    return {
        "source_video": str(source_video_path),
        "output_video": str(output_path),
        "metadata_jsonl": str(json_path) if export_json else "",
        "review_sheet": str(review_sheet_path or ""),
        "run_manifest": str(run_manifest_path),
        "run_dir": str(run_dir),
        "bad_case_counts": dict(sorted(bad_case_counts.items())),
        "string_geometry_counts": string_geometry_counts,
        "weights": str(weights_path),
        "pose_weights": (
            str(pose_model.pose_path)
            if pose_model is not None
            else pose_error or str(pose_weights_path or "")
            if enable_pose
            else ""
        ),
        "string_model": string_model_status,
        "orientation_model": orientation_model_status,
        "frame_count": processed_frames,
        "output_width": output_width,
        "output_height": output_height,
        "string_inference_frame_count": string_inference_frames,
        "string_adaptive_trigger_frame": string_adaptive_trigger_frame,
        "string_adaptive_activation_frame": string_adaptive_activation_frame,
        "string_adaptive_gate_metrics": string_adaptive_metrics,
        "orientation_inference_frame_count": orientation_inference_frames,
        "orientation_burst_inference_frame_count": orientation_burst_inference_frames,
        "orientation_summary": _orientation_summary(records),
        "tracking_loop_seconds": round(loop_seconds, 4),
        "tracking_loop_fps": round(loop_fps, 4),
        "fps": fps,
        "confidence": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track yoyo objects and export reviewable frame metadata.")
    parser.add_argument("video", help="Input video path.")
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.weights_path))
    parser.add_argument("--output-dir", default=str(TRACKING_CONFIG.output_dir))
    parser.add_argument("--conf", type=float, default=TRACKING_CONFIG.confidence)
    parser.add_argument("--iou", type=float, default=TRACKING_CONFIG.iou)
    parser.add_argument("--imgsz", type=int, default=TRACKING_CONFIG.imgsz)
    parser.add_argument("--device", default=TRACKING_CONFIG.device)
    parser.add_argument(
        "--visualization-max-width",
        type=int,
        default=TRACKING_CONFIG.visualization_max_width,
        help="Maximum annotated preview width; 0 preserves source resolution.",
    )
    parser.add_argument("--pose-weights", default=str(TRACKING_CONFIG.pose_weights_path))
    parser.add_argument("--pose-detector", default=str(TRACKING_CONFIG.pose_detector_path))
    parser.add_argument(
        "--pose",
        action=argparse.BooleanOptionalAction,
        default=TRACKING_CONFIG.enable_pose,
        help="Run RTMPose-m WholeBody inference; defaults to tracking.enable_pose.",
    )
    parser.add_argument("--string-weights", default=str(TRACKING_CONFIG.string_weights_path))
    parser.add_argument(
        "--string-ensemble-weights",
        default="",
        help="Secondary semantic checkpoint; the configured default is used with the default primary.",
    )
    parser.add_argument(
        "--string-ensemble-alpha",
        type=float,
        default=TRACKING_CONFIG.string_ensemble_alpha,
    )
    parser.add_argument(
        "--string-ensemble-candidate-threshold",
        type=float,
        default=TRACKING_CONFIG.string_ensemble_candidate_threshold,
    )
    parser.add_argument(
        "--string-adaptive-weights",
        default="",
        help="Weak-domain primary checkpoint; the configured default is used with the default primary.",
    )
    parser.add_argument(
        "--string-adaptive-ensemble-alpha",
        type=float,
        default=TRACKING_CONFIG.string_adaptive_ensemble_alpha,
    )
    parser.add_argument(
        "--string-adaptive-window-frames",
        type=int,
        default=TRACKING_CONFIG.string_adaptive_window_frames,
    )
    parser.add_argument(
        "--string-adaptive-max-color-accepts",
        type=int,
        default=TRACKING_CONFIG.string_adaptive_max_color_accepts,
    )
    parser.add_argument(
        "--string-adaptive-max-mean-confidence",
        type=float,
        default=TRACKING_CONFIG.string_adaptive_max_mean_confidence,
    )
    parser.add_argument(
        "--string-adaptive-min-mean-distance-ratio",
        type=float,
        default=TRACKING_CONFIG.string_adaptive_min_mean_distance_ratio,
    )
    parser.add_argument("--no-string-model", action="store_true")
    parser.add_argument("--string-conf", type=float, default=TRACKING_CONFIG.string_confidence)
    parser.add_argument(
        "--string-inference-scale",
        type=float,
        default=TRACKING_CONFIG.string_inference_scale,
        help="Semantic input scale in [0.5, 2.0]; 2.0 preserves more thin-string pixels at higher cost.",
    )
    parser.add_argument(
        "--string-inference-fps",
        type=float,
        default=TRACKING_CONFIG.string_inference_fps,
        help="Target semantic-model cadence; 0 runs semantic inference on every frame.",
    )
    parser.add_argument(
        "--no-string-color-probability-augment",
        action="store_true",
        default=not TRACKING_CONFIG.string_color_probability_augment,
        help="Disable probability-gated color/Hough augmentation of semantic string observations.",
    )
    parser.add_argument(
        "--string-color-probability-min-mean",
        type=float,
        default=TRACKING_CONFIG.string_color_probability_min_mean,
    )
    parser.add_argument(
        "--string-color-semantic-prefilter",
        action=argparse.BooleanOptionalAction,
        default=TRACKING_CONFIG.string_color_semantic_prefilter,
        help="Restrict color/Hough candidates to the semantic probability neighborhood.",
    )
    parser.add_argument(
        "--string-color-probability-min-fraction",
        type=float,
        default=TRACKING_CONFIG.string_color_probability_min_fraction,
    )
    parser.add_argument("--string-max-propagation-frames", type=int, default=TRACKING_CONFIG.string_max_propagation_frames)
    parser.add_argument("--string-flow-fb-max-error", type=float, default=TRACKING_CONFIG.string_flow_fb_max_error)
    parser.add_argument(
        "--yoyo-division",
        choices=["1A", "2A", "3A", "4A", "5A"],
        default=TRACKING_CONFIG.yoyo_division,
    )
    parser.add_argument("--orientation-weights", default=str(TRACKING_CONFIG.orientation_weights_path))
    parser.add_argument("--no-orientation-model", action="store_true")
    parser.add_argument("--orientation-imgsz", type=int, default=TRACKING_CONFIG.orientation_imgsz)
    parser.add_argument(
        "--orientation-inference-fps",
        type=float,
        default=TRACKING_CONFIG.orientation_inference_fps,
        help="Target coarse-orientation classifier cadence; 0 runs it on every frame.",
    )
    parser.add_argument("--no-orientation-adaptive-inference", action="store_true")
    parser.add_argument(
        "--orientation-burst-inference-fps",
        type=float,
        default=TRACKING_CONFIG.orientation_burst_inference_fps,
    )
    parser.add_argument(
        "--orientation-adaptive-min-confidence",
        type=float,
        default=TRACKING_CONFIG.orientation_adaptive_min_confidence,
    )
    parser.add_argument(
        "--orientation-adaptive-stable-observations",
        type=int,
        default=TRACKING_CONFIG.orientation_adaptive_stable_observations,
    )
    parser.add_argument("--no-orientation-temporal-filter", action="store_true")
    parser.add_argument("--orientation-ema-alpha", type=float, default=TRACKING_CONFIG.orientation_ema_alpha)
    parser.add_argument("--orientation-switch-margin", type=float, default=TRACKING_CONFIG.orientation_switch_margin)
    parser.add_argument(
        "--orientation-switch-confirmations",
        type=int,
        default=TRACKING_CONFIG.orientation_switch_confirmations,
    )
    parser.add_argument(
        "--orientation-strong-switch-confidence",
        type=float,
        default=TRACKING_CONFIG.orientation_strong_switch_confidence,
    )
    parser.add_argument(
        "--orientation-strong-switch-margin",
        type=float,
        default=TRACKING_CONFIG.orientation_strong_switch_margin,
    )
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    use_default_string_bundle = (
        Path(args.string_weights).resolve() == TRACKING_CONFIG.string_weights_path.resolve()
    )
    string_ensemble_weights = (
        args.string_ensemble_weights.strip()
        or (str(TRACKING_CONFIG.string_ensemble_weights_path) if use_default_string_bundle else "")
    )
    string_adaptive_weights = (
        args.string_adaptive_weights.strip()
        or (str(TRACKING_CONFIG.string_adaptive_weights_path) if use_default_string_bundle else "")
    )
    result = track_video(
        source_video_path=args.video,
        weights_path=args.weights,
        output_dir=args.output_dir,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        visualization_max_width=args.visualization_max_width,
        pose_weights_path=args.pose_weights or None,
        pose_detector_path=args.pose_detector or None,
        enable_pose=args.pose,
        string_weights_path=args.string_weights,
        string_ensemble_weights_path=string_ensemble_weights or None,
        string_ensemble_alpha=args.string_ensemble_alpha,
        string_ensemble_candidate_threshold=args.string_ensemble_candidate_threshold,
        string_adaptive_weights_path=string_adaptive_weights or None,
        string_adaptive_ensemble_alpha=args.string_adaptive_ensemble_alpha,
        string_adaptive_window_frames=args.string_adaptive_window_frames,
        string_adaptive_max_color_accepts=args.string_adaptive_max_color_accepts,
        string_adaptive_max_mean_confidence=args.string_adaptive_max_mean_confidence,
        string_adaptive_min_mean_distance_ratio=args.string_adaptive_min_mean_distance_ratio,
        enable_string_model=not args.no_string_model,
        string_confidence=args.string_conf,
        string_inference_scale=args.string_inference_scale,
        string_inference_fps=args.string_inference_fps,
        string_color_probability_augment=not args.no_string_color_probability_augment,
        string_color_semantic_prefilter=args.string_color_semantic_prefilter,
        string_color_probability_min_mean=args.string_color_probability_min_mean,
        string_color_probability_min_fraction=args.string_color_probability_min_fraction,
        string_max_propagation_frames=args.string_max_propagation_frames,
        string_flow_fb_max_error=args.string_flow_fb_max_error,
        yoyo_division=args.yoyo_division,
        orientation_weights_path=args.orientation_weights,
        enable_orientation_model=not args.no_orientation_model,
        orientation_imgsz=args.orientation_imgsz,
        orientation_inference_fps=args.orientation_inference_fps,
        orientation_adaptive_inference=not args.no_orientation_adaptive_inference,
        orientation_burst_inference_fps=args.orientation_burst_inference_fps,
        orientation_adaptive_min_confidence=args.orientation_adaptive_min_confidence,
        orientation_adaptive_stable_observations=args.orientation_adaptive_stable_observations,
        orientation_temporal_filter=not args.no_orientation_temporal_filter,
        orientation_ema_alpha=args.orientation_ema_alpha,
        orientation_switch_margin=args.orientation_switch_margin,
        orientation_switch_confirmations=args.orientation_switch_confirmations,
        orientation_strong_switch_confidence=args.orientation_strong_switch_confidence,
        orientation_strong_switch_margin=args.orientation_strong_switch_margin,
        export_json=not args.no_json,
        start_seconds=args.start_seconds,
        max_frames=args.max_frames,
    )
    logger.info("Done: %s", result["output_video"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
