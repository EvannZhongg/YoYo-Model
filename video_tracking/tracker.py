"""YOYO video tracking, metadata export, and heuristic trick segmentation.

The detector is intentionally kept separate from the annotation protocol.  A
tracking run always produces machine-readable per-frame records, even when a
pose model is unavailable.  This makes failed/ambiguous frames reviewable and
allows a later string segmentation model to consume the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import BASE_DIR, TRACKING_CONFIG
from string_segmentation.semantic_model import (
    is_semantic_checkpoint,
    load_checkpoint as load_semantic_checkpoint,
    predict_letterboxed,
    semantic_mask_observation,
)
from video_tracking.review_sheet import make_tracking_review_sheet
from video_tracking.string_tracker import estimate_string
from video_tracking.tokenize import export_tracking_features
from video_tracking.trick_tokens import export_trick_tokens


LOG_FILE = BASE_DIR / "track_video.log"
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


def _load_pose_model(weights_path: str | Path | None, auto_download: bool):
    if not weights_path and not auto_download:
        return None, None
    from ultralytics import YOLO

    requested = str(weights_path or "yolo11n-pose.pt")
    try:
        model = YOLO(requested)
    except Exception as exc:
        logger.warning("Pose model unavailable (%s): %s", requested, exc)
        return None, str(exc)
    return model, requested


def _load_string_model(weights_path: str | Path | None, enabled: bool, device: str = ""):
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
            return (
                {
                    "kind": "semantic",
                    "model": model,
                    "checkpoint": checkpoint,
                    "device": semantic_device,
                    "path": str(path),
                },
                f"semantic:{path}",
            )
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


def _predict_string_model(
    model,
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    confidence: float,
    imgsz: int,
    device: str,
    attachment_class: str,
) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, dict) and model.get("kind") == "semantic":
        checkpoint = model["checkpoint"]
        model_device = model["device"]
        model_config = checkpoint["model_config"]
        probability, meta = predict_letterboxed(
            model["model"],
            frame,
            int(model_config["input_width"]),
            int(model_config["input_height"]),
            model_device,
        )
        threshold = max(float(checkpoint.get("threshold", 0.5)), float(confidence))
        return semantic_mask_observation(
            probability,
            meta,
            threshold=threshold,
            yoyo=yoyo,
            attachment_class=attachment_class,
            min_component_pixels=8,
        )
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
    if attachment_class == "hand_and_yoyo_attached":
        candidates.sort(key=lambda item: (item[0], item[1]))
    else:
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


def _predict_pose(model, frame: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model is None:
        return [], []
    try:
        result = model.predict(source=frame, verbose=False)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return [], []
        points = keypoints.xy[0].cpu().numpy()
        confidence = None
        if getattr(keypoints, "conf", None) is not None:
            confidence = keypoints.conf[0].cpu().numpy()
        # COCO pose indexes: left/right wrist are 9/10.
        wrists: list[dict[str, Any]] = []
        for index, name in ((9, "left_wrist"), (10, "right_wrist")):
            if index >= len(points):
                continue
            conf = float(confidence[index]) if confidence is not None and index < len(confidence) else 1.0
            if conf < 0.20:
                continue
            wrists.append({"name": name, "x": float(points[index][0]), "y": float(points[index][1]), "confidence": conf})
        pose = []
        for index, point in enumerate(points):
            conf = float(confidence[index]) if confidence is not None and index < len(confidence) else 1.0
            pose.append({"index": index, "x": float(point[0]), "y": float(point[1]), "confidence": conf})
        return wrists, pose
    except Exception as exc:
        logger.debug("Pose inference failed: %s", exc)
        return [], []


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
) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
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
                cv2.polylines(canvas, [np.asarray(trace, dtype=np.int32)], False, color, line_thickness)
    for wrist in wrists:
        point = (int(wrist["x"]), int(wrist["y"]))
        cv2.circle(canvas, point, 6, (0, 220, 255), -1)
        cv2.putText(canvas, wrist["name"], (point[0] + 7, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    if string is not None:
        polygons = string.get("polygons") or ([string["polygon"]] if string.get("polygon") else [])
        if polygons:
            mask_layer = canvas.copy()
            polygon_arrays = [np.asarray(polygon, dtype=np.float32).round().astype(np.int32) for polygon in polygons]
            cv2.fillPoly(mask_layer, polygon_arrays, (255, 80, 30))
            canvas = cv2.addWeighted(mask_layer, 0.28, canvas, 0.72, 0)
            cv2.polylines(canvas, polygon_arrays, True, (255, 80, 30), 1)
        polylines = string.get("polylines") or ([string["points"]] if string.get("points") else [])
        point_arrays = [np.asarray(points, dtype=np.float32).round().astype(np.int32) for points in polylines if len(points) >= 2]
        for points in point_arrays:
            cv2.polylines(canvas, [points], False, (255, 80, 30), 2)
        label = f"string {string.get('method', 'estimate')} {float(string.get('confidence', 0.0)):.2f} / review"
        label_point = tuple(point_arrays[0][0]) if point_arrays else (12, 42)
        cv2.putText(canvas, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 30), 1, cv2.LINE_AA)
    return canvas


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _pick_yoyo(detections: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
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
    return distinct[0], flags


def _is_trick_active(
    yoyo: dict[str, Any] | None,
    speed_px_s: float,
    distance_to_hand_px: float | None,
    width: int,
    height: int,
    speed_diagonal_per_s: float,
) -> bool:
    """Use a resolution/FPS-independent speed threshold for clip candidates."""
    if yoyo is None:
        return False
    diagonal = math.hypot(width, height)
    speed_threshold = max(25.0, max(0.0, float(speed_diagonal_per_s)) * diagonal)
    separated_from_hand = distance_to_hand_px is not None and distance_to_hand_px >= 0.08 * diagonal
    return bool(speed_px_s >= speed_threshold or separated_from_hand)


def _segments_from_records(
    records: list[dict[str, Any]],
    fps: float,
    padding_seconds: float,
    min_segment_seconds: float,
    max_gap_seconds: float,
    max_segment_seconds: float,
) -> list[dict[str, Any]]:
    if not records:
        return []
    # Only exported valid segments are capped; source-video processing is not.
    max_segment_seconds = 180.0 if max_segment_seconds <= 0 else min(float(max_segment_seconds), 180.0)
    active = [bool(item.get("active")) for item in records]
    max_gap = max(1, int(round(fps * max_gap_seconds)))
    min_length = max(1, int(round(fps * min_segment_seconds)))
    candidates: list[tuple[int, int]] = []
    start = None
    gap = 0
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
            gap = 0
        elif start is not None and not is_active:
            gap += 1
            if gap > max_gap:
                end = index - gap
                if end - start + 1 >= min_length:
                    candidates.append((start, end))
                start = None
                gap = 0
    if start is not None:
        end = len(records) - 1 - gap
        if end - start + 1 >= min_length:
            candidates.append((start, end))
    padding = int(round(fps * padding_seconds))
    # Padding is part of the exported clip, so reserve room for both sides.
    max_export_length = max(min_length, int(round(fps * max_segment_seconds))) if max_segment_seconds > 0 else 0
    max_active_length = max(min_length, max_export_length - 2 * padding) if max_export_length else 0
    result = []
    bounded_candidates: list[tuple[int, int, bool]] = []
    for start, end in candidates:
        if not max_active_length or end - start + 1 <= max_active_length:
            bounded_candidates.append((start, end, False))
            continue
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + max_active_length - 1)
            bounded_candidates.append((chunk_start, chunk_end, True))
            chunk_start = chunk_end + 1

    for segment_index, (start, end, duration_limited) in enumerate(bounded_candidates, start=1):
        padded_start = max(0, start - padding)
        padded_end = min(len(records) - 1, end + padding)
        if max_export_length and padded_end - padded_start + 1 > max_export_length:
            padded_end = padded_start + max_export_length - 1
        result.append(
            {
                "segment_id": segment_index,
                "start_frame": records[padded_start]["frame_index"],
                "end_frame": records[padded_end]["frame_index"],
                "start_time_s": records[padded_start]["timestamp_s"],
                "end_time_s": records[padded_end]["timestamp_s"],
                "duration_s": (records[padded_end]["frame_index"] - records[padded_start]["frame_index"] + 1) / fps,
                "reason": "activity_with_duration_limit" if duration_limited else "yoyo_motion_or_hand_distance",
                "needs_review": True,
                "review_status": "auto_candidate_needs_review",
                "trick_label": "",
                "review_notes": "",
            }
        )
    return result


def _write_segments(source: Path, output_dir: Path, segments: list[dict[str, Any]], fps: float, width: int, height: int) -> list[dict[str, Any]]:
    if not segments:
        return []
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for segment in segments:
        path = clip_dir / f"{source.stem}_trick_{segment['segment_id']:03d}.mp4"
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_index = segment["start_frame"]
        while frame_index <= segment["end_frame"]:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index += 1
        writer.release()
        capture.release()
        segment = dict(segment)
        segment["output_video"] = str(path)
        outputs.append(segment)
    return outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    pose_weights_path: str | Path | None = None,
    enable_pose: bool = False,
    auto_download_pose: bool = False,
    string_weights_path: str | Path | None = None,
    enable_string_model: bool = TRACKING_CONFIG.enable_string_model,
    string_confidence: float = TRACKING_CONFIG.string_confidence,
    string_max_propagation_frames: int = TRACKING_CONFIG.string_max_propagation_frames,
    string_flow_fb_max_error: float = TRACKING_CONFIG.string_flow_fb_max_error,
    string_fusion_distance_px: float = TRACKING_CONFIG.string_fusion_distance_px,
    string_attachment_class: str = TRACKING_CONFIG.string_attachment_class,
    export_json: bool = True,
    export_clips: bool = True,
    activity_speed_diagonal_per_s: float = TRACKING_CONFIG.activity_speed_diagonal_per_s,
    padding_seconds: float = 0.4,
    min_segment_seconds: float = 0.5,
    max_gap_seconds: float = 0.4,
    max_segment_seconds: float = TRACKING_CONFIG.max_segment_seconds,
    start_seconds: float = 0.0,
    max_frames: int = 0,
) -> dict[str, Any]:
    allowed_attachment_classes = {"hand_and_yoyo_attached", "yoyo_detached", "hand_detached", "unknown"}
    if string_attachment_class not in allowed_attachment_classes:
        raise ValueError(f"Unsupported string attachment class: {string_attachment_class}")
    max_segment_seconds = 180.0 if max_segment_seconds <= 0 else min(float(max_segment_seconds), 180.0)
    source_video_path, weights_path, output_dir = Path(source_video_path), Path(weights_path), Path(output_dir)
    if not source_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    class_names = {int(key): str(value) for key, value in dict(getattr(model, "names", {}) or {}).items()}
    pose_model, pose_error = _load_pose_model(pose_weights_path, auto_download_pose) if enable_pose else (None, None)
    string_model, string_model_status = _load_string_model(string_weights_path, enable_string_model, device)
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
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(round(float(start_seconds) * fps)))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
    previous_gray: np.ndarray | None = None
    previous_string: dict[str, Any] | None = None
    traces: dict[int, list[tuple[int, int]]] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    frame_index = start_frame
    processed_frames = 0
    metadata_file = open(json_path, "w", encoding="utf-8") if export_json else None
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and processed_frames >= max_frames):
            break
        kwargs: dict[str, Any] = {"source": frame, "conf": confidence, "iou": iou, "imgsz": imgsz, "verbose": False}
        if device:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]
        detections_raw = _extract_detections(result, class_names)
        ultralytics_detections = sv.Detections.from_ultralytics(result)
        tracked = tracker.update_with_detections(ultralytics_detections)
        ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for index, detection in enumerate(detections_raw):
            if index < len(ids):
                detection["track_id"] = int(ids[index])
        yoyo, flags = _pick_yoyo(detections_raw)
        center = tuple(yoyo["center"]) if yoyo else None
        speed = 0.0 if center is None or previous_center is None else math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) * fps
        wrists, pose = _predict_pose(pose_model, frame)
        distance_to_hand = None
        if center and wrists:
            distance_to_hand = min(math.hypot(item["x"] - center[0], item["y"] - center[1]) for item in wrists)
        model_string = _predict_string_model(
            string_model,
            frame,
            yoyo,
            string_confidence,
            imgsz,
            device,
            string_attachment_class,
        )
        string = estimate_string(
            frame,
            yoyo,
            wrists,
            previous_gray,
            previous_string,
            string_attachment_class,
            observation=model_string,
            max_propagation_frames=string_max_propagation_frames,
            max_forward_backward_error=string_flow_fb_max_error,
            fusion_distance_px=string_fusion_distance_px,
        )
        if yoyo and yoyo["confidence"] < 0.35:
            flags.append("low_confidence")
        edge_clipped = bool(yoyo and (yoyo["bbox"][0] <= 1 or yoyo["bbox"][1] <= 1 or yoyo["bbox"][2] >= width - 1 or yoyo["bbox"][3] >= height - 1))
        if edge_clipped:
            flags.append("edge_clipped")
        if enable_pose and not wrists:
            flags.append("pose_missing")
        if string is not None:
            if string.get("needs_review", True):
                flags.append("string_needs_review")
            if float(string.get("confidence", 0.0)) < 0.35:
                flags.append("string_low_confidence")
            if not yoyo:
                flags.append("string_without_yoyo")
            if string.get("temporal_conflict"):
                flags.append("string_temporal_conflict")
            if string.get("spatially_ambiguous"):
                flags.append("string_spatially_ambiguous")
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
        active = _is_trick_active(
            yoyo,
            speed,
            distance_to_hand,
            width,
            height,
            activity_speed_diagonal_per_s,
        )
        record = {
            "schema_version": "1.1",
            "frame_index": frame_index,
            "timestamp_s": frame_index / fps,
            "detections": detections_raw,
            "yoyo": yoyo,
            "hands": wrists,
            "pose": pose,
            "string": string,
            "string_attachment_class": string_attachment_class,
            "visibility": {
                "state": visibility_state,
                "missing_streak_frames": missing_streak,
                "last_seen_frame": last_seen_frame,
            },
            "motion_speed_px_s": speed,
            "distance_to_hand_px": distance_to_hand,
            "active": active,
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
            )
        )
        previous_center = center
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # A visible string can persist briefly while the yoyo is occluded or
        # outside the frame; its record remains explicitly review-only.
        previous_string = (
            string
            if string is not None and not string.get("spatially_ambiguous")
            else None
        )
        frame_index += 1
        processed_frames += 1
    capture.release()
    writer.release()
    if metadata_file:
        metadata_file.close()
    segments = _segments_from_records(
        records,
        fps,
        padding_seconds,
        min_segment_seconds,
        max_gap_seconds,
        max_segment_seconds,
    )
    if export_clips:
        segments = _write_segments(source_video_path, run_dir, segments, fps, width, height)
    segments_path = run_dir / "segments.json"
    segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        review_sheet_path = make_tracking_review_sheet(run_dir)
    except Exception as exc:
        logger.warning("Could not create tracking review sheet: %s", exc)
        review_sheet_path = None
    try:
        frame_feature_outputs = export_tracking_features(records, run_dir, width, height, fps)
    except Exception as exc:
        logger.warning("Could not export tracking frame features: %s", exc)
        frame_feature_outputs = {"jsonl": "", "npz": "", "manifest": ""}
    bad_case_counts = Counter(flag for record in records for flag in record["bad_case"])
    run_manifest = {
        "schema_version": "1.1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source_video_path.resolve()),
        "source_video_sha256": _sha256(source_video_path),
        "weights": str(weights_path.resolve()),
        "weights_sha256": _sha256(weights_path),
        "string_model_kind": (
            string_model.get("kind", "yolo_segmentation") if isinstance(string_model, dict) else "yolo_segmentation"
        ) if string_model is not None else "disabled_or_unavailable",
        "string_weights_sha256": (
            _sha256(Path(string_weights_path or TRACKING_CONFIG.string_weights_path))
            if string_model is not None else ""
        ),
        "parameters": {
            "confidence": confidence,
            "iou": iou,
            "imgsz": imgsz,
            "device": device,
            "pose_enabled": enable_pose,
            "pose_weights": str(pose_weights_path or ""),
            "string_model_enabled": bool(enable_string_model),
            "string_weights": str(string_weights_path or TRACKING_CONFIG.string_weights_path),
            "string_confidence": string_confidence,
            "string_max_propagation_frames": int(string_max_propagation_frames),
            "string_flow_fb_max_error": float(string_flow_fb_max_error),
            "string_fusion_distance_px": float(string_fusion_distance_px),
            "string_attachment_class": string_attachment_class,
            "activity_speed_diagonal_per_s": float(activity_speed_diagonal_per_s),
            "padding_seconds": padding_seconds,
            "min_segment_seconds": min_segment_seconds,
            "max_gap_seconds": max_gap_seconds,
            "max_segment_seconds": max_segment_seconds,
            "start_seconds": start_seconds,
            "max_frames": max_frames,
        },
        "frame_count": processed_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "segments_count": len(segments),
        "bad_case_counts": dict(sorted(bad_case_counts.items())),
        "outputs": {
            "tracked_video": str(output_path),
            "frames_jsonl": str(json_path) if export_json else "",
            "segments_json": str(segments_path),
            "clips": [item.get("output_video", "") for item in segments if item.get("output_video")],
            "review_sheet": str(review_sheet_path or ""),
            "review_index": str(run_dir / "tracking_review_index.json") if review_sheet_path else "",
            "frame_features_jsonl": frame_feature_outputs["jsonl"],
            "frame_features_npz": frame_feature_outputs["npz"],
            "frame_feature_manifest": frame_feature_outputs["manifest"],
        },
        "limitations": [
            "String observations are review-only; model/color observations are fused with forward/backward-checked optical flow and propagation is capped by string_max_propagation_frames.",
            "string_without_yoyo marks frames where a visible string estimate persists while the yoyo is out of frame or occluded; these frames require manual review.",
            "not_visible_or_occluded does not distinguish occlusion from an off-camera yoyo without manual review.",
            "Segments are heuristic candidates; only approved valid segments become clip-tokens and irrelevant intervals are excluded.",
        ],
    }
    run_manifest_path = run_dir / "run.json"
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        trick_token_outputs = export_trick_tokens(segments_path)
    except Exception as exc:
        logger.warning("Could not export trick clip-token manifest: %s", exc)
        trick_token_outputs = {"jsonl": "", "manifest": "", "token_count": 0}
    run_manifest["outputs"].update(
        {
            "trick_tokens_jsonl": trick_token_outputs["jsonl"],
            "trick_token_manifest": trick_token_outputs["manifest"],
        }
    )
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Tracking complete: frames=%s video=%s metadata=%s segments=%s", processed_frames, output_path, json_path if export_json else "disabled", len(segments))
    return {
        "source_video": str(source_video_path),
        "output_video": str(output_path),
        "metadata_jsonl": str(json_path) if export_json else "",
        "segments_json": str(segments_path),
        "review_sheet": str(review_sheet_path or ""),
        "frame_features_jsonl": frame_feature_outputs["jsonl"],
        "frame_features_npz": frame_feature_outputs["npz"],
        "frame_feature_manifest": frame_feature_outputs["manifest"],
        "trick_tokens_jsonl": trick_token_outputs["jsonl"],
        "trick_token_manifest": trick_token_outputs["manifest"],
        "trick_token_count": trick_token_outputs["token_count"],
        "segments": segments,
        "run_manifest": str(run_manifest_path),
        "run_dir": str(run_dir),
        "bad_case_counts": dict(sorted(bad_case_counts.items())),
        "weights": str(weights_path),
        "pose_weights": pose_error or str(pose_weights_path or "") if enable_pose else "",
        "string_model": string_model_status,
        "frame_count": processed_frames,
        "fps": fps,
        "confidence": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track yoyo objects and export reviewable frame metadata.")
    parser.add_argument("video", help="Input video path.")
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.weights_path))
    parser.add_argument("--output-dir", default=str(TRACKING_CONFIG.output_dir))
    parser.add_argument("--conf", type=float, default=TRACKING_CONFIG.confidence)
    parser.add_argument("--iou", type=float, default=TRACKING_CONFIG.iou)
    parser.add_argument("--imgsz", type=int, default=TRACKING_CONFIG.imgsz)
    parser.add_argument("--device", default=TRACKING_CONFIG.device)
    parser.add_argument("--pose-weights", default="")
    parser.add_argument("--pose", action="store_true", help="Run optional YOLO pose inference for wrists/body landmarks.")
    parser.add_argument("--auto-download-pose", action="store_true")
    parser.add_argument("--string-weights", default=str(TRACKING_CONFIG.string_weights_path))
    parser.add_argument("--no-string-model", action="store_true")
    parser.add_argument("--string-conf", type=float, default=TRACKING_CONFIG.string_confidence)
    parser.add_argument("--string-max-propagation-frames", type=int, default=TRACKING_CONFIG.string_max_propagation_frames)
    parser.add_argument("--string-flow-fb-max-error", type=float, default=TRACKING_CONFIG.string_flow_fb_max_error)
    parser.add_argument("--string-fusion-distance-px", type=float, default=TRACKING_CONFIG.string_fusion_distance_px)
    parser.add_argument(
        "--string-attachment-class",
        choices=["hand_and_yoyo_attached", "yoyo_detached", "hand_detached", "unknown"],
        default=TRACKING_CONFIG.string_attachment_class,
    )
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument(
        "--activity-speed-diagonal-per-s",
        type=float,
        default=TRACKING_CONFIG.activity_speed_diagonal_per_s,
        help="Minimum active-trick yoyo speed in image diagonals per second.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--padding-seconds", type=float, default=0.4)
    parser.add_argument("--min-segment-seconds", type=float, default=0.5)
    parser.add_argument("--max-gap-seconds", type=float, default=0.4)
    parser.add_argument("--max-segment-seconds", type=float, default=TRACKING_CONFIG.max_segment_seconds)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = track_video(
        source_video_path=args.video,
        weights_path=args.weights,
        output_dir=args.output_dir,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        pose_weights_path=args.pose_weights or None,
        enable_pose=args.pose,
        auto_download_pose=args.auto_download_pose,
        string_weights_path=args.string_weights,
        enable_string_model=not args.no_string_model,
        string_confidence=args.string_conf,
        string_max_propagation_frames=args.string_max_propagation_frames,
        string_flow_fb_max_error=args.string_flow_fb_max_error,
        string_fusion_distance_px=args.string_fusion_distance_px,
        string_attachment_class=args.string_attachment_class,
        export_json=not args.no_json,
        export_clips=not args.no_clips,
        activity_speed_diagonal_per_s=args.activity_speed_diagonal_per_s,
        padding_seconds=args.padding_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_gap_seconds=args.max_gap_seconds,
        max_segment_seconds=args.max_segment_seconds,
        start_seconds=args.start_seconds,
        max_frames=args.max_frames,
    )
    logger.info("Done: %s", result["output_video"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
