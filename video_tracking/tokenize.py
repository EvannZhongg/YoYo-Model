"""Convert full-video tracking metadata into fixed-width frame features."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STRING_METHODS = (
    "yolo_segmentation",
    "semantic_segmentation",
    "color_hough_observation",
    "lucas_kanade_optical_flow",
    "temporal_fusion",
    "hand_to_yoyo_geometric_prior",
)
STRING_ATTACHMENT_CLASSES = ("unknown", "hand_and_yoyo_attached", "yoyo_detached", "hand_detached")
STRING_HAND_ANCHOR_STATUSES = (
    "unknown",
    "not_applicable",
    "no_visible_wrist",
    "no_geometry",
    "matched",
    "mismatch",
)
VISIBILITY_STATES = ("visible", "edge_clipped", "likely_out_of_frame", "not_visible_or_occluded")
BAD_CASE_VOCAB = (
    "no_yoyo",
    "multiple_yoyo",
    "low_confidence",
    "edge_clipped",
    "pose_missing",
    "pose_identity_needs_review",
    "string_needs_review",
    "string_low_confidence",
    "string_not_observed",
    "string_tracking_lost",
    "string_without_yoyo",
    "string_temporal_conflict",
    "string_spatially_ambiguous",
    "string_hand_anchor_mismatch",
    "likely_out_of_frame",
    "not_visible_or_occluded",
)
STRING_POINT_COUNT = 8
STRING_COMPONENT_COUNT = 8
STRING_COMPONENT_POINT_COUNT = 4
POSE_POINT_COUNT = 17
ORIENTATION_CLASSES = ("horizontal", "normal", "not_applicable")


def feature_names() -> list[str]:
    names = [
        "dt_s",
        "yoyo_present",
        "yoyo_center_x",
        "yoyo_center_y",
        "yoyo_width",
        "yoyo_height",
        "yoyo_confidence",
        "yoyo_velocity_x_per_s",
        "yoyo_velocity_y_per_s",
        "yoyo_speed_diag_per_s",
    ]
    names.extend(f"visibility_{state}" for state in VISIBILITY_STATES)
    names.extend(["string_present", "string_confidence", "string_length_diag"])
    names.extend(f"string_method_{method}" for method in STRING_METHODS)
    names.extend(f"string_attachment_{name}" for name in STRING_ATTACHMENT_CLASSES)
    for index in range(STRING_POINT_COUNT):
        names.extend((f"string_{index}_x", f"string_{index}_y", f"string_{index}_present"))
    names.extend(
        (
            "string_component_count_ratio",
            "string_hand_supported_component_count_ratio",
            "string_flow_component_count_ratio",
            "string_flow_source_component_count_ratio",
            "string_flow_partial_component_loss",
            "string_distance_to_nearest_wrist_diag",
            "string_hand_anchor_threshold_diag",
        )
    )
    names.extend(f"string_hand_anchor_status_{status}" for status in STRING_HAND_ANCHOR_STATUSES)
    for component_index in range(STRING_COMPONENT_COUNT):
        names.extend(
            (
                f"string_component_{component_index}_present",
                f"string_component_{component_index}_length_diag",
            )
        )
        for point_index in range(STRING_COMPONENT_POINT_COUNT):
            names.extend(
                (
                    f"string_component_{component_index}_{point_index}_x",
                    f"string_component_{component_index}_{point_index}_y",
                    f"string_component_{component_index}_{point_index}_present",
                )
            )
    for side in ("left", "right"):
        names.extend((f"{side}_hand_present", f"{side}_hand_x", f"{side}_hand_y", f"{side}_hand_confidence"))
    for index in range(POSE_POINT_COUNT):
        names.extend((f"pose_{index}_x", f"pose_{index}_y", f"pose_{index}_confidence"))
    names.extend(("orientation_present", "orientation_confidence"))
    names.extend(f"orientation_{name}" for name in ORIENTATION_CLASSES)
    names.append("orientation_age_s")
    names.extend(f"bad_case_{name}" for name in BAD_CASE_VOCAB)
    return names


def _resample_polyline(points: list[list[float]], count: int) -> list[list[float]]:
    if not points:
        return []
    values = np.asarray(points, dtype=np.float32)
    if len(values) == 1:
        return values.repeat(count, axis=0).tolist()
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-6:
        return values[:1].repeat(count, axis=0).tolist()
    samples = np.linspace(0.0, cumulative[-1], count)
    x = np.interp(samples, cumulative, values[:, 0])
    y = np.interp(samples, cumulative, values[:, 1])
    return np.stack((x, y), axis=1).tolist()


def _polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def _valid_polyline(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append([x, y])
    return points if len(points) >= 2 else []


def _string_components(string: dict[str, Any] | None) -> list[list[list[float]]]:
    if not string:
        return []
    components = [
        points
        for value in (string.get("polylines") or [])
        if (points := _valid_polyline(value))
    ]
    if not components:
        primary = _valid_polyline(string.get("points"))
        if primary:
            components = [primary]
    return components


def _count_ratio(value: Any, maximum: int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(float(maximum), numeric)) / max(1, maximum)


def _hand_map(hands: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for hand in hands:
        name = str(hand.get("name", "")).lower()
        if "left" in name:
            result["left"] = hand
        elif "right" in name:
            result["right"] = hand
    return result


def tracking_records_to_features(records: list[dict[str, Any]], width: int, height: int, fps: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if width <= 0 or height <= 0:
        raise ValueError("Tokenization requires a positive video width and height")
    diagonal = math.hypot(width, height)
    vectors: list[list[float]] = []
    json_rows: list[dict[str, Any]] = []
    previous_center: tuple[float, float] | None = None
    previous_timestamp: float | None = None
    for record in records:
        timestamp = float(record.get("timestamp_s", 0.0))
        dt = max(1.0 / max(fps, 1.0), timestamp - previous_timestamp) if previous_timestamp is not None else 1.0 / max(fps, 1.0)
        yoyo = record.get("yoyo") or None
        center = tuple(float(value) for value in yoyo["center"]) if yoyo else None
        bbox = [float(value) for value in yoyo["bbox"]] if yoyo else [0.0, 0.0, 0.0, 0.0]
        velocity_x = ((center[0] - previous_center[0]) / width / dt) if center is not None and previous_center is not None else 0.0
        velocity_y = ((center[1] - previous_center[1]) / height / dt) if center is not None and previous_center is not None else 0.0
        speed_norm = math.hypot((center[0] - previous_center[0]) / dt, (center[1] - previous_center[1]) / dt) / diagonal if center is not None and previous_center is not None else 0.0
        visibility = str((record.get("visibility") or {}).get("state", "not_visible_or_occluded"))
        vector = [
            dt,
            1.0 if yoyo else 0.0,
            center[0] / width if center else 0.0,
            center[1] / height if center else 0.0,
            (bbox[2] - bbox[0]) / width if yoyo else 0.0,
            (bbox[3] - bbox[1]) / height if yoyo else 0.0,
            float(yoyo.get("confidence", 0.0)) if yoyo else 0.0,
            velocity_x,
            velocity_y,
            speed_norm,
        ]
        vector.extend(1.0 if visibility == state else 0.0 for state in VISIBILITY_STATES)
        string = record.get("string") or None
        string_components = _string_components(string)
        string_points = _valid_polyline((string or {}).get("points"))
        if not string_points and string_components:
            string_points = string_components[0]
        resampled = _resample_polyline(string_points, STRING_POINT_COUNT) if string_points else []
        vector.extend((1.0 if string else 0.0, float((string or {}).get("confidence", 0.0)), _polyline_length(string_points) / diagonal))
        method = str((string or {}).get("method", ""))
        vector.extend(1.0 if method == name else 0.0 for name in STRING_METHODS)
        attachment_class = str(record.get("string_attachment_class", "unknown"))
        if attachment_class not in STRING_ATTACHMENT_CLASSES:
            attachment_class = "unknown"
        vector.extend(1.0 if attachment_class == name else 0.0 for name in STRING_ATTACHMENT_CLASSES)
        for index in range(STRING_POINT_COUNT):
            if index < len(resampled):
                vector.extend((resampled[index][0] / width, resampled[index][1] / height, 1.0))
            else:
                vector.extend((0.0, 0.0, 0.0))
        anchor_status = str((string or {}).get("hand_anchor_status", "unknown"))
        if anchor_status not in STRING_HAND_ANCHOR_STATUSES:
            anchor_status = "unknown"
        vector.extend(
            (
                _count_ratio(len(string_components), STRING_COMPONENT_COUNT),
                _count_ratio((string or {}).get("hand_supported_component_count", 0), STRING_COMPONENT_COUNT),
                _count_ratio((string or {}).get("flow_component_count", 0), STRING_COMPONENT_COUNT),
                _count_ratio((string or {}).get("flow_source_component_count", 0), STRING_COMPONENT_COUNT),
                1.0 if (string or {}).get("flow_partial_component_loss") else 0.0,
                float((string or {}).get("distance_to_nearest_wrist_px") or 0.0) / diagonal,
                float((string or {}).get("hand_anchor_threshold_px") or 0.0) / diagonal,
            )
        )
        vector.extend(1.0 if anchor_status == status else 0.0 for status in STRING_HAND_ANCHOR_STATUSES)
        for component_index in range(STRING_COMPONENT_COUNT):
            if component_index < len(string_components):
                component = string_components[component_index]
                component_points = _resample_polyline(component, STRING_COMPONENT_POINT_COUNT)
                vector.extend((1.0, _polyline_length(component) / diagonal))
                for point in component_points:
                    vector.extend((point[0] / width, point[1] / height, 1.0))
            else:
                vector.extend((0.0, 0.0))
                for _ in range(STRING_COMPONENT_POINT_COUNT):
                    vector.extend((0.0, 0.0, 0.0))
        hands = _hand_map(record.get("hands") or [])
        for side in ("left", "right"):
            hand = hands.get(side)
            vector.extend((1.0 if hand else 0.0, float(hand.get("x", 0.0)) / width if hand else 0.0, float(hand.get("y", 0.0)) / height if hand else 0.0, float(hand.get("confidence", 0.0)) if hand else 0.0))
        pose_by_index = {int(point.get("index", -1)): point for point in (record.get("pose") or [])}
        for index in range(POSE_POINT_COUNT):
            point = pose_by_index.get(index)
            vector.extend((float(point.get("x", 0.0)) / width if point else 0.0, float(point.get("y", 0.0)) / height if point else 0.0, float(point.get("confidence", 0.0)) if point else 0.0))
        orientation = record.get("trick_orientation") or None
        orientation_label = str((orientation or {}).get("label", ""))
        if orientation_label not in ORIENTATION_CLASSES:
            orientation_label = ""
        vector.extend(
            (
                1.0 if orientation else 0.0,
                float((orientation or {}).get("confidence", 0.0)),
            )
        )
        vector.extend(1.0 if orientation_label == name else 0.0 for name in ORIENTATION_CLASSES)
        vector.append(float((orientation or {}).get("age_frames", 0)) / max(fps, 1.0))
        bad_cases = set(record.get("bad_case") or [])
        vector.extend(1.0 if name in bad_cases else 0.0 for name in BAD_CASE_VOCAB)
        semantic = {
            "frame_index": int(record.get("frame_index", len(json_rows))),
            "timestamp_s": timestamp,
            "vector": [round(float(value), 7) for value in vector],
            "yoyo_present": bool(yoyo),
            "string_present": bool(string),
            "string_method": method or None,
            "string_attachment_class": attachment_class,
            "string_component_count": len(string_components),
            "string_encoded_component_count": min(len(string_components), STRING_COMPONENT_COUNT),
            "string_hand_supported_component_count": int(
                (string or {}).get("hand_supported_component_count", 0) or 0
            ),
            "string_hand_anchor_status": anchor_status,
            "string_flow_partial_component_loss": bool(
                (string or {}).get("flow_partial_component_loss", False)
            ),
            "visibility": visibility,
            "trick_orientation": orientation_label or None,
            "orientation_confidence": float((orientation or {}).get("confidence", 0.0)),
            "orientation_age_frames": int((orientation or {}).get("age_frames", 0)),
            "bad_case": sorted(bad_cases),
        }
        vectors.append(vector)
        json_rows.append(semantic)
        previous_center = center
        previous_timestamp = timestamp
    array = np.asarray(vectors, dtype=np.float32)
    if array.size and array.shape[1] != len(feature_names()):
        raise AssertionError(f"Frame feature width mismatch: {array.shape[1]} != {len(feature_names())}")
    return array, json_rows


def export_tracking_features(records: list[dict[str, Any]], run_dir: str | Path, width: int, height: int, fps: float) -> dict[str, str]:
    run_dir = Path(run_dir)
    vectors, rows = tracking_records_to_features(records, width, height, fps)
    jsonl_path = run_dir / "frame_features.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    npz_path = run_dir / "frame_features.npz"
    np.savez_compressed(
        npz_path,
        vectors=vectors,
        frame_indices=np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
        timestamps_s=np.asarray([row["timestamp_s"] for row in rows], dtype=np.float32),
        feature_names=np.asarray(feature_names()),
    )
    manifest = {
        "schema_version": "yoyo_tracking_frame_features_v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(rows),
        "feature_count": len(feature_names()),
        "feature_names": feature_names(),
        "string_point_count": STRING_POINT_COUNT,
        "string_component_count": STRING_COMPONENT_COUNT,
        "string_component_point_count": STRING_COMPONENT_POINT_COUNT,
        "string_component_order_policy": "Preserve tracking observation order: yoyo-side primary first, then independently observed hand-supported components; never interpolate across components.",
        "string_methods": list(STRING_METHODS),
        "string_attachment_classes": list(STRING_ATTACHMENT_CLASSES),
        "string_hand_anchor_statuses": list(STRING_HAND_ANCHOR_STATUSES),
        "pose_point_count": POSE_POINT_COUNT,
        "orientation_classes": list(ORIENTATION_CLASSES),
        "bad_case_vocab": list(BAD_CASE_VOCAB),
        "image_size": [width, height],
        "fps": fps,
        "missing_value_policy": "numeric zeros with explicit present/confidence/visibility features",
        "sequence_policy": "Full-video frame features aligned one-to-one with tracking metadata.",
        "outputs": {"jsonl": str(jsonl_path), "npz": str(npz_path)},
    }
    manifest_path = run_dir / "frame_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "npz": str(npz_path), "manifest": str(manifest_path)}


def export_run_features(run_dir: str | Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    width, height = int(run_manifest.get("width", 0)), int(run_manifest.get("height", 0))
    if width <= 0 or height <= 0:
        capture = cv2.VideoCapture(str(run_dir / "tracked.mp4"))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
    return export_tracking_features(records, run_dir, width, height, float(run_manifest.get("fps", 30.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fixed-width frame features from one tracking run.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(export_run_features(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
