"""Metrics for evaluating tracking outputs on a consecutive annotated dataset.

The evaluator deliberately works on the final review geometry in ``frames.jsonl``.
For strings, it compares sampled centerlines instead of the buffered segmentation
mask, so the tolerance is expressed in source-image pixels.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


KNOWN_REVIEW_STATES = {"reviewed", "approved", "confirmed"}
ORIENTATION_CLASSES = ("horizontal", "normal", "not_applicable")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON: {path}") from exc


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    x1, y1, x2, y2 = result
    return result if x2 > x1 and y2 > y1 else None


def _points(value: Any, minimum: int = 2) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            result.append([x, y])
    return result if len(result) >= minimum else []


def _polylines(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, (list, tuple)):
        return []
    if _points(value, 2):
        return [_points(value, 2)]
    result = []
    for item in value:
        line = _points(item, 2)
        if line:
            result.append(line)
    return result


def _annotation_polylines(annotation: dict[str, Any]) -> list[list[list[float]]]:
    lines = _polylines(annotation.get("string_polylines_pixel"))
    if lines:
        return lines
    lines = _polylines(annotation.get("string_polyline_pixel"))
    if lines:
        return lines
    paths = []
    for path in (annotation.get("string_path") or {}).get("paths") or []:
        points = _points(path.get("points_pixel"), 2) if isinstance(path, dict) else []
        if points:
            paths.append(points)
    return paths


def _prediction_polylines(prediction: dict[str, Any] | None) -> list[list[list[float]]]:
    if not isinstance(prediction, dict):
        return []
    lines = _polylines(prediction.get("polylines"))
    if lines:
        return lines
    return _polylines(prediction.get("points"))


def sample_polyline(points: Iterable[Iterable[float]], spacing_px: float = 2.0) -> np.ndarray:
    """Sample a polyline at approximately uniform source-pixel spacing."""
    values = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(values) == 1:
        return values.copy()
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return values[:1].copy()
    spacing = max(0.25, float(spacing_px))
    count = max(2, int(math.ceil(total / spacing)) + 1)
    samples = np.linspace(0.0, total, count)
    return np.stack(
        (
            np.interp(samples, cumulative, values[:, 0]),
            np.interp(samples, cumulative, values[:, 1]),
        ),
        axis=1,
    ).astype(np.float32)


def sample_polylines(lines: Iterable[Iterable[Iterable[float]]], spacing_px: float = 2.0) -> np.ndarray:
    sampled = [sample_polyline(line, spacing_px) for line in lines]
    sampled = [item for item in sampled if len(item)]
    return np.concatenate(sampled, axis=0) if sampled else np.empty((0, 2), dtype=np.float32)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(target) == 0:
        return np.full((len(source),), np.inf, dtype=np.float32)
    result = np.empty((len(source),), dtype=np.float32)
    for start in range(0, len(source), 2048):
        chunk = source[start : start + 2048]
        distances = np.linalg.norm(chunk[:, None, :] - target[None, :, :], axis=2)
        result[start : start + len(chunk)] = distances.min(axis=1)
    return result


def centerline_pair_metrics(
    target_lines: Iterable[Iterable[Iterable[float]]],
    prediction_lines: Iterable[Iterable[Iterable[float]]],
    tolerance_px: Iterable[float] = (2.0, 4.0, 8.0),
    spacing_px: float = 2.0,
) -> dict[str, Any]:
    """Return symmetric centerline distances and tolerance coverage."""
    target = sample_polylines(target_lines, spacing_px)
    prediction = sample_polylines(prediction_lines, spacing_px)
    target_to_prediction = _nearest_distances(target, prediction)
    prediction_to_target = _nearest_distances(prediction, target)
    distances = np.concatenate((target_to_prediction, prediction_to_target))
    finite = distances[np.isfinite(distances)]
    result: dict[str, Any] = {
        "target_samples": int(len(target)),
        "prediction_samples": int(len(prediction)),
        "target_to_prediction_mean_px": None if not len(target_to_prediction) else round(float(np.mean(target_to_prediction)), 4),
        "prediction_to_target_mean_px": None if not len(prediction_to_target) else round(float(np.mean(prediction_to_target)), 4),
        "chamfer_mean_px": None if not len(distances) or not len(finite) else round(float(np.mean(distances)), 4),
        "hd95_px": None if not len(distances) or not len(finite) else round(float(np.percentile(distances, 95)), 4),
        "tolerances": {},
    }
    for tolerance in tolerance_px:
        value = float(tolerance)
        target_hits = int(np.count_nonzero(target_to_prediction <= value))
        prediction_hits = int(np.count_nonzero(prediction_to_target <= value))
        precision = prediction_hits / len(prediction) if len(prediction) else 0.0
        recall = target_hits / len(target) if len(target) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["tolerances"][str(value).rstrip("0").rstrip(".")] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "target_hits": target_hits,
            "prediction_hits": prediction_hits,
        }
    return result


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _bbox_center(box: list[float]) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _presence_summary(rows: list[dict[str, Any]], target_key: str, prediction_key: str) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        target = row.get(target_key)
        prediction = row.get(prediction_key)
        if target is None:
            continue
        if target and prediction:
            tp += 1
        elif prediction:
            fp += 1
        elif target:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "known_frames": tp + fp + fn + tn,
    }


def _longest_missing_streak(rows: list[dict[str, Any]], target_key: str, prediction_key: str) -> dict[str, int]:
    current = longest = gaps = 0
    for row in rows:
        if row.get(target_key) is not True:
            current = 0
            continue
        if row.get(prediction_key):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
            if current == 1:
                gaps += 1
    return {"longest_missing_streak_frames": longest, "missing_episode_count": gaps}


def _recovery_summary(rows: list[dict[str, Any]], target_key: str, prediction_key: str) -> dict[str, Any]:
    """Summarize how quickly a visible target returns after a prediction gap."""
    ordered = sorted(rows, key=lambda item: (str(item.get("source_group", "")), int(item["frame_index"])))
    latencies: list[int] = []
    unresolved = 0
    current_group: str | None = None
    missing_frames = 0
    for row in ordered:
        group = str(row.get("source_group", ""))
        if group != current_group:
            unresolved += int(missing_frames > 0)
            missing_frames = 0
            current_group = group
        if row.get(target_key) is not True:
            unresolved += int(missing_frames > 0)
            missing_frames = 0
        elif row.get(prediction_key):
            if missing_frames:
                latencies.append(missing_frames)
                missing_frames = 0
        else:
            missing_frames += 1
    unresolved += int(missing_frames > 0)
    return {
        "recovered_episode_count": len(latencies),
        "unrecovered_episode_count": unresolved,
        "mean_recovery_latency_frames": round(float(np.mean(latencies)), 4) if latencies else None,
        "max_recovery_latency_frames": max(latencies) if latencies else None,
    }


def _track_identity_summary(rows: list[dict[str, Any]], track_id_key: str) -> dict[str, int]:
    """Count selected-target tracker ID changes across each source sequence."""
    ordered = sorted(rows, key=lambda item: (str(item.get("source_group", "")), int(item["frame_index"])))
    previous_group: str | None = None
    previous_track_id: int | str | None = None
    switches = assigned = 0
    unique_ids: set[tuple[str, str]] = set()
    for row in ordered:
        group = str(row.get("source_group", ""))
        if group != previous_group:
            previous_track_id = None
            previous_group = group
        track_id = row.get(track_id_key)
        if track_id is None:
            continue
        assigned += 1
        unique_ids.add((group, str(track_id)))
        if previous_track_id is not None and track_id != previous_track_id:
            switches += 1
        previous_track_id = track_id
    return {
        "track_id_assigned_frames": assigned,
        "unique_track_ids": len(unique_ids),
        "track_id_switch_count": switches,
    }


def _sequence_motion(rows: list[dict[str, Any]], target_box_key: str, prediction_box_key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: (str(item.get("source_group", "")), int(item["frame_index"])))
    vector_errors: list[float] = []
    acceleration_errors: list[float] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    previous_velocity: tuple[np.ndarray, np.ndarray] | None = None
    for row in ordered:
        target = row.get(target_box_key)
        prediction = row.get(prediction_box_key)
        if not target or not prediction:
            previous = None
            previous_velocity = None
            continue
        target_center = _bbox_center(target)
        prediction_center = _bbox_center(prediction)
        if previous is not None:
            target_delta = target_center - previous[0]
            prediction_delta = prediction_center - previous[1]
            vector_errors.append(float(np.linalg.norm(prediction_delta - target_delta)))
            if previous_velocity is not None:
                target_acceleration = target_delta - previous_velocity[0]
                prediction_acceleration = prediction_delta - previous_velocity[1]
                acceleration_errors.append(float(np.linalg.norm(prediction_acceleration - target_acceleration)))
            previous_velocity = (target_delta, prediction_delta)
        previous = (target_center, prediction_center)
    return {
        "matched_motion_pairs": len(vector_errors),
        "mean_motion_vector_error_px": round(float(np.mean(vector_errors)), 4) if vector_errors else None,
        "p95_motion_vector_error_px": round(float(np.percentile(vector_errors, 95)), 4) if vector_errors else None,
        "mean_acceleration_error_px": round(float(np.mean(acceleration_errors)), 4) if acceleration_errors else None,
    }


def _polyline_length(lines: list[list[list[float]]]) -> float:
    total = 0.0
    for line in lines:
        points = np.asarray(line, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 2:
            total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return total


def _string_sequence_motion(rows: list[dict[str, Any]], spacing_px: float) -> dict[str, Any]:
    """Compare frame-to-frame centerline centroid motion and length change."""
    ordered = sorted(rows, key=lambda item: (str(item.get("source_group", "")), int(item["frame_index"])))
    centroid_errors: list[float] = []
    length_change_errors: list[float] = []
    previous: tuple[str, int, np.ndarray, np.ndarray, float, float] | None = None
    for row in ordered:
        target_lines = row.get("target_string") or []
        prediction_lines = row.get("predicted_string") or []
        target_points = sample_polylines(target_lines, spacing_px)
        prediction_points = sample_polylines(prediction_lines, spacing_px)
        group = str(row.get("source_group", ""))
        frame_index = int(row["frame_index"])
        if not len(target_points) or not len(prediction_points):
            previous = None
            continue
        target_centroid = target_points.mean(axis=0)
        prediction_centroid = prediction_points.mean(axis=0)
        target_length = _polyline_length(target_lines)
        prediction_length = _polyline_length(prediction_lines)
        if previous is not None and previous[0] == group and frame_index == previous[1] + 1:
            target_motion = target_centroid - previous[2]
            prediction_motion = prediction_centroid - previous[3]
            centroid_errors.append(float(np.linalg.norm(prediction_motion - target_motion)))
            length_change_errors.append(abs((prediction_length - previous[5]) - (target_length - previous[4])))
        previous = (group, frame_index, target_centroid, prediction_centroid, target_length, prediction_length)
    return {
        "matched_motion_pairs": len(centroid_errors),
        "mean_centroid_motion_error_px": round(float(np.mean(centroid_errors)), 4) if centroid_errors else None,
        "p95_centroid_motion_error_px": round(float(np.percentile(centroid_errors, 95)), 4) if centroid_errors else None,
        "mean_length_change_error_px": round(float(np.mean(length_change_errors)), 4) if length_change_errors else None,
    }


def _load_group_frames(dataset_dir: Path, group_id: str | None) -> list[dict[str, Any]]:
    metadata_path = dataset_dir / "consecutive_groups.json"
    entries: list[dict[str, Any]] = []
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        groups = metadata.get("groups") if isinstance(metadata, dict) else None
        for group in groups or []:
            if not isinstance(group, dict):
                continue
            current_group_id = str(group.get("group_id") or group.get("source_group") or "")
            if group_id and current_group_id != group_id:
                continue
            for frame in group.get("frames") or []:
                if not isinstance(frame, dict):
                    continue
                item = dict(frame)
                item["group_id"] = current_group_id
                item["source_group"] = str(group.get("source_group") or current_group_id)
                entries.append(item)
    if not entries:
        labels_root = dataset_dir / "canonical" / "labels"
        for label_path in sorted(labels_root.rglob("*.json")) if labels_root.is_dir() else []:
            annotation = _load_json(label_path)
            entries.append(
                {
                    "sample_key": str(label_path.relative_to(labels_root).as_posix()),
                    "frame_index": annotation.get("frame_index"),
                    "timestamp_s": annotation.get("timestamp_s"),
                    "group_id": str(annotation.get("source_group") or "default"),
                    "source_group": str(annotation.get("source_group") or "default"),
                }
            )
    if not entries:
        raise ValueError(f"No consecutive frames found under {dataset_dir}")
    result = []
    labels_root = dataset_dir / "canonical" / "labels"
    for entry in entries:
        try:
            frame_index = int(entry["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Frame entry has no integer frame_index: {entry}") from exc
        sample_key = str(entry.get("sample_key") or "")
        label_path = labels_root / Path(sample_key)
        if label_path.suffix.lower() != ".json":
            label_path = label_path.with_suffix(".json")
        if not label_path.is_file():
            raise ValueError(f"Frame label not found for frame {frame_index}: {label_path}")
        annotation = _load_json(label_path)
        result.append(
            {
                "frame_index": frame_index,
                "timestamp_s": entry.get("timestamp_s"),
                "group_id": str(entry.get("group_id") or entry.get("source_group") or "default"),
                "source_group": str(entry.get("source_group") or annotation.get("source_group") or "default"),
                "annotation": annotation,
                "label_path": str(label_path),
            }
        )
    return sorted(result, key=lambda item: (item["source_group"], item["frame_index"]))


def _load_snapshot_frames(snapshot_path: Path, group_id: str | None) -> list[dict[str, Any]]:
    snapshot = _load_json(snapshot_path)
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "yoyo_consecutive_ground_truth_snapshot_v1":
        raise ValueError(f"Unsupported ground-truth snapshot: {snapshot_path}")
    result = []
    for frame in snapshot.get("frames") or []:
        if not isinstance(frame, dict) or not isinstance(frame.get("annotation"), dict):
            continue
        annotation = frame["annotation"]
        source_group = str(annotation.get("source_group") or frame.get("source_group") or "default")
        current_group = str(frame.get("group_id") or source_group)
        if group_id and group_id not in {current_group, source_group}:
            continue
        try:
            frame_index = int(annotation.get("frame_index", frame.get("frame_index")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Snapshot frame has no integer frame_index: {frame}") from exc
        result.append(
            {
                "frame_index": frame_index,
                "timestamp_s": annotation.get("timestamp_s", frame.get("timestamp_s")),
                "group_id": current_group,
                "source_group": source_group,
                "annotation": annotation,
                "label_path": str(frame.get("label_path") or frame.get("sample_key") or "snapshot"),
            }
        )
    if not result:
        raise ValueError(f"No matching frames in ground-truth snapshot: {snapshot_path}")
    return sorted(result, key=lambda item: (item["source_group"], item["frame_index"]))


def _known_targets(annotation: dict[str, Any]) -> tuple[bool, list[float] | None, bool, list[list[list[float]]]]:
    active_yoyo = annotation.get("active_yoyo") or {}
    visibility = str(active_yoyo.get("visibility") or "")
    not_visible_reason = str(active_yoyo.get("not_visible_reason") or "")
    bbox = _valid_bbox(active_yoyo.get("bbox_pixel"))
    bbox_state = str(active_yoyo.get("bbox_review_status") or "")
    if visibility in {"visible", "partial"}:
        yoyo_known = bbox is not None and bbox_state in KNOWN_REVIEW_STATES
    elif visibility == "not_visible":
        yoyo_known = (
            bbox is None
            and bbox_state in KNOWN_REVIEW_STATES
            and not_visible_reason in {"occluded", "out_of_frame", "absent"}
        )
    else:
        yoyo_known = False
    lines = _annotation_polylines(annotation)
    string_visibility = str(annotation.get("string_visibility") or "")
    string_state = str(annotation.get("string_review_status") or annotation.get("review_status") or "")
    string_known = bool(lines) or (string_visibility == "not_visible" and string_state in KNOWN_REVIEW_STATES)
    return yoyo_known, bbox, string_known, lines


def _orientation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row.get("target_orientation") in ORIENTATION_CLASSES]
    recalls: dict[str, float | None] = {}
    confusion: dict[str, dict[str, int]] = {}
    for target in ORIENTATION_CLASSES:
        class_rows = [row for row in known if row["target_orientation"] == target]
        recalls[target] = (
            round(
                sum(row.get("predicted_orientation") == target for row in class_rows) / len(class_rows),
                6,
            )
            if class_rows else None
        )
        confusion[target] = dict(sorted(Counter(
            str(row.get("predicted_orientation") or "unknown") for row in class_rows
        ).items()))

    predicted_switches = target_switches = isolated_flips = 0
    transition_latencies: list[int] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in known:
        by_group.setdefault(str(row.get("source_group") or "default"), []).append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: int(row["frame_index"]))
        targets = [str(row["target_orientation"]) for row in group_rows]
        predictions = [str(row.get("predicted_orientation") or "unknown") for row in group_rows]
        target_switches += sum(left != right for left, right in zip(targets, targets[1:]))
        predicted_switches += sum(left != right for left, right in zip(predictions, predictions[1:]))
        isolated_flips += sum(
            predictions[index - 1] == predictions[index + 1] != predictions[index]
            and targets[index - 1] == targets[index] == targets[index + 1]
            for index in range(1, len(group_rows) - 1)
        )
        transition_indices = [
            index for index in range(1, len(group_rows)) if targets[index] != targets[index - 1]
        ]
        for transition_number, start in enumerate(transition_indices):
            end = transition_indices[transition_number + 1] if transition_number + 1 < len(transition_indices) else len(group_rows)
            recovered = next(
                (index for index in range(start, end) if predictions[index] == targets[start]),
                None,
            )
            if recovered is not None:
                transition_latencies.append(
                    int(group_rows[recovered]["frame_index"]) - int(group_rows[start]["frame_index"])
                )

    valid_recalls = [value for value in recalls.values() if value is not None]
    correct = sum(row.get("predicted_orientation") == row["target_orientation"] for row in known)
    return {
        "known_frames": len(known),
        "accuracy": round(correct / len(known), 6) if known else None,
        "macro_recall": round(float(np.mean(valid_recalls)), 6) if valid_recalls else None,
        "per_class_recall": recalls,
        "confusion_predicted_by_target": confusion,
        "unknown_prediction_frames": sum(not row.get("predicted_orientation") for row in known),
        "temporal": {
            "target_switch_count": target_switches,
            "predicted_switch_count": predicted_switches,
            "excess_switch_count": max(0, predicted_switches - target_switches),
            "isolated_flip_count": isolated_flips,
            "matched_transition_count": len(transition_latencies),
            "mean_transition_latency_frames": (
                round(float(np.mean(transition_latencies)), 4) if transition_latencies else None
            ),
            "max_transition_latency_frames": max(transition_latencies) if transition_latencies else None,
        },
    }


def evaluate_sequence(
    dataset_dir: str | Path,
    predictions_path: str | Path,
    group_id: str | None = None,
    tolerances_px: Iterable[float] = (2.0, 4.0, 8.0),
    sample_spacing_px: float = 2.0,
    include_frames: bool = False,
    ground_truth_snapshot: str | Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    predictions_path = Path(predictions_path).resolve()
    if predictions_path.is_dir():
        predictions_path = predictions_path / "frames.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Tracking frames JSONL not found: {predictions_path}")
    snapshot_path = Path(ground_truth_snapshot).resolve() if ground_truth_snapshot else None
    dataset_frames = (
        _load_snapshot_frames(snapshot_path, group_id)
        if snapshot_path is not None
        else _load_group_frames(dataset_dir, group_id)
    )
    if start_frame is not None or end_frame is not None:
        lower = int(start_frame) if start_frame is not None else None
        upper = int(end_frame) if end_frame is not None else None
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("start_frame must not be greater than end_frame")
        dataset_frames = [
            item
            for item in dataset_frames
            if (lower is None or int(item["frame_index"]) >= lower)
            and (upper is None or int(item["frame_index"]) <= upper)
        ]
        if not dataset_frames:
            raise ValueError("No frames remain after applying the frame range")
    predictions: dict[int, dict[str, Any]] = {}
    method_counts: Counter[str] = Counter()
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read predictions: {predictions_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            frame_index = int(record["frame_index"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid tracking record at line {line_number}") from exc
        predictions[frame_index] = record
        string = record.get("string") or {}
        method = str(string.get("method") or "none")
        method_counts[method] += 1

    rows: list[dict[str, Any]] = []
    for item in dataset_frames:
        annotation = item["annotation"]
        yoyo_known, target_bbox, string_known, target_lines = _known_targets(annotation)
        prediction = predictions.get(item["frame_index"]) or {}
        predicted_yoyo = _valid_bbox((prediction.get("yoyo") or {}).get("bbox"))
        predicted_lines = _prediction_polylines(prediction.get("string"))
        row: dict[str, Any] = {
            "frame_index": item["frame_index"],
            "source_group": item["source_group"],
            "yoyo_known": yoyo_known,
            "target_yoyo": target_bbox,
            "predicted_yoyo": predicted_yoyo,
            "predicted_yoyo_track_id": (prediction.get("yoyo") or {}).get("track_id"),
            "string_known": string_known,
            "target_string": target_lines if string_known else [],
            "predicted_string": predicted_lines,
            "prediction_method": str((prediction.get("string") or {}).get("method") or "none"),
            "string_low_confidence_rescue": bool((prediction.get("string") or {}).get("low_confidence_rescue")),
            "yoyo_low_confidence_rescue": bool(
                (prediction.get("yoyo") or {}).get("selection_source") == "low_confidence_temporal_rescue"
            ),
            "string_propagation_age_frames": int((prediction.get("string") or {}).get("propagation_age_frames") or 0),
            "bad_case": sorted(str(value) for value in (prediction.get("bad_case") or [])),
            "target_orientation": (annotation.get("active_yoyo") or {"trick_orientation": annotation.get("trick_orientation")}).get("trick_orientation"),
            "predicted_orientation": (prediction.get("trick_orientation") or {}).get("label"),
        }
        if yoyo_known:
            row["yoyo_present_target"] = target_bbox is not None
            row["yoyo_present_prediction"] = predicted_yoyo is not None
            if target_bbox and predicted_yoyo:
                row["yoyo_iou"] = _bbox_iou(target_bbox, predicted_yoyo)
                row["yoyo_center_error_px"] = round(float(np.linalg.norm(_bbox_center(target_bbox) - _bbox_center(predicted_yoyo))), 4)
        if string_known:
            row["string_present_target"] = bool(target_lines)
            row["string_present_prediction"] = bool(predicted_lines)
            if target_lines or predicted_lines:
                row["centerline"] = centerline_pair_metrics(target_lines, predicted_lines, tolerances_px, sample_spacing_px)
        rows.append(row)

    yoyo_rows = [row for row in rows if row.get("yoyo_known")]
    string_rows = [row for row in rows if row.get("string_known")]
    yoyo_matches = [row for row in yoyo_rows if row.get("yoyo_iou") is not None]
    string_pairs = [row["centerline"] for row in string_rows if row.get("centerline")]
    yoyo_iou = [float(row["yoyo_iou"]) for row in yoyo_matches]
    yoyo_center = [float(row["yoyo_center_error_px"]) for row in yoyo_matches]
    centerline_summary: dict[str, Any] = {
        "known_frames": len(string_rows),
        "positive_target_frames": sum(bool(row.get("string_present_target")) for row in string_rows),
        "positive_prediction_frames": sum(bool(row.get("string_present_prediction")) for row in string_rows),
        "pair_frames": len(string_pairs),
        "chamfer_mean_px": round(float(np.mean([item["chamfer_mean_px"] for item in string_pairs if item["chamfer_mean_px"] is not None])), 4) if any(item["chamfer_mean_px"] is not None for item in string_pairs) else None,
        "hd95_mean_px": round(float(np.mean([item["hd95_px"] for item in string_pairs if item["hd95_px"] is not None])), 4) if any(item["hd95_px"] is not None for item in string_pairs) else None,
        "tolerances": {},
    }
    for tolerance in tolerances_px:
        key = str(float(tolerance)).rstrip("0").rstrip(".")
        target_samples = sum(item["target_samples"] for item in string_pairs)
        prediction_samples = sum(item["prediction_samples"] for item in string_pairs)
        target_hits = sum(item["tolerances"][key]["target_hits"] for item in string_pairs)
        prediction_hits = sum(item["tolerances"][key]["prediction_hits"] for item in string_pairs)
        precision = prediction_hits / prediction_samples if prediction_samples else 0.0
        recall = target_hits / target_samples if target_samples else 0.0
        centerline_summary["tolerances"][key] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
            "target_samples": target_samples,
            "prediction_samples": prediction_samples,
        }
    result: dict[str, Any] = {
        "schema_version": "yoyo_consecutive_tracking_metrics_v2",
        "task": "consecutive_tracking_evaluation",
        "dataset_dir": str(dataset_dir),
        "ground_truth_snapshot": str(snapshot_path) if snapshot_path is not None else None,
        "predictions": str(predictions_path),
        "group_id": group_id,
        "frame_range": {
            "start": int(start_frame) if start_frame is not None else None,
            "end": int(end_frame) if end_frame is not None else None,
        },
        "frame_count": len(rows),
        "excluded_unknown": {
            "yoyo": len(rows) - len(yoyo_rows),
            "string": len(rows) - len(string_rows),
            "orientation": sum(row.get("target_orientation") not in ORIENTATION_CLASSES for row in rows),
        },
        "yoyo": {
            "presence": _presence_summary(
                [{"target": row.get("yoyo_present_target"), "prediction": row.get("yoyo_present_prediction")} for row in yoyo_rows],
                "target",
                "prediction",
            ),
            "localization": {
                "matched_frames": len(yoyo_matches),
                "mean_iou": round(float(np.mean(yoyo_iou)), 6) if yoyo_iou else None,
                "median_iou": round(float(np.median(yoyo_iou)), 6) if yoyo_iou else None,
                "iou_50_rate": round(float(np.mean(np.asarray(yoyo_iou) >= 0.5)), 6) if yoyo_iou else 0.0,
                "mean_center_error_px": round(float(np.mean(yoyo_center)), 4) if yoyo_center else None,
                "p95_center_error_px": round(float(np.percentile(yoyo_center, 95)), 4) if yoyo_center else None,
            },
            "temporal": {
                **_longest_missing_streak(
                    [{"source_group": row["source_group"], "frame_index": row["frame_index"], "target": row.get("yoyo_present_target"), "prediction": row.get("yoyo_present_prediction")} for row in yoyo_rows],
                    "target",
                    "prediction",
                ),
                **_recovery_summary(
                    [{"source_group": row["source_group"], "frame_index": row["frame_index"], "target": row.get("yoyo_present_target"), "prediction": row.get("yoyo_present_prediction")} for row in yoyo_rows],
                    "target",
                    "prediction",
                ),
                **_track_identity_summary(yoyo_rows, "predicted_yoyo_track_id"),
                **_sequence_motion(
                    [{"source_group": row["source_group"], "frame_index": row["frame_index"], "target": row.get("target_yoyo"), "prediction": row.get("predicted_yoyo")} for row in rows],
                    "target",
                    "prediction",
                ),
            },
        },
        "string": {
            "presence": _presence_summary(
                [{"target": row.get("string_present_target"), "prediction": row.get("string_present_prediction")} for row in string_rows],
                "target",
                "prediction",
            ),
            "centerline": centerline_summary,
            "temporal": {
                **_longest_missing_streak(
                    [{"source_group": row["source_group"], "frame_index": row["frame_index"], "target": row.get("string_present_target"), "prediction": row.get("string_present_prediction")} for row in string_rows],
                    "target",
                    "prediction",
                ),
                **_recovery_summary(
                    [{"source_group": row["source_group"], "frame_index": row["frame_index"], "target": row.get("string_present_target"), "prediction": row.get("string_present_prediction")} for row in string_rows],
                    "target",
                    "prediction",
                ),
                **_string_sequence_motion(string_rows, sample_spacing_px),
            },
            "prediction_method_counts": dict(sorted(method_counts.items())),
            "propagated_frames": sum(row["string_propagation_age_frames"] > 0 for row in rows),
            "temporal_fusion_frames": sum(row["prediction_method"] == "temporal_fusion" for row in rows),
            "low_confidence_rescue_frames": sum(row["string_low_confidence_rescue"] for row in rows),
        },
        "orientation": _orientation_metrics(rows),
        "experiments": {
            "yoyo_low_confidence_rescue_frames": sum(row["yoyo_low_confidence_rescue"] for row in rows),
            "string_low_confidence_rescue_frames": sum(row["string_low_confidence_rescue"] for row in rows),
        },
    }
    if include_frames:
        result["frames"] = rows
    return result
