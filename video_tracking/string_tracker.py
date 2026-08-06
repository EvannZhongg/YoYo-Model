"""Conservative visual string tracking helpers.

This is deliberately a reviewable baseline, not a string ground-truth model.
It combines a saturated-line observation with Lucas-Kanade propagation and
returns an explicit confidence/method so downstream training can gate it.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


_COLOR_SEMANTIC_SUPPORT_KERNEL = np.ones((31, 31), dtype=np.uint8)
_COLOR_MASK_KERNEL = np.ones((3, 3), dtype=np.uint8)
_COLOR_HSV_LOWER = np.asarray([35, 70, 55], dtype=np.uint8)
_COLOR_HSV_UPPER = np.asarray([179, 255, 255], dtype=np.uint8)
_COLOR_MASK_DEPENDENCY_RADIUS = 4


def update_adaptive_string_domain_gate(
    history: list[tuple[bool, float, float]],
    observation: dict[str, Any] | None,
    frame_width: int,
    frame_height: int,
    window_size: int,
    max_color_accepts: int,
    max_mean_confidence: float,
    min_mean_distance_ratio: float,
) -> tuple[list[tuple[bool, float, float]], bool, dict[str, float | int]]:
    """Detect a persistent low-confidence, weak-color string domain."""
    diagonal = max(1.0, math.hypot(float(frame_width), float(frame_height)))
    value = observation or {}
    updated = [
        *history,
        (
            str(value.get("method") or "") == "semantic_color_probability_union",
            float(value.get("confidence") or 0.0),
            float(value.get("distance_to_yoyo_px") or 0.0) / diagonal,
        ),
    ][-max(1, int(window_size)) :]
    color_accepts = sum(int(item[0]) for item in updated)
    mean_confidence = float(np.mean([item[1] for item in updated]))
    mean_distance_ratio = float(np.mean([item[2] for item in updated]))
    metrics: dict[str, float | int] = {
        "observations": len(updated),
        "color_accepts": color_accepts,
        "mean_confidence": mean_confidence,
        "mean_distance_ratio": mean_distance_ratio,
    }
    triggered = bool(
        len(updated) == max(1, int(window_size))
        and color_accepts <= max(0, int(max_color_accepts))
        and mean_confidence < float(max_mean_confidence)
        and mean_distance_ratio > float(min_mean_distance_ratio)
    )
    return updated, triggered, metrics


def _clip_points(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [
        [
            max(0.0, min(float(width - 1), float(point[0]))),
            max(0.0, min(float(height - 1), float(point[1]))),
        ]
        for point in points
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]


def _resample_polyline(points: list[list[float]], count: int) -> np.ndarray:
    """Return evenly spaced points so observations with different widths align."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(values) == 1 or count <= 1:
        return np.repeat(values[:1], max(1, count), axis=0)
    distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-6:
        return np.repeat(values[:1], count, axis=0)
    samples = np.linspace(0.0, float(cumulative[-1]), count)
    return np.stack(
        (
            np.interp(samples, cumulative, values[:, 0]),
            np.interp(samples, cumulative, values[:, 1]),
        ),
        axis=1,
    ).astype(np.float32)


def _saturated_line_mask(
    roi: np.ndarray,
    semantic_support: np.ndarray | None = None,
) -> np.ndarray:
    """Build the color mask, cropping only work that semantic support removes."""
    crop_x1 = crop_y1 = 0
    crop_x2, crop_y2 = roi.shape[1], roi.shape[0]
    if semantic_support is not None:
        if semantic_support.shape != roi.shape[:2]:
            raise ValueError("semantic support must match the color ROI")
        support_x, support_y, support_width, support_height = cv2.boundingRect(
            semantic_support,
        )
        if support_width <= 0 or support_height <= 0:
            return np.zeros(roi.shape[:2], dtype=np.uint8)
        radius = _COLOR_MASK_DEPENDENCY_RADIUS
        crop_x1 = max(0, support_x - radius)
        crop_y1 = max(0, support_y - radius)
        crop_x2 = min(roi.shape[1], support_x + support_width + radius)
        crop_y2 = min(roi.shape[0], support_y + support_height + radius)

    color_roi = roi[crop_y1:crop_y2, crop_x1:crop_x2]
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, _COLOR_HSV_LOWER, _COLOR_HSV_UPPER)
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_OPEN, _COLOR_MASK_KERNEL, iterations=1,
    )
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_CLOSE, _COLOR_MASK_KERNEL, iterations=1,
    )
    if semantic_support is None:
        return color_mask

    color_mask = cv2.bitwise_and(
        color_mask,
        semantic_support[crop_y1:crop_y2, crop_x1:crop_x2],
    )
    result = np.zeros(roi.shape[:2], dtype=np.uint8)
    result[crop_y1:crop_y2, crop_x1:crop_x2] = color_mask
    return result


def _color_line_observation(
    frame: np.ndarray,
    yoyo: dict[str, Any],
    require_yoyo_proximity: bool,
    mark_far_ambiguous: bool = False,
    reference_points: list[list[float]] | None = None,
    semantic_probability: np.ndarray | None = None,
    semantic_meta: Any | None = None,
    semantic_min_probability: float = 0.10,
) -> dict[str, Any] | None:
    """Find a saturated line segment in the yoyo-centered search region."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in yoyo["bbox"]]
    center = tuple(float(value) for value in yoyo["center"])
    scale = max(x2 - x1, y2 - y1, 12.0)
    margin = max(80.0, 10.0 * scale)
    rx1 = max(0, int(x1 - margin))
    ry1 = max(0, int(y1 - margin))
    rx2 = min(width, int(x2 + margin))
    ry2 = min(height, int(y2 + margin))
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    support = None
    if semantic_probability is not None and semantic_meta is not None:
        probability = np.asarray(semantic_probability, dtype=np.float32)
        px1 = max(0, int(math.floor(rx1 * semantic_meta.scale + semantic_meta.pad_x)))
        py1 = max(0, int(math.floor(ry1 * semantic_meta.scale + semantic_meta.pad_y)))
        px2 = min(
            probability.shape[1],
            int(math.ceil(rx2 * semantic_meta.scale + semantic_meta.pad_x)),
        )
        py2 = min(
            probability.shape[0],
            int(math.ceil(ry2 * semantic_meta.scale + semantic_meta.pad_y)),
        )
        if px2 <= px1 or py2 <= py1:
            return None
        support = (probability[py1:py2, px1:px2] >= float(semantic_min_probability)).astype(
            np.uint8
        )
        support = cv2.resize(
            support, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST,
        )
        # A 15 px source-space radius retains weak string edges while removing
        # unrelated saturated stage lines before the expensive Hough search.
        support = cv2.dilate(support, _COLOR_SEMANTIC_SUPPORT_KERNEL, iterations=1)
    # Strings are commonly saturated and brighter than the black stage. This
    # intentionally errs on the side of missing a line rather than inventing it.
    mask = _saturated_line_mask(roi, support)
    diag = math.hypot(roi.shape[1], roi.shape[0])
    lines = cv2.HoughLinesP(
        mask,
        1.0,
        np.pi / 180.0,
        threshold=max(12, int(diag * 0.025)),
        minLineLength=max(12, int(diag * 0.035)),
        maxLineGap=max(8, int(diag * 0.02)),
    )
    if lines is None:
        return None
    reference_pair = (
        _resample_polyline(reference_points, 2)
        if reference_points and len(reference_points) >= 2
        else None
    )
    best: tuple[float, list[list[float]], float, float, float] | None = None
    for line in lines[:, 0, :]:
        ax, ay, bx, by = [float(value) for value in line]
        a = [ax + rx1, ay + ry1]
        b = [bx + rx1, by + ry1]
        da = math.hypot(a[0] - center[0], a[1] - center[1])
        db = math.hypot(b[0] - center[0], b[1] - center[1])
        near, far, near_distance = (a, b, da) if da <= db else (b, a, db)
        length = math.hypot(a[0] - b[0], a[1] - b[1])
        if require_yoyo_proximity and near_distance > max(30.0, 2.5 * scale):
            continue
        midpoint = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        edge_distance = min(midpoint[0], midpoint[1], width - midpoint[0], height - midpoint[1])
        distance_penalty = 0.35 if require_yoyo_proximity else (0.28 if mark_far_ambiguous else 0.05)
        edge_penalty = 1.5 * max(0.0, 2.5 * scale - edge_distance)
        temporal_penalty = 0.0
        if reference_pair is not None:
            direct_distance = (
                math.hypot(
                    far[0] - reference_pair[0, 0], far[1] - reference_pair[0, 1]
                )
                + math.hypot(
                    near[0] - reference_pair[1, 0], near[1] - reference_pair[1, 1]
                )
            )
            reversed_distance = (
                math.hypot(
                    far[0] - reference_pair[1, 0], far[1] - reference_pair[1, 1]
                )
                + math.hypot(
                    near[0] - reference_pair[0, 0], near[1] - reference_pair[0, 1]
                )
            )
            temporal_penalty = 0.225 * min(direct_distance, reversed_distance)
        score = length - distance_penalty * near_distance - edge_penalty - temporal_penalty
        confidence = min(0.72, max(0.18, 0.18 + length / max(1.0, diag)))
        candidate = (score, [far, near], confidence, near_distance, edge_distance)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    spatially_ambiguous = bool(mark_far_ambiguous and best[3] > max(120.0, 6.0 * scale))
    return {
        "points": best[1],
        "confidence": round(float(min(best[2], 0.24) if spatially_ambiguous else best[2]), 4),
        "method": "color_hough_observation",
        "needs_review": True,
        "distance_to_yoyo_px": round(float(best[3]), 2),
        "distance_to_frame_edge_px": round(float(best[4]), 2),
        "spatially_ambiguous": spatially_ambiguous,
    }


def propagate_optical_flow(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    previous_points: list[list[float]] | None,
    width: int,
    height: int,
    max_forward_backward_error: float = 4.0,
    allow_full_frame_fallback: bool = True,
) -> dict[str, Any] | None:
    if previous_gray is None or not previous_points or len(previous_points) < 2:
        return None
    # A two-point centerline is easy to lose at an endpoint. Track interior
    # samples as well, then return them as a short polyline for later fusion.
    track_points = _resample_polyline(previous_points, max(4, min(16, len(previous_points) * 4)))

    def calculate(
        previous_image: np.ndarray,
        current_image: np.ndarray,
        offset: np.ndarray,
        region: str,
        region_fraction: float,
    ) -> dict[str, Any] | None:
        local_track_points = track_points - offset
        p0 = local_track_points.reshape(-1, 1, 2)
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            previous_image,
            current_image,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < max(2, len(track_points) - 2):
            return None
        local_points = p1.reshape(-1, 2)
        mean_error = float(np.mean(error.reshape(-1)[valid])) if error is not None else 10.0
        if not np.isfinite(mean_error) or mean_error > 35.0:
            return None
        # Forward-backward consistency rejects drift onto a nearby background edge.
        reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            current_image,
            previous_image,
            local_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if reverse is None or reverse_status is None:
            return None
        backward_valid = valid & reverse_status.reshape(-1).astype(bool)
        if int(backward_valid.sum()) < max(2, int(valid.sum()) - 1):
            return None
        fb_error = np.linalg.norm(reverse.reshape(-1, 2) - local_track_points, axis=1)
        mean_fb_error = float(np.mean(fb_error[backward_valid]))
        if not np.isfinite(mean_fb_error) or mean_fb_error > max(0.5, float(max_forward_backward_error)):
            return None
        points = local_points + offset
        clipped = _clip_points(points.tolist(), width, height)
        if len(clipped) < 2:
            return None
        confidence = min(0.72, max(0.20, 0.72 - mean_error / 100.0 - mean_fb_error / 25.0))
        return {
            "points": clipped,
            "confidence": round(float(confidence), 4),
            "method": "lucas_kanade_optical_flow",
            "needs_review": True,
            "flow_error": round(mean_error, 3),
            "flow_forward_backward_error": round(mean_fb_error, 3),
            "flow_valid_point_ratio": round(float(backward_valid.sum() / max(1, len(track_points))), 4),
            "flow_region": region,
            "flow_region_fraction": round(float(region_fraction), 6),
        }

    margin = 192
    x1 = max(0, int(math.floor(float(track_points[:, 0].min()))) - margin)
    y1 = max(0, int(math.floor(float(track_points[:, 1].min()))) - margin)
    x2 = min(width, int(math.ceil(float(track_points[:, 0].max()))) + margin + 1)
    y2 = min(height, int(math.ceil(float(track_points[:, 1].max()))) + margin + 1)
    region_fraction = float(max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height))
    if x2 > x1 and y2 > y1 and region_fraction < 0.80:
        offset = np.asarray([x1, y1], dtype=np.float32)
        result = calculate(
            previous_gray[y1:y2, x1:x2],
            gray[y1:y2, x1:x2],
            offset,
            "roi",
            region_fraction,
        )
        if result is not None:
            return result
        if not allow_full_frame_fallback:
            return None
    return calculate(
        previous_gray,
        gray,
        np.zeros(2, dtype=np.float32),
        "full_frame_fallback" if region_fraction < 0.80 else "full_frame",
        1.0,
    )


def _propagate_string_geometry(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    previous_string: dict[str, Any],
    width: int,
    height: int,
    max_forward_backward_error: float,
    allow_full_frame_fallback: bool,
) -> dict[str, Any] | None:
    polylines = previous_string.get("polylines") or [previous_string.get("points") or []]
    propagated_components = []
    for index, points in enumerate(polylines):
        propagated = propagate_optical_flow(
            previous_gray,
            gray,
            points,
            width,
            height,
            max_forward_backward_error=max_forward_backward_error,
            allow_full_frame_fallback=allow_full_frame_fallback,
        )
        # The first polyline is the yoyo-side primary component. Losing it
        # removes the observation anchor and requires semantic reacquisition.
        if index == 0 and propagated is None:
            return None
        if propagated is not None:
            propagated_components.append(propagated)
    if not propagated_components:
        return None
    result = dict(propagated_components[0])
    result["points"] = propagated_components[0]["points"]
    result["polylines"] = [item["points"] for item in propagated_components]
    result["confidence"] = round(
        float(np.mean([float(item.get("confidence", 0.0)) for item in propagated_components])),
        4,
    )
    result["flow_component_count"] = len(propagated_components)
    result["flow_source_component_count"] = len(polylines)
    result["flow_partial_component_loss"] = len(propagated_components) < len(polylines)
    result["flow_forward_backward_error"] = round(
        max(float(item.get("flow_forward_backward_error", 0.0)) for item in propagated_components),
        3,
    )
    result["flow_regions"] = [str(item.get("flow_region", "unknown")) for item in propagated_components]
    if previous_string.get("component_selection"):
        result["source_component_selection"] = previous_string["component_selection"]
    if previous_string.get("hand_supported_component_count") is not None:
        result["source_hand_supported_component_count"] = int(
            previous_string["hand_supported_component_count"]
        )
    return result


def _annotate_observation(observation: dict[str, Any], propagation_age: int = 0) -> dict[str, Any]:
    result = dict(observation)
    result["propagation_age_frames"] = int(max(0, propagation_age))
    result.setdefault("source_methods", [str(result.get("method", "unknown"))])
    return result


def _valid_geometry_sequence(value: Any, minimum_points: int) -> list[list[float]]:
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
    return points if len(points) >= minimum_points else []


def _observation_geometry(observation: dict[str, Any]) -> list[tuple[list[list[float]], bool]]:
    """Collect review geometry as (points, closed) without inventing connections."""
    geometry: list[tuple[list[list[float]], bool]] = []
    points = _valid_geometry_sequence(observation.get("points"), 1)
    if points:
        geometry.append((points, False))
    for polyline in observation.get("polylines") or []:
        points = _valid_geometry_sequence(polyline, 1)
        if points:
            geometry.append((points, False))
    polygons = observation.get("polygons") or (
        [observation["polygon"]] if observation.get("polygon") else []
    )
    for polygon in polygons:
        points = _valid_geometry_sequence(polygon, 3)
        if points:
            geometry.append((points, True))
    return geometry


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = max(0.0, min(1.0, float(np.dot(point - start, segment) / length_squared)))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def _geometry_to_wrist_distance(
    geometry: list[tuple[list[list[float]], bool]],
    wrists: list[dict[str, Any]],
) -> float | None:
    distances: list[float] = []
    for wrist in wrists:
        try:
            point = np.asarray([float(wrist["x"]), float(wrist["y"])], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(point).all():
            continue
        for points, closed in geometry:
            array = np.asarray(points, dtype=np.float32)
            if closed and cv2.pointPolygonTest(array, (float(point[0]), float(point[1])), False) >= 0:
                distances.append(0.0)
                continue
            distances.extend(float(np.linalg.norm(point - vertex)) for vertex in array)
            pairs = list(zip(array, array[1:]))
            if closed:
                pairs.append((array[-1], array[0]))
            distances.extend(_point_to_segment_distance(point, start, end) for start, end in pairs)
    return min(distances) if distances else None


def _annotate_hand_anchor(
    observation: dict[str, Any],
    wrists: list[dict[str, Any]],
    yoyo_division: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    result = dict(observation)
    threshold = max(48.0, 0.025 * math.hypot(width, height))
    distance = None
    status = "not_applicable"
    mismatch = False
    result.update(
        {
            "hand_anchor_status": status,
            "distance_to_nearest_wrist_px": round(distance, 2) if distance is not None else None,
            "hand_anchor_threshold_px": round(threshold, 2),
            "hand_anchor_mismatch": mismatch,
        }
    )
    if mismatch:
        result["needs_review"] = True
    return result


def estimate_string(
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    wrists: list[dict[str, Any]],
    previous_gray: np.ndarray | None,
    previous_string: dict[str, Any] | None,
    yoyo_division: str = "1A",
    observation: dict[str, Any] | None = None,
    max_propagation_frames: int = 12,
    max_forward_backward_error: float = 4.0,
    allow_color_fallback: bool = True,
    allow_unanchored_semantic: bool = False,
    current_gray: np.ndarray | None = None,
    previous_frame: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Estimate a string while preserving uncertainty in the returned record."""
    width, height = frame.shape[1], frame.shape[0]

    def finalize(result: dict[str, Any]) -> dict[str, Any]:
        return _annotate_hand_anchor(result, wrists, yoyo_division, width, height)

    observed = _annotate_observation(observation) if observation is not None else None
    propagated = None
    # Fresh model/color geometry is authoritative. Optical flow is only useful
    # when the current frame has no observation to carry across the gap.
    if observed is None and previous_string and previous_string.get("points"):
        previous_age = int(previous_string.get("propagation_age_frames", 0))
        if previous_age < max(0, int(max_propagation_frames)):
            gray = current_gray if current_gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_previous_gray = previous_gray
            if flow_previous_gray is None and previous_frame is not None:
                flow_previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            propagated = _propagate_string_geometry(
                flow_previous_gray,
                gray,
                previous_string,
                width,
                height,
                max_forward_backward_error,
                yoyo is None,
            )
            if propagated is not None:
                propagated = _annotate_observation(propagated, previous_age + 1)
    if observed is None and yoyo is not None and allow_color_fallback:
        color_observation = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=False,
            mark_far_ambiguous=False,
            reference_points=(propagated or {}).get("points"),
        )
        if color_observation is not None:
            observed = _annotate_observation(color_observation)
    # A full-frame semantic model can produce background components before the
    # first yoyo detection. Without either a yoyo anchor or an existing track,
    # accepting that proposal creates false strings in intro/outro frames. A
    # previously anchored track is still allowed to persist while the yoyo is
    # temporarily occluded or out of frame.
    if (
        observed is not None
        and yoyo is None
        and previous_string is None
        and not allow_unanchored_semantic
        and str(observed.get("method", "")) == "semantic_segmentation"
    ):
        observed = None
    if observed is not None:
        return finalize(observed)
    if propagated is not None:
        return finalize(propagated)
    # Wrist/yoyo proximity is useful metadata, but it is not visual evidence
    # that a string segment is present between those points.
    return None
