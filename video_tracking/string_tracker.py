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


def _orient_like(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resolve the endpoint ordering ambiguity before fusing two polylines."""
    if len(points) != len(reference) or len(points) == 0:
        return points
    direct = float(np.linalg.norm(points[0] - reference[0]) + np.linalg.norm(points[-1] - reference[-1]))
    reversed_distance = float(np.linalg.norm(points[0] - reference[-1]) + np.linalg.norm(points[-1] - reference[0]))
    return points[::-1].copy() if reversed_distance < direct else points


def _color_line_observation(
    frame: np.ndarray,
    yoyo: dict[str, Any],
    require_yoyo_proximity: bool,
    mark_far_ambiguous: bool = False,
    reference_points: list[list[float]] | None = None,
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
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Strings are commonly saturated and brighter than the black stage. This
    # intentionally errs on the side of missing a line rather than inventing it.
    mask = cv2.inRange(hsv, np.array([35, 70, 55], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
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
        if reference_points and len(reference_points) >= 2:
            observed_pair = np.asarray([far, near], dtype=np.float32)
            reference_pair = _resample_polyline(reference_points, 2)
            observed_pair = _orient_like(observed_pair, reference_pair)
            temporal_penalty = 0.45 * float(np.mean(np.linalg.norm(observed_pair - reference_pair, axis=1)))
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
) -> dict[str, Any] | None:
    if previous_gray is None or not previous_points or len(previous_points) < 2:
        return None
    # A two-point centerline is easy to lose at an endpoint. Track interior
    # samples as well, then return them as a short polyline for later fusion.
    track_points = _resample_polyline(previous_points, max(4, min(16, len(previous_points) * 4)))
    p0 = track_points.reshape(-1, 1, 2)
    p1, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
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
    points = p1.reshape(-1, 2)
    mean_error = float(np.mean(error.reshape(-1)[valid])) if error is not None else 10.0
    if not np.isfinite(mean_error) or mean_error > 35.0:
        return None
    # Forward-backward consistency rejects drift onto a nearby background edge.
    reverse, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        points.reshape(-1, 1, 2),
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
    fb_error = np.linalg.norm(reverse.reshape(-1, 2) - track_points, axis=1)
    mean_fb_error = float(np.mean(fb_error[backward_valid]))
    if not np.isfinite(mean_fb_error) or mean_fb_error > max(0.5, float(max_forward_backward_error)):
        return None
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
    }


def _annotate_observation(observation: dict[str, Any], propagation_age: int = 0) -> dict[str, Any]:
    result = dict(observation)
    result["propagation_age_frames"] = int(max(0, propagation_age))
    result.setdefault("source_methods", [str(result.get("method", "unknown"))])
    return result


def _fuse_observations(
    observation: dict[str, Any],
    propagated: dict[str, Any],
    width: int,
    height: int,
    max_distance_px: float,
) -> dict[str, Any] | None:
    observation_points = observation.get("points") or []
    propagated_points = propagated.get("points") or []
    if len(observation_points) < 2 or len(propagated_points) < 2:
        return None
    count = max(2, min(16, max(len(observation_points), len(propagated_points))))
    observed = _resample_polyline(observation_points, count)
    flow = _orient_like(_resample_polyline(propagated_points, count), observed)
    disagreement = float(np.mean(np.linalg.norm(observed - flow, axis=1)))
    distance_limit = max(1.0, float(max_distance_px))
    if disagreement > distance_limit:
        return None
    observation_confidence = float(observation.get("confidence", 0.0))
    flow_confidence = float(propagated.get("confidence", 0.0))
    total = max(1e-6, observation_confidence + flow_confidence)
    weight_observation = observation_confidence / total
    fused_points = observed * weight_observation + flow * (1.0 - weight_observation)
    result = dict(observation)
    result.update(
        {
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in fused_points],
            "method": "temporal_fusion",
            "confidence": round(
                min(0.9, max(observation_confidence, flow_confidence) + 0.08 * (1.0 - disagreement / distance_limit)),
                4,
            ),
            "needs_review": True,
            "source_methods": [str(observation.get("method", "observation")), str(propagated.get("method", "optical_flow"))],
            "fusion_disagreement_px": round(disagreement, 3),
            "flow_forward_backward_error": propagated.get("flow_forward_backward_error"),
            "propagation_age_frames": 0,
        }
    )
    return result


def estimate_string(
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    wrists: list[dict[str, Any]],
    previous_gray: np.ndarray | None,
    previous_string: dict[str, Any] | None,
    attachment_class: str = "unknown",
    observation: dict[str, Any] | None = None,
    max_propagation_frames: int = 12,
    max_forward_backward_error: float = 4.0,
    fusion_distance_px: float = 48.0,
) -> dict[str, Any] | None:
    """Estimate a string while preserving uncertainty in the returned record."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width, height = frame.shape[1], frame.shape[0]
    propagated = None
    if previous_string and previous_string.get("points"):
        previous_age = int(previous_string.get("propagation_age_frames", 0))
        if previous_age < max(0, int(max_propagation_frames)):
            propagated = propagate_optical_flow(
                previous_gray,
                gray,
                previous_string["points"],
                width,
                height,
                max_forward_backward_error=max_forward_backward_error,
            )
            if propagated is not None:
                propagated = _annotate_observation(propagated, previous_age + 1)
    observed = _annotate_observation(observation) if observation is not None else None
    if observed is None and yoyo is not None:
        color_observation = _color_line_observation(
            frame,
            yoyo,
            require_yoyo_proximity=attachment_class in {"hand_and_yoyo_attached", "hand_detached"},
            mark_far_ambiguous=attachment_class == "unknown",
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
        and str(observed.get("method", "")) == "semantic_segmentation"
    ):
        observed = None
    if observed is not None and propagated is not None:
        fused = _fuse_observations(observed, propagated, width, height, fusion_distance_px)
        if fused is not None:
            return fused
        # A fresh observation anchors the track after disagreement, but keep
        # the conflict visible for manual review instead of hiding it.
        if float(observed.get("confidence", 0.0)) >= float(propagated.get("confidence", 0.0)) or int(propagated.get("propagation_age_frames", 0)) >= 3:
            observed = dict(observed)
            observed.update(
                {
                    "temporal_conflict": True,
                    "fusion_disagreement_px": round(
                        float(np.mean(np.linalg.norm(_resample_polyline(observed["points"], 2) - _orient_like(_resample_polyline(propagated["points"], 2), _resample_polyline(observed["points"], 2)), axis=1))),
                        3,
                    ),
                }
            )
            return observed
        propagated["temporal_conflict"] = True
        return propagated
    if observed is not None:
        return observed
    if propagated is not None:
        return propagated
    if yoyo is not None and wrists and attachment_class == "hand_and_yoyo_attached":
        center = yoyo["center"]
        wrist = min(wrists, key=lambda item: (item["x"] - center[0]) ** 2 + (item["y"] - center[1]) ** 2)
        distance = math.hypot(wrist["x"] - center[0], wrist["y"] - center[1])
        if distance >= 8:
            return _annotate_observation({
                "points": [[float(wrist["x"]), float(wrist["y"])], [float(center[0]), float(center[1])]],
                "confidence": 0.20,
                "method": "hand_to_yoyo_geometric_prior",
                "needs_review": True,
            })
    return None
