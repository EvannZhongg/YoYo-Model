"""Presentation-neutral tracking review view models."""

from __future__ import annotations

from pathlib import Path

from video_tracking.frame_review import tracking_review_gallery_items


def tracking_review_caption(record: dict) -> str:
    string = record.get("string") or {}
    method = str(string.get("method") or "none")
    confidence = string.get("confidence")
    confidence_text = f"{float(confidence):.3f}" if confidence is not None else "-"
    components = int(
        string.get("component_count")
        or string.get("flow_component_count")
        or len(string.get("polylines") or [])
        or int(bool(string.get("points")))
    )
    anchor_status = str(string.get("hand_anchor_status") or "-")
    distance = string.get("distance_to_nearest_wrist_px")
    threshold = string.get("hand_anchor_threshold_px")
    anchor_distance = (
        f" {float(distance):.0f}/{float(threshold):.0f}px"
        if distance is not None and threshold is not None
        else ""
    )
    pose_person = record.get("pose_person") or {}
    pose_status = str(pose_person.get("status") or "missing")
    if pose_person.get("needs_review"):
        reasons = ",".join(str(value) for value in (pose_person.get("review_reasons") or []))
        pose_status = f"review:{reasons or 'identity'}"
    bad_cases = ",".join(str(value) for value in (record.get("bad_case") or [])) or "ok"
    return (
        f"f{int(record.get('frame_index', 0))} | {float(record.get('timestamp_s', 0.0)):.2f}s"
        f" | yoyo={'yes' if record.get('yoyo') else 'no'}"
        f" | string={method}:{confidence_text} components={components}"
        f" | hand={anchor_status}{anchor_distance} | pose={pose_status} | bad={bad_cases}"
    )


def tracking_review_gallery(run_dir: str | Path | None) -> list[tuple[str, str]]:
    if not run_dir:
        return []
    return [
        (item["image"], f"{tracking_review_caption(item['frame'])} | view={item['view']}")
        for item in tracking_review_gallery_items(run_dir)
    ]
