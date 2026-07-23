"""Small, explicit status editor for review-gated frame annotations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = {"auto_labeled_needs_review", "partially_reviewed", "reviewed", "approved", "rejected", "unresolved"}
COMPONENTS = {"all", "bbox", "string"}
STRING_ATTACHMENT_CLASSES = {"hand_and_yoyo_attached", "yoyo_detached", "hand_detached", "unknown"}
STRING_VISIBILITY = {"visible", "partial", "not_visible", "uncertain"}
YOYO_VISIBILITY = {"visible", "partially_visible", "occluded", "out_of_frame", "absent", "uncertain"}
SCENE_LABELS = {"trick", "transition", "non_trick", "unknown"}
ACCEPTED_REVIEW = {"approved", "reviewed"}


def _overall_status(data: dict) -> str:
    """Derive the frame status from independent bbox/string decisions."""
    bbox_status = data.get("bbox_review_status", data.get("review_status", "auto_labeled_needs_review"))
    string_status = data.get("string_review_status", data.get("review_status", "auto_labeled_needs_review"))
    if bbox_status in {"approved", "reviewed"} and string_status in {"approved", "reviewed"}:
        return "approved" if bbox_status == string_status == "approved" else "reviewed"
    if bbox_status == string_status == "rejected":
        return "rejected"
    if bbox_status == string_status == "unresolved":
        return "unresolved"
    return "partially_reviewed"


def _valid_bbox(data: dict) -> bool:
    bbox = data.get("yoyo_bbox_pixel")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        boxes = data.get("bbox") or []
        bbox = boxes[0].get("bbox_pixel") if boxes and isinstance(boxes[0], dict) else None
    try:
        return bool(
            isinstance(bbox, list)
            and len(bbox) == 4
            and float(bbox[2]) > float(bbox[0])
            and float(bbox[3]) > float(bbox[1])
        )
    except (TypeError, ValueError):
        return False


def _valid_string_geometry(data: dict) -> bool:
    strokes = data.get("string_polylines_pixel")
    if not strokes and data.get("string_polyline_pixel"):
        strokes = [data["string_polyline_pixel"]]
    for stroke in strokes or []:
        if isinstance(stroke, list) and len(stroke) >= 2:
            return True
    for polygon in data.get("string_mask_polygons_pixel") or []:
        if isinstance(polygon, list) and len(polygon) >= 3:
            return True
    return False


def validate_review_gate(data: dict, component: str) -> list[str]:
    """Return human-readable reasons that prevent a component becoming training truth."""
    issues: list[str] = []
    if component in {"all", "bbox"}:
        visibility = str(data.get("visibility", "uncertain"))
        has_bbox = _valid_bbox(data)
        if visibility not in YOYO_VISIBILITY:
            issues.append(f"unsupported yoyo visibility: {visibility}")
        elif visibility in {"visible", "partially_visible"} and not has_bbox:
            issues.append(f"yoyo visibility={visibility} requires a valid bbox")
        elif visibility in {"absent", "out_of_frame"} and has_bbox:
            issues.append(f"yoyo visibility={visibility} must not retain a bbox")
        elif visibility == "uncertain":
            issues.append("yoyo visibility must be resolved before review approval")

    if component in {"all", "string"}:
        visibility = str(data.get("string_visibility", "uncertain"))
        has_geometry = _valid_string_geometry(data)
        if visibility not in STRING_VISIBILITY:
            issues.append(f"unsupported string visibility: {visibility}")
        elif visibility in {"visible", "partial"} and not has_geometry:
            issues.append(f"string visibility={visibility} requires a reviewed stroke or mask")
        elif visibility == "not_visible" and has_geometry:
            issues.append("string visibility=not_visible must not retain string geometry")
        elif visibility == "uncertain":
            issues.append("string visibility must be resolved before review approval")
    return issues


def _append_review_event(annotation_path: Path, data: dict, component: str, status: str) -> None:
    annotations_root = next((parent for parent in annotation_path.resolve().parents if parent.name == "annotations"), None)
    if annotations_root is None:
        return
    dataset_root = annotations_root.parent
    try:
        label_path = str(annotation_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        label_path = str(annotation_path.resolve())
    event = {
        "created_at_utc": data["reviewed_at_utc"],
        "label_path": label_path,
        "component": component,
        "status": status,
        "reviewer": data.get("reviewer", "manual"),
        "reason": data.get("review_notes", ""),
        "string_visibility": data.get("string_visibility"),
        "yoyo_visibility": data.get("visibility"),
        "scene_label": data.get("scene_label", "unknown"),
    }
    with (dataset_root / "manual_review_log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _refresh_visualization(annotation_path: Path, data: dict) -> None:
    """Keep the stored review overlay in sync with the accepted JSON state."""
    annotations_root = next((parent for parent in annotation_path.resolve().parents if parent.name == "annotations"), None)
    source_image = Path(str(data.get("source_image", "")))
    if annotations_root is None or not source_image.is_file():
        return
    labels_root = annotations_root / "labels"
    try:
        relative = annotation_path.resolve().relative_to(labels_root.resolve())
    except ValueError:
        return
    from annotation.video_frame_annotator import draw_visualization

    output_path = annotations_root / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
    draw_visualization(source_image, data, output_path)


def update_annotation_status(
    path: str | Path,
    status: str,
    reviewer: str = "manual",
    notes: str | None = None,
    component: str = "all",
    string_attachment_class: str | None = None,
    string_visibility: str | None = None,
    yoyo_visibility: str | None = None,
    scene_label: str | None = None,
    clear_string_mask: bool = False,
) -> dict:
    if status not in ALLOWED:
        raise ValueError(f"Unsupported review status: {status}")
    if component not in COMPONENTS:
        raise ValueError(f"Unsupported review component: {component}")
    if string_attachment_class is not None and string_attachment_class not in STRING_ATTACHMENT_CLASSES:
        raise ValueError(f"Unsupported string attachment class: {string_attachment_class}")
    if string_visibility is not None and string_visibility not in STRING_VISIBILITY:
        raise ValueError(f"Unsupported string visibility: {string_visibility}")
    if yoyo_visibility is not None and yoyo_visibility not in YOYO_VISIBILITY:
        raise ValueError(f"Unsupported yoyo visibility: {yoyo_visibility}")
    if scene_label is not None and scene_label not in SCENE_LABELS:
        raise ValueError(f"Unsupported scene label: {scene_label}")
    annotation_path = Path(path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if component in {"all", "bbox"} and yoyo_visibility is not None:
        data["visibility"] = yoyo_visibility
        if yoyo_visibility in {"absent", "out_of_frame"}:
            data["yoyo_bbox_pixel"] = None
            data["yoyo_bbox_2d"] = None
            data["bbox"] = []
    if scene_label is not None:
        data["scene_label"] = scene_label
        bad_case = set(data.get("bad_case") or [])
        bad_case.discard("non_trick_scene")
        bad_case.discard("transition_scene")
        if scene_label == "non_trick":
            bad_case.add("non_trick_scene")
        elif scene_label == "transition":
            bad_case.add("transition_scene")
        data["bad_case"] = sorted(bad_case)
    if component in {"all", "string"} and string_attachment_class is not None:
        data["string_attachment_class"] = string_attachment_class
    if component in {"all", "string"} and string_visibility is not None:
        data["string_visibility"] = string_visibility
        if string_visibility == "not_visible":
            for key in (
                "string_polylines_pixel",
                "string_polylines_2d",
                "string_polyline_pixel",
                "string_polyline_2d",
                "string_mask_polygons_pixel",
                "string_prelabel",
            ):
                data.pop(key, None)
            data["bad_case"] = sorted(set(data.get("bad_case", []) + ["string_not_visible"]))
    if component in {"all", "string"} and clear_string_mask:
        data.pop("string_mask_polygons_pixel", None)
        data.pop("string_prelabel", None)
    if status in ACCEPTED_REVIEW:
        issues = validate_review_gate(data, component)
        if issues:
            raise ValueError("review gate failed: " + "; ".join(issues))
    if component == "all":
        data["review_status"] = status
        data["bbox_review_status"] = status
        data["string_review_status"] = status
    else:
        data[f"{component}_review_status"] = status
        data["review_status"] = _overall_status(data)
    data["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["reviewer"] = reviewer
    if notes is not None:
        data["review_notes"] = notes
    annotation_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _refresh_visualization(annotation_path, data)
    _append_review_event(annotation_path, data, component, status)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Set the review status of one annotation JSON file.")
    parser.add_argument("label", help="Path to one annotation JSON file.")
    parser.add_argument("status", choices=sorted(ALLOWED))
    parser.add_argument("--reviewer", default="manual")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--component", choices=sorted(COMPONENTS), default="all")
    parser.add_argument("--string-attachment-class", choices=sorted(STRING_ATTACHMENT_CLASSES), default=None)
    parser.add_argument("--string-visibility", choices=sorted(STRING_VISIBILITY), default=None)
    parser.add_argument("--yoyo-visibility", choices=sorted(YOYO_VISIBILITY), default=None)
    parser.add_argument("--scene-label", choices=sorted(SCENE_LABELS), default=None)
    parser.add_argument("--clear-string-mask", action="store_true")
    args = parser.parse_args()
    path = Path(args.label)
    data = update_annotation_status(
        path,
        args.status,
        args.reviewer,
        args.notes,
        args.component,
        args.string_attachment_class,
        args.string_visibility,
        args.yoyo_visibility,
        args.scene_label,
        args.clear_string_mask,
    )
    print(f"updated {path}: {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
