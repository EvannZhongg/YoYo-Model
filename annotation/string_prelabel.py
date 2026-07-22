"""Generate review-only color-mask proposals for visible yoyo strings.

This is an assistive prelabeler, never a source of training truth. It uses a
small ROI around the reviewed VLM geometry and keeps the original annotation
status unchanged so every proposal still passes visual review.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from annotation.video_frame_annotator import draw_visualization


def _points(annotation: dict[str, Any]) -> list[list[float]]:
    values: list[list[float]] = []
    polylines = annotation.get("string_polylines_pixel")
    if not polylines and annotation.get("string_polyline_pixel"):
        polylines = [annotation["string_polyline_pixel"]]
    for stroke in polylines or []:
        if isinstance(stroke, list):
            values.extend(point for point in stroke if isinstance(point, list) and len(point) == 2)
    for point in (annotation.get("hands_pixel") or {}).values():
        if isinstance(point, list) and len(point) == 2:
            values.append(point)
    bbox = annotation.get("yoyo_bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        values.append([(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2])
    return values


def _mask_for_annotation(image: np.ndarray, annotation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    anchors = _points(annotation)
    if len(anchors) < 1:
        return np.zeros((height, width), dtype=np.uint8), {"reason": "no_geometry_anchors"}
    points = np.asarray(anchors, dtype=np.float32)
    margin = max(80, int(round(0.07 * np.hypot(width, height))))
    x1 = max(0, int(points[:, 0].min()) - margin)
    y1 = max(0, int(points[:, 1].min()) - margin)
    x2 = min(width, int(points[:, 0].max()) + margin)
    y2 = min(height, int(points[:, 1].max()) + margin)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = cv2.inRange(hsv, np.array([22, 35, 45], np.uint8), np.array([65, 255, 255], np.uint8))

    # Include low-saturation bright pixels only in a narrow corridor around the
    # VLM line. This recovers white strings without turning clothing into masks.
    corridor = np.zeros((height, width), dtype=np.uint8)
    polylines = annotation.get("string_polylines_pixel")
    if not polylines and annotation.get("string_polyline_pixel"):
        polylines = [annotation["string_polyline_pixel"]]
    for stroke in polylines or []:
        if isinstance(stroke, list) and len(stroke) >= 2:
            cv2.polylines(corridor, [np.asarray(stroke, np.float32).round().astype(np.int32)], False, 255, max(12, int(round(0.004 * np.hypot(width, height)))))
    bright = cv2.inRange(hsv, np.array([0, 15, 65], np.uint8), np.array([179, 255, 255], np.uint8))
    low_saturation = cv2.bitwise_and(bright, corridor)
    mask = cv2.bitwise_or(saturated, low_saturation)
    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi)

    # Hands and the yoyo are anchors, not string regions. Removing their cores
    # avoids training skin/ball masks while preserving line pixels around them.
    for hand in (annotation.get("hands_pixel") or {}).values():
        if isinstance(hand, list) and len(hand) == 2:
            cv2.circle(mask, (int(hand[0]), int(hand[1])), max(18, int(round(0.012 * np.hypot(width, height)))), 0, -1)
    bbox = annotation.get("yoyo_bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        cv2.rectangle(mask, tuple(int(round(float(v))) for v in bbox), 0, -1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    min_area = max(20, int(0.000002 * width * height))
    max_area = max(min_area + 1, int(0.012 * (x2 - x1) * (y2 - y1)))
    kept = 0
    for index in range(1, component_count):
        _, _, comp_width, comp_height, area = stats[index]
        if min_area <= area <= max_area and max(comp_width, comp_height) >= 8:
            clean[components == index] = 255
            kept += 1
    roi_area = max(1, (x2 - x1) * (y2 - y1))
    mask_pixels = int(cv2.countNonZero(clean))
    return clean, {
        "roi_pixel": [x1, y1, x2, y2],
        "hsv_range": [22, 35, 45, 65, 255, 255],
        "kept_components": kept,
        "mask_pixels": mask_pixels,
        "mask_roi_ratio": round(mask_pixels / roi_area, 6),
    }


def _mask_polygons(mask: np.ndarray, width: int, height: int) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        epsilon = max(0.8, 0.015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        polygons.append([[round(float(x), 2), round(float(y), 2)] for x, y in polygon])
    return polygons


def _draw_mask_preview(source: Path, annotation: dict[str, Any], output: Path) -> None:
    draw_visualization(source, annotation, output)


def prelabel_annotation(label_path: Path, dataset_dir: Path, force: bool = False) -> dict[str, Any]:
    annotation = json.loads(label_path.read_text(encoding="utf-8"))
    if annotation.get("string_review_status") in {"approved", "reviewed", "rejected"} and not force:
        return {"label": str(label_path), "status": "skipped_reviewed"}
    if annotation.get("string_visibility") not in {"visible", "partial"}:
        return {"label": str(label_path), "status": "skipped_visibility"}
    source = Path(annotation["source_image"])
    image = cv2.imread(str(source))
    if image is None:
        return {"label": str(label_path), "status": "source_missing"}
    mask, details = _mask_for_annotation(image, annotation)
    polygons = _mask_polygons(mask, image.shape[1], image.shape[0])
    quality_reason = ""
    if not polygons:
        quality_reason = "no_mask"
    elif int(details.get("kept_components", 0)) > 15:
        quality_reason = "too_many_components"
    elif float(details.get("mask_roi_ratio", 0.0)) > 0.015:
        quality_reason = "mask_area_too_large"
    if quality_reason:
        annotation.pop("string_mask_polygons_pixel", None)
        annotation["string_prelabel"] = {
            "status": quality_reason,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        label_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        relative = label_path.relative_to(dataset_dir / "annotations" / "labels")
        output = dataset_dir / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
        draw_visualization(source, annotation, output)
        return {"label": str(label_path), "status": quality_reason, **details}
    annotation["string_mask_polygons_pixel"] = polygons
    annotation["string_prelabel"] = {
        "status": "auto_color_mask_needs_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "hsv_roi_color_components",
        **details,
    }
    annotation["string_review_status"] = "auto_labeled_needs_review"
    annotation["review_status"] = "partially_reviewed"
    label_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    relative = label_path.relative_to(dataset_dir / "annotations" / "labels")
    output = dataset_dir / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
    _draw_mask_preview(source, annotation, output)
    return {"label": str(label_path), "status": "updated", "polygon_count": len(polygons), **details}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create review-only color-mask proposals for yoyo strings.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    paths = sorted((dataset_dir / "annotations" / "labels").rglob("*.json"))
    selected = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if args.split != "all" and data.get("split") != args.split:
            continue
        selected.append(path)
    if args.limit > 0:
        selected = selected[: args.limit]
    counts: dict[str, int] = {}
    for path in selected:
        result = prelabel_annotation(path, dataset_dir, args.force)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        print(json.dumps(result, ensure_ascii=False))
    print(json.dumps({"counts": counts, "selected": len(selected)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
