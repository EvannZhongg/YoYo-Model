"""Convert reviewed string centerlines into a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from config import STRING_SEGMENTATION_CONFIG


ACCEPTED_REVIEW = {"approved", "reviewed"}
POSITIVE_VISIBILITY = {"visible", "partial"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _centerline_polygon(
    points: list[list[float]], width: int, height: int, line_width_px: int
) -> list[list[float]] | None:
    if len(points) < 2 or width <= 0 or height <= 0:
        return None
    array = np.asarray(points, dtype=np.float32)
    array[:, 0] = np.clip(array[:, 0], 0, width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    adaptive_width = max(int(line_width_px), int(round(np.hypot(width, height) * 0.0015)))
    cv2.polylines(mask, [array.round().astype(np.int32)], False, 255, adaptive_width, cv2.LINE_8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, adaptive_width * 0.12)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    return [[float(x) / width, float(y) / height] for x, y in polygon]


def _reviewed_mask_polygons(annotation: dict[str, Any], width: int, height: int) -> list[list[list[float]]]:
    polygons = annotation.get("string_mask_polygons_pixel")
    if not isinstance(polygons, list):
        return []
    normalized = []
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x = max(0.0, min(float(width - 1), float(point[0]))) / width
            y = max(0.0, min(float(height - 1), float(point[1]))) / height
            points.append([x, y])
        if len(points) >= 3:
            normalized.append(points)
    return normalized


def prepare_string_dataset(
    annotations_dir: Path,
    output_dir: Path,
    line_width_px: int = 8,
    clear: bool = False,
) -> dict[str, Any]:
    labels_root = annotations_dir / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Annotation labels not found: {labels_root}")
    if clear and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(labels_root.rglob("*.json"))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped: list[dict[str, str]] = []
    source_groups: dict[str, set[str]] = defaultdict(set)
    attachment_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    used_annotations: list[Path] = []

    for label_path in label_paths:
        try:
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"label": str(label_path), "reason": f"invalid_json: {exc}"})
            continue
        review = str(annotation.get("string_review_status", "auto_labeled_needs_review")).lower()
        if review not in ACCEPTED_REVIEW:
            skipped.append({"label": str(label_path), "reason": f"string_review_status={review}"})
            continue
        split = str(annotation.get("split", "")).lower()
        if split not in {"train", "val", "test"}:
            skipped.append({"label": str(label_path), "reason": f"invalid_split={split}"})
            continue
        visibility = str(annotation.get("string_visibility", "uncertain")).lower()
        if visibility == "uncertain":
            skipped.append({"label": str(label_path), "reason": "string_visibility=uncertain"})
            continue
        source = Path(str(annotation.get("source_image", "")))
        if not source.exists():
            skipped.append({"label": str(label_path), "reason": f"source_image_missing={source}"})
            continue
        image_size = annotation.get("image_size") or []
        if len(image_size) != 2:
            skipped.append({"label": str(label_path), "reason": "invalid_image_size"})
            continue
        width, height = int(image_size[0]), int(image_size[1])
        polygon = None
        polygons: list[list[list[float]]] = []
        if visibility in POSITIVE_VISIBILITY:
            polygons = _reviewed_mask_polygons(annotation, width, height)
            if not polygons:
                polylines = annotation.get("string_polylines_pixel")
                if not polylines and annotation.get("string_polyline_pixel"):
                    polylines = [annotation["string_polyline_pixel"]]
                polygons = [
                    polygon
                    for points in (polylines or [])
                    if (polygon := _centerline_polygon(points, width, height, line_width_px)) is not None
                ]
            if not polygons:
                skipped.append({"label": str(label_path), "reason": "reviewed_visible_string_without_valid_polyline"})
                continue
        elif visibility != "not_visible":
            skipped.append({"label": str(label_path), "reason": f"unsupported_string_visibility={visibility}"})
            continue

        video_id = str(annotation.get("video_id") or annotation.get("source_group") or "unknown")
        relative = Path(split) / video_id / source.name
        image_target = output_dir / "images" / relative
        yolo_target = output_dir / "labels" / relative.with_suffix(".txt")
        image_target.parent.mkdir(parents=True, exist_ok=True)
        yolo_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_target)
        line = ""
        if polygons:
            line = "".join(
                f"0 {' '.join(f'{value:.6f}' for point in polygon for value in point)}\n"
                for polygon in polygons
            )
            counts[split]["positive"] += 1
            counts[split]["instances"] += len(polygons)
        else:
            counts[split]["negative"] += 1
        yolo_target.write_text(line, encoding="utf-8")
        counts[split]["total"] += 1
        source_groups[split].add(str(annotation.get("source_group") or video_id))
        attachment_class = str(annotation.get("string_attachment_class", "unknown"))
        attachment_class_counts[split][attachment_class] += 1
        used_annotations.append(label_path)

    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(source_groups[left] & source_groups[right])
        if shared:
            overlaps[f"{left}_{right}"] = shared
    if overlaps:
        raise ValueError(f"source_group leakage detected: {overlaps}")

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["string"],
    }
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "task": "segment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_dir": str(annotations_dir.resolve()),
        "annotations_sha256": _annotation_digest(used_annotations),
        "output_dir": str(output_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "line_width_px": int(line_width_px),
        "accepted_review_statuses": sorted(ACCEPTED_REVIEW),
        "counts": {split: dict(values) for split, values in sorted(counts.items())},
        "used_annotation_count": len(used_annotations),
        "source_groups": {split: sorted(groups) for split, groups in sorted(source_groups.items())},
        "string_attachment_class_counts": {
            split: dict(sorted(values.items())) for split, values in sorted(attachment_class_counts.items())
        },
        "skipped": skipped,
        "label_semantics": {
            "visible_or_partial": "reviewed mask polygons when available, otherwise centerline buffered into a thin segmentation polygon",
            "not_visible": "reviewed negative image with an empty label file",
            "uncertain": "excluded",
            "string_attachment_class": "optional reviewed metadata; never used as segmentation inclusion/exclusion logic",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-gated YOLO segmentation dataset for yoyo string.")
    parser.add_argument("--annotations-dir", default=str(STRING_SEGMENTATION_CONFIG.annotations_dir))
    parser.add_argument("--output-dir", default=str(STRING_SEGMENTATION_CONFIG.dataset_dir))
    parser.add_argument("--line-width-px", type=int, default=STRING_SEGMENTATION_CONFIG.line_width_px)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_string_dataset(Path(args.annotations_dir), Path(args.output_dir), args.line_width_px, args.clear)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
