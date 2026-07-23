#!/usr/bin/env python3
"""Independent, review-gated rope annotation pipeline.

The script deliberately does not import the host repository. It emits labels that
remain compatible with the repository's yoyo string training code while keeping
stronger provenance, temporal, reconstruction, and review metadata.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont


SCHEMA_VERSION = "rope_annotation_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {"train", "val", "test"}
STRING_VISIBILITY = {"visible", "partial", "not_visible", "uncertain"}
YOYO_VISIBILITY = {"visible", "partially_visible", "occluded", "out_of_frame", "absent", "uncertain"}
ATTACHMENT_CLASSES = {"hand_and_yoyo_attached", "yoyo_detached", "hand_detached", "unknown"}
SCENE_LABELS = {"trick", "transition", "non_trick", "unknown"}
TOPOLOGIES = {"open", "loop", "branched", "multiple", "uncertain"}
RECONSTRUCTION_STATUS = {"complete", "partial", "uncertain", "not_applicable"}
EDGE_EVIDENCE = {"observed", "temporal", "inferred"}
PATH_ANCHORS = {"left_hand", "right_hand", "yoyo", "unknown"}
ACCEPTED_REVIEW = {"approved", "reviewed"}
CORE_FIELDS = (
    "visibility",
    "yoyo_bbox_pixel",
    "string_visibility",
    "string_polylines_pixel",
    "string_mask_polygons_pixel",
    "hands_pixel",
    "string_attachment_class",
    "scene_label",
    "string_path",
    "bad_case",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_digest(label: dict[str, Any]) -> str:
    payload = {key: label.get(key) for key in CORE_FIELDS}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or "images"


def parse_frame_index(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def image_info(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [x, y]


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result if result[2] > result[0] and result[3] > result[1] else None


def normalized_to_pixel(value: list[float], width: int, height: int) -> list[float]:
    return [value[0] / 999.0 * width, value[1] / 999.0 * height]


def pixel_to_normalized(value: list[float], width: int, height: int) -> list[float]:
    return [round(value[0] / width * 999.0, 3), round(value[1] / height * 999.0, 3)]


def normalize_points(values: Any, width: int, height: int, normalized: bool = False) -> list[list[float]]:
    result: list[list[float]] = []
    if not isinstance(values, list):
        return result
    for raw in values:
        parsed = point(raw)
        if parsed is None:
            continue
        if normalized:
            parsed = normalized_to_pixel(parsed, width, height)
        result.append([round(parsed[0], 3), round(parsed[1], 3)])
    return result


def normalize_polylines(candidate: dict[str, Any], width: int, height: int) -> list[list[list[float]]]:
    raw = candidate.get("string_polylines_pixel")
    normalized = False
    if raw is None:
        raw = candidate.get("string_polylines_2d")
        normalized = raw is not None
    if raw is None and candidate.get("string_polyline_pixel") is not None:
        raw = [candidate["string_polyline_pixel"]]
    if raw is None and candidate.get("string_polyline_2d") is not None:
        raw = [candidate["string_polyline_2d"]]
        normalized = True
    if not isinstance(raw, list):
        return []
    strokes = []
    for stroke in raw:
        points = normalize_points(stroke, width, height, normalized)
        if points:
            strokes.append(points)
    return strokes


def normalize_path(raw: Any, width: int, height: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "topology": "uncertain",
            "reconstruction_status": "uncertain",
            "paths": [],
            "unresolved_gaps": [],
        }
    paths = []
    for index, raw_path in enumerate(raw.get("paths") or []):
        if not isinstance(raw_path, dict):
            continue
        points_pixel = normalize_points(raw_path.get("points_pixel"), width, height)
        if not points_pixel and raw_path.get("points_2d"):
            points_pixel = normalize_points(raw_path.get("points_2d"), width, height, True)
        edges = []
        for edge_index, raw_edge in enumerate(raw_path.get("edges") or []):
            if not isinstance(raw_edge, dict):
                continue
            try:
                start = int(raw_edge.get("from", edge_index))
                end = int(raw_edge.get("to", edge_index + 1))
                confidence = max(0.0, min(1.0, float(raw_edge.get("confidence", 0.0))))
            except (TypeError, ValueError):
                continue
            evidence = str(raw_edge.get("evidence", "inferred")).lower()
            if evidence not in EDGE_EVIDENCE:
                evidence = "inferred"
            edges.append({"from": start, "to": end, "evidence": evidence, "confidence": round(confidence, 4)})
        paths.append(
            {
                "path_id": str(raw_path.get("path_id") or f"path-{index + 1}"),
                "start_anchor": str(raw_path.get("start_anchor") or "unknown"),
                "end_anchor": str(raw_path.get("end_anchor") or "unknown"),
                "points_pixel": points_pixel,
                "points_2d": [pixel_to_normalized(item, width, height) for item in points_pixel],
                "edges": edges,
            }
        )
    topology = str(raw.get("topology", "uncertain")).lower()
    status = str(raw.get("reconstruction_status", "uncertain")).lower()
    return {
        "topology": topology if topology in TOPOLOGIES else "uncertain",
        "reconstruction_status": status if status in RECONSTRUCTION_STATUS else "uncertain",
        "paths": paths,
        "unresolved_gaps": [str(item) for item in (raw.get("unresolved_gaps") or []) if str(item).strip()],
    }


def normalize_candidate(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    width, height = [int(item) for item in base["image_size"]]
    result = copy.deepcopy(base)
    string_visibility = str(candidate.get("string_visibility", result.get("string_visibility", "uncertain"))).lower()
    result["string_visibility"] = string_visibility if string_visibility in STRING_VISIBILITY else "uncertain"
    yoyo_visibility = str(candidate.get("visibility", result.get("visibility", "uncertain"))).lower()
    result["visibility"] = yoyo_visibility if yoyo_visibility in YOYO_VISIBILITY else "uncertain"

    strokes = normalize_polylines(candidate, width, height)
    result["string_polylines_pixel"] = strokes or None
    result["string_polylines_2d"] = [
        [pixel_to_normalized(item, width, height) for item in stroke] for stroke in strokes
    ] or None
    result["string_polyline_pixel"] = strokes[0] if strokes else None
    result["string_polyline_2d"] = result["string_polylines_2d"][0] if strokes else None

    raw_bbox = bbox(candidate.get("yoyo_bbox_pixel"))
    if raw_bbox is None:
        normalized_bbox = bbox(candidate.get("yoyo_bbox_2d"))
        if normalized_bbox:
            left_top = normalized_to_pixel(normalized_bbox[:2], width, height)
            right_bottom = normalized_to_pixel(normalized_bbox[2:], width, height)
            raw_bbox = left_top + right_bottom
    result["yoyo_bbox_pixel"] = [round(item, 3) for item in raw_bbox] if raw_bbox else None
    result["yoyo_bbox_2d"] = (
        pixel_to_normalized(raw_bbox[:2], width, height) + pixel_to_normalized(raw_bbox[2:], width, height)
        if raw_bbox
        else None
    )
    result["bbox"] = (
        [{"label": "yoyo", "sub_label": "visible yoyo body", "bbox_pixel": result["yoyo_bbox_pixel"], "bbox_2d": result["yoyo_bbox_2d"]}]
        if raw_bbox
        else []
    )

    hands_raw = candidate.get("hands_pixel")
    hands_normalized = False
    if not isinstance(hands_raw, dict):
        hands_raw = candidate.get("hands_2d") if isinstance(candidate.get("hands_2d"), dict) else {}
        hands_normalized = True
    hands: dict[str, list[float] | None] = {}
    for name in ("left", "right"):
        parsed = point(hands_raw.get(name))
        if parsed and hands_normalized:
            parsed = normalized_to_pixel(parsed, width, height)
        hands[name] = [round(item, 3) for item in parsed] if parsed else None
    result["hands_pixel"] = hands
    result["hands_2d"] = {
        name: pixel_to_normalized(value, width, height) if value else None for name, value in hands.items()
    }

    masks = []
    for polygon in candidate.get("string_mask_polygons_pixel") or []:
        points = normalize_points(polygon, width, height)
        if points:
            masks.append(points)
    result["string_mask_polygons_pixel"] = masks or None
    result["string_path"] = normalize_path(candidate.get("string_path"), width, height)
    attachment = str(candidate.get("string_attachment_class", result.get("string_attachment_class", "unknown"))).lower()
    result["string_attachment_class"] = attachment if attachment in ATTACHMENT_CLASSES else "unknown"
    scene = str(candidate.get("scene_label", result.get("scene_label", "unknown"))).lower()
    result["scene_label"] = scene if scene in SCENE_LABELS else "unknown"
    bad_case = candidate.get("bad_case", result.get("bad_case", []))
    result["bad_case"] = sorted({str(item).strip() for item in (bad_case or []) if str(item).strip()})
    result["notes"] = str(candidate.get("notes", result.get("notes", "")))[:2000]
    if result["string_visibility"] == "not_visible":
        result["string_polylines_pixel"] = None
        result["string_polylines_2d"] = None
        result["string_polyline_pixel"] = None
        result["string_polyline_2d"] = None
        result["string_mask_polygons_pixel"] = None
        if "string_not_visible" not in result["bad_case"]:
            result["bad_case"].append("string_not_visible")
    return result


def label_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    labels = root / "labels" if (root / "labels").exists() else root
    return sorted(
        path
        for path in labels.rglob("*.json")
        if path.name not in {"audit.json", "manifest.json", "project.json"}
        and not path.name.endswith("_audit.json")
    )


def initial_label(image_path: Path, split: str, source_group: str, min_approvals: int) -> dict[str, Any]:
    width, height = image_info(image_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_image": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "image_size": [width, height],
        "source_group": source_group,
        "video_id": source_group,
        "split": split,
        "frame_index": parse_frame_index(image_path.stem),
        "timestamp_s": None,
        "visibility": "uncertain",
        "yoyo_bbox_pixel": None,
        "yoyo_bbox_2d": None,
        "bbox": [],
        "string_visibility": "uncertain",
        "string_polylines_pixel": None,
        "string_polylines_2d": None,
        "string_polyline_pixel": None,
        "string_polyline_2d": None,
        "string_mask_polygons_pixel": None,
        "hands_pixel": {"left": None, "right": None},
        "hands_2d": {"left": None, "right": None},
        "string_attachment_class": "unknown",
        "scene_label": "unknown",
        "string_path": {
            "topology": "uncertain",
            "reconstruction_status": "uncertain",
            "paths": [],
            "unresolved_gaps": [],
        },
        "bad_case": [],
        "notes": "",
        "review_status": "auto_labeled_needs_review",
        "bbox_review_status": "auto_labeled_needs_review",
        "string_review_status": "auto_labeled_needs_review",
        "quality": {
            "revision": 0,
            "min_model_approvals": max(1, int(min_approvals)),
            "history": [],
            "reviews": [],
        },
    }


def command_init(args: argparse.Namespace) -> int:
    images_root = Path(args.images).resolve()
    output = Path(args.output).resolve()
    if args.split not in SPLITS:
        raise ValueError(f"Unsupported split: {args.split}")
    paths = [images_root] if images_root.is_file() else sorted(images_root.rglob("*"))
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    created = skipped = 0
    for image_path in paths:
        group = clean_id(args.source_group or image_path.parent.name or images_root.stem)
        target = output / "labels" / args.split / group / f"{image_path.stem}.json"
        if target.exists() and not args.force:
            skipped += 1
            continue
        write_json(target, initial_label(image_path, args.split, group, args.min_approvals))
        created += 1
    manifest = {
        "schema_version": "rope_annotation_project_v1",
        "created_at_utc": utc_now(),
        "images_root": str(images_root),
        "output": str(output),
        "split": args.split,
        "image_count": len(paths),
        "created": created,
        "skipped": skipped,
    }
    write_json(output / "project.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def apply_candidate(
    label_path: Path,
    candidate: dict[str, Any],
    actor: str,
    role: str,
    model: str,
    message: str,
) -> dict[str, Any]:
    before = read_json(label_path)
    before_digest = content_digest(before)
    after = normalize_candidate(before, candidate)
    quality = copy.deepcopy(before.get("quality") or {})
    quality.setdefault("min_model_approvals", 2)
    quality.setdefault("history", [])
    quality.setdefault("reviews", [])
    quality["revision"] = int(quality.get("revision", 0)) + 1
    after["updated_at_utc"] = utc_now()
    after["string_review_status"] = "auto_labeled_needs_review"
    after["review_status"] = "partially_reviewed" if after.get("bbox_review_status") in ACCEPTED_REVIEW else "auto_labeled_needs_review"
    after["reviewed_at_utc"] = None
    after["reviewer"] = None
    after_digest = content_digest(after)
    quality["history"].append(
        {
            "revision": quality["revision"],
            "created_at_utc": utc_now(),
            "actor": actor,
            "role": role,
            "model": model or None,
            "message": message,
            "before_sha256": before_digest,
            "after_sha256": after_digest,
            "previous_content": {key: before.get(key) for key in CORE_FIELDS},
        }
    )
    after["quality"] = quality
    write_json(label_path, after)
    return after


def command_apply(args: argparse.Namespace) -> int:
    label_path = Path(args.label).resolve()
    candidate = read_json(Path(args.candidate).resolve())
    result = apply_candidate(label_path, candidate, args.actor, args.role, args.model, args.message)
    print(json.dumps({"label": str(label_path), "revision": result["quality"]["revision"], "content_sha256": content_digest(result)}, indent=2))
    return 0


def distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(stroke: list[list[float]]) -> float:
    return sum(distance(left, right) for left, right in zip(stroke, stroke[1:]))


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2.0


def polyline_mask_support(
    strokes: list[list[list[float]]],
    masks: list[list[list[float]]],
    width: int,
    height: int,
    radius: int = 3,
    sample_step: float = 4.0,
) -> tuple[float, int, int]:
    """Measure how much centerline geometry is supported by supplied mask pixels."""
    if not strokes or not masks:
        return 1.0, 0, 0
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in masks:
        if len(polygon) >= 3:
            draw.polygon([(round(item[0]), round(item[1])) for item in polygon], fill=255)
    if radius > 0:
        canvas = canvas.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    pixels = canvas.load()
    supported = total = 0
    for stroke in strokes:
        for left, right in zip(stroke, stroke[1:]):
            length = distance(left, right)
            samples = max(2, int(math.ceil(length / max(1.0, sample_step))) + 1)
            for index in range(samples):
                ratio = index / (samples - 1)
                x = round(left[0] + (right[0] - left[0]) * ratio)
                y = round(left[1] + (right[1] - left[1]) * ratio)
                total += 1
                supported += int(0 <= x < width and 0 <= y < height and pixels[x, y] > 0)
    return (supported / total if total else 1.0), supported, total


def centerlines_from_mask_polygons(
    polygons: list[list[list[float]]],
    width: int,
    height: int,
    min_component_pixels: int = 8,
    max_points: int = 64,
) -> list[list[list[float]]]:
    """Trace thin polygon ribbons without creating endpoint shortcut chords."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to derive mask centerlines") from exc

    def resample_arc(values: Any, count: int) -> Any:
        segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        if cumulative[-1] <= 1e-6:
            return np.repeat(values[:1], count, axis=0)
        targets = np.linspace(0.0, cumulative[-1], count)
        output = []
        for target in targets:
            index = min(len(segments) - 1, int(np.searchsorted(cumulative, target, side="right") - 1))
            local_length = max(1e-6, segments[index])
            ratio = (target - cumulative[index]) / local_length
            output.append(values[index] + (values[index + 1] - values[index]) * ratio)
        return np.asarray(output, dtype=np.float32)

    strokes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        if len(points) < 4 or polygon_area(points.tolist()) < max(1.0, float(min_component_pixels)):
            continue
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.sum(deltas * deltas, axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        if first > second:
            first, second = second, first
        first_arc = points[first : second + 1]
        second_arc = np.concatenate((points[second:], points[: first + 1]), axis=0)[::-1]
        if len(first_arc) < 2 or len(second_arc) < 2:
            continue
        sample_count = max(
            3,
            min(
                max(3, int(max_points)),
                int(math.ceil(max(
                    np.linalg.norm(np.diff(first_arc, axis=0), axis=1).sum(),
                    np.linalg.norm(np.diff(second_arc, axis=0), axis=1).sum(),
                ) / 4.0)) + 1,
            ),
        )
        centerline = (resample_arc(first_arc, sample_count) + resample_arc(second_arc, sample_count)) * 0.5
        centerline[:, 0] = np.clip(centerline[:, 0], 0, width - 1)
        centerline[:, 1] = np.clip(centerline[:, 1], 0, height - 1)
        approximated = cv2.approxPolyDP(centerline.reshape(-1, 1, 2), 1.5, False).reshape(-1, 2)
        stroke = [[round(float(px), 3), round(float(py), 3)] for px, py in approximated]
        if len(stroke) >= 2 and polyline_length(stroke) >= 3.0:
            strokes.append(stroke)
    return strokes


def point_box_distance(value: list[float], raw_bbox: list[float]) -> float:
    dx = max(raw_bbox[0] - value[0], 0.0, value[0] - raw_bbox[2])
    dy = max(raw_bbox[1] - value[1], 0.0, value[1] - raw_bbox[3])
    return math.hypot(dx, dy)


def infer_path_anchor(
    value: list[float],
    hands: dict[str, list[float] | None] | None,
    yoyo_bbox: list[float] | None,
    threshold: float,
) -> str:
    candidates: list[tuple[float, str]] = []
    for name in ("left", "right"):
        hand = (hands or {}).get(name)
        if hand:
            candidates.append((distance(value, hand), f"{name}_hand"))
    if yoyo_bbox:
        candidates.append((point_box_distance(value, yoyo_bbox), "yoyo"))
    if not candidates:
        return "unknown"
    nearest_distance, anchor = min(candidates)
    return anchor if nearest_distance <= threshold else "unknown"


def observed_path_from_strokes(
    strokes: list[list[list[float]]],
    hands: dict[str, list[float] | None] | None = None,
    yoyo_bbox: list[float] | None = None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    width, height = image_size or (1, 1)
    threshold = max(8.0, math.hypot(float(width), float(height)) * 0.06)
    paths = []
    unresolved = []
    for index, stroke in enumerate(strokes):
        start_anchor = infer_path_anchor(stroke[0], hands, yoyo_bbox, threshold)
        end_anchor = infer_path_anchor(stroke[-1], hands, yoyo_bbox, threshold)
        paths.append(
            {
                "path_id": f"observed-stroke-{index + 1}",
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "points_pixel": stroke,
                "edges": [
                    {"from": edge, "to": edge + 1, "evidence": "observed", "confidence": 0.9}
                    for edge in range(len(stroke) - 1)
                ],
            }
        )
        if start_anchor == "unknown" or end_anchor == "unknown":
            unresolved.append(f"stroke {index + 1} has an unanchored endpoint")
    return {
        "topology": "open" if len(paths) == 1 else "multiple" if paths else "uncertain",
        "reconstruction_status": "partial" if paths else "uncertain",
        "paths": paths,
        "unresolved_gaps": unresolved,
    }


def command_derive_centerlines(args: argparse.Namespace) -> int:
    label_path = Path(args.label).resolve()
    label = read_json(label_path)
    width, height = [int(item) for item in label["image_size"]]
    polygons = label.get("string_mask_polygons_pixel") or []
    if not polygons:
        raise ValueError("label has no string_mask_polygons_pixel geometry")
    strokes = centerlines_from_mask_polygons(
        polygons,
        width,
        height,
        min_component_pixels=args.min_component_pixels,
        max_points=args.max_points,
    )
    if not strokes:
        raise ValueError("mask geometry did not yield a valid centerline")
    candidate = copy.deepcopy(label)
    candidate["string_polylines_pixel"] = strokes
    candidate["string_path"] = observed_path_from_strokes(
        strokes,
        hands=label.get("hands_pixel"),
        yoyo_bbox=label.get("yoyo_bbox_pixel"),
        image_size=(width, height),
    )
    candidate["notes"] = (
        str(candidate.get("notes") or "")
        + " Centerlines derived from paired mask-boundary arcs; no endpoint shortcut chords."
    ).strip()
    result = apply_candidate(
        label_path,
        candidate,
        actor=args.actor,
        role="mask-centerline-deriver",
        model=args.model,
        message=args.message,
    )
    support, supported, total = polyline_mask_support(
        result.get("string_polylines_pixel") or [],
        result.get("string_mask_polygons_pixel") or [],
        width,
        height,
    )
    print(
        json.dumps(
            {
                "label": str(label_path),
                "revision": result["quality"]["revision"],
                "stroke_count": len(strokes),
                "mask_support_fraction": round(support, 6),
                "supported_samples": supported,
                "total_samples": total,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def validate_points(
    points: Any,
    width: int,
    height: int,
    name: str,
    minimum: int,
    errors: list[str],
    warnings: list[str],
) -> list[list[float]]:
    if not isinstance(points, list):
        errors.append(f"{name}: expected a list")
        return []
    parsed = []
    for index, value in enumerate(points):
        item = point(value)
        if item is None:
            errors.append(f"{name}[{index}]: invalid point")
            continue
        if not (0 <= item[0] < width and 0 <= item[1] < height):
            errors.append(f"{name}[{index}]: point outside image")
        if parsed and distance(parsed[-1], item) < 0.5:
            errors.append(f"{name}[{index}]: consecutive duplicate point")
        parsed.append(item)
    if len(parsed) < minimum:
        errors.append(f"{name}: requires at least {minimum} points")
    diagonal = math.hypot(width, height)
    for index, (left, right) in enumerate(zip(parsed, parsed[1:])):
        if distance(left, right) > diagonal * 0.8:
            warnings.append(f"{name}[{index}:{index + 1}]: unusually long segment")
    return parsed


def current_approvals(label: dict[str, Any]) -> list[dict[str, Any]]:
    digest = content_digest(label)
    reviews = (label.get("quality") or {}).get("reviews") or []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict) or review.get("content_sha256") != digest:
            continue
        key = (str(review.get("reviewer")), str(review.get("role")))
        latest[key] = review
    return [review for review in latest.values() if review.get("decision") == "approve"]


def validate_label(label: dict[str, Any], check_image: bool = True, check_reviews: bool = True) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        width, height = [int(item) for item in label.get("image_size", [])]
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"errors": ["image_size must contain positive width and height"], "warnings": []}
    source = Path(str(label.get("source_image", "")))
    if check_image:
        if not source.exists():
            errors.append("source_image does not exist")
        else:
            if image_info(source) != (width, height):
                errors.append("image_size does not match source_image")
            expected_hash = str(label.get("image_sha256", ""))
            if expected_hash and sha256_file(source) != expected_hash:
                errors.append("image_sha256 does not match source_image")
    if str(label.get("split")) not in SPLITS:
        errors.append("split must be train, val, or test")
    visibility = str(label.get("string_visibility", "uncertain"))
    if visibility not in STRING_VISIBILITY:
        errors.append(f"unsupported string_visibility={visibility}")
    if str(label.get("visibility", "uncertain")) not in YOYO_VISIBILITY:
        errors.append("unsupported yoyo visibility")
    strokes = label.get("string_polylines_pixel") or []
    valid_strokes = []
    if not isinstance(strokes, list):
        errors.append("string_polylines_pixel must be a list or null")
    else:
        for index, stroke in enumerate(strokes):
            valid_strokes.append(validate_points(stroke, width, height, f"string_polylines_pixel[{index}]", 2, errors, warnings))
    masks = label.get("string_mask_polygons_pixel") or []
    if not isinstance(masks, list):
        errors.append("string_mask_polygons_pixel must be a list or null")
        masks = []
    mask_area = 0.0
    for index, polygon in enumerate(masks):
        parsed = validate_points(polygon, width, height, f"string_mask_polygons_pixel[{index}]", 3, errors, warnings)
        mask_area += polygon_area(parsed)
    if mask_area / max(1, width * height) > 0.15:
        warnings.append("string mask covers more than 15% of the image")
    has_geometry = bool(valid_strokes or masks)
    if visibility in {"visible", "partial"} and not has_geometry:
        errors.append(f"string_visibility={visibility} requires a centerline or mask")
    if visibility == "not_visible" and has_geometry:
        errors.append("string_visibility=not_visible must not retain visible geometry")
    if visibility == "uncertain" and has_geometry:
        warnings.append("uncertain label retains candidate geometry and cannot be approved")
    if visibility in {"visible", "partial"}:
        total_length = sum(polyline_length(stroke) for stroke in valid_strokes)
        if valid_strokes and total_length < max(3.0, math.hypot(width, height) * 0.002):
            warnings.append("visible centerline is unusually short")
    if valid_strokes and masks:
        support, supported_samples, total_samples = polyline_mask_support(
            valid_strokes,
            masks,
            width,
            height,
        )
        if support < 0.8:
            errors.append(
                "centerline has only "
                f"{support:.1%} support from mask geometry "
                f"({supported_samples}/{total_samples} samples); remove unsupported shortcut edges"
            )
        elif support < 0.895:
            warnings.append(
                "centerline-mask support is marginal at "
                f"{support:.1%} ({supported_samples}/{total_samples} samples)"
            )

    path_data = label.get("string_path") or {}
    if not isinstance(path_data, dict):
        errors.append("string_path must be an object")
        path_data = {}
    if str(path_data.get("topology", "uncertain")) not in TOPOLOGIES:
        errors.append("string_path.topology is unsupported")
    if str(path_data.get("reconstruction_status", "uncertain")) not in RECONSTRUCTION_STATUS:
        errors.append("string_path.reconstruction_status is unsupported")
    observed_edge_count = 0
    yoyo_anchored = False
    anchor_threshold = max(8.0, math.hypot(width, height) * 0.08)
    label_hands = label.get("hands_pixel") or {}
    label_yoyo_bbox = label.get("yoyo_bbox_pixel")
    for path_index, path_item in enumerate(path_data.get("paths") or []):
        if not isinstance(path_item, dict):
            errors.append(f"string_path.paths[{path_index}] must be an object")
            continue
        points = validate_points(path_item.get("points_pixel"), width, height, f"string_path.paths[{path_index}].points_pixel", 2, errors, warnings)
        for endpoint_name, point_index in (("start_anchor", 0), ("end_anchor", -1)):
            anchor = str(path_item.get(endpoint_name, "unknown"))
            if anchor not in PATH_ANCHORS:
                errors.append(
                    f"string_path.paths[{path_index}].{endpoint_name}={anchor} is unsupported; "
                    "use left_hand, right_hand, yoyo, or unknown"
                )
                continue
            if anchor == "unknown" or not points:
                continue
            endpoint = points[point_index]
            if anchor in {"left_hand", "right_hand"}:
                hand = label_hands.get(anchor.removesuffix("_hand"))
                if not hand:
                    errors.append(f"string_path.paths[{path_index}].{endpoint_name} names missing {anchor}")
                elif distance(endpoint, hand) > anchor_threshold:
                    errors.append(
                        f"string_path.paths[{path_index}].{endpoint_name} is too far from {anchor}"
                    )
            elif anchor == "yoyo":
                if not label_yoyo_bbox:
                    errors.append(f"string_path.paths[{path_index}].{endpoint_name} names missing yoyo bbox")
                elif point_box_distance(endpoint, label_yoyo_bbox) > anchor_threshold:
                    errors.append(
                        f"string_path.paths[{path_index}].{endpoint_name} is too far from yoyo"
                    )
        edges = path_item.get("edges") or []
        if len(edges) != max(0, len(points) - 1):
            errors.append(f"string_path.paths[{path_index}].edges must describe every consecutive point pair")
        for edge_index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"string_path.paths[{path_index}].edges[{edge_index}] must be an object")
                continue
            if edge.get("from") != edge_index or edge.get("to") != edge_index + 1:
                errors.append(f"string_path.paths[{path_index}].edges[{edge_index}] must connect consecutive points")
            evidence = str(edge.get("evidence", ""))
            if evidence not in EDGE_EVIDENCE:
                errors.append(f"string_path.paths[{path_index}].edges[{edge_index}] has unsupported evidence")
            observed_edge_count += int(evidence == "observed")
            try:
                confidence = float(edge.get("confidence"))
                if not 0 <= confidence <= 1:
                    raise ValueError
                if evidence == "inferred" and confidence > 0.75:
                    warnings.append(f"string_path.paths[{path_index}].edges[{edge_index}] inferred confidence is unusually high")
            except (TypeError, ValueError):
                errors.append(f"string_path.paths[{path_index}].edges[{edge_index}] confidence must be in [0,1]")
        if str(path_item.get("end_anchor")) == "yoyo" or str(path_item.get("start_anchor")) == "yoyo":
            yoyo_anchored = True
    if observed_edge_count and not has_geometry:
        errors.append("observed string_path edges require visible training geometry")
    if visibility in {"visible", "partial"} and not (path_data.get("paths") or []):
        errors.append("visible string requires an ordered string_path reconstruction")
    if visibility in {"visible", "partial"} and observed_edge_count == 0:
        errors.append("visible string_path requires at least one current-frame observed edge")
    if label.get("string_attachment_class") == "hand_and_yoyo_attached" and visibility in {"visible", "partial"} and not yoyo_anchored:
        errors.append("attached string path must be anchored to yoyo")

    if check_reviews:
        quality = label.get("quality") or {}
        minimum = max(1, int(quality.get("min_model_approvals", 2)))
        approvals = current_approvals(label)
        if label.get("string_review_status") in ACCEPTED_REVIEW:
            if visibility == "uncertain":
                errors.append("approved label cannot have uncertain visibility")
            if len(approvals) < minimum:
                errors.append(f"approved label has {len(approvals)} current approvals; requires {minimum}")
            roles = {str(review.get("role")) for review in approvals}
            if minimum >= 2 and len(roles) < 2:
                errors.append("approved label requires approvals from at least two review roles")
    return {"errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def command_review(args: argparse.Namespace) -> int:
    path = Path(args.label).resolve()
    label = read_json(path)
    gate = validate_label(label, check_image=True, check_reviews=False)
    if args.decision == "approve" and gate["errors"]:
        raise ValueError("cannot approve label: " + "; ".join(gate["errors"]))
    quality = label.setdefault("quality", {})
    quality.setdefault("history", [])
    quality.setdefault("reviews", [])
    quality.setdefault("min_model_approvals", 2)
    review = {
        "created_at_utc": utc_now(),
        "reviewer": args.reviewer,
        "role": args.role,
        "model": args.model or None,
        "decision": args.decision,
        "notes": args.notes,
        "content_sha256": content_digest(label),
    }
    quality["reviews"].append(review)
    if args.decision == "approve":
        approvals = current_approvals(label)
        required = max(1, int(quality.get("min_model_approvals", 2)))
        distinct_roles = {str(item.get("role")) for item in approvals}
        if len(approvals) >= required and (required == 1 or len(distinct_roles) >= 2):
            label["string_review_status"] = "approved"
            label["reviewed_at_utc"] = utc_now()
            label["reviewer"] = ",".join(sorted({str(item.get("reviewer")) for item in approvals}))
        else:
            label["string_review_status"] = "auto_labeled_needs_review"
    elif args.decision == "reject":
        label["string_review_status"] = "rejected"
    else:
        label["string_review_status"] = "auto_labeled_needs_review"
    label["review_status"] = "approved" if label.get("bbox_review_status") in ACCEPTED_REVIEW and label["string_review_status"] == "approved" else "partially_reviewed"
    label["updated_at_utc"] = utc_now()
    write_json(path, label)
    print(json.dumps({"label": str(path), "string_review_status": label["string_review_status"], "current_approvals": len(current_approvals(label)), "gate": gate}, ensure_ascii=False, indent=2))
    return 0


def scale_point(value: list[float], scale: float) -> tuple[int, int]:
    return round(value[0] * scale), round(value[1] * scale)


def draw_dashed(draw: ImageDraw.ImageDraw, left: tuple[int, int], right: tuple[int, int], fill: str, width: int) -> None:
    length = math.hypot(right[0] - left[0], right[1] - left[1])
    if length <= 0:
        return
    segments = max(1, int(length / 16))
    for index in range(0, segments, 2):
        start = index / segments
        end = min(1.0, (index + 1) / segments)
        a = (round(left[0] + (right[0] - left[0]) * start), round(left[1] + (right[1] - left[1]) * start))
        b = (round(left[0] + (right[0] - left[0]) * end), round(left[1] + (right[1] - left[1]) * end))
        draw.line([a, b], fill=fill, width=width)


def render_layer(
    image: Image.Image,
    label: dict[str, Any],
    scale: float,
    include_grid: bool,
    grid_origin: tuple[int, int] = (0, 0),
    grid_reference_size: tuple[int, int] | None = None,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = [int(item) for item in label["image_size"]]
    line_width = max(2, round(min(canvas.size) / 400))
    font = ImageFont.load_default()
    if include_grid:
        reference_width, reference_height = grid_reference_size or (width, height)
        step = max(50, int(round(max(reference_width, reference_height) / 12 / 50) * 50))
        first_x = math.ceil(grid_origin[0] / step) * step
        first_y = math.ceil(grid_origin[1] / step) * step
        for x in range(first_x, grid_origin[0] + width, step):
            sx = round((x - grid_origin[0]) * scale)
            draw.line([(sx, 0), (sx, canvas.height)], fill=(255, 255, 0, 90), width=1)
            draw.text((sx + 2, 2), str(x), fill=(255, 255, 0, 255), font=font)
        for y in range(first_y, grid_origin[1] + height, step):
            sy = round((y - grid_origin[1]) * scale)
            draw.line([(0, sy), (canvas.width, sy)], fill=(255, 255, 0, 90), width=1)
            draw.text((2, sy + 2), str(y), fill=(255, 255, 0, 255), font=font)
    for polygon in label.get("string_mask_polygons_pixel") or []:
        points = [scale_point(item, scale) for item in polygon]
        if len(points) >= 3:
            draw.polygon(points, fill=(0, 180, 255, 70), outline=(0, 255, 255, 230))
    for stroke in label.get("string_polylines_pixel") or []:
        points = [scale_point(item, scale) for item in stroke]
        if len(points) >= 2:
            draw.line(points, fill=(0, 255, 255, 255), width=line_width + 2, joint="curve")
            for item in points:
                draw.ellipse([item[0] - 3, item[1] - 3, item[0] + 3, item[1] + 3], fill=(0, 90, 255, 255))
    colors = {"temporal": "#FF9F1C", "inferred": "#FF4FD8"}
    for path_item in (label.get("string_path") or {}).get("paths") or []:
        points = [scale_point(item, scale) for item in path_item.get("points_pixel") or []]
        for edge in path_item.get("edges") or []:
            start, end = int(edge.get("from", -1)), int(edge.get("to", -1))
            if not (0 <= start < len(points) and 0 <= end < len(points)):
                continue
            evidence = str(edge.get("evidence", "inferred"))
            # Observed path edges must duplicate current-frame visible geometry,
            # which is already rendered in cyan. Drawing them again makes open
            # formations look closed and hides centerline drift.
            if evidence == "observed":
                continue
            if evidence == "inferred":
                draw_dashed(draw, points[start], points[end], colors[evidence], line_width)
            else:
                draw.line([points[start], points[end]], fill=colors.get(evidence, "white"), width=line_width)
    raw_bbox = label.get("yoyo_bbox_pixel")
    if raw_bbox:
        draw.rectangle([round(item * scale) for item in raw_bbox], outline=(50, 255, 70, 255), width=line_width + 1)
    for name, raw in (label.get("hands_pixel") or {}).items():
        if raw:
            x, y = scale_point(raw, scale)
            radius = max(4, line_width * 2)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 230, 0, 230))
            draw.text((x + radius + 2, y), name, fill=(255, 230, 0, 255), font=font)
    header = (
        f"string={label.get('string_visibility')} status={label.get('string_review_status')} "
        f"rev={(label.get('quality') or {}).get('revision', 0)} digest={content_digest(label)[:10]}"
    )
    draw.rectangle((0, 0, min(canvas.width, 620), 24), fill=(0, 0, 0, 190))
    draw.text((6, 6), header, fill=(255, 255, 255, 255), font=font)
    return canvas


def geometry_bounds(label: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = []
    for stroke in label.get("string_polylines_pixel") or []:
        points.extend(stroke)
    for item in (label.get("hands_pixel") or {}).values():
        if item:
            points.append(item)
    for path_item in (label.get("string_path") or {}).get("paths") or []:
        points.extend(path_item.get("points_pixel") or [])
    raw_bbox = label.get("yoyo_bbox_pixel")
    if raw_bbox:
        points.extend([raw_bbox[:2], raw_bbox[2:]])
    if not points:
        return None
    return min(item[0] for item in points), min(item[1] for item in points), max(item[0] for item in points), max(item[1] for item in points)


def command_render(args: argparse.Namespace) -> int:
    label_path = Path(args.label).resolve()
    label = read_json(label_path)
    source = Path(label["source_image"])
    output = Path(args.output).resolve() if args.output else label_path.parent / "review"
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        raw = opened.convert("RGB")
    scale = min(1.0, args.max_side / max(raw.size))
    resized = raw.resize((round(raw.width * scale), round(raw.height * scale)), Image.Resampling.LANCZOS) if scale < 1 else raw.copy()
    grid = render_layer(resized, label, scale, True)
    overlay = render_layer(resized, label, scale, False)
    grid_path = output / f"{label_path.stem}_grid.jpg"
    overlay_path = output / f"{label_path.stem}_overlay.jpg"
    grid.save(grid_path, quality=94)
    overlay.save(overlay_path, quality=94)
    detail_path = None
    bounds = geometry_bounds(label)
    if bounds:
        width, height = raw.size
        margin = max(32, round(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.18))
        crop_box = (
            max(0, math.floor(bounds[0] - margin)),
            max(0, math.floor(bounds[1] - margin)),
            min(width, math.ceil(bounds[2] + margin)),
            min(height, math.ceil(bounds[3] + margin)),
        )
        detail_label = copy.deepcopy(label)
        offset_x, offset_y = crop_box[0], crop_box[1]
        for stroke in detail_label.get("string_polylines_pixel") or []:
            for item in stroke:
                item[0] -= offset_x
                item[1] -= offset_y
        for polygon in detail_label.get("string_mask_polygons_pixel") or []:
            for item in polygon:
                item[0] -= offset_x
                item[1] -= offset_y
        for item in (detail_label.get("hands_pixel") or {}).values():
            if item:
                item[0] -= offset_x
                item[1] -= offset_y
        if detail_label.get("yoyo_bbox_pixel"):
            box = detail_label["yoyo_bbox_pixel"]
            detail_label["yoyo_bbox_pixel"] = [box[0] - offset_x, box[1] - offset_y, box[2] - offset_x, box[3] - offset_y]
        for path_item in (detail_label.get("string_path") or {}).get("paths") or []:
            for item in path_item.get("points_pixel") or []:
                item[0] -= offset_x
                item[1] -= offset_y
        crop = raw.crop(crop_box)
        detail_scale = min(2.0, args.max_side / max(crop.size))
        crop = crop.resize((round(crop.width * detail_scale), round(crop.height * detail_scale)), Image.Resampling.LANCZOS)
        detail_label["image_size"] = [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]]
        detail = render_layer(
            crop,
            detail_label,
            detail_scale,
            True,
            grid_origin=(offset_x, offset_y),
            grid_reference_size=(width, height),
        )
        detail_path = output / f"{label_path.stem}_detail.jpg"
        detail.save(detail_path, quality=96)
    metadata_path = output / f"{label_path.stem}_render.json"
    result = {
        "grid": str(grid_path),
        "overlay": str(overlay_path),
        "detail": str(detail_path) if detail_path else None,
        "render_metadata": str(metadata_path),
    }
    write_json(
        metadata_path,
        {
            "schema_version": "rope_review_render_v1",
            "created_at_utc": utc_now(),
            "label": str(label_path),
            "content_sha256": content_digest(label),
            "revision": int((label.get("quality") or {}).get("revision", 0)),
            "string_review_status": label.get("string_review_status"),
            "artifacts": {key: result[key] for key in ("grid", "overlay", "detail")},
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def flow_points(previous_gray: Any, current_gray: Any, points: list[list[float]], max_error: float) -> tuple[list[list[float] | None], list[float | None]]:
    import cv2
    import numpy as np

    if not points:
        return [], []
    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    forward, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        array,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        forward,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    fb = np.linalg.norm(backward.reshape(-1, 2) - array.reshape(-1, 2), axis=1)
    result: list[list[float] | None] = []
    errors: list[float | None] = []
    for index, item in enumerate(forward.reshape(-1, 2)):
        valid = bool(status[index, 0]) and bool(back_status[index, 0]) and float(fb[index]) <= max_error
        result.append([round(float(item[0]), 3), round(float(item[1]), 3)] if valid else None)
        errors.append(round(float(fb[index]), 4) if valid else None)
    return result, errors


def command_propagate(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for temporal propagation: pip install opencv-python") from exc
    previous_path = Path(args.previous_label).resolve()
    target_path = Path(args.target_label).resolve()
    previous = read_json(previous_path)
    target = read_json(target_path)
    if previous.get("image_size") != target.get("image_size"):
        raise ValueError("temporal propagation requires frames with identical dimensions")
    previous_gray = cv2.imread(str(previous["source_image"]), cv2.IMREAD_GRAYSCALE)
    current_gray = cv2.imread(str(target["source_image"]), cv2.IMREAD_GRAYSCALE)
    if previous_gray is None or current_gray is None:
        raise RuntimeError("could not read source or target frame")
    candidate = copy.deepcopy(previous)
    candidate["source_image"] = target["source_image"]
    # Masks cannot be transported by sparse point flow. Keep only geometry that
    # this command explicitly tracks and require a new current-frame mask review.
    candidate["string_mask_polygons_pixel"] = None
    all_errors: list[float] = []
    attempted_measurements = 0
    propagated_strokes = []
    for stroke in previous.get("string_polylines_pixel") or []:
        tracked, errors = flow_points(previous_gray, current_gray, stroke, args.max_error)
        attempted_measurements += len(tracked)
        all_errors.extend(item for item in errors if item is not None)
        chunk: list[list[float]] = []
        for item in tracked + [None]:
            if item is not None:
                chunk.append(item)
            else:
                if len(chunk) >= 2:
                    propagated_strokes.append(chunk)
                chunk = []
    candidate["string_polylines_pixel"] = propagated_strokes or None
    if not propagated_strokes and previous.get("string_visibility") in {"visible", "partial"}:
        candidate["string_polyline_pixel"] = None
        candidate["string_polylines_2d"] = None
        candidate["string_polyline_2d"] = None
        candidate["string_visibility"] = "uncertain"
        candidate["bad_case"] = sorted(set((candidate.get("bad_case") or []) + ["temporal_propagation_failed"]))
    elif propagated_strokes and previous.get("string_visibility") == "visible":
        candidate["string_visibility"] = "partial"
    previous_hands = previous.get("hands_pixel") or {}
    hand_names = [name for name in ("left", "right") if previous_hands.get(name)]
    tracked_hands, hand_errors = flow_points(previous_gray, current_gray, [previous_hands[name] for name in hand_names], args.max_error)
    attempted_measurements += len(tracked_hands)
    all_errors.extend(item for item in hand_errors if item is not None)
    candidate["hands_pixel"] = {"left": None, "right": None}
    for name, tracked in zip(hand_names, tracked_hands):
        candidate["hands_pixel"][name] = tracked
    raw_bbox = previous.get("yoyo_bbox_pixel")
    if raw_bbox:
        corners = [raw_bbox[:2], raw_bbox[2:]]
        tracked_box, box_errors = flow_points(previous_gray, current_gray, corners, args.max_error)
        attempted_measurements += len(tracked_box)
        all_errors.extend(item for item in box_errors if item is not None)
        if all(item is not None for item in tracked_box):
            candidate["yoyo_bbox_pixel"] = tracked_box[0] + tracked_box[1]
        else:
            candidate["yoyo_bbox_pixel"] = None
    propagated_path = copy.deepcopy(previous.get("string_path") or {})
    for path_item in propagated_path.get("paths") or []:
        tracked, errors = flow_points(previous_gray, current_gray, path_item.get("points_pixel") or [], args.max_error)
        attempted_measurements += len(tracked)
        all_errors.extend(item for item in errors if item is not None)
        old_points = path_item.get("points_pixel") or []
        path_item["points_pixel"] = [tracked[index] if tracked[index] is not None else old_points[index] for index in range(len(old_points))]
        for edge in path_item.get("edges") or []:
            start, end = int(edge.get("from", -1)), int(edge.get("to", -1))
            endpoints_tracked = (
                0 <= start < len(tracked)
                and 0 <= end < len(tracked)
                and tracked[start] is not None
                and tracked[end] is not None
            )
            if edge.get("evidence") != "inferred" and endpoints_tracked:
                edge["evidence"] = "temporal"
                edge["confidence"] = round(min(0.85, float(edge.get("confidence", 0.0)) * 0.9), 4)
            elif not endpoints_tracked:
                edge["evidence"] = "inferred"
                edge["confidence"] = round(min(0.25, float(edge.get("confidence", 0.0)) * 0.5), 4)
    candidate["string_path"] = propagated_path
    candidate["temporal_seed"] = {
        "method": "pyramidal_lucas_kanade_forward_backward",
        "source_label": str(previous_path),
        "source_content_sha256": content_digest(previous),
        "mean_forward_backward_error": round(sum(all_errors) / len(all_errors), 4) if all_errors else None,
        "tracked_measurement_count": len(all_errors),
        "attempted_measurement_count": attempted_measurements,
        "tracked_fraction": round(len(all_errors) / attempted_measurements, 4) if attempted_measurements else 0.0,
        "requires_current_frame_model_review": True,
    }
    candidate["notes"] = (str(candidate.get("notes", "")) + " Temporal seed; refine against the current frame before approval.").strip()
    result = apply_candidate(target_path, candidate, args.actor, "temporal-propagator", args.model, args.message)
    result["temporal_seed"] = candidate["temporal_seed"]
    write_json(target_path, result)
    print(json.dumps({"target_label": str(target_path), "temporal_seed": candidate["temporal_seed"], "stroke_count": len(propagated_strokes)}, ensure_ascii=False, indent=2))
    return 0


def audit_collection(root: Path, check_image: bool = True, require_approved: bool = False) -> dict[str, Any]:
    paths = label_files(root)
    records = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for path in paths:
        try:
            label = read_json(path)
            gate = validate_label(label, check_image=check_image, check_reviews=True)
            if require_approved and label.get("string_review_status") not in ACCEPTED_REVIEW:
                gate["errors"].append(f"final gate requires approved label; found {label.get('string_review_status')}")
                gate["errors"] = sorted(set(gate["errors"]))
        except Exception as exc:
            records.append({"label": str(path), "errors": [f"could not validate label: {exc}"], "warnings": []})
            counts["labels"] += 1
            counts["labels_with_errors"] += 1
            continue
        split = str(label.get("split", "unknown"))
        group = str(label.get("source_group", ""))
        digest = str(label.get("image_sha256", ""))
        split_groups[split].add(group)
        if digest:
            hash_splits[digest].add(split)
        counts["labels"] += 1
        counts[f"split:{split}"] += 1
        counts[f"visibility:{label.get('string_visibility', 'missing')}"] += 1
        counts[f"status:{label.get('string_review_status', 'missing')}"] += 1
        counts["labels_with_errors"] += int(bool(gate["errors"]))
        counts["labels_with_warnings"] += int(bool(gate["warnings"]))
        records.append({"label": str(path), **gate})
    collection_errors = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_groups[left] & split_groups[right])
        if overlap:
            collection_errors.append({"kind": "source_group_split_leakage", "splits": [left, right], "source_groups": overlap})
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            collection_errors.append({"kind": "duplicate_image_split_leakage", "image_sha256": digest, "splits": sorted(splits)})
    return {
        "schema_version": "rope_annotation_audit_v1",
        "created_at_utc": utc_now(),
        "root": str(root.resolve()),
        "ok": counts["labels_with_errors"] == 0 and not collection_errors,
        "counts": dict(sorted(counts.items())),
        "collection_errors": collection_errors,
        "records": records,
    }


def command_audit(args: argparse.Namespace) -> int:
    root = Path(args.labels).resolve()
    report = audit_collection(root, check_image=not args.skip_image_check, require_approved=args.require_approved)
    output = Path(args.output).resolve() if args.output else (root / "audit.json" if root.is_dir() else root.with_name(root.stem + "_audit.json"))
    write_json(output, report)
    summary = {"ok": report["ok"], "counts": report["counts"], "collection_errors": len(report["collection_errors"]), "output": str(output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.strict and not report["ok"] else 0


def command_export(args: argparse.Namespace) -> int:
    source_root = Path(args.labels).resolve()
    output = Path(args.output).resolve()
    existing_labels = output / "labels"
    if existing_labels.exists() and any(existing_labels.rglob("*.json")):
        raise ValueError(f"output already contains labels; choose an empty snapshot directory: {output}")
    audit = audit_collection(source_root, check_image=True)
    record_by_path = {item["label"]: item for item in audit["records"]}
    if audit["collection_errors"]:
        raise ValueError("collection leakage prevents export: " + json.dumps(audit["collection_errors"], ensure_ascii=False))
    exported = []
    excluded = []
    for path in label_files(source_root):
        label = read_json(path)
        reasons = list(record_by_path.get(str(path), {}).get("errors") or [])
        if label.get("string_review_status") not in ACCEPTED_REVIEW:
            reasons.append(f"string_review_status={label.get('string_review_status')}")
        approvals = current_approvals(label)
        required = args.min_approvals if args.min_approvals is not None else int((label.get("quality") or {}).get("min_model_approvals", 2))
        if len(approvals) < required:
            reasons.append(f"current_approvals={len(approvals)} requires={required}")
        if reasons:
            excluded.append({"label": str(path), "reasons": sorted(set(reasons))})
            continue
        split = str(label["split"])
        group = clean_id(str(label.get("source_group") or "images"))
        target = output / "labels" / split / group / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        exported.append({"source": str(path), "target": str(target), "split": split, "visibility": label.get("string_visibility")})
    manifest = {
        "schema_version": "rope_annotation_export_v1",
        "created_at_utc": utc_now(),
        "source": str(source_root),
        "output": str(output),
        "exported_count": len(exported),
        "excluded_count": len(excluded),
        "counts": dict(Counter(f"{item['split']}:{item['visibility']}" for item in exported)),
        "exported": exported,
        "excluded": excluded,
        "label_semantics": {
            "string_polylines_pixel": "model-reviewed visible centerlines only",
            "string_mask_polygons_pixel": "optional reviewed visible-area polygons",
            "string_path": "ordered reconstruction with per-edge evidence; temporal/inferred edges are not segmentation truth",
            "not_visible": "reviewed negative with no visible geometry",
            "uncertain": "excluded",
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("exported_count", "excluded_count", "counts", "output")}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create draft labels for an image collection.")
    init.add_argument("--images", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--split", choices=sorted(SPLITS), default="train")
    init.add_argument("--source-group", default="")
    init.add_argument("--min-approvals", type=int, default=2)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=command_init)

    apply = subparsers.add_parser("apply", help="Apply a model/manual candidate as a new label revision.")
    apply.add_argument("--label", required=True)
    apply.add_argument("--candidate", required=True)
    apply.add_argument("--actor", required=True)
    apply.add_argument("--role", default="model-annotator")
    apply.add_argument("--model", default="")
    apply.add_argument("--message", default="candidate geometry update")
    apply.set_defaults(func=command_apply)

    derive = subparsers.add_parser(
        "derive-centerlines",
        help="Trace mask-ribbon midlines without creating shortcut chords.",
    )
    derive.add_argument("--label", required=True)
    derive.add_argument("--actor", default="model-mask-centerline-deriver")
    derive.add_argument("--model", default="")
    derive.add_argument("--message", default="derive centerlines from connected mask skeletons")
    derive.add_argument("--min-component-pixels", type=int, default=8)
    derive.add_argument("--max-points", type=int, default=64)
    derive.set_defaults(func=command_derive_centerlines)

    render = subparsers.add_parser("render", help="Create grid, overlay, and detail images for model review.")
    render.add_argument("--label", required=True)
    render.add_argument("--output", default="")
    render.add_argument("--max-side", type=int, default=1800)
    render.set_defaults(func=command_render)

    propagate = subparsers.add_parser("propagate", help="Seed a consecutive frame using forward/backward optical flow.")
    propagate.add_argument("--previous-label", required=True)
    propagate.add_argument("--target-label", required=True)
    propagate.add_argument("--actor", default="model-temporal-propagator")
    propagate.add_argument("--model", default="")
    propagate.add_argument("--message", default="temporal seed from previous reviewed frame")
    propagate.add_argument("--max-error", type=float, default=2.5)
    propagate.set_defaults(func=command_propagate)

    review = subparsers.add_parser("review", help="Record a digest-bound model review decision.")
    review.add_argument("--label", required=True)
    review.add_argument("--decision", choices=("approve", "request_changes", "reject", "unresolved"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--role", choices=("geometry-critic", "semantic-critic", "temporal-verifier", "final-verifier"), required=True)
    review.add_argument("--model", default="")
    review.add_argument("--notes", required=True)
    review.set_defaults(func=command_review)

    audit = subparsers.add_parser("audit", help="Validate labels, provenance, reviews, and split isolation.")
    audit.add_argument("--labels", required=True)
    audit.add_argument("--output", default="")
    audit.add_argument("--skip-image-check", action="store_true")
    audit.add_argument("--require-approved", action="store_true", help="Treat every pending/rejected label as a final-gate error.")
    audit.add_argument("--strict", action="store_true")
    audit.set_defaults(func=command_audit)

    export = subparsers.add_parser("export", help="Export only current, fully approved training labels.")
    export.add_argument("--labels", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--min-approvals", type=int)
    export.set_defaults(func=command_export)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
