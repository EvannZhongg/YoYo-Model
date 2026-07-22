"""Auto-label sampled video frames with review-gated yoyo/string geometry."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from annotation.annotator import (
    _call_model_once,
    _image_token_limits,
    _messages_for_image,
    image_url_for_model,
)
from annotation.prompts import VIDEO_FRAME_ANNOTATION_PROMPT
from config import MODEL_CONFIG, OSS_CONFIG


VISIBILITY = {"visible", "partially_visible", "occluded", "out_of_frame", "absent", "uncertain"}
STRING_VISIBILITY = {"visible", "partial", "not_visible", "uncertain"}
STRING_ATTACHMENT_CLASSES = {
    "hand_and_yoyo_attached",
    "yoyo_detached",
    "hand_detached",
    "unknown",
}


def _extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    if "```" in text:
        candidates.extend(part.strip() for part in text.split("```") if "{" in part)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        candidate = candidate.removeprefix("json").strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a valid JSON object")


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return [max(0.0, min(999.0, float(value[0]))), max(0.0, min(999.0, float(value[1])))]
    except (TypeError, ValueError):
        return None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        coords = [max(0.0, min(999.0, float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    return coords if coords[2] > coords[0] and coords[3] > coords[1] else None


def normalize_annotation(raw: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    visibility = str(raw.get("visibility", "uncertain")).lower()
    if visibility not in VISIBILITY:
        visibility = "uncertain"
    string_visibility = str(raw.get("string_visibility", "uncertain")).lower()
    if string_visibility not in STRING_VISIBILITY:
        string_visibility = "uncertain"
    bbox_2d = _bbox(raw.get("yoyo_bbox_2d"))
    polylines_raw = raw.get("string_polylines_2d")
    if not isinstance(polylines_raw, list):
        legacy = raw.get("string_polyline_2d")
        polylines_raw = [legacy] if isinstance(legacy, list) else []
    polylines = []
    for stroke in polylines_raw:
        if not isinstance(stroke, list):
            continue
        points = [_point(value) for value in stroke]
        points = [point for point in points if point is not None]
        if len(points) >= 2:
            polylines.append(points)
    hands_raw = raw.get("hands_2d") if isinstance(raw.get("hands_2d"), dict) else {}
    hands = {"left": _point(hands_raw.get("left")), "right": _point(hands_raw.get("right"))}

    def pixel_point(point: list[float]) -> list[int]:
        return [round(point[0] / 999.0 * width), round(point[1] / 999.0 * height)]

    bbox_pixel = None
    if bbox_2d:
        bbox_pixel = [round(bbox_2d[0] / 999.0 * width), round(bbox_2d[1] / 999.0 * height), round(bbox_2d[2] / 999.0 * width), round(bbox_2d[3] / 999.0 * height)]
    bad_case = sorted({str(value).strip() for value in raw.get("bad_case", []) if str(value).strip()}) if isinstance(raw.get("bad_case"), list) else []
    if bbox_2d is None and visibility not in {"absent", "out_of_frame"}:
        bad_case.append("yoyo_not_visible")
    if not polylines:
        if string_visibility in {"visible", "partial"}:
            string_visibility = "uncertain"
            bad_case.append("string_ambiguous")
    return {
        "visibility": visibility,
        "string_visibility": string_visibility,
        "yoyo_bbox_2d": bbox_2d,
        "yoyo_bbox_pixel": bbox_pixel,
        "string_polylines_2d": polylines or None,
        "string_polylines_pixel": [[pixel_point(point) for point in stroke] for stroke in polylines] or None,
        "string_polyline_2d": polylines[0] if polylines else None,
        "string_polyline_pixel": [pixel_point(point) for point in polylines[0]] if polylines else None,
        "hands_2d": hands,
        "hands_pixel": {name: pixel_point(point) if point else None for name, point in hands.items()},
        "bad_case": sorted(set(bad_case)),
        "notes": str(raw.get("notes", ""))[:500],
    }


def draw_visualization(image_path: Path, annotation: dict[str, Any], output_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read frame: {image_path}")
    bbox = annotation.get("yoyo_bbox_pixel")
    if bbox:
        bbox_int = [int(round(float(value))) for value in bbox]
        cv2.rectangle(frame, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0, 220, 0), 3)
    polylines = annotation.get("string_polylines_pixel")
    if not polylines and annotation.get("string_polyline_pixel"):
        polylines = [annotation["string_polyline_pixel"]]
    if polylines:
        import numpy as np
        for polyline in polylines:
            cv2.polylines(frame, [np.asarray(polyline, dtype=np.float32).round().astype(np.int32)], False, (255, 80, 30), 3)
    mask_polygons = annotation.get("string_mask_polygons_pixel") or []
    if mask_polygons:
        import numpy as np

        arrays = [np.asarray(polygon, dtype=np.float32).round().astype(np.int32) for polygon in mask_polygons if isinstance(polygon, list) and len(polygon) >= 3]
        if arrays:
            overlay = frame.copy()
            # Thin strings disappear at 4K when rendered with a faint outline;
            # keep the fill translucent but make every proposed component easy
            # to distinguish during manual review.
            cv2.fillPoly(overlay, arrays, (0, 120, 255))
            cv2.polylines(overlay, arrays, True, (255, 255, 0), 5)
            frame = cv2.addWeighted(overlay, 0.36, frame, 0.64, 0)
    for point in annotation.get("hands_pixel", {}).values():
        if point:
            cv2.circle(frame, tuple(int(round(float(value))) for value in point), 8, (0, 220, 255), -1)
    bad_case = ",".join(annotation.get("bad_case", [])) or "none"
    qa_warnings = ",".join((annotation.get("qa") or {}).get("warnings", [])) or "none"
    header = f"VLM REVIEW ONLY | visibility={annotation.get('visibility', 'uncertain')} | string={annotation.get('string_visibility', 'uncertain')}"
    review_line = (
        f"bbox_review={annotation.get('bbox_review_status', 'needs_review')} | "
        f"string_review={annotation.get('string_review_status', 'needs_review')} | "
        f"overall={annotation.get('review_status', 'needs_review')}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 98), (0, 0, 0), -1)
    cv2.putText(frame, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, review_line, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"bad_case={bad_case} | qa={qa_warnings}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    if mask_polygons:
        cv2.putText(frame, "COLOR MASK PROPOSAL | REVIEW REQUIRED", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def annotate_record(record: dict[str, Any], dataset_dir: Path, model: str, min_tokens: str, max_tokens: str) -> dict[str, Any]:
    frame_path = Path(record["frame_path"])
    width, height = Image.open(frame_path).size
    min_pixels, max_pixels = _image_token_limits(min_tokens, max_tokens)
    object_name = f"{OSS_CONFIG.object_prefix}/video-frame-{uuid.uuid4().hex}{frame_path.suffix}"
    image_url = image_url_for_model(frame_path, object_name)
    messages = _messages_for_image(image_url, VIDEO_FRAME_ANNOTATION_PROMPT, min_pixels, max_pixels)
    raw_response = _call_model_once(messages, model)
    normalized = normalize_annotation(_extract_json_object(raw_response), width, height)
    bbox = []
    if normalized["yoyo_bbox_2d"]:
        bbox.append({"label": "yoyo", "sub_label": "visible yoyo body", "bbox_2d": normalized["yoyo_bbox_2d"], "bbox_pixel": normalized["yoyo_bbox_pixel"]})
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_status": "auto_labeled_needs_review",
        "bbox_review_status": "auto_labeled_needs_review",
        "string_review_status": "auto_labeled_needs_review",
        "string_attachment_class": "unknown",
        "model": model,
        "source_image": str(frame_path.resolve()),
        "source_video": record["source_video"],
        "source_video_sha256": record["source_video_sha256"],
        "source_group": record["source_group"],
        "action_group": record.get("action_group", "1A"),
        "video_id": record["video_id"],
        "split": record["split"],
        "frame_index": record["frame_index"],
        "timestamp_s": record["timestamp_s"],
        "image_size": [width, height],
        "bbox": bbox,
        **normalized,
        "raw_response": raw_response,
    }
    relative = Path(record["split"]) / record["video_id"] / f"frame_{record['frame_index']:08d}"
    label_path = dataset_dir / "annotations" / "labels" / relative.with_suffix(".json")
    vis_path = dataset_dir / "annotations" / "visualizations" / relative.with_name(f"{relative.name}_vis.jpg")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_visualization(frame_path, {**normalized, **{key: result[key] for key in ("review_status", "bbox_review_status", "string_review_status")}}, vis_path)
    return {"label_path": str(label_path), "visualization_path": str(vis_path), **result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label sampled video frames with yoyo/string geometry.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--model", default=MODEL_CONFIG.default_model)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API requests.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    records = [json.loads(line) for line in (dataset_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        record
        for record in records
        if (args.split == "all" or record["split"] == args.split)
        and (not args.candidates_only or record.get("candidate_only"))
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    results_path = dataset_dir / "auto_annotations.jsonl"
    pending = []
    for index, record in enumerate(selected, start=1):
        relative = Path(record["split"]) / record["video_id"] / f"frame_{record['frame_index']:08d}.json"
        label_path = dataset_dir / "annotations" / "labels" / relative
        if label_path.exists() and not args.force:
            print(f"[{index}/{len(selected)}] skip {label_path}")
            continue
        pending.append((index, record))

    with results_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    annotate_record,
                    record,
                    dataset_dir,
                    args.model,
                    MODEL_CONFIG.min_image_tokens,
                    MODEL_CONFIG.max_image_tokens,
                ): (index, record)
                for index, record in pending
            }
            for future in as_completed(futures):
                index, record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    error = {"frame_path": record["frame_path"], "error": str(exc)}
                    output.write(json.dumps(error, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"[{index}/{len(selected)}] failed: {exc}")
                    continue
                output.write(json.dumps({"label_path": result["label_path"], "review_status": result["review_status"]}, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{index}/{len(selected)}] {result['label_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
