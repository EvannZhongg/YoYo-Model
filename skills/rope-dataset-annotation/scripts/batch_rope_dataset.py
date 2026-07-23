"""Build dispersed, model-assisted rope annotation batches for visual review.

The semantic and detector outputs produced here are candidates only.  Labels are
created through ``rope_pipeline`` and remain pending until independent visual
review passes approve the current content digest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rope_pipeline  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def stable_source_group(path: Path) -> str:
    identity = f"{path.resolve()}|{path.stat().st_size}".encode("utf-8")
    return f"{rope_pipeline.clean_id(path.stem)[:48]}_{hashlib.sha256(identity).hexdigest()[:10]}"


def polyline_length(stroke: list[list[float]]) -> float:
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(stroke, stroke[1:])
    )


def sample_indices(frame_count: int, count: int, edge_fraction: float) -> list[int]:
    if frame_count <= 0:
        return []
    start = max(0, int(round(frame_count * edge_fraction)))
    stop = min(frame_count - 1, int(round(frame_count * (1.0 - edge_fraction))))
    if stop <= start:
        return [frame_count // 2]
    return sorted({int(round(value)) for value in np.linspace(start, stop, count)})


def observed_path(strokes: list[list[list[float]]], confidence: float) -> dict[str, Any]:
    paths = []
    for index, stroke in enumerate(strokes):
        paths.append(
            {
                "path_id": f"visible-segment-{index + 1}",
                "start_anchor": "unknown",
                "end_anchor": "unknown",
                "points_pixel": stroke,
                "edges": [
                    {
                        "from": edge_index,
                        "to": edge_index + 1,
                        "evidence": "observed",
                        "confidence": round(float(confidence), 4),
                    }
                    for edge_index in range(len(stroke) - 1)
                ],
            }
        )
    return {
        "topology": "multiple" if len(paths) > 1 else "open",
        "reconstruction_status": "partial",
        "paths": paths,
        "unresolved_gaps": [
            "Only current-frame visible segments are labeled; the full hand-to-yoyo route is not reconstructed."
        ],
    }


def choose_diverse(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return sorted(records, key=lambda item: item["frame_index"])
    chosen = [max(records, key=lambda item: item["score"])]
    remaining = [item for item in records if item is not chosen[0]]
    duration = max(1, max(item["frame_index"] for item in records) - min(item["frame_index"] for item in records))
    while remaining and len(chosen) < limit:
        def rank(item: dict[str, Any]) -> float:
            separation = min(abs(item["frame_index"] - other["frame_index"]) for other in chosen) / duration
            return float(item["score"]) + 0.85 * separation

        next_item = max(remaining, key=rank)
        chosen.append(next_item)
        remaining.remove(next_item)
    return sorted(chosen, key=lambda item: item["frame_index"])


def allocate(records: list[dict[str, Any]], target: int, max_per_source: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source_group"]].append(record)
    queues = {
        source: choose_diverse(items, max_per_source)
        for source, items in grouped.items()
    }
    selected: list[dict[str, Any]] = []
    for depth in range(max_per_source):
        candidates = [items[depth] for items in queues.values() if len(items) > depth]
        candidates.sort(key=lambda item: (-item["score"], item["source_group"]))
        for item in candidates:
            if len(selected) >= target:
                return selected
            selected.append(item)
    return selected


def prepare(args: argparse.Namespace) -> int:
    videos_dir = Path(args.videos).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.rglob("*.json")):
        raise ValueError(f"Output already contains JSON artifacts; choose an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    from ultralytics import YOLO
    from video_tracking.tracker import (
        _extract_detections,
        _load_string_model,
        _pick_yoyo,
        _predict_string_model,
    )

    detector = YOLO(str(Path(args.detector_weights).resolve()))
    class_names = {int(key): str(value) for key, value in dict(detector.names or {}).items()}
    string_model, status = _load_string_model(args.string_weights, True, args.device)
    if string_model is None:
        raise RuntimeError(f"Could not load semantic string model: {status}")

    eligible: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for video_number, video_path in enumerate(videos, start=1):
        source_group = stable_source_group(video_path)
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        target_indices = sample_indices(frame_count, args.candidates_per_video, args.edge_fraction)
        previous_index = None
        for frame_index in target_indices:
            # Decode targets in timeline order. Repeated random seeks in 4K H.264
            # streams can retain large FFmpeg buffers on Windows.
            try:
                if previous_index is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                else:
                    for _ in range(previous_index + 1, frame_index):
                        if not capture.grab():
                            break
                ok, frame = capture.read()
            except cv2.error:
                ok, frame = False, None
            previous_index = frame_index
            if not ok or frame is None:
                continue
            result = detector.predict(
                source=frame,
                conf=args.detector_confidence,
                imgsz=args.detector_imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections = _extract_detections(result, class_names)
            yoyo, detector_flags = _pick_yoyo(detections)
            observation = None
            if yoyo is not None:
                observation = _predict_string_model(
                    string_model,
                    frame,
                    yoyo,
                    args.string_confidence,
                    args.detector_imgsz,
                    args.device,
                    "hand_and_yoyo_attached",
                    args.semantic_scale,
                )
            row: dict[str, Any] = {
                "video": str(video_path),
                "video_name": video_path.name,
                "source_group": source_group,
                "frame_index": frame_index,
                "timestamp_s": round(frame_index / fps, 4) if fps else None,
                "fps": fps,
                "frame_count": frame_count,
                "image_size": [width, height],
                "detector_flags": detector_flags,
                "yoyo": yoyo,
                "observation": observation,
                "eligible": False,
            }
            if yoyo is not None and observation is not None:
                strokes = [stroke for stroke in observation.get("polylines") or [] if len(stroke) >= 2]
                total_length = sum(polyline_length(stroke) for stroke in strokes)
                component_count = int(observation.get("component_count", len(strokes)))
                body_overlap = float(observation.get("yoyo_body_overlap_fraction", 0.0))
                confidence = float(observation.get("confidence", 0.0))
                detector_confidence = float(yoyo.get("confidence", 0.0))
                diagonal = math.hypot(width, height)
                eligible_now = bool(
                    strokes
                    and total_length >= args.min_length_fraction * diagonal
                    and component_count <= args.max_components
                    and body_overlap <= args.max_yoyo_overlap
                    and not detector_flags
                )
                score = (
                    2.0 * confidence
                    + detector_confidence
                    + 0.15 * math.log1p(total_length)
                    - 0.10 * max(0, component_count - 1)
                    - body_overlap
                )
                row.update(
                    {
                        "eligible": eligible_now,
                        "score": round(score, 6),
                        "total_length_px": round(total_length, 2),
                    }
                )
                if eligible_now:
                    eligible.append(row)
            inference_rows.append(row)
            del result, detections, frame, observation
        capture.release()
        gc.collect()
        print(
            f"[{video_number}/{len(videos)}] {video_path.name}: "
            f"eligible={sum(item['source_group'] == source_group for item in eligible)}",
            flush=True,
        )

    selected = allocate(eligible, args.preliminary_count, args.max_per_source)
    if len(selected) < args.minimum_required:
        raise RuntimeError(
            f"Only {len(selected)} eligible dispersed candidates were found; "
            f"minimum_required={args.minimum_required}"
        )

    images_root = output / "project" / "images" / "train"
    labels_root = output / "project" / "labels" / "train"
    candidates_root = output / "project" / "candidates"
    review_root = output / "project" / "review" / "draft"
    selected_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        selected_by_video[record["video"]].append(record)

    serial = 0
    for video_value, records in selected_by_video.items():
        capture = cv2.VideoCapture(video_value)
        for record in sorted(records, key=lambda item: item["frame_index"]):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not re-extract selected frame {record['frame_index']} from {video_value}")
            serial += 1
            stem = f"sample_{serial:03d}_f{int(record['frame_index']):06d}"
            group = record["source_group"]
            image_path = images_root / group / f"{stem}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise OSError(f"Could not write {image_path}")

            refined_observation = _predict_string_model(
                string_model,
                frame,
                record["yoyo"],
                args.string_confidence,
                args.detector_imgsz,
                args.device,
                "hand_and_yoyo_attached",
                args.refine_semantic_scale,
            )
            if refined_observation is not None:
                record["observation"] = refined_observation
                record["refined_at_scale"] = args.refine_semantic_scale
            else:
                record["refined_at_scale"] = None

            label_path = labels_root / group / f"{stem}.json"
            label = rope_pipeline.initial_label(image_path.resolve(), "train", group, 2)
            label["frame_index"] = int(record["frame_index"])
            label["timestamp_s"] = record["timestamp_s"]
            rope_pipeline.write_json(label_path, label)

            observation = record["observation"]
            strokes = observation["polylines"]
            bad_case = ["partial_reconstruction"]
            if len(strokes) > 1:
                bad_case.append("multiple_visible_components")
            candidate = {
                "visibility": "visible",
                "yoyo_bbox_pixel": record["yoyo"]["bbox"],
                "string_visibility": "partial",
                "string_polylines_pixel": strokes,
                "string_mask_polygons_pixel": None,
                "hands_pixel": {"left": None, "right": None},
                "string_attachment_class": "unknown",
                "scene_label": "competition",
                "string_path": observed_path(strokes, observation["confidence"]),
                "bad_case": bad_case,
                "notes": (
                    "Semantic v17 candidate anchored to a detected yoyo. "
                    "Visible centerline segments only; visual review is required."
                ),
            }
            candidate_path = candidates_root / group / f"{stem}.json"
            rope_pipeline.write_json(candidate_path, candidate)
            rope_pipeline.apply_candidate(
                label_path,
                candidate,
                "semantic-v17-batch-annotator",
                "model-annotator",
                "yoyo_string_semantic_v17_reviewed_expansion_hn005",
                "initial dispersed current-frame trace",
            )
            rope_pipeline.command_render(
                SimpleNamespace(label=str(label_path), output=str(review_root), max_side=args.render_max_side)
            )
            record.update(
                {
                    "serial": serial,
                    "stem": stem,
                    "image": str(image_path.resolve()),
                    "label": str(label_path.resolve()),
                    "candidate": str(candidate_path.resolve()),
                }
            )
        capture.release()

    manifest = {
        "schema_version": "rope_batch_selection_v1",
        "videos_dir": str(videos_dir),
        "video_count": len(videos),
        "inference_count": len(inference_rows),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "requested_final_count": args.minimum_required,
        "sampling": {
            "candidates_per_video": args.candidates_per_video,
            "edge_fraction": args.edge_fraction,
            "max_per_source": args.max_per_source,
            "semantic_scale": args.semantic_scale,
        },
        "selected": sorted(selected, key=lambda item: item["serial"]),
        "inference": inference_rows,
    }
    rope_pipeline.write_json(output / "project" / "selection_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("video_count", "inference_count", "eligible_count", "selected_count")}, ensure_ascii=False, indent=2))
    return 0


def migrate_existing(args: argparse.Namespace) -> int:
    source_root = Path(args.labels).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.rglob("*.json")):
        raise ValueError(f"Output already contains JSON artifacts; choose an empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    accepted = []
    for source_label_path in rope_pipeline.label_files(source_root):
        source_label = rope_pipeline.read_json(source_label_path)
        if str(source_label.get("string_review_status")) not in {"reviewed", "approved"}:
            continue
        if str(source_label.get("string_visibility")) == "uncertain":
            continue
        accepted.append((source_label_path, source_label))
    accepted.sort(
        key=lambda item: (
            str(item[1].get("source_group")),
            int(item[1].get("frame_index") or 0),
        )
    )
    if len(accepted) < args.minimum_required:
        raise RuntimeError(f"Only {len(accepted)} accepted legacy labels were found")

    migrated = []
    review_root = output / "project" / "review" / "draft"
    for serial, (source_label_path, source_label) in enumerate(accepted, start=1):
        source_image = Path(str(source_label["source_image"])).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        split = str(source_label.get("split", "train"))
        group = str(source_label.get("source_group") or source_label.get("video_id") or "unknown")
        frame_index = int(source_label.get("frame_index") or 0)
        stem = f"sample_{serial:03d}_f{frame_index:06d}"
        image_path = output / "project" / "images" / split / group / f"{stem}{source_image.suffix.lower()}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_path)

        label_path = output / "project" / "labels" / split / group / f"{stem}.json"
        label = rope_pipeline.initial_label(image_path, split, group, 2)
        label["frame_index"] = frame_index
        label["timestamp_s"] = source_label.get("timestamp_s")
        rope_pipeline.write_json(label_path, label)

        visibility = str(source_label.get("string_visibility"))
        strokes = source_label.get("string_polylines_pixel") or []
        masks = source_label.get("string_mask_polygons_pixel") or []
        # Prefer an already reviewed centerline. For mask-only legacy labels,
        # retain the reviewed mask and let rope_pipeline derive its midline.
        candidate_masks = masks if not strokes else None
        candidate = {
            "visibility": source_label.get("visibility", "uncertain"),
            "yoyo_bbox_pixel": source_label.get("yoyo_bbox_pixel"),
            "string_visibility": visibility,
            "string_polylines_pixel": strokes or None,
            "string_mask_polygons_pixel": candidate_masks,
            "hands_pixel": source_label.get("hands_pixel") or {"left": None, "right": None},
            "string_attachment_class": source_label.get("string_attachment_class", "unknown"),
            "scene_label": source_label.get("scene_label", "competition"),
            "string_path": (
                observed_path(strokes, 1.0)
                if strokes
                else {
                    "topology": "uncertain",
                    "reconstruction_status": "uncertain",
                    "paths": [],
                    "unresolved_gaps": [],
                }
            ),
            "bad_case": source_label.get("bad_case") or [],
            "notes": (
                "Migrated from a legacy reviewed label for independent current-protocol review. "
                f"Legacy label: {source_label_path.name}."
            ),
        }
        candidate_path = output / "project" / "candidates" / split / group / f"{stem}.json"
        rope_pipeline.write_json(candidate_path, candidate)
        rope_pipeline.apply_candidate(
            label_path,
            candidate,
            "legacy-review-migrator",
            "model-annotator",
            "legacy-reviewed-v1",
            "migrate evidence for current-protocol review",
        )
        if visibility in {"visible", "partial"} and not strokes and candidate_masks:
            rope_pipeline.command_derive_centerlines(
                SimpleNamespace(
                    label=str(label_path),
                    actor="model-mask-centerline-deriver",
                    model="rope-pipeline-mask-midline",
                    message="derive centerline from legacy reviewed mask",
                    min_component_pixels=8,
                    max_points=64,
                )
            )
        rope_pipeline.command_render(
            SimpleNamespace(label=str(label_path), output=str(review_root), max_side=args.render_max_side)
        )
        migrated.append(
            {
                "serial": serial,
                "source_label": str(source_label_path.resolve()),
                "source_group": group,
                "split": split,
                "frame_index": frame_index,
                "visibility": visibility,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "candidate": str(candidate_path.resolve()),
            }
        )

    manifest = {
        "schema_version": "rope_batch_selection_v1",
        "selection_mode": "legacy-reviewed-migration-for-current-protocol-review",
        "selected_count": len(migrated),
        "requested_final_count": args.minimum_required,
        "source_group_count": len({item["source_group"] for item in migrated}),
        "selected": migrated,
    }
    rope_pipeline.write_json(output / "project" / "selection_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("selected_count", "requested_final_count", "source_group_count")}, ensure_ascii=False, indent=2))
    return 0


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        canvas = Image.new("RGB", size, "#101214")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), "no detail geometry", fill="#c8cdd2", font=ImageFont.load_default())
        return canvas
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#101214")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def sheets(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    labels = rope_pipeline.label_files(output / "project" / "labels")
    render_root = output / "project" / "review" / args.render_set
    sheet_root = output / "project" / "review" / f"{args.phase}_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cell_width, cell_height = args.cell_width, round(args.cell_width * 9 / 16)
    columns = ("raw", "grid", "overlay", "detail")
    for batch_index in range(0, len(labels), args.batch_size):
        batch = labels[batch_index : batch_index + args.batch_size]
        header_height = 34
        row_height = cell_height + header_height
        canvas = Image.new("RGB", (cell_width * len(columns), row_height * len(batch)), "#151719")
        draw = ImageDraw.Draw(canvas)
        for row_index, label_path in enumerate(batch):
            label = rope_pipeline.read_json(label_path)
            serial = label_path.stem.split("_")[1]
            caption = (
                f"S{serial} | {Path(label['source_image']).parent.name[:34]} | "
                f"f={label.get('frame_index')} t={label.get('timestamp_s')} | {args.phase}"
            )
            y = row_index * row_height
            draw.rectangle((0, y, canvas.width, y + header_height), fill="#202428")
            draw.text((8, y + 9), caption, fill="white", font=font)
            artifacts = {
                "raw": Path(label["source_image"]),
                "grid": render_root / f"{label_path.stem}_grid.jpg",
                "overlay": render_root / f"{label_path.stem}_overlay.jpg",
                "detail": render_root / f"{label_path.stem}_detail.jpg",
            }
            for column_index, name in enumerate(columns):
                artifact = artifacts[name]
                cell = fit_image(artifact, (cell_width, cell_height))
                x = column_index * cell_width
                canvas.paste(cell, (x, y + header_height))
                draw.rectangle((x + 4, y + header_height + 4, x + 68, y + header_height + 22), fill="#000000")
                draw.text((x + 8, y + header_height + 7), name, fill="white", font=font)
        path = sheet_root / f"{args.phase}_{batch_index // args.batch_size + 1:03d}.jpg"
        canvas.save(path, quality=94)
    print(json.dumps({"phase": args.phase, "labels": len(labels), "sheets": math.ceil(len(labels) / args.batch_size), "output": str(sheet_root)}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--videos", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--detector-weights", required=True)
    prepare_parser.add_argument("--string-weights", required=True)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument("--detector-imgsz", type=int, default=1280)
    prepare_parser.add_argument("--detector-confidence", type=float, default=0.20)
    prepare_parser.add_argument("--string-confidence", type=float, default=0.0)
    prepare_parser.add_argument("--semantic-scale", type=float, default=2.0)
    prepare_parser.add_argument("--refine-semantic-scale", type=float, default=2.0)
    prepare_parser.add_argument("--candidates-per-video", type=int, default=10)
    prepare_parser.add_argument("--edge-fraction", type=float, default=0.10)
    prepare_parser.add_argument("--preliminary-count", type=int, default=130)
    prepare_parser.add_argument("--minimum-required", type=int, default=100)
    prepare_parser.add_argument("--max-per-source", type=int, default=5)
    prepare_parser.add_argument("--max-components", type=int, default=4)
    prepare_parser.add_argument("--max-yoyo-overlap", type=float, default=0.45)
    prepare_parser.add_argument("--min-length-fraction", type=float, default=0.003)
    prepare_parser.add_argument("--render-max-side", type=int, default=1800)
    prepare_parser.set_defaults(func=prepare)

    migrate_parser = subparsers.add_parser("migrate-existing")
    migrate_parser.add_argument("--labels", required=True)
    migrate_parser.add_argument("--output", required=True)
    migrate_parser.add_argument("--minimum-required", type=int, default=100)
    migrate_parser.add_argument("--render-max-side", type=int, default=1800)
    migrate_parser.set_defaults(func=migrate_existing)

    sheets_parser = subparsers.add_parser("sheets")
    sheets_parser.add_argument("--output", required=True)
    sheets_parser.add_argument("--phase", choices=("geometry", "semantic"), required=True)
    sheets_parser.add_argument("--render-set", default="draft")
    sheets_parser.add_argument("--batch-size", type=int, default=5)
    sheets_parser.add_argument("--cell-width", type=int, default=440)
    sheets_parser.set_defaults(func=sheets)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
