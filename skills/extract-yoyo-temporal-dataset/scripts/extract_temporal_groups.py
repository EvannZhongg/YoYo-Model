#!/usr/bin/env python3
"""Extract fixed-size consecutive frame groups for temporal yoyo/string models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ANNOTATION_SCHEMA = "agent_yoyo_string_annotation_v5"
DATASET_SCHEMA = "yoyo_consecutive_annotation_dataset_v1"
SAMPLING_SCHEMA = "agent_video_sampling_v1"
GROUP_SCHEMA = "yoyo_consecutive_groups_v1"
TEMPORAL_REVIEW_SCHEMA = "yoyo_temporal_review_v1"
TEMPORAL_REVIEW_FILENAME = "temporal_review.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "-", value, flags=re.UNICODE).strip("-.")
    return cleaned or "video"


def discover_videos(value: str) -> list[Path]:
    root = Path(value).expanduser().resolve()
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    else:
        raise FileNotFoundError(f"video path not found: {root}")
    if not paths:
        raise ValueError(f"no supported videos found under: {root}")
    return paths


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if frames < 1 or width < 1 or height < 1:
        raise RuntimeError(f"video has invalid metadata: {path}")
    return {"frame_count": frames, "fps": fps if fps > 0 else 30.0, "image_size": [width, height]}


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_provenance(paths: list[Path]) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for path in paths:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"reference manifest not found: {manifest_path}")
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in document.get("records") or []:
            if not isinstance(record, dict):
                continue
            source_hash_value = record.get("source_video_sha256")
            frame_index = record.get("frame_index")
            # The unified 1Ayoyo manifest keeps provenance in each label rather
            # than repeating it in every manifest record; read that fallback.
            if not source_hash_value or frame_index is None:
                label_value = record.get("canonical_label") or record.get("label")
                if label_value:
                    label_path = Path(str(label_value))
                    if not label_path.is_absolute():
                        label_path = path / label_path
                    if label_path.is_file():
                        label = json.loads(label_path.read_text(encoding="utf-8"))
                        source_hash_value = label.get("source_video_sha256")
                        frame_index = label.get("frame_index")
            if source_hash_value and frame_index is not None:
                result.add((str(source_hash_value).lower(), int(frame_index)))
    return result


def allocate_counts(infos: list[dict[str, Any]], total: int | None, per_video: int | None, group_size: int) -> list[int]:
    capacities = [int(info["frame_count"]) // group_size for info in infos]
    if per_video is not None:
        if per_video < 1:
            raise ValueError("--groups-per-video must be positive")
        if any(per_video > capacity for capacity in capacities):
            raise ValueError(f"at least one video cannot hold {per_video} non-overlapping groups of {group_size} frames")
        return [per_video] * len(infos)
    if total is None or total < 1:
        raise ValueError("provide a positive --groups value or --groups-per-video")
    if total > sum(capacities):
        raise ValueError(f"requested {total} groups but sources hold at most {sum(capacities)} groups")
    counts = [0] * len(infos)
    remaining = total
    while remaining:
        progressed = False
        for index, capacity in enumerate(capacities):
            if counts[index] < capacity and remaining:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("could not allocate groups across sources")
    return counts


def choose_starts(frame_count: int, group_size: int, count: int, source_hash_value: str, reference: set[tuple[str, int]]) -> list[int]:
    max_start = frame_count - group_size
    if max_start < 0:
        raise ValueError("video is shorter than --frames-per-group")
    starts: list[int] = []
    # Even spacing is deterministic and exposes early, middle, and late motion.
    candidates = [round(index * max_start / max(1, count - 1)) for index in range(count)]
    candidates = sorted(set(int(value) for value in candidates))
    # Always retain a deterministic fallback scan: an evenly spaced candidate
    # can be rejected by reference provenance or overlap, even when the
    # initial candidate list already has the requested length.
    candidates.extend(index for index in range(max_start + 1) if index not in candidates)
    for start in candidates:
        if len(starts) >= count:
            break
        window = range(start, start + group_size)
        if any((source_hash_value, frame) in reference for frame in window):
            continue
        if any(start < old + group_size and old < start + group_size for old in starts):
            continue
        starts.append(start)
    if len(starts) != count:
        raise ValueError("not enough non-overlapping, non-reference windows; reduce group count or disable the reference check")
    return sorted(starts)


def read_frame(cap: cv2.VideoCapture, index: int) -> Any:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {index}")
    return frame


def initial_label(record: dict[str, Any], sampling_hash: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "created_at_utc": now,
        "updated_at_utc": now,
        "source_image": Path(record["image"]).relative_to("canonical").as_posix(),
        "image_sha256": record["image_sha256"],
        "image_size": record["image_size"],
        "source_video": record["source_video"],
        "source_video_sha256": record["source_video_sha256"],
        "source_group": record["source_group"],
        "video_id": record["source_group"],
        "frame_index": record["frame_index"],
        "timestamp_s": record["timestamp_s"],
        "sequence_id": record["sequence_id"],
        "group_id": record["group_id"],
        "sampling_role": "anchor",
        "anchor_frame_index": record["frame_index"],
        "sampling_manifest_sha256": sampling_hash,
        "active_yoyo": {"visibility": "visible", "not_visible_reason": None, "trick_orientation": "normal", "presentation_orientation": "frontal", "bbox_pixel": None, "bbox_2d": None, "bbox_review_status": "needs_review"},
        "backup_yoyos": [],
        "string_visibility": "partial",
        "string_polylines_pixel": None,
        "string_polylines_2d": None,
        "string_polyline_pixel": None,
        "string_polyline_2d": None,
        "string_mask_polygons_pixel": None,
        "yoyo_division": "1A",
        "scene_label": "unknown",
        "string_path": {"topology": "uncertain", "reconstruction_status": "uncertain", "paths": [], "unresolved_gaps": []},
        "bad_case": [],
        "review_status": "needs_review",
        "string_review_status": "unresolved",
        "quality": {"revision": 0, "min_model_approvals": 1, "history": [], "reviews": []},
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    videos = discover_videos(args.videos)
    infos = [video_info(path) for path in videos]
    counts = allocate_counts(infos, args.groups, args.groups_per_video, args.frames_per_group)
    reference = load_reference_provenance([Path(value).expanduser().resolve() for value in args.reference_dataset])
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    (output / "canonical" / "images").mkdir(parents=True)
    (output / "canonical" / "labels").mkdir(parents=True)
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for video, info, count in zip(videos, infos, counts):
        digest = source_hash(video)
        source_group = f"{clean_id(video.stem)[:64]}-{digest[:10]}"
        starts = choose_starts(info["frame_count"], args.frames_per_group, count, digest, reference)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video}")
        source_groups = []
        try:
            for group_index, start in enumerate(starts, 1):
                sequence_id = f"group-{group_index:03d}"
                group_id = f"{source_group}--{sequence_id}"
                group_frames = []
                for offset in range(args.frames_per_group):
                    frame_index = start + offset
                    frame = read_frame(cap, frame_index)
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    if not ok:
                        raise RuntimeError(f"could not encode frame {frame_index}")
                    jpeg = encoded.tobytes()
                    image_hash = sha256_bytes(jpeg)
                    filename = f"{sequence_id}_frame_{frame_index:08d}-{image_hash[:10]}.jpg"
                    image_rel = Path("canonical") / "images" / source_group / filename
                    label_rel = Path("canonical") / "labels" / source_group / filename.replace(".jpg", ".json")
                    (output / image_rel).parent.mkdir(parents=True, exist_ok=True)
                    (output / label_rel).parent.mkdir(parents=True, exist_ok=True)
                    (output / image_rel).write_bytes(jpeg)
                    record = {"source_video": str(video), "source_video_sha256": digest, "source_group": source_group, "sequence_id": sequence_id, "group_id": group_id, "group_index": group_index, "frame_offset": offset, "frame_index": frame_index, "timestamp_s": round(frame_index / info["fps"], 6), "image_size": info["image_size"], "image": image_rel.as_posix(), "label": label_rel.as_posix(), "image_sha256": image_hash}
                    records.append(record)
                    group_frames.append({"sample_key": (Path(source_group) / label_rel.name).as_posix(), "image": image_rel.as_posix(), "frame_index": frame_index, "timestamp_s": record["timestamp_s"]})
                group = {"group_id": group_id, "source_video": str(video), "source_video_sha256": digest, "source_group": source_group, "sequence_id": sequence_id, "original_start_frame": start, "original_end_frame": start + args.frames_per_group - 1, "selected_start_frame": start, "selected_end_frame": start + args.frames_per_group - 1, "start_sample_key": group_frames[0]["sample_key"], "propagated_from_start_at_utc": None, "frames": group_frames}
                groups.append(group)
                source_groups.append({"group_id": group_id, "sequence_id": sequence_id, "start_frame": start, "end_frame": start + args.frames_per_group - 1})
        finally:
            cap.release()
        sources.append({"source_video": str(video), "source_video_sha256": digest, "source_group": source_group, **info, "group_count": count, "groups": source_groups})
    sampling = {"schema_version": SAMPLING_SCHEMA, "created_at_utc": utc_now(), "sampling_method": "evenly_spaced_non_overlapping_consecutive_groups", "recognition_model_used": False, "videos_root": str(Path(args.videos).expanduser().resolve()), "parameters": {"frames_per_group": args.frames_per_group, "groups": args.groups, "groups_per_video": args.groups_per_video, "jpeg_quality": args.jpeg_quality, "reference_dataset": args.reference_dataset}, "source_count": len(sources), "group_count": len(groups), "record_count": len(records), "sources": sources, "records": records, "failures": []}
    sampling_path = output / "sampling_manifest.json"
    write_json(sampling_path, sampling)
    sampling_hash = sha256_bytes(sampling_path.read_bytes())
    for record in records:
        write_json(output / record["label"], initial_label(record, sampling_hash))
    manifest = {"schema_version": DATASET_SCHEMA, "dataset_id": output.name, "created_at_utc": utc_now(), "output_dir": str(output), "annotation_schema_version": ANNOTATION_SCHEMA, "sampling_manifest_sha256": sampling_hash, "recognition_model_used": False, "agent_annotation_used": False, "model_review_used": False, "reference_datasets": list(args.reference_dataset), "deduplication": {"source_frame_identity": bool(reference), "image_sha256": False, "within_group_repeated_frames_allowed": True}, "source_count": len(sources), "group_count": len(groups), "sample_count": len(records), "records": [{"image": r["image"], "label": r["label"], "image_sha256": r["image_sha256"], "source_video_sha256": r["source_video_sha256"], "source_group": r["source_group"], "frame_index": r["frame_index"], "group_id": r["group_id"]} for r in records]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "consecutive_groups.json", {"schema_version": GROUP_SCHEMA, "dataset_id": output.name, "created_at_utc": utc_now(), "updated_at_utc": utc_now(), "frame_selection_policy": "source_frames", "groups": groups})
    write_json(
        output / TEMPORAL_REVIEW_FILENAME,
        {
            "schema_version": TEMPORAL_REVIEW_SCHEMA,
            "dataset_id": output.name,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "minimum_selected_frames": 3,
            "groups": {
                group["group_id"]: {
                    "status": "unresolved",
                    "selected_sample_keys": [frame["sample_key"] for frame in group["frames"]],
                    "selected_frame_indices": [int(frame["frame_index"]) for frame in group["frames"]],
                    "reviewer": None,
                    "confirmed_at_utc": None,
                }
                for group in groups
            },
        },
    )
    return {"ok": True, "dataset": str(output), "source_count": len(sources), "group_count": len(groups), "sample_count": len(records), "groups_per_source": counts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, help="one video file or a directory searched recursively")
    parser.add_argument("--output", required=True, help="new dataset directory, normally under datasets/")
    parser.add_argument("--groups", type=int, help="total groups; directory inputs are allocated round-robin")
    parser.add_argument("--groups-per-video", type=int, help="fixed group count for every source video")
    parser.add_argument("--frames-per-group", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--reference-dataset", action="append", default=None, help="dataset root whose manifest source frames are excluded; repeatable (default: datasets/1Ayoyo_dataset and datasets/1Ayoyo_consecutive); pass --reference-dataset \"\" to disable")
    args = parser.parse_args(argv)
    if bool(args.groups) == bool(args.groups_per_video):
        parser.error("provide exactly one of --groups or --groups-per-video")
    if args.frames_per_group < 1 or args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("frames-per-group must be positive and jpeg-quality must be in [1, 100]")
    if args.groups is not None and args.groups < 1:
        parser.error("--groups must be positive")
    if args.reference_dataset is None:
        args.reference_dataset = ["datasets/1Ayoyo_dataset", "datasets/1Ayoyo_consecutive"]
    elif args.reference_dataset == [""]:
        args.reference_dataset = []
    elif "" in args.reference_dataset:
        parser.error("an empty --reference-dataset cannot be combined with other references")
    return args


if __name__ == "__main__":
    try:
        print(json.dumps(extract(parse_args()), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
