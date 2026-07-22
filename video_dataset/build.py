"""Build a reviewable, video-first frame dataset.

Frames intentionally start as ``unreviewed`` records. An empty annotation is
not treated as a negative sample until a human confirms that the yoyo is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path: Path, videos_dir: Path, action_group: str = "1A") -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    relative_path = path.resolve().relative_to(videos_dir.resolve()).as_posix()
    video_id = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
    return {
        "video_id": video_id,
        "path": str(path.resolve()),
        "filename": path.name,
        "action_group": action_group,
        "source_group": video_id,
        "subject_id": None,
        "sha256": sha256_file(path),
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_s": round(frames / fps, 3) if fps else None,
    }


def assign_splits(sources: list[dict[str, Any]], seed: int, val_ratio: float, test_ratio: float) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        groups.setdefault(source["source_group"], []).append(source)
    names = list(groups)
    random.Random(seed).shuffle(names)
    test_count = max(1, round(len(names) * test_ratio)) if len(names) > 2 else 0
    val_count = max(1, round(len(names) * val_ratio)) if len(names) > 3 else 0
    test_names = set(names[:test_count])
    val_names = set(names[test_count : test_count + val_count])
    for source in sources:
        source["split"] = "test" if source["source_group"] in test_names else "val" if source["source_group"] in val_names else "train"


def write_source_manifest(
    videos_dir: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    action_group: str = "1A",
) -> dict[str, Any]:
    videos = sorted(path for path in videos_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    sources = [video_info(path, videos_dir, action_group) for path in videos]
    assign_splits(sources, seed, val_ratio, test_ratio)
    manifest = {
        "schema_version": "1.0",
        "current_action_group": action_group,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos_dir": str(videos_dir.resolve()),
        "seed": seed,
        "split_strategy": "source_group",
        "source_group_policy": "Each video is isolated by default. Set the same source_group manually for related videos before rebuilding frames.",
        "ratios": {"val": val_ratio, "test": test_ratio},
        "sources": sources,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def extract_frames(manifest: dict[str, Any], output_dir: Path, sample_fps: float, split: str, max_videos: int, max_frames_per_video: int) -> dict[str, Any]:
    frame_root = output_dir / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frames.jsonl"
    existing_records = []
    if manifest_path.exists():
        existing_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records_by_path = {record["frame_path"]: record for record in existing_records}
    new_count = 0
    selected = [source for source in manifest["sources"] if split == "all" or source["split"] == split]
    if max_videos > 0:
        selected = selected[:max_videos]
    for source in selected:
        capture = cv2.VideoCapture(source["path"])
        stride = max(1, int(round(source["fps"] / sample_fps))) if sample_fps > 0 and source["fps"] else 1
        video_dir = frame_root / source["split"] / source["video_id"]
        video_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        saved = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frame_path = video_dir / f"frame_{index:08d}.jpg"
                cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                record = {
                    "schema_version": "1.0",
                    "frame_path": str(frame_path.resolve()),
                    "source_video": source["path"],
                    "source_video_sha256": source["sha256"],
                    "video_id": source["video_id"],
                    "source_group": source["source_group"],
                    "action_group": source.get("action_group", manifest.get("current_action_group", "1A")),
                    "subject_id": source.get("subject_id"),
                    "split": source["split"],
                    "frame_index": index,
                    "timestamp_s": round(index / source["fps"], 4) if source["fps"] else None,
                    "annotation_status": "unreviewed",
                    "visibility": "unknown",
                    "yoyo_bbox": None,
                    "string_polyline": None,
                    "hands": None,
                    "pose": None,
                    "bad_case": [],
                    "review_notes": "",
                }
                if record["frame_path"] not in records_by_path:
                    new_count += 1
                records_by_path[record["frame_path"]] = record
                saved += 1
                if max_frames_per_video > 0 and saved >= max_frames_per_video:
                    break
            index += 1
        capture.release()
    records = sorted(records_by_path.values(), key=lambda item: (item["split"], item["video_id"], item["frame_index"]))
    with manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(), "frames_jsonl": str(manifest_path.resolve()), "frame_count": len(records), "new_frame_count": new_count, "split": split, "sample_fps": sample_fps}
    (output_dir / "frames_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a video-first yoyo frame dataset and performer-isolated splits.")
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--output-dir", default="datasets/video_v1")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--rebuild-sources", action="store_true")
    parser.add_argument("--action-group", default="1A", help="Current dataset group (videos are 1A for this project).")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    sources_path = output_dir / "sources.json"
    manifest = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() and not args.rebuild_sources else write_source_manifest(Path(args.videos_dir), output_dir, args.seed, args.val_ratio, args.test_ratio, args.action_group)
    summary = extract_frames(manifest, output_dir, args.sample_fps, args.split, args.max_videos, args.max_frames_per_video)
    print(json.dumps({"sources": len(manifest["sources"]), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
