#!/usr/bin/env python3
"""Sample source-balanced, temporally dispersed video frames without recognition models."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v3 as iio
except ImportError:
    iio = None


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}
SCHEMA_VERSION = "agent_video_sampling_v1"
HASH_CACHE_SCHEMA_VERSION = "agent_video_sha256_cache_v1"
SAMPLED_FRAME_PATTERN = re.compile(
    r"(?P<sequence>seq-(?P<sequence_number>\d{3})-anchor-(?P<anchor>\d{8}))_"
    r"(?P<role>anchor|temporal_context)_frame_(?P<frame>\d{8})\.jpg"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quick_file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash small, dispersed byte ranges before trusting a cached full digest."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)}):
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def load_hash_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": HASH_CACHE_SCHEMA_VERSION, "entries": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": HASH_CACHE_SCHEMA_VERSION, "entries": {}}
    if value.get("schema_version") != HASH_CACHE_SCHEMA_VERSION or not isinstance(value.get("entries"), dict):
        return {"schema_version": HASH_CACHE_SCHEMA_VERSION, "entries": {}}
    return value


def cached_sha256(path: Path, entries: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    resolved = str(path.resolve())
    stat = path.stat()
    fingerprint = quick_file_fingerprint(path)
    cached = entries.get(resolved)
    if (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("quick_fingerprint") == fingerprint
        and isinstance(cached.get("sha256"), str)
        and len(cached["sha256"]) == 64
    ):
        digest = cached["sha256"].lower()
        return digest, True, cached
    digest = sha256_file(path)
    entry = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "quick_fingerprint": fingerprint,
        "sha256": digest,
    }
    return digest, False, entry


def resolve_source_hashes(
    videos: list[Path], cache_path: Path, workers: int
) -> tuple[dict[Path, str], dict[str, int]]:
    cache = load_hash_cache(cache_path)
    entries = cache["entries"]

    def calculate(path: Path) -> tuple[Path, str, bool, dict[str, Any]]:
        digest, hit, entry = cached_sha256(path, entries)
        return path, digest, hit, entry

    results: dict[Path, str] = {}
    hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for path, digest, hit, entry in executor.map(calculate, videos):
            results[path] = digest
            entries[str(path.resolve())] = entry
            hits += int(hit)
    cache["updated_at_utc"] = utc_now()
    write_json(cache_path, cache)
    return results, {"hit_count": hits, "computed_count": len(videos) - hits}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_.-" else "-" for character in value)
    return cleaned.strip("-.") or "video"


def parse_offsets(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        return sorted({int(item.strip()) for item in value.split(",") if int(item.strip()) != 0})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("neighbor offsets must be comma-separated frame offsets") from exc


def descriptor(frame: np.ndarray) -> np.ndarray:
    """Return a cheap appearance descriptor; it performs no object recognition."""
    image = Image.fromarray(ensure_rgb(frame)).resize((48, 27), Image.Resampling.BILINEAR)
    small = np.asarray(image, dtype=np.float32)
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32)
    histograms = [
        np.histogram(small[:, :, channel], bins=bins, range=(0, 256))[0].astype(np.float32)
        for channel, bins in ((0, 12), (1, 8), (2, 8))
    ]
    appearance = np.asarray(gray_image.resize((16, 9), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1)
    gy, gx = np.gradient(gray)
    edges = np.hypot(gx, gy)
    features = np.concatenate([*(item.astype(np.float32) for item in histograms), appearance, [float(edges.mean())]])
    norm = float(np.linalg.norm(features))
    return features / norm if norm else features


def appearance_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


@dataclass
class Candidate:
    frame_index: int
    frame: np.ndarray
    descriptor: np.ndarray


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    if iio is None:
        raise RuntimeError("video decoding requires OpenCV or imageio")
    frames = [ensure_rgb(frame) for frame in iio.imiter(path)]
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    fps = 0.0
    try:
        metadata = iio.immeta(path)
        fps = float(metadata.get("fps") or 0.0)
    except Exception:
        fps = 0.0
    if fps <= 0:
        fps = 10.0
    return frames, fps


def read_frames_cv2(capture: Any, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode only requested frames through container-backed random seeks."""
    frames: dict[int, np.ndarray] = {}
    requested = sorted(set(indices))
    if not requested:
        return frames

    # Neighbor windows contain short spans separated by an omitted anchor.
    # Seek once per span and decode the intervening frame instead of forcing a
    # costly container seek for every individual neighbor.
    spans: list[tuple[int, int]] = []
    span_start = previous = requested[0]
    for frame_index in requested[1:]:
        if frame_index - previous > 2:
            spans.append((span_start, previous))
            span_start = frame_index
        previous = frame_index
    spans.append((span_start, previous))

    requested_set = set(requested)
    for start, stop in spans:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
            continue
        for frame_index in range(start, stop + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index in requested_set:
                frames[frame_index] = frame[:, :, ::-1]
    return frames


def read_candidates_cv2(capture: Any, indices: list[int]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for frame_index, frame in read_frames_cv2(capture, indices).items():
        candidates.append(Candidate(frame_index, frame, descriptor(frame)))
    return candidates


def read_candidates(frames: list[np.ndarray], indices: list[int]) -> list[Candidate]:
    return [
        Candidate(frame_index, frames[frame_index], descriptor(frames[frame_index]))
        for frame_index in sorted(set(indices))
        if 0 <= frame_index < len(frames)
    ]


def select_anchors(candidates: list[Candidate], frame_count: int, count: int) -> list[Candidate]:
    """Choose one frame per temporal stratum, preferring appearance diversity."""
    if not candidates or count <= 0:
        return []
    count = min(count, len(candidates))
    selected: list[Candidate] = []
    for stratum in range(count):
        start = frame_count * stratum / count
        stop = frame_count * (stratum + 1) / count
        pool = [item for item in candidates if start <= item.frame_index < stop and item not in selected]
        if not pool:
            pool = [item for item in candidates if item not in selected]
        center = (start + stop) / 2.0

        def score(item: Candidate) -> tuple[float, float, int]:
            diversity = min((appearance_distance(item.descriptor, prior.descriptor) for prior in selected), default=0.0)
            temporal_centering = 1.0 - abs(item.frame_index - center) / max(1.0, stop - start)
            return diversity, temporal_centering, -item.frame_index

        selected.append(max(pool, key=score))
    return sorted(selected, key=lambda item: item.frame_index)


def candidate_indices(frame_count: int, count: int, oversample: int, edge_fraction: float) -> list[int]:
    if frame_count <= 0:
        return []
    start = min(frame_count - 1, max(0, round(frame_count * edge_fraction)))
    stop = min(frame_count - 1, max(start, round((frame_count - 1) * (1.0 - edge_fraction))))
    sample_count = min(max(count, count * oversample), stop - start + 1)
    return sorted({int(round(value)) for value in np.linspace(start, stop, sample_count)})


def extract_frame(frames: list[np.ndarray], frame_index: int) -> np.ndarray | None:
    return frames[frame_index] if 0 <= frame_index < len(frames) else None


def save_contact_sheet(
    records: list[dict[str, Any]],
    output: Path,
    cell_width: int = 360,
    cell_height: int = 235,
    crop_center: bool = False,
) -> None:
    anchors = [record for record in records if record["role"] == "anchor"]
    if not anchors:
        return
    columns = min(4, len(anchors))
    rows = math.ceil(len(anchors) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#16181b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(anchors):
        with Image.open(record["output_image"]) as opened:
            image = opened.convert("RGB")
        if crop_center:
            width, height = image.size
            crop_box = (
                int(width * 0.18),
                int(height * 0.03),
                int(width * 0.82),
                int(height * 0.94),
            )
            image = image.crop(crop_box)
        image.thumbnail((cell_width, cell_height - 35), Image.Resampling.LANCZOS)
        x = index % columns * cell_width
        y = index // columns * cell_height
        sheet.paste(image, (x + (cell_width - image.width) // 2, y + 30))
        timestamp = record.get("timestamp_s")
        time_text = f"{timestamp:.2f}s" if isinstance(timestamp, (int, float)) else "unknown"
        prefix = "crop " if crop_center else ""
        caption = f"{prefix}{record['source_group']} f={record['frame_index']} t={time_text}"
        draw.rectangle((x, y, x + cell_width, y + 29), fill="#25292e")
        draw.text((x + 7, y + 9), caption[:55], fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def process_video(
    path: Path,
    source_sha256: str,
    anchor_count: int,
    output: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_group = f"{clean_id(path.stem)[:40]}-{source_sha256[:10]}"
    frames: list[np.ndarray] | None = None
    capture = None
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        else:
            capture.release()
            capture = None
    if capture is None:
        frames, fps = read_video_frames(path)
        frame_count = len(frames)
        height, width = frames[0].shape[:2]
    indices = candidate_indices(frame_count, anchor_count, args.oversample_factor, args.edge_fraction)
    candidates = read_candidates_cv2(capture, indices) if capture is not None else read_candidates(frames or [], indices)
    anchors = select_anchors(candidates, frame_count, anchor_count)
    context_frames: dict[int, np.ndarray] = {}
    if capture is not None:
        context_indices = [
            anchor.frame_index + offset
            for anchor in anchors
            for offset in args.neighbor_offsets
            if 0 <= anchor.frame_index + offset < frame_count
        ]
        context_frames = read_frames_cv2(capture, context_indices)
    records: list[dict[str, Any]] = []
    written: set[int] = set()
    anchor_indices = {item.frame_index for item in anchors}
    for sequence_number, anchor in enumerate(anchors, start=1):
        sequence_id = f"seq-{sequence_number:03d}-anchor-{anchor.frame_index:08d}"
        for frame_index in [anchor.frame_index, *(anchor.frame_index + offset for offset in args.neighbor_offsets)]:
            is_anchor = frame_index == anchor.frame_index
            if frame_index < 0 or frame_index >= frame_count or frame_index in written:
                continue
            if not is_anchor and frame_index in anchor_indices:
                continue
            if frame_index == anchor.frame_index:
                frame = anchor.frame
            elif capture is not None:
                frame = context_frames.get(frame_index)
            else:
                frame = extract_frame(frames or [], frame_index)
            if frame is None:
                continue
            role = "anchor" if is_anchor else "temporal_context"
            filename = f"{sequence_id}_{role}_frame_{frame_index:08d}.jpg"
            role_root = "context" if args.separate_context and not is_anchor else "images"
            image_path = output / role_root / source_group / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ensure_rgb(frame)).save(image_path, quality=args.jpeg_quality)
            written.add(frame_index)
            records.append(
                {
                    "source_video": str(path),
                    "source_video_sha256": source_sha256,
                    "source_group": source_group,
                    "sequence_id": sequence_id,
                    "role": role,
                    "anchor_frame_index": anchor.frame_index,
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 6) if fps else None,
                    "image_size": [width, height],
                    "output_image": str(image_path.resolve()),
                    "output_image_sha256": sha256_file(image_path),
                }
            )
    if capture is not None:
        capture.release()
    source = {
        "source_video": str(path),
        "source_video_sha256": source_sha256,
        "source_group": source_group,
        "fps": fps,
        "frame_count": frame_count,
        "image_size": [width, height],
        "candidate_count": len(candidates),
        "anchor_count": len(anchors),
        "written_count": len(records),
    }
    return source, records


def recover_completed_source(
    path: Path,
    source_sha256: str,
    anchor_count: int,
    output: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Rebuild provenance for one fully written source without decoding sampled pixels again."""
    source_group = f"{clean_id(path.stem)[:40]}-{source_sha256[:10]}"
    anchor_root = output / "images" / source_group
    context_root = output / ("context" if args.separate_context else "images") / source_group
    anchor_files = sorted(anchor_root.glob("*.jpg")) if anchor_root.is_dir() else []
    parsed_anchors: list[tuple[int, int, Path]] = []
    for image_path in anchor_files:
        match = SAMPLED_FRAME_PATTERN.fullmatch(image_path.name)
        if match and match.group("role") == "anchor":
            anchor_index = int(match.group("anchor"))
            if int(match.group("frame")) != anchor_index:
                return None
            parsed_anchors.append((int(match.group("sequence_number")), anchor_index, image_path))
    if len(parsed_anchors) != anchor_count:
        return None
    parsed_anchors.sort()
    if [number for number, _, _ in parsed_anchors] != list(range(1, anchor_count + 1)):
        return None

    capture = cv2.VideoCapture(str(path)) if cv2 is not None else None
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        return None

    records: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    written: set[int] = set()
    anchor_indices = {frame_index for _, frame_index, _ in parsed_anchors}
    for sequence_number, anchor_index, _ in parsed_anchors:
        sequence_id = f"seq-{sequence_number:03d}-anchor-{anchor_index:08d}"
        for frame_index in [anchor_index, *(anchor_index + offset for offset in args.neighbor_offsets)]:
            is_anchor = frame_index == anchor_index
            if frame_index < 0 or frame_index >= frame_count or frame_index in written:
                continue
            if not is_anchor and frame_index in anchor_indices:
                continue
            role = "anchor" if is_anchor else "temporal_context"
            filename = f"{sequence_id}_{role}_frame_{frame_index:08d}.jpg"
            role_root = "context" if args.separate_context and not is_anchor else "images"
            image_path = output / role_root / source_group / filename
            if not image_path.is_file():
                return None
            with Image.open(image_path) as opened:
                if opened.size != (width, height):
                    return None
            expected_paths.add(image_path.resolve())
            written.add(frame_index)
            records.append(
                {
                    "source_video": str(path),
                    "source_video_sha256": source_sha256,
                    "source_group": source_group,
                    "sequence_id": sequence_id,
                    "role": role,
                    "anchor_frame_index": anchor_index,
                    "frame_index": frame_index,
                    "timestamp_s": round(frame_index / fps, 6),
                    "image_size": [width, height],
                    "output_image": str(image_path.resolve()),
                    "output_image_sha256": sha256_file(image_path),
                }
            )

    existing_paths = {
        image_path.resolve()
        for root in {anchor_root, context_root}
        if root.is_dir()
        for image_path in root.glob("*.jpg")
    }
    if existing_paths != expected_paths:
        return None
    source = {
        "source_video": str(path),
        "source_video_sha256": source_sha256,
        "source_group": source_group,
        "fps": fps,
        "frame_count": frame_count,
        "image_size": [width, height],
        "candidate_count": len(candidate_indices(frame_count, anchor_count, args.oversample_factor, args.edge_fraction)),
        "anchor_count": len(parsed_anchors),
        "written_count": len(records),
        "recovered_from_existing_files": True,
    }
    return source, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--videos", help="One video or a directory searched recursively.")
    sources.add_argument(
        "--videos-list",
        help="UTF-8 text file containing one explicit video path per line.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-video", type=int, default=12, help="Number of temporally stratified anchor frames.")
    parser.add_argument(
        "--total-anchors",
        type=int,
        help="Distribute exactly this many anchors as evenly as possible across all sources.",
    )
    parser.add_argument("--oversample-factor", type=int, default=5, help="Candidates per selected anchor before diversity selection.")
    parser.add_argument("--neighbor-offsets", type=parse_offsets, default=parse_offsets("-2,-1,1,2"))
    parser.add_argument("--edge-fraction", type=float, default=0.04)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument(
        "--separate-context",
        action="store_true",
        help="Write temporal context under OUTPUT/context so OUTPUT/images contains anchors only.",
    )
    parser.add_argument(
        "--hash-cache",
        help="Reusable full-video SHA-256 cache (default: OUTPUT/source_hash_cache.json).",
    )
    parser.add_argument("--hash-workers", type=int, default=2, help="Parallel workers for uncached full-file hashing.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recover fully written sources from a partial output and decode only missing or incomplete sources.",
    )
    args = parser.parse_args()
    if args.frames_per_video < 1 or args.oversample_factor < 1:
        parser.error("frames-per-video and oversample-factor must be positive")
    if args.total_anchors is not None and args.total_anchors < 1:
        parser.error("total-anchors must be positive")
    if args.hash_workers < 1:
        parser.error("hash-workers must be positive")
    if not 0 <= args.edge_fraction < 0.5:
        parser.error("edge-fraction must be in [0, 0.5)")

    videos_root = Path(args.videos_list or args.videos).resolve()
    output = Path(args.output).resolve()
    manifest_path = output / "sampling_manifest.json"
    if manifest_path.exists():
        parser.error(f"output already contains a sampling manifest: {manifest_path}")
    existing_frames = list((output / "images").rglob("*.jpg")) if (output / "images").is_dir() else []
    if existing_frames and not args.resume:
        parser.error("output contains partial sampled frames; pass --resume or choose a new output")
    if args.videos_list:
        videos = []
        seen = set()
        for line in videos_root.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value).expanduser().resolve()
            if path not in seen:
                videos.append(path)
                seen.add(path)
        missing = [str(path) for path in videos if not path.is_file()]
        unsupported = [str(path) for path in videos if path.suffix.lower() not in VIDEO_EXTENSIONS]
        if missing:
            parser.error("video list contains missing files: " + ", ".join(missing))
        if unsupported:
            parser.error("video list contains unsupported files: " + ", ".join(unsupported))
    else:
        videos = [videos_root] if videos_root.is_file() else sorted(
            path for path in videos_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    if not videos:
        parser.error(f"no supported videos found under {videos_root}")
    if args.total_anchors is not None and args.total_anchors < len(videos):
        parser.error("total-anchors must be at least the number of videos for source-balanced sampling")

    if args.total_anchors is None:
        anchor_counts = [args.frames_per_video] * len(videos)
    else:
        base, remainder = divmod(args.total_anchors, len(videos))
        anchor_counts = [base + int(index < remainder) for index in range(len(videos))]
    hash_cache_path = Path(args.hash_cache).resolve() if args.hash_cache else output / "source_hash_cache.json"
    source_hashes, hash_cache_stats = resolve_source_hashes(videos, hash_cache_path, args.hash_workers)

    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    recovered_source_count = 0
    for path, anchor_count in zip(videos, anchor_counts):
        try:
            recovered = (
                recover_completed_source(path, source_hashes[path], anchor_count, output, args)
                if args.resume
                else None
            )
            if recovered is not None:
                source, source_records = recovered
                recovered_source_count += 1
            else:
                source, source_records = process_video(path, source_hashes[path], anchor_count, output, args)
        except Exception as exc:
            failures.append({"source_video": str(path), "error": str(exc)})
            continue
        sources.append(source)
        records.extend(source_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "sampling_method": "source-balanced temporal strata plus non-semantic appearance diversity",
        "recognition_model_used": False,
        "videos_root": str(videos_root),
        "output": str(output),
        "parameters": {
            "frames_per_video": args.frames_per_video,
            "total_anchors": args.total_anchors,
            "anchors_per_source": anchor_counts,
            "oversample_factor": args.oversample_factor,
            "neighbor_offsets": args.neighbor_offsets,
            "edge_fraction": args.edge_fraction,
            "decode_strategy": "random_seek_requested_frames_only" if cv2 is not None else "full_decode_fallback",
            "separate_context": args.separate_context,
            "resume": args.resume,
        },
        "resume_stats": {
            "recovered_source_count": recovered_source_count,
            "decoded_source_count": len(sources) - recovered_source_count,
        },
        "source_hash_cache": {
            "path": str(hash_cache_path),
            **hash_cache_stats,
        },
        "source_count": len(sources),
        "record_count": len(records),
        "failure_count": len(failures),
        "sources": sources,
        "records": records,
        "failures": failures,
    }
    write_json(manifest_path, manifest)
    save_contact_sheet(records, output / "anchor_contact_sheet.jpg")
    save_contact_sheet(records, output / "anchor_contact_sheet_large.jpg", cell_width=720, cell_height=450)
    save_contact_sheet(records, output / "anchor_center_crop_sheet.jpg", cell_width=720, cell_height=560, crop_center=True)
    print(json.dumps({key: manifest[key] for key in ("source_count", "record_count", "failure_count", "output")}, ensure_ascii=False, indent=2))
    return 1 if not sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
