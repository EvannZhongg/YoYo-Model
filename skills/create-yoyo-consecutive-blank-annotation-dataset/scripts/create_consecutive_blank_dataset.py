#!/usr/bin/env python3
"""Create a deduplicated blank yoyo/string dataset from consecutive video runs."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSITION_BIASES = ("middle", "front", "back")
ANNOTATION_SCHEMA_VERSION = "agent_yoyo_string_annotation_v5"
SAMPLING_SCHEMA_VERSION = "agent_video_sampling_v1"
DATASET_SCHEMA_VERSION = "yoyo_consecutive_annotation_dataset_v1"
CONSECUTIVE_SCHEMA_VERSION = "yoyo_consecutive_groups_v1"
CONSECUTIVE_FILENAME = "consecutive_groups.json"
REFERENCE_DATASET_SCHEMAS = {
    DATASET_SCHEMA_VERSION,
    "yoyo_blank_annotation_dataset_v1",
}
HASH_CACHE_SCHEMA_VERSION = "agent_video_sha256_cache_v1"
FRAME_CACHE_SCHEMA_VERSION = "agent_video_frame_jpeg_cache_v1"
REVIEW_SCHEMA_VERSION = "yoyo_dataset_review_v3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_revision(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def quick_file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as handle:
        offsets = {0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)}
        for offset in sorted(offsets):
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def clean_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_.-" else "-" for character in value)
    return cleaned.strip("-.") or "video"


def validate_dataset_name(value: str) -> str:
    if (
        not 1 <= len(value) <= 80
        or value in {".", ".."}
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise ValueError("dataset-name must start with a letter or digit and use only letters, digits, dots, underscores, or hyphens")
    return value


def create_staging_directory(datasets_root: Path, dataset_name: str) -> Path:
    """Create a private staging directory without breaking Windows ACL inheritance.

    Python applies a restrictive owner-only DACL for ``0o700`` directories on
    Windows. ``tempfile.mkdtemp`` uses that mode, and an atomic directory
    rename would carry the restrictive DACL into the published dataset. Use
    the normal inherited mode on Windows; keep staging private under POSIX
    where the mode has the expected meaning.
    """
    mode = 0o777 if os.name == "nt" else 0o700
    for _ in range(128):
        candidate = datasets_root / f".{dataset_name}.building-{secrets.token_hex(8)}"
        try:
            candidate.mkdir(mode=mode)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique staging directory for {dataset_name}")


def parse_time_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError("start-time must be SS, MM:SS, or HH:MM:SS")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("start-time contains a non-numeric component") from exc
    if any(number < 0 for number in numbers):
        raise argparse.ArgumentTypeError("start-time cannot be negative")
    if len(numbers) > 1 and any(number >= 60 for number in numbers[1:]):
        raise argparse.ArgumentTypeError("minutes and seconds components must be below 60")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def discover_videos(value: str | None, list_file: str | None) -> list[Path]:
    if bool(value) == bool(list_file):
        raise ValueError("provide exactly one of --videos or --videos-list")
    if list_file:
        source = Path(list_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"video list not found: {source}")
        raw_paths = [
            Path(line.strip()).expanduser().resolve()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        source = Path(str(value)).expanduser().resolve()
        raw_paths = [source] if source.is_file() else sorted(
            path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    videos: list[Path] = []
    seen: set[Path] = set()
    for path in raw_paths:
        if not path.is_file():
            raise FileNotFoundError(f"video not found: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {path}")
        if path not in seen:
            videos.append(path)
            seen.add(path)
    if not videos:
        raise ValueError("no supported videos found")
    return videos


def load_hash_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": HASH_CACHE_SCHEMA_VERSION, "entries": {}}
    value = read_json(path)
    if value.get("schema_version") != HASH_CACHE_SCHEMA_VERSION or not isinstance(value.get("entries"), dict):
        raise ValueError(f"unsupported video hash cache: {path}")
    return value


def resolve_video_hashes(videos: list[Path], cache_path: Path, workers: int) -> tuple[dict[Path, str], dict[str, int]]:
    cache = load_hash_cache(cache_path)
    entries = cache["entries"]

    def calculate(path: Path) -> tuple[Path, str, bool, dict[str, Any]]:
        stat = path.stat()
        fingerprint = quick_file_fingerprint(path)
        cached = entries.get(str(path))
        if (
            isinstance(cached, dict)
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and cached.get("quick_fingerprint") == fingerprint
            and isinstance(cached.get("sha256"), str)
            and len(cached["sha256"]) == 64
        ):
            return path, cached["sha256"].lower(), True, cached
        digest = sha256_file(path)
        return path, digest, False, {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "quick_fingerprint": fingerprint,
            "sha256": digest,
        }

    resolved: dict[Path, str] = {}
    hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for path, digest, hit, entry in executor.map(calculate, videos):
            resolved[path] = digest
            entries[str(path)] = entry
            hits += int(hit)
    cache["updated_at_utc"] = utc_now()
    write_json(cache_path, cache)
    return resolved, {"hit_count": hits, "computed_count": len(videos) - hits}


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass
class ReferenceInventory:
    dataset_paths: list[Path] = field(default_factory=list)
    provenance: set[tuple[str, int]] = field(default_factory=set)
    image_sha256: set[str] = field(default_factory=set)
    difference_hashes: list[int] = field(default_factory=list)


@dataclass
class ProtectedDataset:
    root: Path
    manifest: dict[str, Any]
    manifest_bytes: bytes
    label_revisions: dict[str, tuple[int, int]]
    image_revisions: dict[str, tuple[int, int]]
    review_map_path: Path
    review_map_bytes: bytes | None
    review_entry_count: int
    consecutive_manifest: dict[str, Any]
    consecutive_bytes: bytes


def resolve_inside(root: Path, relative_value: str) -> Path:
    raw = Path(relative_value)
    if raw.is_absolute():
        raise ValueError(f"dataset manifest path must be relative: {raw}")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"dataset manifest path escapes its root: {raw}")
    return resolved


def load_protected_dataset(output: Path, review_map_path: Path, dataset_key: str) -> ProtectedDataset:
    manifest_path = output / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"incremental append supports only datasets created by this skill: {output}")
    if manifest.get("dataset_id") != output.name:
        raise ValueError(f"dataset_id does not match directory name: {manifest_path}")
    consecutive_path = output / CONSECUTIVE_FILENAME
    if not consecutive_path.is_file():
        raise ValueError(f"consecutive group metadata is missing: {consecutive_path}")
    consecutive_bytes = consecutive_path.read_bytes()
    consecutive_manifest = read_json(consecutive_path)
    if consecutive_manifest.get("schema_version") != CONSECUTIVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported consecutive group metadata: {consecutive_path}")
    if consecutive_manifest.get("dataset_id") != output.name:
        raise ValueError(f"consecutive metadata dataset_id does not match: {consecutive_path}")
    sampling_path = output / "sampling_manifest.json"
    expected_sampling_hash = str(manifest.get("sampling_manifest_sha256") or "").lower()
    if not sampling_path.is_file() or sha256_file(sampling_path) != expected_sampling_hash:
        raise ValueError(f"initial sampling manifest is missing or changed: {sampling_path}")
    runs = manifest.get("generation_runs")
    if runs is not None:
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"generation_runs must be a non-empty array: {manifest_path}")
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                raise ValueError(f"generation_runs[{index}] must be an object")
            run_path = resolve_inside(output, str(run.get("sampling_manifest") or ""))
            run_hash = str(run.get("sampling_manifest_sha256") or "").lower()
            if not run_path.is_file() or sha256_file(run_path) != run_hash:
                raise ValueError(f"generation_runs[{index}] sampling manifest is missing or changed")
    labels_root, images_root = annotation_roots(output)
    labels = sorted(labels_root.rglob("*.json"))
    images = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(labels) or len(images) != len(labels):
        raise ValueError(f"existing manifest/image/label counts disagree: {output}")
    label_revisions: dict[str, tuple[int, int]] = {}
    image_revisions: dict[str, tuple[int, int]] = {}
    record_labels: set[str] = set()
    record_images: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"existing records[{index}] must be an object")
        label_path = resolve_inside(output, str(record.get("label") or ""))
        image_path = resolve_inside(output, str(record.get("image") or ""))
        if not label_path.is_file() or not image_path.is_file():
            raise ValueError(f"existing records[{index}] points to a missing file")
        label_relative = label_path.relative_to(labels_root).as_posix()
        image_relative = image_path.relative_to(images_root).as_posix()
        if Path(label_relative).with_suffix("").as_posix() != Path(image_relative).with_suffix("").as_posix():
            raise ValueError(f"existing records[{index}] image/label pair does not match")
        label = read_json(label_path)
        if label.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported existing label schema: {label_path}")
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash != str(label.get("image_sha256") or "").lower():
            raise ValueError(f"existing image hash differs from its label: {image_path}")
        if actual_image_hash != str(record.get("image_sha256") or "").lower():
            raise ValueError(f"existing image hash differs from manifest: {image_path}")
        label_revisions[label_relative] = file_revision(label_path)
        image_revisions[image_relative] = file_revision(image_path)
        record_labels.add(label_relative)
        record_images.add(image_relative)
    actual_labels = {path.relative_to(labels_root).as_posix() for path in labels}
    actual_images = {path.relative_to(images_root).as_posix() for path in images}
    if record_labels != actual_labels or record_images != actual_images:
        raise ValueError(f"existing dataset contains untracked image or label files: {output}")

    review_bytes = review_map_path.read_bytes() if review_map_path.is_file() else None
    review_count = 0
    if review_bytes is not None:
        review_document = read_json(review_map_path)
        if review_document.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported Workbench review map: {review_map_path}")
        datasets = review_document.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError(f"Workbench review map datasets must be an object: {review_map_path}")
        dataset_reviews = datasets.get(dataset_key)
        if dataset_reviews is not None:
            if not isinstance(dataset_reviews, dict) or not isinstance(dataset_reviews.get("samples"), dict):
                raise ValueError(f"Workbench review entry is invalid for dataset: {dataset_key}")
            for key, review in dataset_reviews["samples"].items():
                if not isinstance(review, dict):
                    raise ValueError(f"Workbench review sample must be an object: {key}")
                label_path = (labels_root / Path(str(key))).resolve()
                if not label_path.is_relative_to(labels_root.resolve()) or not label_path.is_file():
                    raise ValueError(f"Workbench review points to a missing label: {key}")
                label_size_bytes, label_mtime_ns = file_revision(label_path)
                if (
                    review.get("label_size_bytes") != label_size_bytes
                    or review.get("label_mtime_ns") != label_mtime_ns
                ):
                    raise ValueError(f"Workbench review is stale before append: {key}")
                review_count += 1
    return ProtectedDataset(
        root=output,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        label_revisions=label_revisions,
        image_revisions=image_revisions,
        review_map_path=review_map_path,
        review_map_bytes=review_bytes,
        review_entry_count=review_count,
        consecutive_manifest=consecutive_manifest,
        consecutive_bytes=consecutive_bytes,
    )


def assert_protected_unchanged(
    state: ProtectedDataset,
    check_manifest: bool = True,
    check_consecutive: bool = True,
) -> None:
    manifest_path = state.root / "manifest.json"
    if check_manifest and manifest_path.read_bytes() != state.manifest_bytes:
        raise ValueError("existing manifest changed during incremental generation")
    consecutive_path = state.root / CONSECUTIVE_FILENAME
    if check_consecutive and consecutive_path.read_bytes() != state.consecutive_bytes:
        raise ValueError("existing consecutive group metadata changed during incremental generation")
    labels_root, images_root = annotation_roots(state.root)
    for relative, expected in state.label_revisions.items():
        path = labels_root / Path(relative)
        if not path.is_file() or file_revision(path) != expected:
            raise ValueError(f"protected Workbench label changed during append: {relative}")
    for relative, expected in state.image_revisions.items():
        path = images_root / Path(relative)
        if not path.is_file() or file_revision(path) != expected:
            raise ValueError(f"protected Workbench image changed during append: {relative}")
    current_review_bytes = state.review_map_path.read_bytes() if state.review_map_path.is_file() else None
    if current_review_bytes != state.review_map_bytes:
        raise ValueError("Workbench review map changed during incremental generation")


def annotation_roots(dataset: Path) -> tuple[Path, Path]:
    for root in (dataset / "canonical", dataset):
        if (root / "labels").is_dir() and (root / "images").is_dir():
            return root / "labels", root / "images"
    raise ValueError(f"reference dataset lacks images and labels: {dataset}")


def collect_reference_datasets(datasets_root: Path, root_reference: Path, extra: Iterable[str]) -> list[Path]:
    candidates = [root_reference.resolve(), *(Path(value).expanduser().resolve() for value in extra)]
    for manifest_path in sorted(datasets_root.glob("*/manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except ValueError:
            continue
        if manifest.get("schema_version") in REFERENCE_DATASET_SCHEMAS:
            candidates.append(manifest_path.parent.resolve())
    result: list[Path] = []
    for path in candidates:
        if path not in result:
            if not path.is_dir():
                raise FileNotFoundError(f"reference dataset not found: {path}")
            result.append(path)
    return result


def build_reference_inventory(dataset_paths: list[Path]) -> ReferenceInventory:
    inventory = ReferenceInventory(dataset_paths=dataset_paths)
    for dataset in dataset_paths:
        labels_root, images_root = annotation_roots(dataset)
        labels = sorted(labels_root.rglob("*.json"))
        images = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not labels or not images:
            raise ValueError(f"reference dataset is empty: {dataset}")
        for label_path in labels:
            label = read_json(label_path)
            video_hash = str(label.get("source_video_sha256") or "").lower()
            frame_index = label.get("frame_index")
            image_hash = str(label.get("image_sha256") or "").lower()
            if len(video_hash) != 64 or not isinstance(frame_index, int) or len(image_hash) != 64:
                raise ValueError(f"reference label lacks deduplication provenance: {label_path}")
            inventory.provenance.add((video_hash, frame_index))
            inventory.image_sha256.add(image_hash)
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    inventory.difference_hashes.append(difference_hash(image))
            except (OSError, ValueError) as exc:
                raise ValueError(f"unreadable reference image: {image_path}") from exc
            inventory.image_sha256.add(sha256_file(image_path))
    return inventory


@dataclass
class Candidate:
    frame_index: int
    jpeg: bytes
    image_sha256: str
    difference_hash: int


def ordered_block_starts(
    frame_count: int,
    desired: int,
    edge_fraction: float,
    position_bias: str = "middle",
) -> list[int]:
    start = min(frame_count - 1, max(0, int(frame_count * edge_fraction)))
    stop = max(start + 1, min(frame_count, int(math.ceil(frame_count * (1.0 - edge_fraction)))))
    last_start = stop - desired
    if last_start < start:
        return []
    if position_bias == "front":
        return list(range(start, last_start + 1))
    if position_bias == "back":
        return list(range(last_start, start - 1, -1))
    if position_bias != "middle":
        raise ValueError(f"unsupported position bias: {position_bias}")
    center = (start + last_start) / 2.0
    return sorted(range(start, last_start + 1), key=lambda value: (abs(value - center), value))


def middle_out_block_starts(frame_count: int, desired: int, edge_fraction: float) -> list[int]:
    return ordered_block_starts(frame_count, desired, edge_fraction, "middle")


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def frame_cache_path(root: Path, video_hash: str, frame_index: int, jpeg_quality: int) -> Path:
    return (
        root
        / FRAME_CACHE_SCHEMA_VERSION
        / video_hash[:2]
        / video_hash
        / f"q{jpeg_quality:03d}"
        / f"frame-{frame_index:012d}.jpg"
    )


def candidate_from_jpeg(frame_index: int, jpeg: bytes, expected_size: tuple[int, int]) -> Candidate:
    with Image.open(io.BytesIO(jpeg)) as encoded:
        encoded.load()
        if encoded.size != expected_size:
            raise ValueError(
                f"cached frame size mismatch at frame {frame_index}: "
                f"actual={encoded.size} expected={expected_size}"
            )
        dhash = difference_hash(encoded)
    return Candidate(frame_index, jpeg, sha256_bytes(jpeg), dhash)


def decode_frame_candidate(
    capture: cv2.VideoCapture,
    video_hash: str,
    frame_index: int,
    image_size: tuple[int, int],
    args: argparse.Namespace,
    cache_stats: dict[str, int],
    capture_position: list[int | None],
) -> Candidate | None:
    cache_root = getattr(args, "frame_cache_root", None)
    cache_file = (
        frame_cache_path(cache_root, video_hash, frame_index, args.jpeg_quality)
        if isinstance(cache_root, Path)
        else None
    )
    if cache_file is not None and cache_file.is_file():
        try:
            candidate = candidate_from_jpeg(frame_index, cache_file.read_bytes(), image_size)
            cache_stats["hit_count"] += 1
            return candidate
        except (OSError, ValueError):
            cache_stats["invalid_count"] += 1

    cache_stats["miss_count"] += 1
    if capture_position[0] != frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        capture_position[0] = frame_index
        cache_stats["video_seek_count"] += 1
    ok, frame = capture.read()
    capture_position[0] = frame_index + 1 if ok else None
    if not ok or frame is None:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    jpeg = encode_jpeg(image, args.jpeg_quality)
    candidate = candidate_from_jpeg(frame_index, jpeg, image_size)
    if cache_file is not None:
        write_bytes_atomic(cache_file, jpeg)
    return candidate


def block_overlaps_reference(
    reference_frames: list[int], block_start: int, desired: int, window: int
) -> bool:
    position = bisect.bisect_left(reference_frames, block_start - window)
    block_end = block_start + desired - 1 + window
    return position < len(reference_frames) and reference_frames[position] <= block_end


def is_perceptual_duplicate(value: int, references: Iterable[int], threshold: int) -> bool:
    return any(hamming_distance(value, other) <= threshold for other in references)


def decode_candidates(
    video: Path,
    video_hash: str,
    desired: int,
    inventory: ReferenceInventory,
    args: argparse.Namespace,
    batch_hashes: set[str],
    batch_dhashes: list[int],
) -> tuple[list[Candidate], dict[str, int], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {video}")
    rejected = {"provenance": 0, "image_sha256": 0, "perceptual": 0, "decode": 0}
    cache_stats = {"hit_count": 0, "miss_count": 0, "invalid_count": 0, "video_seek_count": 0}
    reference_frames = sorted(index for digest, index in inventory.provenance if digest == video_hash)
    selected: list[Candidate] = []
    blocked_frames: set[int] = set()
    capture_position: list[int | None] = [None]
    block_candidates = 0
    provenance_filtered_blocks = 0
    blocked_filtered_blocks = 0
    requested_start_time = getattr(args, "start_time", None)
    if requested_start_time is None:
        block_starts = ordered_block_starts(
            frame_count,
            desired,
            args.edge_fraction,
            getattr(args, "position_bias", "middle"),
        )
        requested_start_frame = None
    else:
        requested_start_frame = int(round(float(requested_start_time) * fps))
        if requested_start_frame < 0 or requested_start_frame + desired > frame_count:
            capture.release()
            raise ValueError(
                f"requested run {requested_start_frame}-{requested_start_frame + desired - 1} "
                f"is outside video frame range 0-{frame_count - 1}: {video}"
            )
        block_starts = [requested_start_frame]
    try:
        for block_start in block_starts:
            block_candidates += 1
            if block_overlaps_reference(
                reference_frames, block_start, desired, args.exclude_frame_window
            ):
                provenance_filtered_blocks += 1
                rejected["provenance"] += 1
                continue
            if any(block_start <= value < block_start + desired for value in blocked_frames):
                blocked_filtered_blocks += 1
                continue
            current: list[Candidate] = []
            current_hashes: set[str] = set()
            for frame_index in range(block_start, block_start + desired):
                candidate = decode_frame_candidate(
                    capture,
                    video_hash,
                    frame_index,
                    (width, height),
                    args,
                    cache_stats,
                    capture_position,
                )
                if candidate is None:
                    rejected["decode"] += 1
                    blocked_frames.add(frame_index)
                    break
                if (
                    candidate.image_sha256 in inventory.image_sha256
                    or candidate.image_sha256 in batch_hashes
                    or candidate.image_sha256 in current_hashes
                ):
                    rejected["image_sha256"] += 1
                    blocked_frames.add(frame_index)
                    break
                if is_perceptual_duplicate(
                    candidate.difference_hash,
                    inventory.difference_hashes,
                    args.perceptual_hamming_threshold,
                ) or is_perceptual_duplicate(
                    candidate.difference_hash,
                    batch_dhashes,
                    args.perceptual_hamming_threshold,
                ):
                    rejected["perceptual"] += 1
                    blocked_frames.add(frame_index)
                    break
                current.append(candidate)
                current_hashes.add(candidate.image_sha256)
            if len(current) == desired:
                selected = current
                break
    finally:
        capture.release()
    if not selected:
        raise ValueError(
            f"{video} has no deduplicated run of {desired} consecutive frames in the eligible region"
        )
    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "image_size": [width, height],
        "block_candidates_considered": block_candidates,
        "provenance_filtered_blocks": provenance_filtered_blocks,
        "blocked_filtered_blocks": blocked_filtered_blocks,
        "requested_start_time_s": requested_start_time,
        "requested_start_frame": requested_start_frame,
        "frame_cache": cache_stats,
    }
    return selected, rejected, metadata


def selected_conflicts_with_batch(
    selected: list[Candidate],
    batch_hashes: set[str],
    batch_dhashes: list[int],
    perceptual_threshold: int,
) -> bool:
    return any(
        candidate.image_sha256 in batch_hashes
        or is_perceptual_duplicate(
            candidate.difference_hash, batch_dhashes, perceptual_threshold
        )
        for candidate in selected
    )


def combine_retry_results(
    first: tuple[list[Candidate], dict[str, int], dict[str, Any]],
    second: tuple[list[Candidate], dict[str, int], dict[str, Any]],
) -> tuple[list[Candidate], dict[str, int], dict[str, Any]]:
    selected, rejected, metadata = second
    first_rejected = first[1]
    rejected = {
        key: int(first_rejected.get(key, 0)) + int(rejected.get(key, 0))
        for key in set(first_rejected) | set(rejected)
    }
    first_cache = first[2].get("frame_cache") or {}
    cache = metadata.get("frame_cache") or {}
    metadata["frame_cache"] = {
        key: int(first_cache.get(key, 0)) + int(cache.get(key, 0))
        for key in set(first_cache) | set(cache)
    }
    metadata["speculative_retry"] = True
    return selected, rejected, metadata


def decode_sources_in_order(
    videos: list[Path],
    counts: list[int],
    video_hashes: dict[Path, str],
    inventory: ReferenceInventory,
    args: argparse.Namespace,
    batch_hashes: set[str],
    batch_dhashes: list[int],
) -> Iterable[tuple[Path, str, list[Candidate], dict[str, int], dict[str, Any]]]:
    workers = min(len(videos), max(1, int(getattr(args, "decode_workers", 1))))
    if workers == 1:
        for video, desired in zip(videos, counts):
            video_hash = video_hashes[video]
            result = decode_candidates(
                video, video_hash, desired, inventory, args, batch_hashes, batch_dhashes
            )
            yield video, video_hash, *result
        return

    def speculative(index: int) -> tuple[list[Candidate], dict[str, int], dict[str, Any]]:
        video = videos[index]
        return decode_candidates(
            video,
            video_hashes[video],
            counts[index],
            inventory,
            args,
            set(),
            [],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            index: executor.submit(speculative, index)
            for index in range(min(workers, len(videos)))
        }
        for index, video in enumerate(videos):
            result = futures.pop(index).result()
            next_index = index + workers
            if next_index < len(videos):
                futures[next_index] = executor.submit(speculative, next_index)
            if selected_conflicts_with_batch(
                result[0],
                batch_hashes,
                batch_dhashes,
                args.perceptual_hamming_threshold,
            ):
                retry = decode_candidates(
                    video,
                    video_hashes[video],
                    counts[index],
                    inventory,
                    args,
                    batch_hashes,
                    batch_dhashes,
                )
                result = combine_retry_results(result, retry)
            yield video, video_hashes[video], *result


def initial_label(record: dict[str, Any], sampling_manifest_sha256: str) -> dict[str, Any]:
    created = utc_now()
    source_group = str(record["source_group"])
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "created_at_utc": created,
        "updated_at_utc": created,
        "source_image": Path(str(record["output_image"])).relative_to("canonical").as_posix(),
        "image_sha256": str(record["output_image_sha256"]),
        "image_size": list(record["image_size"]),
        "source_video": str(record["source_video"]),
        "source_video_sha256": str(record["source_video_sha256"]),
        "source_group": source_group,
        "video_id": source_group,
        "frame_index": int(record["frame_index"]),
        "timestamp_s": float(record["timestamp_s"]),
        "sequence_id": str(record["sequence_id"]),
        "sampling_role": "anchor",
        "anchor_frame_index": int(record["frame_index"]),
        "sampling_manifest_sha256": sampling_manifest_sha256,
        "visibility": "visible",
        "yoyo_bbox_pixel": None,
        "yoyo_bbox_2d": None,
        "bbox": [],
        "string_visibility": "partial",
        "string_polylines_pixel": None,
        "string_polylines_2d": None,
        "string_polyline_pixel": None,
        "string_polyline_2d": None,
        "string_mask_polygons_pixel": None,
        "yoyo_division": "1A",
        "scene_label": "unknown",
        "trick_orientation": "normal",
        "string_path": {
            "topology": "uncertain",
            "reconstruction_status": "uncertain",
            "paths": [],
            "unresolved_gaps": [],
        },
        "bad_case": [],
        "review_status": "needs_review",
        "bbox_review_status": "needs_review",
        "string_review_status": "unresolved",
        "quality": {"revision": 0, "min_model_approvals": 1, "history": [], "reviews": []},
    }


def validate_staged_dataset(
    staging: Path,
    inventory: ReferenceInventory,
    expected: int,
    perceptual_threshold: int,
    exclude_frame_window: int,
) -> None:
    labels_root = staging / "canonical" / "labels"
    images_root = staging / "canonical" / "images"
    labels = sorted(labels_root.rglob("*.json"))
    images = sorted(images_root.rglob("*.jpg"))
    if len(labels) != expected or len(images) != expected:
        raise ValueError(f"pair count mismatch: labels={len(labels)} images={len(images)} expected={expected}")
    seen_provenance: set[tuple[str, int]] = set()
    seen_hashes: set[str] = set()
    seen_dhashes: list[tuple[str, int]] = []
    for label_path in labels:
        label = read_json(label_path)
        if label.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"wrong annotation schema: {label_path}")
        relative = label_path.relative_to(labels_root).with_suffix(".jpg")
        image_path = images_root / relative
        if not image_path.is_file():
            raise ValueError(f"paired image missing: {image_path}")
        image_hash = sha256_file(image_path)
        if image_hash != label.get("image_sha256"):
            raise ValueError(f"image hash mismatch: {image_path}")
        key = (str(label.get("source_video_sha256")), int(label.get("frame_index")))
        overlaps_reference_frame = any(
            digest == key[0] and abs(index - key[1]) <= exclude_frame_window
            for digest, index in inventory.provenance
        )
        if overlaps_reference_frame or key in seen_provenance:
            raise ValueError(f"provenance overlap survived generation: {key}")
        if image_hash in inventory.image_sha256 or image_hash in seen_hashes:
            raise ValueError(f"image overlap survived generation: {image_path}")
        seen_provenance.add(key)
        seen_hashes.add(image_hash)
        with Image.open(image_path) as image:
            if list(image.size) != label.get("image_size"):
                raise ValueError(f"image size mismatch: {image_path}")
            dhash = difference_hash(image)
        if is_perceptual_duplicate(dhash, inventory.difference_hashes, perceptual_threshold):
            raise ValueError(f"perceptual overlap survived generation: {image_path}")
        other_source_dhashes = [value for source_hash, value in seen_dhashes if source_hash != key[0]]
        if is_perceptual_duplicate(dhash, other_source_dhashes, perceptual_threshold):
            raise ValueError(f"cross-run perceptual overlap survived generation: {image_path}")
        seen_dhashes.append((key[0], dhash))


def copy_file_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as exc:
        raise ValueError(f"incremental append refuses to overwrite: {destination}") from exc
    try:
        with os.fdopen(descriptor_fd, "wb") as target, source.open("rb") as current:
            shutil.copyfileobj(current, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def remove_created_files(paths: list[Path], output: Path) -> None:
    for path in reversed(paths):
        if path.is_file():
            path.unlink()
    parents = sorted(
        {parent for path in paths for parent in path.parents if parent != output and parent.is_relative_to(output)},
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for parent in parents:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def generation_run(
    manifest: dict[str, Any],
    sampling_manifest: str,
    operation: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "created_at_utc": str(manifest.get("created_at_utc") or utc_now()),
        "sampling_manifest": sampling_manifest,
        "sampling_manifest_sha256": str(manifest["sampling_manifest_sha256"]),
        "added_sample_count": int(manifest["sample_count"]),
        "reference_datasets": list(manifest.get("reference_datasets") or []),
        "deduplication": dict(manifest.get("deduplication") or {}),
        "source_frame_cache": dict(manifest.get("source_frame_cache") or {}),
    }


def build_consecutive_manifest(
    dataset_id: str,
    sources: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the Workbench-owned ordering map for uninterrupted frame groups."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["source_group"]), str(record["sequence_id"]))
        grouped.setdefault(key, []).append(record)
    source_by_group = {str(source["source_group"]): source for source in sources}
    groups: list[dict[str, Any]] = []
    for (source_group, sequence_id), group_records in grouped.items():
        ordered = sorted(group_records, key=lambda item: int(item["frame_index"]))
        indices = [int(item["frame_index"]) for item in ordered]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"generated group is not consecutive: {source_group}/{sequence_id}")
        source = source_by_group[source_group]
        frames = []
        for record in ordered:
            image = Path(str(record["output_image"]))
            sample_key = (
                Path(source_group) / image.with_suffix(".json").name
            ).as_posix()
            frames.append({
                "sample_key": sample_key,
                "image": image.as_posix(),
                "frame_index": int(record["frame_index"]),
                "timestamp_s": float(record["timestamp_s"]),
            })
        groups.append({
            "group_id": f"{source_group}--{sequence_id}",
            "source_video": str(source["source_video"]),
            "source_video_sha256": str(source["source_video_sha256"]),
            "source_group": source_group,
            "sequence_id": sequence_id,
            "original_start_frame": indices[0],
            "original_end_frame": indices[-1],
            "selected_start_frame": indices[0],
            "selected_end_frame": indices[-1],
            "start_sample_key": frames[0]["sample_key"],
            "propagated_from_start_at_utc": None,
            "frames": frames,
        })
    return {
        "schema_version": CONSECUTIVE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "groups": groups,
    }


def merge_consecutive_manifests(
    existing: dict[str, Any], addition: dict[str, Any]
) -> dict[str, Any]:
    old_groups = existing.get("groups")
    new_groups = addition.get("groups")
    if not isinstance(old_groups, list) or not isinstance(new_groups, list):
        raise ValueError("consecutive group manifests must contain group arrays")
    combined = [*old_groups, *new_groups]
    ids = [str(group.get("group_id") or "") for group in combined if isinstance(group, dict)]
    if len(ids) != len(combined) or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("consecutive append contains an invalid or duplicate group_id")
    merged = dict(existing)
    merged["updated_at_utc"] = utc_now()
    merged["groups"] = combined
    return merged


def merge_incremental_manifest(
    existing: dict[str, Any],
    addition: dict[str, Any],
    run_manifest_path: str,
) -> dict[str, Any]:
    old_records = existing.get("records")
    new_records = addition.get("records")
    if not isinstance(old_records, list) or not isinstance(new_records, list):
        raise ValueError("incremental manifests must contain record arrays")
    combined = [*old_records, *new_records]
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    seen_hashes: set[str] = set()
    seen_provenance: set[tuple[str, int]] = set()
    for index, record in enumerate(combined):
        if not isinstance(record, dict):
            raise ValueError(f"merged records[{index}] must be an object")
        image = str(record.get("image") or "")
        label = str(record.get("label") or "")
        image_hash = str(record.get("image_sha256") or "").lower()
        provenance = (str(record.get("source_video_sha256") or "").lower(), int(record.get("frame_index")))
        if image in seen_images or label in seen_labels:
            raise ValueError(f"incremental append contains a path collision: {image or label}")
        if image_hash in seen_hashes or provenance in seen_provenance:
            raise ValueError(f"incremental append contains duplicate sample identity: {image_hash}")
        seen_images.add(image)
        seen_labels.add(label)
        seen_hashes.add(image_hash)
        seen_provenance.add(provenance)

    runs = existing.get("generation_runs")
    if not isinstance(runs, list):
        runs = [generation_run(existing, "sampling_manifest.json", "create")]
    append_run = generation_run(addition, run_manifest_path, "append")
    merged = dict(existing)
    merged.update({
        "updated_at_utc": utc_now(),
        "records": combined,
        "sample_count": len(combined),
        "source_count": len({str(record.get("source_group") or "") for record in combined}),
        "generation_runs": [*runs, append_run],
        "last_append": append_run,
        "source_hash_cache": dict(addition.get("source_hash_cache") or {}),
        "source_frame_cache": dict(addition.get("source_frame_cache") or {}),
    })
    merged["reference_datasets"] = list(dict.fromkeys([
        *(str(value) for value in existing.get("reference_datasets") or []),
        *(str(value) for value in addition.get("reference_datasets") or []),
    ]))
    return merged


def publish_incremental(
    staging: Path,
    output: Path,
    protected: ProtectedDataset,
    addition_manifest: dict[str, Any],
) -> dict[str, Any]:
    assert_protected_unchanged(protected)
    sampling_hash = str(addition_manifest["sampling_manifest_sha256"])
    run_relative = Path("provenance") / f"sampling_manifest-{sampling_hash[:16]}.json"
    merged = merge_incremental_manifest(protected.manifest, addition_manifest, run_relative.as_posix())
    addition_consecutive = read_json(staging / CONSECUTIVE_FILENAME)
    merged_consecutive = merge_consecutive_manifests(
        protected.consecutive_manifest, addition_consecutive
    )
    created: list[Path] = []
    manifest_written = False
    consecutive_written = False
    try:
        run_destination = output / run_relative
        copy_file_exclusive(staging / "sampling_manifest.json", run_destination)
        created.append(run_destination)
        for record in addition_manifest["records"]:
            for field in ("image", "label"):
                source = resolve_inside(staging, str(record[field]))
                destination = resolve_inside(output, str(record[field]))
                copy_file_exclusive(source, destination)
                created.append(destination)
        assert_protected_unchanged(protected)
        write_json(output / "manifest.json", merged)
        manifest_written = True
        assert_protected_unchanged(protected, check_manifest=False)
        write_json(output / CONSECUTIVE_FILENAME, merged_consecutive)
        consecutive_written = True
        assert_protected_unchanged(
            protected, check_manifest=False, check_consecutive=False
        )
        postflight = load_protected_dataset(
            output,
            protected.review_map_path,
            output.relative_to(output.parent).as_posix(),
        )
        for relative, expected in protected.label_revisions.items():
            if postflight.label_revisions.get(relative) != expected:
                raise ValueError(f"protected Workbench label changed after append: {relative}")
        assert_protected_unchanged(
            protected, check_manifest=False, check_consecutive=False
        )
        return {
            "ok": True,
            "dataset": str(output),
            "operation": "append",
            "added_sample_count": int(addition_manifest["sample_count"]),
            "sample_count": int(merged["sample_count"]),
            "protected_label_count": len(protected.label_revisions),
            "review_entry_count_preserved": protected.review_entry_count,
            "review_map_unchanged": True,
            "sampling_manifest": str(run_destination),
            "rejected": addition_manifest["deduplication"]["rejected"],
        }
    except Exception:
        if manifest_written:
            write_bytes_atomic(output / "manifest.json", protected.manifest_bytes)
        if consecutive_written:
            write_bytes_atomic(
                output / CONSECUTIVE_FILENAME, protected.consecutive_bytes
            )
        remove_created_files(created, output)
        raise


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    repository = script_path.parents[3]
    datasets_root = Path(args.datasets_root).expanduser().resolve() if args.datasets_root else repository / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    dataset_name = validate_dataset_name(args.dataset_name)
    output = (datasets_root / dataset_name).resolve()
    if output.parent != datasets_root.resolve():
        raise ValueError("dataset output escaped datasets root")
    root_reference = (datasets_root / "1Ayoyo_dataset").resolve()
    if output == root_reference:
        raise ValueError("this skill never writes or appends directly to datasets/1Ayoyo_dataset")
    if output.exists() and not args.append:
        raise FileExistsError(f"dataset already exists; pass --append for protected incremental addition: {output}")
    review_map_path = (
        Path(args.review_map).expanduser().resolve()
        if args.review_map
        else datasets_root.parent / "workbench_state" / "dataset_review_status.json"
    )
    if review_map_path.is_relative_to(output):
        raise ValueError("Workbench review map must be outside the annotation dataset")
    dataset_key = output.relative_to(datasets_root).as_posix()
    protected = load_protected_dataset(output, review_map_path, dataset_key) if output.exists() else None
    videos = discover_videos(args.videos, args.videos_list)
    if args.total_frames is not None and args.total_frames < len(videos):
        raise ValueError("total-frames must be at least the number of videos")
    if args.total_frames is None:
        counts = [args.frames_per_video] * len(videos)
    else:
        base, remainder = divmod(args.total_frames, len(videos))
        counts = [base + int(index < remainder) for index in range(len(videos))]
    reference_paths = collect_reference_datasets(datasets_root, root_reference, args.exclude_dataset)
    inventory = build_reference_inventory(reference_paths)
    cache_path = Path(args.hash_cache).expanduser().resolve() if args.hash_cache else repository / "annotations" / "source_video_sha256_cache.json"
    frame_cache_root = None
    if not args.no_frame_cache:
        frame_cache_root = (
            Path(args.frame_cache).expanduser().resolve()
            if args.frame_cache
            else cache_path.parent / "source_frame_jpeg_cache"
        )
        if frame_cache_root.is_relative_to(output):
            raise ValueError("frame cache must be outside the annotation dataset")
    args.frame_cache_root = frame_cache_root
    video_hashes, cache_stats = resolve_video_hashes(videos, cache_path, args.hash_workers)
    staging = create_staging_directory(datasets_root, dataset_name)
    try:
        records: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        rejection_totals = {"provenance": 0, "image_sha256": 0, "perceptual": 0, "decode": 0}
        frame_cache_totals = {"hit_count": 0, "miss_count": 0, "invalid_count": 0, "video_seek_count": 0}
        batch_hashes: set[str] = set()
        batch_dhashes: list[int] = []
        for video, video_hash, selected, rejected, metadata in decode_sources_in_order(
            videos,
            counts,
            video_hashes,
            inventory,
            args,
            batch_hashes,
            batch_dhashes,
        ):
            source_group = f"{clean_id(video.stem)[:40]}-{video_hash[:10]}"
            for key, value in rejected.items():
                rejection_totals[key] += value
            for key, value in (metadata.get("frame_cache") or {}).items():
                frame_cache_totals[key] += int(value)
            sequence_id = f"run-{selected[0].frame_index:08d}-{selected[-1].frame_index:08d}"
            for candidate in selected:
                filename = f"{sequence_id}_frame_{candidate.frame_index:08d}.jpg"
                relative_image = Path("canonical") / "images" / source_group / filename
                image_path = staging / relative_image
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(candidate.jpeg)
                record = {
                    "source_video": str(video),
                    "source_video_sha256": video_hash,
                    "source_group": source_group,
                    "sequence_id": sequence_id,
                    "role": "anchor",
                    "anchor_frame_index": candidate.frame_index,
                    "frame_index": candidate.frame_index,
                    "timestamp_s": round(candidate.frame_index / metadata["fps"], 6),
                    "image_size": metadata["image_size"],
                    "output_image": relative_image.as_posix(),
                    "output_image_sha256": candidate.image_sha256,
                }
                records.append(record)
                batch_hashes.add(candidate.image_sha256)
                batch_dhashes.append(candidate.difference_hash)
            sources.append({
                "source_video": str(video),
                "source_video_sha256": video_hash,
                "source_group": source_group,
                **metadata,
                "run_start_frame": selected[0].frame_index,
                "run_end_frame": selected[-1].frame_index,
                "anchor_count": len(selected),
            })
        sampling_manifest = {
            "schema_version": SAMPLING_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "sampling_method": (
                "explicit-time consecutive runs with non-overlap checks"
                if args.start_time is not None
                else f"{args.position_bias}-preferred consecutive runs with non-overlap checks"
            ),
            "recognition_model_used": False,
            "videos_root": str(Path(args.videos_list or args.videos).expanduser().resolve()),
            "output": ".",
            "parameters": {
                "frames_per_video": args.frames_per_video,
                "total_frames": args.total_frames,
                "frames_per_source": counts,
                "edge_fraction": args.edge_fraction,
                "position_bias": args.position_bias,
                "start_time_s": args.start_time,
                "provenance_window_prefilter": True,
                "decode_workers": args.decode_workers,
                "jpeg_quality": args.jpeg_quality,
                "single_frame_only": True,
                "consecutive_frames": True,
                "within_run_perceptual_similarity_allowed": True,
            },
            "source_count": len(sources),
            "record_count": len(records),
            "failure_count": 0,
            "sources": sources,
            "records": records,
            "failures": [],
        }
        sampling_path = staging / "sampling_manifest.json"
        write_json(sampling_path, sampling_manifest)
        sampling_hash = sha256_file(sampling_path)
        for record in records:
            relative_image = Path(record["output_image"])
            relative_label = Path("canonical") / "labels" / relative_image.relative_to(Path("canonical") / "images")
            write_json((staging / relative_label).with_suffix(".json"), initial_label(record, sampling_hash))
        manifest_records = [
            {
                "image": record["output_image"],
                "label": (Path("canonical") / "labels" / Path(record["output_image"]).relative_to(Path("canonical") / "images")).with_suffix(".json").as_posix(),
                "image_sha256": record["output_image_sha256"],
                "source_video_sha256": record["source_video_sha256"],
                "source_group": record["source_group"],
                "frame_index": record["frame_index"],
            }
            for record in records
        ]
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": dataset_name,
            "created_at_utc": utc_now(),
            "output_dir": str(output),
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "sampling_manifest_sha256": sampling_hash,
            "recognition_model_used": False,
            "agent_annotation_used": False,
            "model_review_used": False,
            "reference_datasets": [str(path) for path in reference_paths],
            "deduplication": {
                "source_frame_identity": True,
                "exclude_frame_window": args.exclude_frame_window,
                "image_sha256": True,
                "difference_hash_bits": 64,
                "perceptual_hamming_threshold": args.perceptual_hamming_threshold,
                "rejected": rejection_totals,
            },
            "source_hash_cache": {"path": str(cache_path), **cache_stats},
            "source_frame_cache": {
                "enabled": frame_cache_root is not None,
                "path": str(frame_cache_root) if frame_cache_root is not None else None,
                "schema_version": FRAME_CACHE_SCHEMA_VERSION,
                **frame_cache_totals,
            },
            "source_count": len(sources),
            "sample_count": len(records),
            "records": manifest_records,
        }
        manifest["generation_runs"] = [generation_run(manifest, "sampling_manifest.json", "create")]
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / CONSECUTIVE_FILENAME,
            build_consecutive_manifest(dataset_name, sources, records),
        )
        validate_staged_dataset(
            staging,
            inventory,
            sum(counts),
            args.perceptual_hamming_threshold,
            args.exclude_frame_window,
        )
        if protected is not None:
            result = publish_incremental(staging, output, protected, manifest)
            shutil.rmtree(staging)
            return result
        os.replace(staging, output)
        return {
            "ok": True,
            "dataset": str(output),
            "operation": "create",
            "added_sample_count": len(records),
            "sample_count": len(records),
            "protected_label_count": 0,
            "review_entry_count_preserved": 0,
            "review_map_unchanged": True,
            "rejected": rejection_totals,
        }
    except Exception:
        if staging.is_dir() and staging.parent.resolve() == datasets_root.resolve():
            shutil.rmtree(staging)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--videos", help="One video or a directory searched recursively")
    sources.add_argument("--videos-list", help="UTF-8 file with one video path per line")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--append", action="store_true", help="Add only new samples to an existing skill-created dataset")
    parser.add_argument("--datasets-root", help="Override the repository datasets directory (mainly for testing)")
    parser.add_argument("--review-map", help="Override Workbench review map path (mainly for testing)")
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Additional reference dataset; repeatable")
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--total-frames", type=int)
    parser.add_argument("--edge-fraction", type=float, default=0.04)
    parser.add_argument(
        "--position-bias",
        choices=POSITION_BIASES,
        default="middle",
        help="Prefer eligible runs near the temporal middle, front, or back.",
    )
    parser.add_argument(
        "--start-time",
        type=parse_time_seconds,
        help="Require the run to start exactly at SS, MM:SS, or HH:MM:SS.",
    )
    parser.add_argument("--exclude-frame-window", type=int, default=0)
    parser.add_argument("--perceptual-hamming-threshold", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--hash-cache")
    parser.add_argument("--hash-workers", type=int, default=2)
    parser.add_argument("--frame-cache", help="Override the persistent decoded-frame cache directory")
    parser.add_argument("--no-frame-cache", action="store_true", help="Disable the persistent decoded-frame cache")
    parser.add_argument("--decode-workers", type=int, default=2, help="Videos decoded concurrently")
    args = parser.parse_args(argv)
    if args.frames_per_video < 1 or args.hash_workers < 1 or args.decode_workers < 1:
        parser.error("frame counts, hash-workers, and decode-workers must be positive")
    if args.frame_cache and args.no_frame_cache:
        parser.error("--frame-cache and --no-frame-cache cannot be used together")
    if args.start_time is not None and args.position_bias != "middle":
        parser.error("--start-time cannot be combined with --position-bias")
    if args.total_frames is not None and args.total_frames < 1:
        parser.error("total-frames must be positive")
    if not 0 <= args.edge_fraction < 0.5:
        parser.error("edge-fraction must be in [0, 0.5)")
    if args.exclude_frame_window < 0:
        parser.error("exclude-frame-window cannot be negative")
    if not 0 <= args.perceptual_hamming_threshold <= 64:
        parser.error("perceptual-hamming-threshold must be in [0, 64]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be in [1, 100]")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        result = build_dataset(parse_args(argv))
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
