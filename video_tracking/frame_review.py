"""Append digest-bound human reviews for tracking frame records."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.files import sha256_file


ALLOWED_DECISIONS = {"correct", "incorrect", "unresolved"}
BINDING_SCHEMA = "tracking_frame_review_binding_v1"
REVIEW_SCHEMA = "tracking_frame_review_v1"


def frame_record_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}") from exc


def _resolve_run(metadata_path: str | Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    path = Path(metadata_path).resolve()
    if path.name != "frames.jsonl" or not path.is_file():
        raise ValueError("Tracking metadata must be an existing frames.jsonl file")
    run_dir = path.parent
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise ValueError(f"Tracking run manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or not str(manifest.get("run_id") or "").strip():
        raise ValueError("Tracking run manifest has no run_id")
    declared = str((manifest.get("outputs") or {}).get("frames_jsonl") or "").strip()
    if declared:
        declared_path = Path(declared)
        candidates = (
            [declared_path.resolve()]
            if declared_path.is_absolute()
            else [(Path.cwd() / declared_path).resolve(), (run_dir / declared_path).resolve()]
        )
        if path not in candidates:
            raise ValueError("frames.jsonl does not belong to the selected tracking run")
    return run_dir, path, manifest_path, manifest


def tracking_review_gallery_items(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return valid raw/overlay items in the same order shown by the Gallery."""
    root = Path(run_dir).resolve()
    index_path = root / "tracking_review_index.json"
    if not index_path.is_file():
        return []
    try:
        entries = _load_json(index_path)
    except ValueError:
        return []
    if not isinstance(entries, list):
        return []
    items: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("frame"), dict):
            continue
        frame = entry["frame"]
        try:
            frame_index = int(frame["frame_index"])
            source_frame_index = int(entry.get("source_frame_index", frame_index))
        except (KeyError, TypeError, ValueError):
            continue
        if source_frame_index != frame_index:
            continue
        variants = []
        if entry.get("raw_image"):
            variants.append(("raw", entry["raw_image"]))
        overlay_value = entry.get("overlay_image") or entry.get("image")
        if overlay_value:
            variants.append(("overlay", overlay_value))
        seen_paths: set[Path] = set()
        for view_name, image_value in variants:
            image_path = Path(str(image_value))
            if not image_path.is_absolute():
                image_path = root / image_path
            try:
                resolved = image_path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved in seen_paths or not resolved.is_file():
                continue
            seen_paths.add(resolved)
            items.append(
                {
                    "entry_index": entry_index,
                    "frame_index": frame_index,
                    "frame": frame,
                    "view": view_name,
                    "image": str(resolved),
                }
            )
    return items


def _load_frame_record(metadata_path: Path, frame_index: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read tracking metadata: {metadata_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in frames.jsonl at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid frame record at line {line_number}")
        try:
            record_index = int(record.get("frame_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Frame record has an invalid frame_index at line {line_number}") from exc
        if record_index == frame_index:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one frame record for frame_index={frame_index}; found {len(matches)}")
    return matches[0]


def load_tracking_frame_selection(metadata_path: str | Path, gallery_index: int) -> dict[str, Any]:
    run_dir, frames_path, manifest_path, manifest = _resolve_run(metadata_path)
    try:
        gallery_index = int(gallery_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gallery selection index must be an integer") from exc
    items = tracking_review_gallery_items(run_dir)
    if gallery_index < 0 or gallery_index >= len(items):
        raise IndexError(f"Gallery selection is out of range: {gallery_index}")
    item = items[gallery_index]
    record = _load_frame_record(frames_path, item["frame_index"])
    if record != item["frame"]:
        raise ValueError("Review index frame does not match the authoritative frames.jsonl record")
    binding = {
        "schema_version": BINDING_SCHEMA,
        "run_id": str(manifest["run_id"]),
        "run_manifest_sha256": sha256_file(manifest_path),
        "frame_index": int(item["frame_index"]),
        "frame_record_sha256": frame_record_digest(record),
        "gallery_index": gallery_index,
        "view": item["view"],
        "selected_image": item["image"],
    }
    return {"frame_record": record, "binding": binding}


def append_tracking_frame_review(
    metadata_path: str | Path,
    binding: dict[str, Any],
    decision: str,
    reviewer: str,
    notes: str = "",
) -> tuple[Path, dict[str, Any]]:
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported frame review decision: {decision}")
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise ValueError("Reviewer is required")
    if not isinstance(binding, dict) or binding.get("schema_version") != BINDING_SCHEMA:
        raise ValueError("Select a tracking review frame before saving")
    current = load_tracking_frame_selection(metadata_path, binding.get("gallery_index"))
    current_binding = current["binding"]
    binding_fields = (
        "run_id",
        "run_manifest_sha256",
        "frame_index",
        "frame_record_sha256",
        "gallery_index",
        "view",
        "selected_image",
    )
    if any(binding.get(field) != current_binding.get(field) for field in binding_fields):
        raise ValueError("Tracking frame selection is stale or does not belong to this run")
    run_dir, _, manifest_path, manifest = _resolve_run(metadata_path)
    event = {
        "schema_version": REVIEW_SCHEMA,
        "review_id": uuid.uuid4().hex,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reviewer": reviewer,
        "notes": str(notes or "").strip(),
        "run_id": current_binding["run_id"],
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": current_binding["run_manifest_sha256"],
        "source_video_sha256": str(manifest.get("source_video_sha256") or ""),
        "frame_index": current_binding["frame_index"],
        "frame_record_sha256": current_binding["frame_record_sha256"],
        "selected_view": current_binding["view"],
        "selected_image": current_binding["selected_image"],
        "frame_record": current["frame_record"],
    }
    output_path = run_dir / "tracking_frame_reviews.jsonl"
    encoded = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with output_path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return output_path, event
