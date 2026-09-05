#!/usr/bin/env python3
"""Transactionally merge confirmed temporal groups into one aggregate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_SCHEMA = "yoyo_temporal_dataset_v1"
SOURCE_DATASET_SCHEMA = "yoyo_consecutive_annotation_dataset_v1"
GROUP_SCHEMA = "yoyo_consecutive_groups_v1"
REVIEW_SCHEMA = "yoyo_temporal_review_v1"
ANNOTATION_SCHEMA = "agent_yoyo_string_annotation_v5"
MIN_FRAMES = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path must be relative to dataset: {value}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes dataset: {value}")
    return resolved


def selected_source(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(source / "manifest.json")
    groups = read_json(source / "consecutive_groups.json")
    review = read_json(source / "temporal_review.json")
    if manifest.get("schema_version") != SOURCE_DATASET_SCHEMA:
        raise ValueError("source is not a skill-generated temporal dataset")
    if groups.get("schema_version") != GROUP_SCHEMA or review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("source temporal metadata schema is unsupported")
    if any(document.get("dataset_id") != source.name for document in (manifest, groups, review)):
        raise ValueError("source dataset_id does not match its directory")
    records_by_label = {str(record.get("label")): record for record in manifest.get("records") or []}
    review_groups = review.get("groups")
    if not isinstance(review_groups, dict):
        raise ValueError("temporal review groups must be an object")
    selected_records: list[dict[str, Any]] = []
    selected_groups: list[dict[str, Any]] = []
    for group in groups.get("groups") or []:
        group_id = str(group.get("group_id") or "")
        entry = review_groups.get(group_id)
        if not isinstance(entry, dict) or entry.get("status") != "confirmed":
            continue
        selected_keys = [str(value) for value in entry.get("selected_sample_keys") or []]
        if len(selected_keys) < MIN_FRAMES or len(set(selected_keys)) != len(selected_keys):
            raise ValueError(f"confirmed group must select at least {MIN_FRAMES} unique frames: {group_id}")
        frames_by_key = {str(frame.get("sample_key")): frame for frame in group.get("frames") or []}
        if not set(selected_keys).issubset(frames_by_key):
            raise ValueError(f"confirmed group selects an unknown frame: {group_id}")
        selected_set = set(selected_keys)
        selected_frames = [frame for frame in group["frames"] if str(frame["sample_key"]) in selected_set]
        indices = [int(frame["frame_index"]) for frame in selected_frames]
        if indices != sorted(set(indices)):
            raise ValueError(f"selected frame indices must be strictly increasing: {group_id}")
        review_indices = entry.get("selected_frame_indices")
        if review_indices is not None and [int(value) for value in review_indices] != indices:
            raise ValueError(f"temporal review frame indices disagree with selection: {group_id}")
        clean_group = {
            key: value for key, value in group.items()
            if not key.startswith("group_review") and key not in {"group_reviewer"}
        }
        clean_group["frames"] = selected_frames
        clean_group["selected_start_frame"] = indices[0]
        clean_group["selected_end_frame"] = indices[-1]
        clean_group["start_sample_key"] = str(selected_frames[0]["sample_key"])
        selected_groups.append(clean_group)
        for frame in selected_frames:
            key = str(frame["sample_key"])
            record = records_by_label.get(f"canonical/labels/{key}")
            if record is None:
                record = next((item for item in manifest.get("records") or [] if str(item.get("label", "")).endswith(key)), None)
            if record is None:
                raise ValueError(f"selected frame has no manifest record: {key}")
            label_path = resolve_inside(source, str(record["label"]))
            image_path = resolve_inside(source, str(record["image"]))
            label = read_json(label_path)
            if label.get("schema_version") != ANNOTATION_SCHEMA:
                raise ValueError(f"unsupported label schema: {label_path}")
            if sha256_file(image_path) != str(record.get("image_sha256") or "").lower():
                raise ValueError(f"image hash mismatch: {image_path}")
            selected_records.append(dict(record))
    if not selected_groups:
        raise ValueError("source has no confirmed temporal groups")
    return selected_records, selected_groups, manifest


def existing_target(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not target.exists():
        return ({"schema_version": DATASET_SCHEMA, "dataset_id": target.name, "created_at_utc": utc_now(), "records": [], "generation_runs": []}, {"schema_version": GROUP_SCHEMA, "dataset_id": target.name, "created_at_utc": utc_now(), "groups": []})
    manifest = read_json(target / "manifest.json")
    groups = read_json(target / "consecutive_groups.json")
    if manifest.get("schema_version") != DATASET_SCHEMA or groups.get("schema_version") != GROUP_SCHEMA:
        raise ValueError("target aggregate schema is unsupported")
    if manifest.get("dataset_id") != target.name or groups.get("dataset_id") != target.name:
        raise ValueError("target dataset_id does not match its directory")
    if (target / "temporal_review.json").exists():
        raise ValueError("aggregate target must not contain temporal_review.json")
    return manifest, groups


def _dataset_key(path: Path) -> str:
    """Return the Workbench review-map key for a dataset root."""
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name.lower() == "datasets":
            return resolved.relative_to(parent).as_posix()
    return resolved.name


def _carry_frame_reviews(
    review_map_path: Path | None,
    source: Path,
    target: Path,
    selected_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build an updated map containing only existing per-frame reviews.

    Group confirmation is intentionally not represented here.  The map is
    optional (older checkouts may not have one), and selected frames without a
    single-frame review simply remain unreviewed in the aggregate.
    """
    if review_map_path is None or not review_map_path.is_file():
        return None
    document = read_json(review_map_path)
    if document.get("schema_version") != "yoyo_dataset_review_v3" or not isinstance(document.get("datasets"), dict):
        raise ValueError("review status map schema is unsupported")
    source_entry = document["datasets"].get(_dataset_key(source)) or {}
    source_samples = source_entry.get("samples", {}) if isinstance(source_entry, dict) else {}
    if not isinstance(source_samples, dict):
        raise ValueError("source review status entry is invalid")
    target_key = _dataset_key(target)
    target_entry = document["datasets"].get(target_key)
    if target_entry is None:
        target_entry = {"samples": {}}
        document["datasets"][target_key] = target_entry
    if not isinstance(target_entry, dict) or not isinstance(target_entry.get("samples"), dict):
        raise ValueError("target review status entry is invalid")
    target_samples = target_entry["samples"]
    for record in selected_records:
        label = Path(str(record["label"]))
        key = label.relative_to(Path("canonical/labels")).as_posix() if "canonical" in label.parts else label.as_posix()
        review = source_samples.get(key)
        if isinstance(review, dict) and review.get("confirmed"):
            target_samples[key] = dict(review)
    document["updated_at_utc"] = utc_now()
    return document


def merge(source: Path, target: Path, review_map_path: Path | None = None) -> dict[str, Any]:
    source, target = source.resolve(), target.resolve()
    if source == target or target.is_relative_to(source):
        raise ValueError("source and target must be separate dataset roots")
    new_records, new_groups, source_manifest = selected_source(source)
    old_manifest, old_groups = existing_target(target)
    if review_map_path is None:
        candidate = Path("workbench_state/dataset_review_status.json")
        review_map_path = candidate if candidate.is_file() else None
    updated_review_map = _carry_frame_reviews(review_map_path, source, target, new_records)
    records = [*old_manifest.get("records", []), *new_records]
    groups = [*old_groups.get("groups", []), *new_groups]
    identities: set[tuple[str, int]] = set()
    hashes: set[str] = set()
    paths: set[str] = set()
    for record in records:
        identity = (str(record.get("source_video_sha256") or "").lower(), int(record["frame_index"]))
        image_hash = str(record.get("image_sha256") or "").lower()
        record_paths = {str(record.get("image") or ""), str(record.get("label") or "")}
        if not identity[0] or identity in identities or not image_hash or image_hash in hashes or "" in record_paths or paths.intersection(record_paths):
            raise ValueError(f"duplicate provenance, hash, or path while merging frame {identity}")
        identities.add(identity)
        hashes.add(image_hash)
        paths.update(record_paths)
    group_ids = [str(group.get("group_id") or "") for group in groups]
    if not all(group_ids) or len(group_ids) != len(set(group_ids)):
        raise ValueError("duplicate or empty temporal group ID")
    staging = target.parent / f".{target.name}.merge-{secrets.token_hex(8)}"
    backup: Path | None = None
    target_existed = target.exists()
    review_original = review_map_path.read_bytes() if updated_review_map is not None and review_map_path is not None and review_map_path.exists() else None
    staging.mkdir(parents=True)
    try:
        if target.exists():
            shutil.copytree(target, staging, dirs_exist_ok=True)
        for record in new_records:
            for field in ("image", "label"):
                source_file = resolve_inside(source, str(record[field]))
                destination = resolve_inside(staging, str(record[field]))
                if destination.exists():
                    raise ValueError(f"merge refuses to overwrite: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
        now = utc_now()
        source_manifest_hash = sha256_file(source / "manifest.json")
        merged_manifest = {
            **old_manifest,
            "schema_version": DATASET_SCHEMA,
            "dataset_id": target.name,
            "updated_at_utc": now,
            "annotation_schema_version": ANNOTATION_SCHEMA,
            "source_count": len({record["source_group"] for record in records}),
            "group_count": len(groups),
            "sample_count": len(records),
            "records": records,
            "generation_runs": [*old_manifest.get("generation_runs", []), {"operation": "merge_confirmed_temporal_groups", "created_at_utc": now, "source_dataset": str(source), "source_manifest_sha256": source_manifest_hash, "added_group_count": len(new_groups), "added_sample_count": len(new_records)}],
        }
        merged_groups = {**old_groups, "schema_version": GROUP_SCHEMA, "dataset_id": target.name, "updated_at_utc": now, "frame_selection_policy": "reviewed_subset_minimum_3", "groups": groups}
        sampling = {"schema_version": "yoyo_temporal_aggregate_sampling_v1", "dataset_id": target.name, "updated_at_utc": now, "generation_runs": merged_manifest["generation_runs"]}
        write_json(staging / "manifest.json", merged_manifest)
        write_json(staging / "consecutive_groups.json", merged_groups)
        write_json(staging / "sampling_manifest.json", sampling)
        if target.exists():
            backup = target.parent / f".{target.name}.previous-{secrets.token_hex(8)}"
            os.replace(target, backup)
            try:
                os.replace(staging, target)
                if updated_review_map is not None and review_map_path is not None:
                    review_tmp = review_map_path.with_name(f".{review_map_path.name}.merge-{secrets.token_hex(8)}")
                    try:
                        write_json(review_tmp, updated_review_map)
                        os.replace(review_tmp, review_map_path)
                    finally:
                        if review_tmp.exists():
                            review_tmp.unlink()
                shutil.rmtree(backup)
                backup = None
            except Exception:
                if backup is not None and backup.exists():
                    if target.exists():
                        shutil.rmtree(target)
                    os.replace(backup, target)
                elif not target_existed and target.exists():
                    shutil.rmtree(target)
                if updated_review_map is not None and review_map_path is not None:
                    if review_original is None:
                        if review_map_path.exists():
                            review_map_path.unlink()
                    else:
                        review_map_path.write_bytes(review_original)
                raise
        else:
            os.replace(staging, target)
            if updated_review_map is not None and review_map_path is not None:
                review_tmp = review_map_path.with_name(f".{review_map_path.name}.merge-{secrets.token_hex(8)}")
                try:
                    write_json(review_tmp, updated_review_map)
                    os.replace(review_tmp, review_map_path)
                finally:
                    if review_tmp.exists():
                        review_tmp.unlink()
        return {"ok": True, "target": str(target), "added_group_count": len(new_groups), "added_sample_count": len(new_records), "group_count": len(groups), "sample_count": len(records)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="datasets/1Ayoyo_temporal")
    parser.add_argument("--review-map", default=None, help="Workbench review map JSON (optional)")
    args = parser.parse_args()
    try:
        result = merge(Path(args.source), Path(args.target), Path(args.review_map) if args.review_map else None)
    except (OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
