"""Remove hand and pose annotations without changing yoyo or string geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.files import file_revision
from config import BASE_DIR
from training_v3.prepare_dataset import POSE_ANNOTATION_FIELDS


LEGACY_HAND_ANCHORS = {"left_hand", "right_hand"}
PATH_ANCHOR_FIELDS = {"start_anchor", "end_anchor"}
CONTENT_DIGEST_FIELDS = (
    "image_sha256",
    "image_size",
    "source_group",
    "visibility",
    "yoyo_bbox_pixel",
    "string_visibility",
    "string_polylines_pixel",
    "string_mask_polygons_pixel",
    "yoyo_division",
    "scene_label",
    "trick_orientation",
    "string_path",
    "bad_case",
    "notes",
)
DIGEST_REFERENCE_FIELDS = {"before_sha256", "after_sha256", "content_sha256"}
REVIEW_SCHEMA_VERSION = "yoyo_dataset_review_v3"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _content_digest(annotation: dict[str, Any]) -> str:
    payload = {key: annotation.get(key) for key in CONTENT_DIGEST_FIELDS}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _clean_nested(
    value: Any,
    removed_counts: Counter[str] | None = None,
    anchor_counts: Counter[str] | None = None,
) -> Any:
    """Return a copy with pose keys removed and legacy hand anchors normalized."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            if key in POSE_ANNOTATION_FIELDS:
                if removed_counts is not None:
                    removed_counts[key] += 1
                continue
            if key in PATH_ANCHOR_FIELDS and child in LEGACY_HAND_ANCHORS:
                if anchor_counts is not None:
                    anchor_counts[str(child)] += 1
                cleaned[key] = "unknown"
                continue
            cleaned[key] = _clean_nested(child, removed_counts, anchor_counts)
        return cleaned
    if isinstance(value, list):
        return [_clean_nested(child, removed_counts, anchor_counts) for child in value]
    return value


def _without_digest_references(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_digest_references(child)
            for key, child in value.items()
            if key not in DIGEST_REFERENCE_FIELDS
        }
    if isinstance(value, list):
        return [_without_digest_references(child) for child in value]
    return value


def _digest_without_pose(annotation: dict[str, Any]) -> str:
    """Digest all non-hand/pose semantics, including normalized path anchors."""
    payload = _without_digest_references(_clean_nested(annotation))
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _replace_digest_references(value: Any, digest_map: dict[str, str]) -> int:
    updated = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in DIGEST_REFERENCE_FIELDS and isinstance(child, str) and child in digest_map:
                replacement = digest_map[child]
                if replacement != child:
                    value[key] = replacement
                    updated += 1
                continue
            updated += _replace_digest_references(child, digest_map)
    elif isinstance(value, list):
        for child in value:
            updated += _replace_digest_references(child, digest_map)
    return updated


def _migrate_digest_references(before: dict[str, Any], after: dict[str, Any]) -> int:
    """Retarget revision/review digests after legacy anchor normalization."""
    digest_map = {_content_digest(before): _content_digest(after)}
    old_history = ((before.get("quality") or {}).get("history") or [])
    new_history = ((after.get("quality") or {}).get("history") or [])
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for old_item, new_item in zip(old_history, new_history):
        if not isinstance(old_item, dict) or not isinstance(new_item, dict):
            continue
        old_previous = old_item.get("previous_content")
        new_previous = new_item.get("previous_content")
        if not isinstance(old_previous, dict) or not isinstance(new_previous, dict):
            continue
        old_before = str(old_item.get("before_sha256") or _content_digest(old_previous))
        new_before = _content_digest(new_previous)
        digest_map[old_before] = new_before
        digest_map[_content_digest(old_previous)] = new_before
        paired.append((old_item, new_item))

    current_after = _content_digest(after)
    for index, (old_item, _) in enumerate(paired):
        if index + 1 < len(paired):
            next_previous = paired[index + 1][1].get("previous_content")
            new_after = _content_digest(next_previous) if isinstance(next_previous, dict) else current_after
        else:
            new_after = current_after
        old_after = old_item.get("after_sha256")
        if isinstance(old_after, str):
            digest_map[old_after] = new_after

    return _replace_digest_references(after, digest_map)


def _serialized(annotation: dict[str, Any]) -> bytes:
    return (json.dumps(annotation, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def strip_pose_fields(
    labels_root: Path,
    *,
    dry_run: bool = False,
    file_revision_changes: dict[Path, tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    labels_root = labels_root.resolve()
    try:
        labels_root_display = str(labels_root.relative_to(BASE_DIR))
    except ValueError:
        labels_root_display = str(labels_root)
    labels = sorted(labels_root.rglob("*.json"))
    updated = unchanged = non_cleanup_changes = digest_reference_updates = 0
    removed_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    for path in labels:
        original_revision = file_revision(path)
        original_bytes = path.read_bytes()
        annotation = json.loads(original_bytes.decode("utf-8"))
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation root must be an object: {path}")
        before = _digest_without_pose(annotation)
        cleaned = _clean_nested(annotation, removed_counts, anchor_counts)
        reference_updates = _migrate_digest_references(annotation, cleaned)
        digest_reference_updates += reference_updates
        if _digest_without_pose(cleaned) != before:
            non_cleanup_changes += 1
            continue
        if cleaned == annotation and reference_updates == 0:
            unchanged += 1
            continue
        output_bytes = _serialized(cleaned)
        if not dry_run:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(output_bytes)
            temporary.replace(path)
        if file_revision_changes is not None:
            output_revision = file_revision(path) if not dry_run else (len(output_bytes), original_revision[1])
            file_revision_changes[path] = (original_revision, output_revision)
        updated += 1
    result = {
        "labels_root": labels_root_display,
        "label_count": len(labels),
        "updated_label_count": updated,
        "unchanged_label_count": unchanged,
        "removed_field_counts": {
            field: removed_counts.get(field, 0) for field in sorted(POSE_ANNOTATION_FIELDS)
        },
        "replaced_anchor_counts": {
            anchor: anchor_counts.get(anchor, 0) for anchor in sorted(LEGACY_HAND_ANCHORS)
        },
        "digest_reference_update_count": digest_reference_updates,
        "non_cleanup_change_count": non_cleanup_changes,
        "other_annotation_fields_changed": False,
        "dry_run": dry_run,
    }
    if non_cleanup_changes:
        raise RuntimeError(f"Non-cleanup annotation values changed: {result}")
    return result


def _dataset_key(labels_root: Path) -> tuple[str, Path] | None:
    labels_root = labels_root.resolve()
    if labels_root.name != "labels":
        return None
    dataset_root = labels_root.parent.parent if labels_root.parent.name == "canonical" else labels_root.parent
    try:
        key = dataset_root.relative_to((BASE_DIR / "datasets").resolve()).as_posix()
    except ValueError:
        return None
    return key, dataset_root


def refresh_review_revisions(
    review_map_path: Path,
    labels_roots: list[Path],
    file_revision_changes: dict[Path, tuple[tuple[int, int], tuple[int, int]]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Keep Workbench confirmations attached to mechanical migrations."""
    if not review_map_path.is_file():
        return {"path": str(review_map_path), "updated": 0, "stale": 0, "missing": True}
    document = json.loads(review_map_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"Invalid Workbench review map schema: {review_map_path}")
    datasets = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid Workbench review map: {review_map_path}")
    updated = stale = 0
    for labels_root in labels_roots:
        dataset_info = _dataset_key(labels_root)
        if dataset_info is None:
            continue
        key, _ = dataset_info
        samples = (datasets.get(key) or {}).get("samples")
        if not isinstance(samples, dict):
            continue
        for path, (old_revision, new_revision) in file_revision_changes.items():
            try:
                relative = path.relative_to(labels_root.resolve()).as_posix()
            except ValueError:
                continue
            review = samples.get(relative)
            if not isinstance(review, dict):
                continue
            if (
                review.get("label_size_bytes") != old_revision[0]
                or review.get("label_mtime_ns") != old_revision[1]
            ):
                stale += 1
                continue
            review["label_size_bytes"] = new_revision[0]
            review["label_mtime_ns"] = new_revision[1]
            updated += 1
    if updated and not dry_run:
        temporary = review_map_path.with_suffix(review_map_path.suffix + ".tmp")
        temporary.write_bytes(_serialized(document))
        temporary.replace(review_map_path)
    return {
        "path": str(review_map_path),
        "updated": updated,
        "stale": stale,
        "missing": False,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        action="append",
        default=[],
        help="Canonical labels directory; repeat for multiple datasets.",
    )
    parser.add_argument("--report", default=str(BASE_DIR / "reports" / "pose_annotation_cleanup.json"))
    parser.add_argument(
        "--review-map",
        default=str(BASE_DIR / "workbench_state" / "dataset_review_status.json"),
        help="Workbench review map to migrate; pass an empty value to skip.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing labels or reviews.")
    args = parser.parse_args()
    roots = [Path(value).resolve() for value in args.labels_root] or [
        BASE_DIR / "datasets" / "1Ayoyo_dataset" / "canonical" / "labels",
        BASE_DIR / "datasets" / "1Ayoyo_consecutive" / "canonical" / "labels",
        BASE_DIR / "datasets" / "2023SouthChina1A_Final_2nd_YangYunfan_consecutive_200" / "canonical" / "labels",
    ]
    file_revision_changes: dict[Path, tuple[tuple[int, int], tuple[int, int]]] = {}
    result = {
        "schema_version": "pose_annotation_cleanup_v2",
        "policy": "remove all hand/pose annotations; preserve yoyo and string geometry",
        "datasets": [
            strip_pose_fields(root, dry_run=args.dry_run, file_revision_changes=file_revision_changes)
            for root in roots
        ],
    }
    if args.review_map:
        result["workbench_review_map"] = refresh_review_revisions(
            Path(args.review_map).resolve(), roots, file_revision_changes, dry_run=args.dry_run
        )
    report = Path(args.report)
    if not args.dry_run:
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + ".tmp")
        temporary.write_bytes(_serialized(result))
        temporary.replace(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
