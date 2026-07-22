"""Read-only integrity audit for the video-first yoyo dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotation.review import validate_review_gate

ACCEPTED_REVIEW = {"approved", "reviewed"}
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record is not an object at {path}:{number}")
        rows.append(value)
    return rows


def audit(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    sources_path = root / "sources.json"
    if not sources_path.exists():
        raise FileNotFoundError(f"sources.json not found: {sources_path}")
    source_manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    sources = source_manifest.get("sources") or []
    current_action_group = str(source_manifest.get("current_action_group", "")).strip()
    by_video = {str(item.get("video_id")): item for item in sources if item.get("video_id")}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    source_counts = Counter(str(item.get("split", "unknown")) for item in sources)
    for source in sources:
        if current_action_group and str(source.get("action_group", "")) != current_action_group:
            errors.append({"kind": "source_action_group_mismatch", "video_id": source.get("video_id"), "expected": current_action_group})
        path = Path(str(source.get("path", "")))
        if not path.exists():
            errors.append({"kind": "missing_source_video", "path": str(path), "video_id": source.get("video_id")})
        elif source.get("sha256") and _sha256(path) != str(source["sha256"]):
            errors.append({"kind": "source_sha256_mismatch", "path": str(path), "video_id": source.get("video_id")})

    frames = _jsonl(root / "frames.jsonl")
    frame_counts = Counter()
    frame_groups: dict[str, set[str]] = defaultdict(set)
    frame_keys: set[tuple[str, int]] = set()
    for row in frames:
        video_id = str(row.get("video_id", ""))
        index = int(row.get("frame_index", -1))
        key = (video_id, index)
        if key in frame_keys:
            errors.append({"kind": "duplicate_frame_record", "video_id": video_id, "frame_index": index})
        frame_keys.add(key)
        split = str(row.get("split", "unknown"))
        frame_counts[split] += 1
        frame_groups[split].add(str(row.get("source_group", "")))
        path = Path(str(row.get("frame_path", "")))
        source = by_video.get(video_id)
        if not path.exists():
            errors.append({"kind": "missing_frame", "path": str(path), "video_id": video_id, "frame_index": index})
        if source is None:
            errors.append({"kind": "frame_unknown_video", "video_id": video_id, "frame_index": index})
        else:
            if current_action_group and str(row.get("action_group", "")) != current_action_group:
                errors.append({"kind": "frame_action_group_mismatch", "video_id": video_id, "frame_index": index, "expected": current_action_group})
            if str(row.get("source_video_sha256", "")) != str(source.get("sha256", "")):
                errors.append({"kind": "frame_source_sha256_mismatch", "video_id": video_id, "frame_index": index})
            if split != str(source.get("split")):
                errors.append({"kind": "frame_split_mismatch", "video_id": video_id, "frame_index": index})

    labels_root = root / "annotations" / "labels"
    labels = sorted(labels_root.rglob("*.json")) if labels_root.exists() else []
    label_counts = Counter()
    bbox_accepted = Counter()
    string_accepted = Counter()
    label_groups: dict[str, set[str]] = defaultdict(set)
    for path in labels:
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"kind": "invalid_label_json", "path": str(path), "detail": str(exc)})
            continue
        split = str(annotation.get("split", "unknown"))
        video_id = str(annotation.get("video_id", ""))
        group = str(annotation.get("source_group", ""))
        label_counts[split] += 1
        label_groups[split].add(group)
        source = by_video.get(video_id)
        image_path = Path(str(annotation.get("source_image", "")))
        if not image_path.exists():
            errors.append({"kind": "missing_label_source_image", "path": str(image_path), "label": str(path)})
        relative = path.relative_to(labels_root)
        visualization = root / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
        if not visualization.exists():
            warnings.append({"kind": "missing_visualization", "path": str(visualization), "label": str(path)})
        if source is None:
            errors.append({"kind": "label_unknown_video", "video_id": video_id, "label": str(path)})
        else:
            if current_action_group and str(annotation.get("action_group", "")) != current_action_group:
                errors.append({"kind": "label_action_group_mismatch", "video_id": video_id, "label": str(path), "expected": current_action_group})
            if group and group != str(source.get("source_group", "")):
                errors.append({"kind": "label_source_group_mismatch", "video_id": video_id, "label": str(path)})
            if split != str(source.get("split")):
                errors.append({"kind": "label_split_mismatch", "video_id": video_id, "label": str(path)})
            if str(annotation.get("source_video_sha256", "")) != str(source.get("sha256", "")):
                errors.append({"kind": "label_source_sha256_mismatch", "video_id": video_id, "label": str(path)})
        if str(annotation.get("bbox_review_status", "auto_labeled_needs_review")) in ACCEPTED_REVIEW:
            bbox_accepted[split] += 1
            for detail in validate_review_gate(annotation, "bbox"):
                errors.append({"kind": "accepted_bbox_failed_review_gate", "label": str(path), "detail": detail})
        if str(annotation.get("string_review_status", "auto_labeled_needs_review")) in ACCEPTED_REVIEW:
            string_accepted[split] += 1
            for detail in validate_review_gate(annotation, "string"):
                errors.append({"kind": "accepted_string_failed_review_gate", "label": str(path), "detail": detail})

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(label_groups[left] & label_groups[right])
        if overlap:
            errors.append({"kind": "label_source_group_leakage", "splits": [left, right], "source_groups": overlap})
        overlap = sorted(frame_groups[left] & frame_groups[right])
        if overlap:
            errors.append({"kind": "frame_source_group_leakage", "splits": [left, right], "source_groups": overlap})
    if not labels:
        warnings.append({"kind": "no_annotations", "path": str(labels_root)})
    for split in SPLITS:
        if bbox_accepted[split] == 0:
            warnings.append({"kind": "no_accepted_bbox", "split": split})
        if string_accepted[split] == 0:
            warnings.append({"kind": "no_accepted_string", "split": split})
    return {
        "schema_version": "yoyo_video_dataset_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "current_action_group": current_action_group or None,
        "sources": {"total": len(sources), "by_split": dict(sorted(source_counts.items()))},
        "frames": {"total": len(frames), "by_split": dict(sorted(frame_counts.items()))},
        "labels": {"total": len(labels), "by_split": dict(sorted(label_counts.items())), "bbox_accepted_by_split": dict(sorted(bbox_accepted.items())), "string_accepted_by_split": dict(sorted(string_accepted.items()))},
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source, frame, and review-gated video dataset integrity.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when errors or warnings exist.")
    args = parser.parse_args()
    report = audit(args.dataset_dir)
    output = Path(args.output) if args.output else Path(args.dataset_dir) / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "errors": len(report["errors"]), "warnings": len(report["warnings"]), "output": str(output.resolve())}, ensure_ascii=False, indent=2))
    return 1 if args.strict and (report["errors"] or report["warnings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
