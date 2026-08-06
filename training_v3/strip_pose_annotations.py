"""Remove hand and pose fields while preserving every other label value."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from config import BASE_DIR
from training_v3.prepare_dataset import POSE_ANNOTATION_FIELDS


def _digest_without_pose(annotation: dict[str, Any]) -> str:
    filtered = {key: value for key, value in annotation.items() if key not in POSE_ANNOTATION_FIELDS}
    payload = json.dumps(filtered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_pose_fields(labels_root: Path) -> dict[str, Any]:
    labels_root = labels_root.resolve()
    try:
        labels_root_display = str(labels_root.relative_to(BASE_DIR))
    except ValueError:
        labels_root_display = str(labels_root)
    labels = sorted(labels_root.rglob("*.json"))
    updated = unchanged = non_pose_changes = 0
    removed_counts = {field: 0 for field in sorted(POSE_ANNOTATION_FIELDS)}
    for path in labels:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        before = _digest_without_pose(annotation)
        removed = False
        for field in POSE_ANNOTATION_FIELDS:
            if field in annotation:
                annotation.pop(field)
                removed_counts[field] += 1
                removed = True
        if _digest_without_pose(annotation) != before:
            non_pose_changes += 1
            continue
        if removed:
            path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
        else:
            unchanged += 1
    result = {
        "labels_root": labels_root_display,
        "label_count": len(labels),
        "updated_label_count": updated,
        "unchanged_label_count": unchanged,
        "removed_field_counts": removed_counts,
        "non_pose_change_count": non_pose_changes,
        "other_annotation_fields_changed": False,
    }
    if non_pose_changes:
        raise RuntimeError(f"Non-pose annotation values changed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        action="append",
        default=[],
        help="Canonical labels directory; repeat for multiple datasets.",
    )
    parser.add_argument("--report", default=str(BASE_DIR / "reports" / "pose_annotation_cleanup.json"))
    args = parser.parse_args()
    roots = [Path(value) for value in args.labels_root] or [
        BASE_DIR / "datasets" / "1Ayoyo_dataset" / "canonical" / "labels",
        BASE_DIR / "datasets" / "1Ayoyo_consecutive" / "canonical" / "labels",
    ]
    result = {
        "schema_version": "pose_annotation_cleanup_v1",
        "policy": "datasets contain yoyo, string, orientation, and sequence labels; pose is runtime-only",
        "datasets": [strip_pose_fields(root) for root in roots],
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
