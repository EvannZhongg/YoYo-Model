"""Migrate legacy accepted labels into the strict component review schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from annotation.review import _valid_bbox, _overall_status


MIGRATION_ID = "review_gate_v2"
ACCEPTED = {"approved", "reviewed"}


def migrate_review_schema(dataset_dir: str | Path, apply: bool = False) -> dict:
    root = Path(dataset_dir)
    labels_root = root / "annotations" / "labels"
    changes: list[dict] = []
    counts: Counter[str] = Counter()

    for path in sorted(labels_root.rglob("*.json")) if labels_root.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        before: dict[str, object] = {}
        after: dict[str, object] = {}

        bbox_status = str(data.get("bbox_review_status", "auto_labeled_needs_review"))
        visibility = str(data.get("visibility", "uncertain"))
        if bbox_status in ACCEPTED and visibility == "uncertain":
            before["visibility"] = data.get("visibility")
            data["visibility"] = "visible" if _valid_bbox(data) else "absent"
            after["visibility"] = data["visibility"]
            counts["bbox_visibility_resolved"] += 1

        string_status = str(data.get("string_review_status", "auto_labeled_needs_review"))
        if string_status in ACCEPTED and str(data.get("string_visibility", "uncertain")) == "uncertain":
            before["string_review_status"] = string_status
            data["string_review_status"] = "auto_labeled_needs_review"
            data["review_status"] = _overall_status(data)
            after["string_review_status"] = data["string_review_status"]
            counts["uncertain_string_requeued"] += 1

        if data.get("scene_label") is None:
            bad_case = set(data.get("bad_case") or [])
            inferred = "non_trick" if "non_trick_scene" in bad_case else "transition" if "transition_scene" in bad_case else None
            if inferred:
                before["scene_label"] = None
                data["scene_label"] = inferred
                after["scene_label"] = inferred
                counts["scene_label_from_flag"] += 1

        if not after:
            continue
        changes.append({"label": str(path), "before": before, "after": after})
        if apply:
            history = list(data.get("schema_migrations") or [])
            if MIGRATION_ID not in history:
                history.append(MIGRATION_ID)
            data["schema_migrations"] = history
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "schema_version": "yoyo_review_schema_migration_v1",
        "migration_id": MIGRATION_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "apply": bool(apply),
        "changed_labels": len(changes),
        "counts": dict(sorted(counts.items())),
        "changes": changes,
    }
    output = root / "review_schema_migration.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate_review_schema(args.dataset_dir, apply=args.apply)
    print(json.dumps({key: report[key] for key in ("apply", "changed_labels", "counts")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
