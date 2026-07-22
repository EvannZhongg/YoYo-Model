"""Audit reviewed string attachment labels without changing annotations.

The current video collection is mostly 1A.  This report makes that limitation
explicit before an attachment classifier or tracker prior is trained.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTACHMENT_CLASSES = (
    "hand_and_yoyo_attached",
    "yoyo_detached",
    "hand_detached",
    "unknown",
)
REVIEWED_STATUSES = {"approved", "reviewed"}


def iter_annotations(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "source_group" in data:
            yield path, data


def audit(root: Path) -> dict[str, Any]:
    totals = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    groups_by_class: dict[str, set[str]] = defaultdict(set)

    for path, data in iter_annotations(root):
        split = str(data.get("split", "unknown"))
        attachment = str(data.get("string_attachment_class", "unknown"))
        string_status = str(data.get("string_review_status", ""))
        string_reviewed = string_status in REVIEWED_STATUSES

        totals["annotations"] += 1
        totals["string_reviewed"] += int(string_reviewed)
        totals["string_pending"] += int(not string_reviewed)
        by_split[split]["annotations"] += 1
        by_split[split]["string_reviewed"] += int(string_reviewed)
        by_split[split]["string_pending"] += int(not string_reviewed)
        by_split[split][f"class:{attachment}"] += 1

        # Unknown is useful for audit completeness but is not a training class.
        if string_reviewed:
            by_class[attachment]["reviewed"] += 1
            by_class[attachment][f"split:{split}"] += 1
            group = str(data.get("source_group", "unknown"))
            groups_by_class[attachment].add(group)
        else:
            by_class[attachment]["pending"] += 1

    reviewed_class_counts = {
        cls: by_class[cls].get("reviewed", 0) for cls in ATTACHMENT_CLASSES
    }
    missing = [
        cls for cls in ATTACHMENT_CLASSES if cls != "unknown" and reviewed_class_counts[cls] == 0
    ]

    return {
        "annotation_root": str(root),
        "totals": dict(totals),
        "by_split": {split: dict(counts) for split, counts in sorted(by_split.items())},
        "reviewed_attachment_classes": reviewed_class_counts,
        "reviewed_source_groups": {
            cls: sorted(groups) for cls, groups in sorted(groups_by_class.items())
        },
        "missing_reviewed_classes": missing,
        "recommendations": [
            "Do not infer 4A/5A from 1A footage or from endpoint geometry.",
            "Collect and review at least one independent source group per attachment class.",
            "Keep unknown, uncertain, rejected, and pending samples out of attachment training.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("datasets/video_v1/annotations/labels"),
        help="Directory containing per-frame annotation JSON files.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    report = audit(args.annotations_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
