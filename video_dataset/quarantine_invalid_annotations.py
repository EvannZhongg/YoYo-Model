"""Quarantine annotation files that do not follow the video-frame schema."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def quarantine(dataset_dir: Path, quarantine_root: Path) -> dict[str, object]:
    annotations = (dataset_dir / "annotations").resolve()
    labels_root = annotations / "labels"
    destination = quarantine_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []

    for label in sorted(labels_root.rglob("*.json")):
        data = json.loads(label.read_text(encoding="utf-8"))
        if data.get("schema_version") and data.get("split"):
            continue
        relative = label.relative_to(labels_root)
        candidates = [
            (label, Path("labels") / relative),
            (annotations / "images" / relative.with_suffix(".jpg"), Path("images") / relative.with_suffix(".jpg")),
            (
                annotations / "visualizations" / relative.with_name(f"{relative.stem}_vis.png"),
                Path("visualizations") / relative.with_name(f"{relative.stem}_vis.png"),
            ),
        ]
        for source, target_relative in candidates:
            source = source.resolve()
            if annotations not in source.parents or not source.exists():
                continue
            target = destination / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(str(source))

    return {"quarantine_dir": str(destination), "moved_files": len(moved), "moved_labels": sum(path.endswith(".json") for path in moved)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine malformed video annotation groups.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--quarantine-root", default="tmp/batch_annotate_quarantine")
    args = parser.parse_args()
    print(json.dumps(quarantine(Path(args.dataset_dir), Path(args.quarantine_root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
