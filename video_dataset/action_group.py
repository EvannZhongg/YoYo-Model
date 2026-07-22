"""Apply one explicit action-group label across a video dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def set_action_group(dataset_dir: str | Path, action_group: str, apply: bool = False) -> dict[str, Any]:
    root = Path(dataset_dir)
    group = str(action_group).strip()
    if not group:
        raise ValueError("action_group must not be empty")

    sources_path = root / "sources.json"
    if not sources_path.exists():
        raise FileNotFoundError(f"sources.json not found: {sources_path}")
    manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources") or []
    source_changes = int(manifest.get("current_action_group") != group)
    source_changes += sum(item.get("action_group") != group for item in sources)

    frames_path = root / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame_changes = sum(item.get("action_group") != group for item in frames)

    labels_root = root / "annotations" / "labels"
    label_paths = sorted(labels_root.rglob("*.json")) if labels_root.exists() else []
    labels = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in label_paths]
    label_changes = sum(item.get("action_group") != group for _, item in labels)

    if apply:
        manifest["current_action_group"] = group
        for item in sources:
            item["action_group"] = group
        sources_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        for item in frames:
            item["action_group"] = group
        with frames_path.open("w", encoding="utf-8") as handle:
            for item in frames:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        for path, item in labels:
            item["action_group"] = group
            path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "dataset_dir": str(root.resolve()),
        "action_group": group,
        "apply": bool(apply),
        "changes": {
            "source_manifest_and_records": source_changes,
            "frames": frame_changes,
            "labels": label_changes,
        },
        "counts": {"sources": len(sources), "frames": len(frames), "labels": len(labels)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Set the current action group across one video dataset.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--action-group", default="1A")
    parser.add_argument("--apply", action="store_true", help="Write changes; otherwise report a dry run.")
    args = parser.parse_args()
    print(json.dumps(set_action_group(args.dataset_dir, args.action_group, args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
