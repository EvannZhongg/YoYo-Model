"""Regenerate visual review images from saved annotation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation.video_frame_annotator import draw_visualization


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate visualizations for video-frame annotations.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    labels_root = dataset_dir / "annotations" / "labels"
    visualizations_root = dataset_dir / "annotations" / "visualizations"
    count = 0
    for label_path in sorted(labels_root.rglob("*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        if args.split != "all" and data.get("split") != args.split:
            continue
        relative = label_path.relative_to(labels_root)
        output_path = visualizations_root / relative.with_name(f"{relative.stem}_vis.jpg")
        draw_visualization(Path(data["source_image"]), data, output_path)
        count += 1
    print(f"regenerated={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
