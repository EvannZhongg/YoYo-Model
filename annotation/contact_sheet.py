"""Create compact visual review sheets for VLM annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def make_contact_sheet(
    dataset_dir: Path,
    split: str,
    columns: int,
    thumb_width: int,
    string_status: str = "all",
) -> Path:
    labels_root = dataset_dir / "annotations" / "labels"
    labels = []
    for path in sorted(labels_root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if split != "all" and data.get("split") != split:
            continue
        if string_status != "all" and data.get("string_review_status", "auto_labeled_needs_review") != string_status:
            continue
        labels.append((path, data))
    thumb_height = round(thumb_width * 9 / 16)
    cell_height = thumb_height + 68
    rows = max(1, (len(labels) + columns - 1) // columns)
    canvas = 255 * __import__("numpy").ones((rows * cell_height, columns * thumb_width, 3), dtype="uint8")
    for index, (path, data) in enumerate(labels):
        relative = path.relative_to(labels_root)
        visualization = dataset_dir / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
        image = cv2.imread(str(visualization if visualization.exists() else data["source_image"]))
        if image is None:
            continue
        image = cv2.resize(image, (thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = image
        text = f"{data.get('video_id','?')[:8]} f{data.get('frame_index','?')} {data.get('visibility','?')}"
        bbox_status = data.get("bbox_review_status", data.get("review_status", "needs_review"))
        string_status = data.get("string_review_status", "needs_review")
        review = f"bbox={bbox_status} string={string_status}"
        warnings = ",".join((data.get("qa") or {}).get("warnings", []))
        bad = warnings or ",".join(data.get("bad_case", [])) or "qa=ok"
        cv2.putText(canvas, text, (x + 4, y + thumb_height + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, review[:62], (x + 4, y + thumb_height + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 100, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, bad[:62], (x + 4, y + thumb_height + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 180), 1, cv2.LINE_AA)
    suffix = "" if string_status == "all" else f"_{string_status}"
    output = dataset_dir / "review_sheets" / f"{split}{suffix}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create VLM annotation contact sheets for visual review.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument(
        "--string-status",
        choices=["all", "auto_labeled_needs_review", "approved", "reviewed", "rejected"],
        default="all",
    )
    args = parser.parse_args()
    print(make_contact_sheet(Path(args.dataset_dir), args.split, args.columns, args.thumb_width, args.string_status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
