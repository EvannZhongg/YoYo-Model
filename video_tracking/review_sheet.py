"""Create visual review artifacts for a tracking run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Could not encode review image: {path}")
    encoded.tofile(str(path))


def _select_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows
    selected: list[int] = []
    seen: set[int] = set()
    method_before = None
    for index, row in enumerate(rows):
        method = (row.get("string") or {}).get("method")
        bad = row.get("bad_case") or []
        important = bool(row.get("yoyo")) or bool(bad) or method != method_before
        if important and index not in seen:
            selected.append(index)
            seen.add(index)
        method_before = method
    uniform = np.linspace(0, len(rows) - 1, num=max_samples, dtype=int).tolist()
    selected.extend(index for index in uniform if index not in seen)
    selected = sorted(set(selected))
    if len(selected) > max_samples:
        selected = selected[:max_samples]
    return [rows[index] for index in selected]


def make_tracking_review_sheet(run_dir: str | Path, max_samples: int = 24, columns: int = 4, thumb_width: int = 480) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "frames.jsonl"
    video_path = run_dir / "tracked.mp4"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = _select_rows(rows, max(1, max_samples))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open tracked video: {video_path}")
    frame_map = {index: row for index, row in enumerate(rows) if row in selected}
    # The output video starts at local frame zero even when tracking started at
    # a non-zero source timestamp.
    images: list[tuple[dict[str, Any], np.ndarray]] = []
    wanted = {rows.index(row) for row in selected}
    ordinal = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if ordinal in wanted:
            images.append((rows[ordinal], frame.copy()))
            wanted.remove(ordinal)
        ordinal += 1
    capture.release()
    if not images:
        raise RuntimeError(f"No frames available in {video_path}")
    aspect = images[0][1].shape[0] / max(1, images[0][1].shape[1])
    thumb_height = max(1, round(thumb_width * aspect))
    cell_height = thumb_height + 78
    rows_count = max(1, (len(images) + columns - 1) // columns)
    canvas = np.full((rows_count * cell_height, columns * thumb_width, 3), 255, dtype=np.uint8)
    frame_dir = run_dir / "review_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for cell, (record, frame) in enumerate(images):
        thumb = cv2.resize(frame, (thumb_width, thumb_height))
        x = (cell % columns) * thumb_width
        y = (cell // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = thumb
        string = record.get("string") or {}
        method = string.get("method", "none")
        confidence = string.get("confidence", "-")
        bad = ",".join(record.get("bad_case") or []) or "ok"
        text = f"f{record.get('frame_index')} t={record.get('timestamp_s', 0):.2f}s yoyo={'yes' if record.get('yoyo') else 'no'}"
        cv2.putText(canvas, text[:72], (x + 4, y + thumb_height + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"string={method} conf={confidence}", (x + 4, y + thumb_height + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 100, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, bad[:72], (x + 4, y + thumb_height + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 180), 1, cv2.LINE_AA)
        frame_path = frame_dir / f"frame_{int(record.get('frame_index', cell)):08d}.jpg"
        _write_jpeg(frame_path, frame, 90)
        index_rows.append({"frame": record, "image": str(frame_path.resolve())})
    output_path = run_dir / "tracking_review_sheet.jpg"
    _write_jpeg(output_path, canvas, 88)
    (run_dir / "tracking_review_index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a visual review sheet for one tracking run.")
    parser.add_argument("run_dir")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=480)
    args = parser.parse_args()
    print(make_tracking_review_sheet(args.run_dir, args.max_samples, args.columns, args.thumb_width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
