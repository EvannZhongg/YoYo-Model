"""Create visual review artifacts for a tracking run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PRIORITY_BAD_CASES = (
    "string_hand_anchor_mismatch",
    "string_temporal_conflict",
    "string_tracking_lost",
    "pose_identity_needs_review",
)


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Could not encode review image: {path}")
    encoded.tofile(str(path))


def _write_source_review_frames(
    source_video_path: str | Path | None,
    frame_indices: list[int],
    output_dir: Path,
) -> dict[int, tuple[Path, list[int]]]:
    if not source_video_path or not frame_indices:
        return {}
    source_path = Path(source_video_path)
    if not source_path.is_file():
        return {}
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        return {}
    indices = sorted(set(max(0, int(value)) for value in frame_indices))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, tuple[Path, list[int]]] = {}

    def save(frame_index: int, frame: np.ndarray) -> None:
        path = output_dir / f"frame_{frame_index:08d}.jpg"
        _write_jpeg(path, frame, 92)
        results[frame_index] = (path, [int(frame.shape[1]), int(frame.shape[0])])

    try:
        span = indices[-1] - indices[0] + 1
        if span <= max(500, len(indices) * 24):
            wanted = set(indices)
            capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
            frame_index = indices[0]
            while wanted and frame_index <= indices[-1]:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in wanted:
                    save(frame_index, frame)
                    wanted.remove(frame_index)
                frame_index += 1
        else:
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if ok:
                    save(frame_index, frame)
    finally:
        capture.release()
    return results


def _select_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows
    priority_indices: set[int] = {0, len(rows) - 1}
    for flag in PRIORITY_BAD_CASES:
        matching = [index for index, row in enumerate(rows) if flag in (row.get("bad_case") or [])]
        if matching:
            priority_indices.add(matching[0])
    mismatch_rows = [
        (index, row.get("string") or {})
        for index, row in enumerate(rows)
        if "string_hand_anchor_mismatch" in (row.get("bad_case") or [])
    ]
    if mismatch_rows:
        worst_index, _ = max(
            mismatch_rows,
            key=lambda item: float(item[1].get("distance_to_nearest_wrist_px") or 0.0)
            / max(1.0, float(item[1].get("hand_anchor_threshold_px") or 0.0)),
        )
        priority_indices.add(worst_index)

    event_indices: set[int] = set(priority_indices)
    previous_signature = None
    for index, row in enumerate(rows):
        method = (row.get("string") or {}).get("method")
        pose_person = row.get("pose_person") or {}
        signature = (
            bool(row.get("yoyo")),
            method,
            tuple(sorted(row.get("bad_case") or [])),
            pose_person.get("status"),
            bool(pose_person.get("needs_review")),
        )
        if signature != previous_signature or pose_person.get("needs_review"):
            event_indices.add(index)
        previous_signature = signature
    selected = sorted(priority_indices)
    if len(selected) > max_samples:
        selected = sorted(
            sorted(
                selected,
                key=lambda index: (
                    0
                    if "string_hand_anchor_mismatch" in (rows[index].get("bad_case") or [])
                    else 1
                    if any(flag in (rows[index].get("bad_case") or []) for flag in PRIORITY_BAD_CASES)
                    else 2,
                    index,
                ),
            )[:max_samples]
        )
    remaining_events = sorted(event_indices - priority_indices)
    available = max(0, max_samples - len(selected))
    if len(remaining_events) > available:
        offsets = np.linspace(0, len(remaining_events) - 1, num=available, dtype=int)
        remaining_events = [remaining_events[int(offset)] for offset in offsets]
    selected.extend(remaining_events)
    uniform = np.linspace(0, len(rows) - 1, num=max_samples, dtype=int).tolist()
    for index in uniform:
        if len(selected) >= max_samples:
            break
        if index not in selected:
            selected.append(index)
    selected = sorted(set(selected))
    return [rows[index] for index in selected]


def _pose_caption(record: dict[str, Any]) -> str:
    pose_person = record.get("pose_person") or {}
    status = str(pose_person.get("status", "missing"))
    if status != "ok":
        return f"pose={status}"
    person_index = pose_person.get("person_index", "-")
    person_count = pose_person.get("person_count", "-")
    mode = "temporal" if pose_person.get("temporal_reference_used") else "cold"
    iou = pose_person.get("temporal_bbox_iou")
    iou_text = f"{float(iou):.3f}" if iou is not None else "-"
    review_reasons = pose_person.get("review_reasons") or []
    review = ",".join(str(value) for value in review_reasons) or "ok"
    return f"pose=p{person_index}/{person_count} {mode} iou={iou_text} review={review}"


def _string_caption(record: dict[str, Any]) -> str:
    string = record.get("string") or {}
    method = string.get("method", "none")
    confidence = string.get("confidence", "-")
    caption = f"string={method} conf={confidence}"
    anchor_status = string.get("hand_anchor_status")
    if anchor_status:
        caption += f" hand={anchor_status}"
    distance = string.get("distance_to_nearest_wrist_px")
    threshold = string.get("hand_anchor_threshold_px")
    if distance is not None and threshold is not None:
        caption += f" {float(distance):.0f}/{float(threshold):.0f}px"
    return caption


def make_tracking_review_sheet(
    run_dir: str | Path,
    max_samples: int = 24,
    columns: int = 4,
    thumb_width: int = 480,
    source_video_path: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "frames.jsonl"
    video_path = run_dir / "tracked.mp4"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = _select_rows(rows, max(1, max_samples))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open tracked video: {video_path}")
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
    cell_height = thumb_height + 99
    rows_count = max(1, (len(images) + columns - 1) // columns)
    canvas = np.full((rows_count * cell_height, columns * thumb_width, 3), 255, dtype=np.uint8)
    frame_dir = run_dir / "review_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_frame_dir = run_dir / "review_raw_frames"
    raw_frames = _write_source_review_frames(
        source_video_path,
        [int(record.get("frame_index", index)) for index, (record, _) in enumerate(images)],
        raw_frame_dir,
    )
    index_rows = []
    for cell, (record, frame) in enumerate(images):
        thumb = cv2.resize(frame, (thumb_width, thumb_height))
        x = (cell % columns) * thumb_width
        y = (cell // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = thumb
        string = record.get("string") or {}
        anchor_mismatch = bool(string.get("hand_anchor_mismatch"))
        bad = ",".join(record.get("bad_case") or []) or "ok"
        text = f"f{record.get('frame_index')} t={record.get('timestamp_s', 0):.2f}s yoyo={'yes' if record.get('yoyo') else 'no'}"
        cv2.putText(canvas, text[:72], (x + 4, y + thumb_height + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            _string_caption(record)[:72],
            (x + 4, y + thumb_height + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 0, 210) if anchor_mismatch else (20, 100, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(canvas, _pose_caption(record)[:72], (x + 4, y + thumb_height + 59), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 75, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, bad[:72], (x + 4, y + thumb_height + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 180), 1, cv2.LINE_AA)
        if anchor_mismatch:
            cv2.rectangle(canvas, (x + 1, y + 1), (x + thumb_width - 2, y + thumb_height - 2), (0, 0, 255), 4)
        frame_path = frame_dir / f"frame_{int(record.get('frame_index', cell)):08d}.jpg"
        _write_jpeg(frame_path, frame, 90)
        source_frame_index = int(record.get("frame_index", cell))
        raw_frame = raw_frames.get(source_frame_index)
        raw_frame_path = raw_frame[0] if raw_frame else None
        raw_image_size = raw_frame[1] if raw_frame else None
        index_rows.append(
            {
                "frame": record,
                "source_frame_index": source_frame_index,
                "image": str(frame_path.resolve()),
                "overlay_image": str(frame_path.resolve()),
                "overlay_image_size": [int(frame.shape[1]), int(frame.shape[0])],
                "raw_image": str(raw_frame_path.resolve()) if raw_frame_path else None,
                "raw_image_size": raw_image_size,
            }
        )
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
    parser.add_argument("--source-video", default="")
    args = parser.parse_args()
    print(
        make_tracking_review_sheet(
            args.run_dir,
            args.max_samples,
            args.columns,
            args.thumb_width,
            args.source_video or None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
