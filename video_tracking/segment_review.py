"""Review and export heuristic trick segments without editing source videos."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ALLOWED_STATUSES = {"auto_candidate_needs_review", "approved", "irrelevant", "rejected", "edited"}


def load_segment_context(segments_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    segments_path = Path(segments_path)
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    run_dir = segments_path.parent
    run_manifest_path = run_dir / "run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
    return segments_path, run_manifest, segments


def _write_clip(source_video: Path, output_video: Path, start_frame: int, end_frame: int, fps: float, width: int, height: int) -> None:
    capture = cv2.VideoCapture(str(source_video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create clip: {output_video}")
    index = start_frame
    try:
        while index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()


def update_segment(
    segments_path: str | Path,
    segment_id: int,
    status: str,
    start_time_s: float,
    end_time_s: float,
    trick_label: str = "",
    review_notes: str = "",
    export_clip: bool = True,
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported segment status: {status}")
    segments_path, run_manifest, segments = load_segment_context(segments_path)
    fps = float(run_manifest.get("fps") or 30.0)
    source_video = Path(run_manifest.get("source_video", ""))
    max_duration = float((run_manifest.get("parameters") or {}).get("max_segment_seconds", 180.0))
    start_time_s = max(0.0, float(start_time_s))
    end_time_s = max(start_time_s, float(end_time_s))
    if end_time_s <= start_time_s:
        raise ValueError("Segment end must be after start")
    if end_time_s - start_time_s > max_duration + 1e-6 or end_time_s - start_time_s > 180.0 + 1e-6:
        raise ValueError("Valid exported segment duration cannot exceed 180 seconds")
    selected = next((item for item in segments if int(item.get("segment_id", -1)) == int(segment_id)), None)
    if selected is None:
        raise KeyError(f"Segment not found: {segment_id}")
    start_frame = max(0, int(round(start_time_s * fps)))
    end_frame = max(start_frame + 1, int(round(end_time_s * fps)) - 1)
    selected.update(
        {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": round(start_time_s, 4),
            "end_time_s": round(end_time_s, 4),
            "duration_s": round(end_time_s - start_time_s, 4),
            "review_status": status,
            "needs_review": status not in {"approved", "irrelevant", "rejected"},
            "trick_label": str(trick_label).strip(),
            "review_notes": str(review_notes).strip(),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if export_clip and source_video.exists():
        capture = cv2.VideoCapture(str(source_video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        output_video = segments_path.parent / "clips" / f"{source_video.stem}_trick_{int(segment_id):03d}.mp4"
        _write_clip(source_video, output_video, start_frame, end_frame, fps, width, height)
        selected["output_video"] = str(output_video)
    segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from video_tracking.trick_tokens import export_trick_tokens

        selected["trick_token_export"] = export_trick_tokens(segments_path)
    except Exception as exc:
        selected["trick_token_export_error"] = str(exc)
    return selected
