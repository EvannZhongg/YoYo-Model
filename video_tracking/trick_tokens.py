"""Build one clip-token per reviewed, valid trick video segment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _resolve_clip(path_value: str, segments_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path.resolve()
    candidate = segments_path.parent / path
    return candidate.resolve() if candidate.exists() else path


def _tracking_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {
            "frame_count": 0,
            "yoyo_visible_ratio": 0.0,
            "string_observed_ratio": 0.0,
            "hands_observed_ratio": 0.0,
            "pose_observed_ratio": 0.0,
            "bad_case_counts": {},
        }
    yoyo_rows = [row for row in records if row.get("yoyo")]
    string_rows = [row for row in records if row.get("string")]
    hand_rows = [row for row in records if row.get("hands")]
    pose_rows = [row for row in records if row.get("pose")]
    bad_cases = Counter(value for row in records for value in (row.get("bad_case") or []))
    return {
        "frame_count": count,
        "yoyo_visible_ratio": round(len(yoyo_rows) / count, 6),
        "string_observed_ratio": round(len(string_rows) / count, 6),
        "hands_observed_ratio": round(len(hand_rows) / count, 6),
        "pose_observed_ratio": round(len(pose_rows) / count, 6),
        "mean_yoyo_confidence": round(sum(float(row["yoyo"].get("confidence", 0.0)) for row in yoyo_rows) / len(yoyo_rows), 6) if yoyo_rows else 0.0,
        "mean_string_confidence": round(sum(float(row["string"].get("confidence", 0.0)) for row in string_rows) / len(string_rows), 6) if string_rows else 0.0,
        "bad_case_counts": dict(sorted(bad_cases.items())),
    }


def export_trick_tokens(segments_path: str | Path) -> dict[str, Any]:
    segments_path = Path(segments_path)
    run_dir = segments_path.parent
    run_manifest_path = run_dir / "run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    frame_metadata_path = run_dir / "frames.jsonl"
    records = [json.loads(line) for line in frame_metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()] if frame_metadata_path.exists() else []
    tokens: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    approved = sorted((item for item in segments if item.get("review_status") == "approved"), key=lambda item: float(item.get("start_time_s", 0.0)))
    source_sha = str(run_manifest.get("source_video_sha256", ""))
    for sequence_index, segment in enumerate(approved):
        duration = float(segment.get("duration_s", 0.0))
        clip = _resolve_clip(str(segment.get("output_video", "")), segments_path)
        if duration <= 0 or duration > 180.0:
            excluded.append({"segment_id": segment.get("segment_id"), "reason": "invalid valid-segment duration"})
            continue
        if not clip.exists():
            excluded.append({"segment_id": segment.get("segment_id"), "reason": f"clip not found: {clip}"})
            continue
        start_time = float(segment.get("start_time_s", 0.0))
        end_time = float(segment.get("end_time_s", start_time + duration))
        segment_records = [row for row in records if start_time <= float(row.get("timestamp_s", 0.0)) <= end_time]
        token_id = f"{source_sha[:12] or 'source'}_trick_{int(segment.get('segment_id', sequence_index + 1)):03d}"
        tokens.append(
            {
                "schema_version": "yoyo_trick_clip_token_v1",
                "token_type": "valid_trick_video_clip",
                "token_id": token_id,
                "sequence_index": len(tokens),
                "source_video": run_manifest.get("source_video", ""),
                "source_video_sha256": source_sha,
                "tracking_run": str(run_dir.resolve()),
                "segment_id": int(segment.get("segment_id", sequence_index + 1)),
                "clip_path": str(clip),
                "start_time_s": start_time,
                "end_time_s": end_time,
                "duration_s": duration,
                "trick_label": str(segment.get("trick_label", "")),
                "review_status": "approved",
                "tracking_summary": _tracking_summary(segment_records),
                "internal_frame_features": {
                    "manifest": str(run_dir / "frame_feature_manifest.json"),
                    "start_frame": int(segment.get("start_frame", 0)),
                    "end_frame": int(segment.get("end_frame", 0)),
                },
                "score_target": None,
            }
        )
    for segment in segments:
        if segment.get("review_status") != "approved":
            excluded.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "review_status": segment.get("review_status", "auto_candidate_needs_review"),
                    "reason": "not an approved valid trick segment",
                }
            )
    jsonl_path = run_dir / "trick_tokens.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for token in tokens:
            file.write(json.dumps(token, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "yoyo_trick_clip_token_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": run_manifest.get("source_video", ""),
        "source_video_sha256": source_sha,
        "token_definition": "One reviewed, continuous, valid trick video segment. Irrelevant or unreviewed intervals are excluded.",
        "token_count": len(tokens),
        "excluded_segment_count": len(excluded),
        "excluded_segments": excluded,
        "ready_for_scoring_dataset": bool(tokens),
        "outputs": {"jsonl": str(jsonl_path)},
    }
    manifest_path = run_dir / "trick_token_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "manifest": str(manifest_path), "token_count": len(tokens)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one clip-token per approved valid trick segment.")
    parser.add_argument("segments_json")
    args = parser.parse_args()
    print(json.dumps(export_trick_tokens(args.segments_json), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
