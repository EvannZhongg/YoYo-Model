"""Select candidate frames with a bootstrap yoyo detector.

The output is deliberately marked ``candidate_only`` and ``unreviewed``. It is
an active-learning queue, never a replacement for human-approved annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO
from video_dataset.split_policy import parse_source_groups


def select_candidates(
    dataset_dir: Path,
    weights: Path,
    sample_fps: float,
    confidence: float,
    imgsz: int,
    split: str,
    max_videos: int,
    max_candidates_per_video: int,
    exclude_source_groups: set[str] | str | None = None,
) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "sources.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    sources = [source for source in sources if split == "all" or source["split"] == split]
    excluded_groups = (
        parse_source_groups(exclude_source_groups)
        if isinstance(exclude_source_groups, str)
        else {str(value).strip() for value in (exclude_source_groups or set()) if str(value).strip()}
    )
    sources = [
        source for source in sources
        if str(source.get("source_group") or source.get("video_id") or "").strip() not in excluded_groups
    ]
    if max_videos > 0:
        sources = sources[:max_videos]
    model = YOLO(str(weights))
    frame_manifest_path = dataset_dir / "frames.jsonl"
    existing = {}
    if frame_manifest_path.exists():
        existing = {
            record["frame_path"]: record
            for record in (
                json.loads(line)
                for line in frame_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    candidate_root = dataset_dir / "candidate_frames"
    candidate_count = 0
    videos_processed = 0
    for source in sources:
        capture = cv2.VideoCapture(source["path"])
        if not capture.isOpened():
            continue
        stride = max(1, int(round(source["fps"] / sample_fps))) if sample_fps > 0 and source["fps"] else 1
        saved_for_video = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            result = model.predict(source=frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            detections = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                for bbox, conf in zip(xyxy, confs):
                    detections.append({"bbox_pixel": [float(value) for value in bbox], "confidence": float(conf)})
            if detections:
                relative = Path(source["split"]) / source["video_id"] / f"frame_{index:08d}.jpg"
                output_path = candidate_root / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                record = {
                    "schema_version": "1.0",
                    "frame_path": str(output_path.resolve()),
                    "source_video": source["path"],
                    "source_video_sha256": source["sha256"],
                    "video_id": source["video_id"],
                    "source_group": source["source_group"],
                    "action_group": source.get("action_group", manifest.get("current_action_group", "1A")),
                    "subject_id": source.get("subject_id"),
                    "split": source["split"],
                    "frame_index": index,
                    "timestamp_s": round(index / source["fps"], 4) if source["fps"] else None,
                    "annotation_status": "unreviewed",
                    "candidate_only": True,
                    "bootstrap_detections": detections,
                    "visibility": "unknown",
                    "yoyo_bbox": None,
                    "string_polyline": None,
                    "hands": None,
                    "pose": None,
                    "bad_case": [],
                    "review_notes": "Candidate selected by bootstrap detector; verify manually.",
                }
                existing[str(output_path.resolve())] = record
                candidate_count += 1
                saved_for_video += 1
                if max_candidates_per_video > 0 and saved_for_video >= max_candidates_per_video:
                    break
            index += 1
        capture.release()
        videos_processed += 1
    records = sorted(existing.values(), key=lambda item: (item["split"], item["video_id"], item["frame_index"], item["frame_path"]))
    with frame_manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "videos_processed": videos_processed,
        "new_candidate_observations": candidate_count,
        "total_frame_records": len(records),
        "frames_jsonl": str(frame_manifest_path.resolve()),
        "weights": str(weights.resolve()),
        "exclude_source_groups": sorted(excluded_groups),
    }
    (dataset_dir / "candidate_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select yoyo candidate frames using a bootstrap detector.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-candidates-per-video", type=int, default=0)
    parser.add_argument("--exclude-source-groups", default="", help="Comma-separated source groups excluded before detector inference.")
    args = parser.parse_args()
    result = select_candidates(
        Path(args.dataset_dir),
        Path(args.weights),
        args.sample_fps,
        args.confidence,
        args.imgsz,
        args.split,
        args.max_videos,
        args.max_candidates_per_video,
        args.exclude_source_groups,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
