"""Mine real-video detector candidates for human hard-negative review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from common.files import sha256_file
from config import BASE_DIR, TRACKING_CONFIG


YOYO_NAMES = {"yoyo", "yo-yo", "yoyo_body"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _video_files(videos_dir: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        result = [path.resolve() for path in explicit]
    else:
        result = sorted(
            path.resolve()
            for path in videos_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    missing = [path for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Video file not found: {missing[0]}")
    if not result:
        raise ValueError(f"No video files found under {videos_dir}")
    return result


def _encode_frame(frame: Any, path: Path, quality: int) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Could not encode frame: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def _extract_candidates(result: Any, names: dict[int, str], confidence: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    candidates = []
    for box, score, class_id in zip(xyxy, confs, classes):
        class_name = str(names.get(int(class_id), "")).lower()
        score = float(score)
        if class_name not in YOYO_NAMES or score < float(confidence):
            continue
        candidates.append({
            "class_name": class_name,
            "confidence": round(score, 6),
            "bbox": [round(float(value), 3) for value in box],
        })
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


def mine_hard_negatives(
    videos_dir: Path,
    output_dir: Path,
    weights: Path,
    explicit_videos: list[Path] | None = None,
    confidence: float = 0.15,
    sample_every: int = 5,
    min_gap_frames: int = 5,
    imgsz: int = 1024,
    device: str = TRACKING_CONFIG.device,
    jpeg_quality: int = 92,
    start_frame: int = 0,
    max_frames_per_video: int = 0,
) -> dict[str, Any]:
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if int(sample_every) < 1 or int(min_gap_frames) < 1:
        raise ValueError("sample_every and min_gap_frames must be positive")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if int(start_frame) < 0:
        raise ValueError("start_frame must be non-negative")
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    videos = _video_files(videos_dir.resolve(), explicit_videos or [])

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(key): str(value) for key, value in dict(getattr(model, "names", {}) or {}).items()}
    records: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    frames_root = output_dir / "frames"
    for video_path in videos:
        video_sha256 = sha256_file(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")
        if int(start_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frame_index = int(start_frame)
        frames_read = 0
        sampled_frames = 0
        candidate_count = 0
        last_saved_frame = -int(min_gap_frames)
        video_records = []
        while True:
            ok, frame = capture.read()
            if not ok or (max_frames_per_video and frame_index >= int(start_frame) + int(max_frames_per_video)):
                break
            frames_read += 1
            if frame_index < int(start_frame) or (frame_index - int(start_frame)) % int(sample_every) != 0:
                frame_index += 1
                continue
            sampled_frames += 1
            kwargs: dict[str, Any] = {"source": frame, "imgsz": int(imgsz), "conf": float(confidence), "verbose": False}
            if str(device).strip():
                kwargs["device"] = str(device).strip()
            result = model.predict(**kwargs)[0]
            detections = _extract_candidates(result, names, confidence)
            if detections and frame_index - last_saved_frame >= int(min_gap_frames):
                digest = hashlib.sha256(f"{video_path}:{frame_index}".encode("utf-8")).hexdigest()[:16]
                frame_path = frames_root / digest[:2] / f"{digest}.jpg"
                frame_sha256 = _encode_frame(frame, frame_path, jpeg_quality)
                record = {
                    "review_status": "needs_review",
                    "source_video": str(video_path),
                    "source_video_sha256": video_sha256,
                    "frame_index": int(frame_index),
                    "timestamp_s": round(float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0, 6),
                    "frame_path": str(frame_path.resolve()),
                    "frame_sha256": frame_sha256,
                    "detections": detections,
                }
                records.append(record)
                video_records.append(record)
                candidate_count += 1
                last_saved_frame = frame_index
            frame_index += 1
        capture.release()
        video_summaries.append({
            "source_video": str(video_path),
            "start_frame": int(start_frame),
            "frame_count": int(frames_read),
            "end_frame_exclusive": int(frame_index),
            "sampled_frames": int(sampled_frames),
            "candidate_frames": int(candidate_count),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "yoyo_real_hard_negative_mining_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "videos_dir": str(videos_dir.resolve()),
        "parameters": {
            "confidence": float(confidence),
            "sample_every": int(sample_every),
            "min_gap_frames": int(min_gap_frames),
            "imgsz": int(imgsz),
            "device": str(device),
            "jpeg_quality": int(jpeg_quality),
            "start_frame": int(start_frame),
            "max_frames_per_video": int(max_frames_per_video),
        },
        "review_policy": "Candidates are not training negatives until a human confirms no yoyo is present.",
        "videos": video_summaries,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path.resolve()), "video_count": len(videos), "candidate_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine real-video yoyo detections for hard-negative review.")
    parser.add_argument("--videos-dir", default=str(BASE_DIR / "videos"))
    parser.add_argument("--video", action="append", default=[], help="Process one video; repeat for multiple videos.")
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.weights_path))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "tmp" / "hard_negative_mining"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--min-gap-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=TRACKING_CONFIG.device)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    args = parser.parse_args()
    result = mine_hard_negatives(
        Path(args.videos_dir), Path(args.output_dir), Path(args.weights), [Path(value) for value in args.video],
        args.confidence, args.sample_every, args.min_gap_frames, args.imgsz, args.device,
        args.jpeg_quality, args.start_frame, args.max_frames_per_video,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
