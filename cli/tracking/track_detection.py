"""Run only the yoyo detector and emit frame-keyed detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from config import DETECTION_CONFIG
from yoyo_detection.inference import load_detector


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect yoyos without loading string or orientation models.")
    parser.add_argument("video")
    parser.add_argument("--weights", default=str(DETECTION_CONFIG.weights_path))
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default=DETECTION_CONFIG.device)
    parser.add_argument("--confidence", type=float, default=DETECTION_CONFIG.confidence)
    parser.add_argument("--iou", type=float, default=DETECTION_CONFIG.iou)
    parser.add_argument("--imgsz", type=int, default=DETECTION_CONFIG.imgsz)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.video).with_suffix(".detection.jsonl")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")
    detector = load_detector(args.weights, args.device)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stream.write(json.dumps({"frame_index": index, "timestamp_s": index / fps, "detections": detector.predict(frame, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)}, ensure_ascii=False) + "\n")
            index += 1
    capture.release()
    print(json.dumps({"output": str(output.resolve()), "frames": index}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
