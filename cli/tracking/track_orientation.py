"""Run only orientation recognition; detector results are optional context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from config import ORIENTATION_CONFIG
from yoyo_orientation.inference import OrientationRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize yoyo orientation without loading detector or string models.")
    parser.add_argument("video")
    parser.add_argument("--weights", default=str(ORIENTATION_CONFIG.weights_path))
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default=ORIENTATION_CONFIG.device or "cpu")
    parser.add_argument("--imgsz", type=int, default=ORIENTATION_CONFIG.imgsz)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.video).with_suffix(".orientation.jsonl")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")
    recognizer = OrientationRecognizer(args.weights, args.device, args.imgsz)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stream.write(json.dumps({"frame_index": index, "timestamp_s": index / fps, "orientation": recognizer.predict(frame)}, ensure_ascii=False) + "\n")
            index += 1
    capture.release()
    print(json.dumps({"output": str(output.resolve()), "frames": index}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
