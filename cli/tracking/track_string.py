"""Run only string recognition; detector results may be joined by frame index later."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from config import STRING_TRACKING_CONFIG
from string_tracking.inference import load_string_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize strings without loading detector or orientation models.")
    parser.add_argument("video")
    parser.add_argument("--weights", default=str(STRING_TRACKING_CONFIG.weights_path))
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default=STRING_TRACKING_CONFIG.device or "cpu")
    parser.add_argument("--confidence", type=float, default=STRING_TRACKING_CONFIG.confidence)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.video).with_suffix(".string.jsonl")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.video}")
    recognizer = load_string_model(args.weights, args.device)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            value = recognizer.predict(frame, confidence=args.confidence)
            stream.write(json.dumps({"frame_index": index, "timestamp_s": index / fps, "string": value}, ensure_ascii=False) + "\n")
            index += 1
    capture.release()
    print(json.dumps({"output": str(output.resolve()), "frames": index}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
