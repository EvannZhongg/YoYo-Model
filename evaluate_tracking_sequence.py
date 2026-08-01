"""Evaluate a tracking JSONL against a consecutive annotated dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_tracking.sequence_metrics import evaluate_sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate yoyo and string tracking on consecutive labels.")
    parser.add_argument("dataset_dir")
    parser.add_argument("predictions", help="frames.jsonl or a tracking run directory")
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--tolerance", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--sample-spacing", type=float, default=2.0)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--ground-truth-snapshot", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate_sequence(
        args.dataset_dir,
        args.predictions,
        args.group_id,
        args.tolerance,
        args.sample_spacing,
        args.include_frames,
        args.ground_truth_snapshot or None,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
