"""Evaluate causal orientation smoothing on reviewed consecutive frames."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from common.files import sha256_file
from config import BASE_DIR, TRACKING_CONFIG
from video_tracking.orientation import (
    ORIENTATION_CLASS_ORDER,
    OrientationTemporalFilter,
    orientation_observation_is_unstable,
    orientation_crop_box,
)


def _load_raw_predictions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "yoyo_orientation_sequence_raw_predictions_v1":
        raise ValueError(f"Unsupported raw orientation predictions: {path}")
    return payload


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise RuntimeError(f"Could not read orientation evaluation image: {path}")
    return image


def _predict_dataset(
    dataset_dir: Path,
    weights: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    groups_path = dataset_dir / "consecutive_groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8")).get("groups") or []
    records: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    for group in groups:
        for item in group.get("frames") or []:
            label_path = dataset_dir / "canonical" / "labels" / Path(item["sample_key"])
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            image_path = dataset_dir / Path(item["image"])
            image = _read_image(image_path)
            height, width = image.shape[:2]
            bbox = annotation.get("yoyo_bbox_pixel")
            yoyo = {"bbox": [float(value) for value in bbox]} if isinstance(bbox, list) and len(bbox) == 4 else None
            left, top, right, bottom = orientation_crop_box(width, height, yoyo)
            crops.append(image[top:bottom, left:right])
            records.append({
                "group_id": str(group["group_id"]),
                "source_group": str(group.get("source_group") or group["group_id"]),
                "frame_index": int(item["frame_index"]),
                "timestamp_s": float(item["timestamp_s"]),
                "target": annotation.get("trick_orientation"),
                "crop_box_pixel": [left, top, right, bottom],
                "image": str(Path(item["image"])),
            })

    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(weights))
    results = model.predict(
        source=crops,
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        verbose=False,
    )
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    if set(names.values()) != set(ORIENTATION_CLASS_ORDER):
        raise ValueError(f"Incompatible orientation classes: {names}")
    for record, result in zip(records, results):
        values = [float(value) for value in result.probs.data.detach().cpu().tolist()]
        record["probabilities"] = {names[index]: values[index] for index in range(len(values))}
        record["predicted"] = names[int(result.probs.top1)]
        record["confidence"] = float(result.probs.top1conf.detach().cpu().item())
    return {
        "schema_version": "yoyo_orientation_sequence_raw_predictions_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "groups_sha256": sha256_file(groups_path),
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "imgsz": int(imgsz),
        "device": str(device),
        "inference_seconds": round(time.perf_counter() - started, 4),
        "records": records,
    }


def _interval_frames(records: list[dict[str, Any]], inference_fps: float) -> int:
    if inference_fps <= 0.0:
        return 1
    deltas = [
        float(right["timestamp_s"]) - float(left["timestamp_s"])
        for left, right in zip(records, records[1:])
        if float(right["timestamp_s"]) > float(left["timestamp_s"])
    ]
    source_fps = 1.0 / float(np.median(deltas)) if deltas else inference_fps
    return max(1, int(round(source_fps / inference_fps)))


def _replay(
    records: list[dict[str, Any]],
    inference_fps: float,
    filter_kwargs: dict[str, Any] | None,
    adaptive_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group_id"])].append(record)
    predictions: dict[str, str] = {}
    inference_count = burst_inference_count = 0
    for group_id, group_records in by_group.items():
        group_records.sort(key=lambda item: int(item["frame_index"]))
        interval = _interval_frames(group_records, inference_fps)
        burst_interval = _interval_frames(
            group_records,
            float((adaptive_kwargs or {}).get("burst_inference_fps", inference_fps)),
        )
        temporal_filter = OrientationTemporalFilter(**filter_kwargs) if filter_kwargs is not None else None
        label: str | None = None
        next_inference = 0
        stable_observations = 0
        for index, record in enumerate(group_records):
            if index >= next_inference:
                raw = {
                    "label": record["predicted"],
                    "confidence": record["confidence"],
                    "probabilities": record["probabilities"],
                }
                prediction = temporal_filter.update(raw) if temporal_filter is not None else raw
                label = str(prediction["label"])
                inference_count += 1
                if adaptive_kwargs is not None:
                    unstable = orientation_observation_is_unstable(
                        prediction,
                        float(adaptive_kwargs["min_confidence"]),
                    )
                    stable_observations = 0 if unstable else stable_observations + 1
                    use_burst = stable_observations < int(adaptive_kwargs["stable_observations"])
                    next_inference = index + (burst_interval if use_burst else interval)
                    burst_inference_count += int(use_burst)
                else:
                    next_inference = index + interval
            predictions[f"{group_id}:{record['frame_index']}"] = str(label or "unknown")
    return {
        "predictions": predictions,
        "inference_count": inference_count,
        "burst_inference_count": burst_inference_count,
    }


def _metrics(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group_id"])].append(record)
    correct = 0
    targets = Counter()
    class_correct = Counter()
    group_metrics: dict[str, Any] = {}
    predicted_switches = target_switches = isolated_flips = 0
    for group_id, group_records in by_group.items():
        group_records.sort(key=lambda item: int(item["frame_index"]))
        target = [str(item["target"]) for item in group_records]
        predicted = [predictions[f"{group_id}:{item['frame_index']}"] for item in group_records]
        group_correct = sum(left == right for left, right in zip(target, predicted))
        group_predicted_switches = sum(left != right for left, right in zip(predicted, predicted[1:]))
        group_target_switches = sum(left != right for left, right in zip(target, target[1:]))
        group_isolated = sum(
            predicted[index - 1] == predicted[index + 1] != predicted[index]
            and target[index - 1] == target[index] == target[index + 1]
            for index in range(1, len(group_records) - 1)
        )
        group_metrics[group_id] = {
            "frame_count": len(group_records),
            "accuracy": round(group_correct / len(group_records), 6),
            "target_switch_count": group_target_switches,
            "predicted_switch_count": group_predicted_switches,
            "isolated_flip_count": group_isolated,
        }
        correct += group_correct
        predicted_switches += group_predicted_switches
        target_switches += group_target_switches
        isolated_flips += group_isolated
        for expected, actual in zip(target, predicted):
            targets[expected] += 1
            class_correct[expected] += int(expected == actual)
    recalls = {
        name: round(class_correct[name] / targets[name], 6) if targets[name] else None
        for name in ORIENTATION_CLASS_ORDER
    }
    valid_recalls = [value for value in recalls.values() if value is not None]
    return {
        "frame_count": len(records),
        "accuracy": round(correct / len(records), 6),
        "macro_recall": round(float(np.mean(valid_recalls)), 6),
        "per_class_recall": recalls,
        "target_switch_count": target_switches,
        "predicted_switch_count": predicted_switches,
        "excess_switch_count": max(0, predicted_switches - target_switches),
        "isolated_flip_count": isolated_flips,
        "groups": group_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate temporal orientation stability on consecutive labels.")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "1Ayoyo_consecutive"))
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.orientation_weights_path))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "runs" / "experiments" / "orientation_temporal"))
    parser.add_argument("--raw-predictions", default="")
    parser.add_argument("--device", default=TRACKING_CONFIG.device)
    parser.add_argument("--imgsz", type=int, default=TRACKING_CONFIG.orientation_imgsz)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--baseline-fps", type=float, default=5.0)
    parser.add_argument("--burst-fps", type=float, default=TRACKING_CONFIG.orientation_burst_inference_fps)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    weights = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = (
        _load_raw_predictions(Path(args.raw_predictions).resolve())
        if args.raw_predictions else
        _predict_dataset(dataset_dir, weights, args.imgsz, args.batch, args.device)
    )
    raw_path = output_dir / "raw_predictions.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    records = list(raw["records"])
    filter_kwargs = {
        "ema_alpha": TRACKING_CONFIG.orientation_ema_alpha,
        "switch_margin": TRACKING_CONFIG.orientation_switch_margin,
        "switch_confirmations": TRACKING_CONFIG.orientation_switch_confirmations,
        "strong_switch_confidence": TRACKING_CONFIG.orientation_strong_switch_confidence,
        "strong_switch_margin": TRACKING_CONFIG.orientation_strong_switch_margin,
    }
    baseline_replay = _replay(records, args.baseline_fps, None)
    cadence_replay = _replay(records, args.burst_fps, None)
    adaptive_kwargs = {
        "burst_inference_fps": float(args.burst_fps),
        "min_confidence": TRACKING_CONFIG.orientation_adaptive_min_confidence,
        "stable_observations": TRACKING_CONFIG.orientation_adaptive_stable_observations,
    }
    candidate_replay = _replay(records, args.baseline_fps, filter_kwargs, adaptive_kwargs)
    baseline = _metrics(records, baseline_replay["predictions"])
    unfiltered_candidate_cadence = _metrics(records, cadence_replay["predictions"])
    candidate = _metrics(records, candidate_replay["predictions"])
    every_group_non_decreasing = all(
        candidate["groups"][group_id]["accuracy"] >= values["accuracy"]
        for group_id, values in baseline["groups"].items()
    )
    promotion_passed = bool(
        candidate["accuracy"] >= baseline["accuracy"]
        and candidate["macro_recall"] >= baseline["macro_recall"]
        and candidate["predicted_switch_count"] <= baseline["predicted_switch_count"]
        and every_group_non_decreasing
    )
    result = {
        "schema_version": "yoyo_orientation_temporal_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "raw_predictions": str(raw_path),
        "weights": raw.get("weights", str(weights)),
        "weights_sha256": raw.get("weights_sha256", sha256_file(weights)),
        "baseline": {
            "inference_fps": float(args.baseline_fps),
            "temporal_filter": False,
            "inference_count": baseline_replay["inference_count"],
            "metrics": baseline,
        },
        "candidate_cadence_ablation": {
            "inference_fps": float(args.burst_fps),
            "temporal_filter": False,
            "inference_count": cadence_replay["inference_count"],
            "metrics": unfiltered_candidate_cadence,
        },
        "candidate": {
            "stable_inference_fps": float(args.baseline_fps),
            "adaptive": adaptive_kwargs,
            "temporal_filter": True,
            "filter": filter_kwargs,
            "inference_count": candidate_replay["inference_count"],
            "burst_inference_count": candidate_replay["burst_inference_count"],
            "metrics": candidate,
        },
        "promotion_gate": {
            "passed": promotion_passed,
            "pooled_accuracy_non_decreasing": candidate["accuracy"] >= baseline["accuracy"],
            "pooled_macro_recall_non_decreasing": candidate["macro_recall"] >= baseline["macro_recall"],
            "switch_count_non_increasing": candidate["predicted_switch_count"] <= baseline["predicted_switch_count"],
            "every_group_accuracy_non_decreasing": every_group_non_decreasing,
        },
    }
    output = output_dir / "metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "promotion_gate": result["promotion_gate"]}, ensure_ascii=False, indent=2))
    return 0 if promotion_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
