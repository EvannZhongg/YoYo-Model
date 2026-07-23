"""Evaluate the deployed detector/string pipeline on reviewed frame splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.files import sha256_file
from config import TRACKING_CONFIG
from string_segmentation.semantic_metrics import metrics_at_threshold
from string_segmentation.semantic_model import image_label_pairs, letterbox, render_yolo_segmentation
from video_tracking.string_tracker import estimate_string
from video_tracking.tracker import (
    _extract_detections,
    _load_string_model,
    _pick_yoyo,
    _predict_string_model,
)


def _truth_bbox(annotation: dict[str, Any]) -> list[float] | None:
    bbox = annotation.get("yoyo_bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]
    for item in annotation.get("bbox") or []:
        bbox = item.get("bbox_pixel") if isinstance(item, dict) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            return [float(value) for value in bbox]
    return None


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _observation_mask(observation: dict[str, Any] | None, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if observation is None:
        return mask
    polygons = observation.get("polygons") or ([observation["polygon"]] if observation.get("polygon") else [])
    arrays = []
    for polygon in polygons:
        array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(array) >= 3:
            array[:, 0] = np.clip(array[:, 0], 0, width - 1)
            array[:, 1] = np.clip(array[:, 1], 0, height - 1)
            arrays.append(array.round().astype(np.int32))
    if arrays:
        cv2.fillPoly(mask, arrays, 1)
        return mask
    polylines = observation.get("polylines") or ([observation["points"]] if observation.get("points") else [])
    for polyline in polylines:
        array = np.asarray(polyline, dtype=np.float32).reshape(-1, 2)
        if len(array) >= 2:
            array[:, 0] = np.clip(array[:, 0], 0, width - 1)
            array[:, 1] = np.clip(array[:, 1], 0, height - 1)
            cv2.polylines(mask, [array.round().astype(np.int32)], False, 1, 4)
    return mask


def detector_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("bbox_truth_accepted")]
    tp = fp = fn = tn = 0
    ious: list[float] = []
    matches = 0
    for row in accepted:
        truth_present = row.get("truth_bbox") is not None
        prediction_present = row.get("predicted_bbox") is not None
        if truth_present and prediction_present:
            tp += 1
            iou = _bbox_iou(row["truth_bbox"], row["predicted_bbox"])
            ious.append(iou)
            matches += iou >= 0.5
        elif prediction_present:
            fp += 1
        elif truth_present:
            fn += 1
        else:
            tn += 1

    def ratio(value: float, denominator: float) -> float:
        return float(value / denominator) if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "accepted_images": len(accepted),
        "presence": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(ratio(2 * precision * recall, precision + recall), 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "mean_iou_when_both_present": round(float(np.mean(ious)) if ious else 0.0, 6),
        "iou50_matches": matches,
        "iou50_recall": round(ratio(matches, sum(row.get("truth_bbox") is not None for row in accepted)), 6),
    }


def _annotation_path(annotations_dir: Path, split: str, image_path: Path) -> Path:
    source_group = image_path.parent.name
    filename = f"{image_path.stem}.json"
    preferred = annotations_dir / "labels" / split / source_group / filename
    if preferred.is_file():
        return preferred

    matches = [
        annotations_dir / "labels" / candidate_split / source_group / filename
        for candidate_split in ("train", "val", "test")
        if candidate_split != split
        and (annotations_dir / "labels" / candidate_split / source_group / filename).is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous canonical annotation for {source_group}/{filename}: {matches}")
    return matches[0] if matches else preferred


def _detector_truth(annotation: dict[str, Any]) -> tuple[bool, list[float] | None]:
    bbox_status = str(annotation.get("bbox_review_status", annotation.get("review_status", "")))
    truth_bbox = _truth_bbox(annotation)
    truth_visibility = str(annotation.get("visibility", "uncertain"))
    if truth_visibility in {"absent", "out_of_frame"}:
        truth_bbox = None
    return bbox_status in {"approved", "reviewed"}, truth_bbox


def backfill_detector_truth(metrics_path: str | Path, annotations_dir: str | Path) -> dict[str, Any]:
    """Repair detector truth metrics from frozen prediction rows without rerunning models."""
    metrics_path = Path(metrics_path)
    annotations_dir = Path(annotations_dir)
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    split = str(result.get("split", ""))
    rows = result.get("string", {}).get("images")
    if split not in {"train", "val", "test"} or not isinstance(rows, list):
        raise ValueError(f"Not a tracking pipeline metrics artifact: {metrics_path}")

    resolved = missing = canonical_split_fallbacks = 0
    for row in rows:
        image_path = Path(str(row["image_path"]))
        annotation_path = _annotation_path(annotations_dir, split, image_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path.is_file() else {}
        accepted, truth_bbox = _detector_truth(annotation)
        row["annotation_path"] = str(annotation_path) if annotation_path.is_file() else None
        row["bbox_truth_accepted"] = accepted
        row["truth_bbox"] = truth_bbox
        row.pop("bbox_iou", None)
        if annotation_path.is_file():
            resolved += 1
            canonical_split_fallbacks += int(annotation_path.parent.parent.name != split)
        else:
            missing += 1
        if accepted and truth_bbox is not None and row.get("predicted_bbox") is not None:
            row["bbox_iou"] = round(_bbox_iou(truth_bbox, row["predicted_bbox"]), 6)

    result["detector_on_accepted_subset"] = detector_metrics(rows)
    result["detector_truth_resolution"] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "frozen_prediction_annotation_backfill",
        "model_inference_rerun": False,
        "resolved_annotations": resolved,
        "missing_annotations": missing,
        "canonical_split_fallbacks": canonical_split_fallbacks,
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_path"] = str(metrics_path.resolve())
    return result


def _make_cell(
    frame: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    yoyo: dict[str, Any] | None,
    row: dict[str, Any],
) -> np.ndarray:
    overlay = frame.copy()
    target_present = target > 0
    prediction_present = prediction > 0
    overlay[target_present] = (40, 210, 40)
    overlay[np.logical_and(prediction_present, np.logical_not(target_present))] = (210, 50, 210)
    overlay[np.logical_and(prediction_present, target_present)] = (40, 220, 240)
    if yoyo is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in yoyo["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 30), 4)
    display = cv2.resize(overlay, (480, 270), interpolation=cv2.INTER_AREA)
    cell = np.full((338, 480, 3), 255, dtype=np.uint8)
    cell[:270] = display
    name = f"{row['source_group']}/{row['frame']}"
    state = f"string truth={int(row['string_truth_present'])} pred={int(row['string_prediction_present'])}"
    detector = f"yoyo={row['detector_confidence']:.3f}" if yoyo is not None else "yoyo=none"
    cv2.putText(cell, name, (6, 289), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cell, f"{state} | {detector}", (6, 309), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(cell, str(row.get("string_method") or "string=none"), (6, 329), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 40, 80), 1, cv2.LINE_AA)
    return cell


def _write_sheet(cells: list[np.ndarray], output: Path) -> None:
    columns = 2
    rows = max(1, (len(cells) + columns - 1) // columns)
    canvas = np.full((rows * 338, columns * 480, 3), 235, dtype=np.uint8)
    for index, cell in enumerate(cells):
        x, y = (index % columns) * 480, (index // columns) * 338
        canvas[y : y + 338, x : x + 480] = cell
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not encode pipeline review sheet: {output}")
    encoded.tofile(str(output))


def evaluate(
    detector_weights: str | Path,
    string_weights: str | Path,
    dataset_dir: str | Path,
    annotations_dir: str | Path,
    split: str,
    output_dir: str | Path,
    device: str = "",
    confidence: float = 0.25,
    imgsz: int = 640,
    string_confidence: float = 0.20,
    string_inference_scale: float = 1.0,
    attachment_class: str = "hand_and_yoyo_attached",
) -> dict[str, Any]:
    detector_weights = Path(detector_weights)
    string_weights = Path(string_weights)
    dataset_dir = Path(dataset_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not detector_weights.is_file() or not string_weights.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Detector weights, string weights, and dataset manifest must exist")

    from ultralytics import YOLO

    detector = YOLO(str(detector_weights))
    class_names = {int(key): str(value) for key, value in dict(detector.names or {}).items()}
    string_model, string_status = _load_string_model(string_weights, True, device)
    if string_model is None:
        raise RuntimeError(f"Could not load learned string model: {string_status}")
    checkpoint = string_model["checkpoint"] if isinstance(string_model, dict) else {}
    model_config = checkpoint.get("model_config") or {}
    target_width = int(model_config.get("input_width", 960))
    target_height = int(model_config.get("input_height", 544))
    min_mask_width = max(1, int(model_config.get("min_mask_width_px", 2)))
    samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cells: list[np.ndarray] = []

    for image_path, label_path in image_label_pairs(dataset_dir, split):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read reviewed frame: {image_path}")
        height, width = frame.shape[:2]
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": float(confidence),
            "imgsz": int(imgsz),
            "verbose": False,
        }
        if str(device).strip():
            kwargs["device"] = device
        detector_result = detector.predict(**kwargs)[0]
        detections = _extract_detections(detector_result, class_names)
        yoyo, detector_flags = _pick_yoyo(detections)
        model_observation = _predict_string_model(
            string_model,
            frame,
            yoyo,
            string_confidence,
            imgsz,
            device,
            attachment_class,
            string_inference_scale,
        )
        string = estimate_string(
            frame,
            yoyo,
            [],
            None,
            None,
            attachment_class=attachment_class,
            observation=model_observation,
            allow_color_fallback=False,
        )
        target_raw = render_yolo_segmentation(label_path, width, height)
        prediction_raw = _observation_mask(string, width, height)
        _, target, _ = letterbox(frame, target_width, target_height, target_raw)
        _, prediction, _ = letterbox(frame, target_width, target_height, prediction_raw)
        assert target is not None and prediction is not None
        if min_mask_width > 1 and np.any(target):
            target = cv2.dilate(target, np.ones((min_mask_width, min_mask_width), dtype=np.uint8), iterations=1)
        samples.append(
            {
                "probability": prediction.astype(np.float32),
                "target": (target > 0).astype(np.uint8),
                "image_path": str(image_path),
            }
        )

        annotation_path = _annotation_path(annotations_dir, split, image_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path.is_file() else {}
        bbox_truth_accepted, truth_bbox = _detector_truth(annotation)
        row = {
            "image_path": str(image_path),
            "annotation_path": str(annotation_path) if annotation_path.is_file() else None,
            "source_group": image_path.parent.name,
            "frame": image_path.stem,
            "string_truth_present": bool(np.any(target)),
            "string_prediction_present": bool(np.any(prediction)),
            "string_method": string.get("method") if string else None,
            "string_confidence": float(string.get("confidence", 0.0)) if string else 0.0,
            "string_distance_to_yoyo_px": string.get("distance_to_yoyo_px") if string else None,
            "string_yoyo_body_overlap_fraction": string.get("yoyo_body_overlap_fraction") if string else None,
            "string_inference_scale": string.get("inference_scale") if string else None,
            "string_inference_size": string.get("inference_size") if string else None,
            "detector_confidence": float(yoyo.get("confidence", 0.0)) if yoyo else 0.0,
            "detector_flags": detector_flags,
            "bbox_truth_accepted": bbox_truth_accepted,
            "truth_bbox": truth_bbox,
            "predicted_bbox": yoyo.get("bbox") if yoyo else None,
        }
        if row["bbox_truth_accepted"] and truth_bbox is not None and yoyo is not None:
            row["bbox_iou"] = round(_bbox_iou(truth_bbox, yoyo["bbox"]), 6)
        rows.append(row)
        cells.append(_make_cell(frame, target_raw, prediction_raw, yoyo, row))

    string_metrics = metrics_at_threshold(samples, 0.5, tolerance_px=3, min_component_pixels=1)
    string_metrics["threshold"] = float(checkpoint.get("threshold", 0.5))
    string_metrics["images"] = [
        {**metric_row, **pipeline_row}
        for metric_row, pipeline_row in zip(string_metrics["images"], rows)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / f"{split}_tracking_pipeline_predictions.jpg"
    _write_sheet(cells, sheet_path)
    result = {
        "schema_version": "yoyo_tracking_pipeline_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "detector_weights": str(detector_weights.resolve()),
        "detector_weights_sha256": sha256_file(detector_weights),
        "string_weights": str(string_weights.resolve()),
        "string_weights_sha256": sha256_file(string_weights),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_dataset_manifest_sha256": str(checkpoint.get("dataset_manifest_sha256", "")),
        "parameters": {
            "detector_confidence": float(confidence),
            "imgsz": int(imgsz),
            "string_confidence_floor": float(string_confidence),
            "string_inference_scale": float(string_inference_scale),
            "string_threshold": float(checkpoint.get("threshold", 0.5)),
            "attachment_class": attachment_class,
            "color_fallback": False,
            "temporal_tracking": False,
        },
        "string": string_metrics,
        "detector_on_accepted_subset": detector_metrics(rows),
        "prediction_sheet": str(sheet_path.resolve()),
        "limitations": [
            "This frame benchmark validates deployed per-frame fusion; it does not measure optical-flow propagation.",
            "The frozen val/test splits are small, so every failed frame must also be visually reviewed.",
        ],
    }
    metrics_path = output_dir / f"{split}_tracking_pipeline_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_path"] = str(metrics_path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-weights", default=str(TRACKING_CONFIG.weights_path))
    parser.add_argument("--string-weights", default=str(TRACKING_CONFIG.string_weights_path))
    parser.add_argument("--dataset-dir", default="datasets/video_v1/string_seg_v17_reviewed_expansion")
    parser.add_argument("--annotations-dir", default="datasets/video_v1/annotations")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", default="runs/pipeline_eval/yoyo_v8_semantic_v17_reviewed_expansion")
    parser.add_argument("--device", default=TRACKING_CONFIG.device)
    parser.add_argument("--confidence", type=float, default=TRACKING_CONFIG.confidence)
    parser.add_argument("--imgsz", type=int, default=TRACKING_CONFIG.imgsz)
    parser.add_argument("--string-confidence", type=float, default=TRACKING_CONFIG.string_confidence)
    parser.add_argument("--string-inference-scale", type=float, default=TRACKING_CONFIG.string_inference_scale)
    parser.add_argument(
        "--backfill-detector-truth",
        default="",
        help="Recompute detector truth metrics in an existing artifact from frozen prediction rows; models are not rerun.",
    )
    parser.add_argument(
        "--attachment-class",
        choices=["hand_and_yoyo_attached", "yoyo_detached", "hand_detached", "unknown"],
        default=TRACKING_CONFIG.string_attachment_class,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.backfill_detector_truth).strip():
        result = backfill_detector_truth(args.backfill_detector_truth, args.annotations_dir)
    else:
        result = evaluate(
            detector_weights=args.detector_weights,
            string_weights=args.string_weights,
            dataset_dir=args.dataset_dir,
            annotations_dir=args.annotations_dir,
            split=args.split,
            output_dir=args.output_dir,
            device=args.device,
            confidence=args.confidence,
            imgsz=args.imgsz,
            string_confidence=args.string_confidence,
            string_inference_scale=args.string_inference_scale,
            attachment_class=args.attachment_class,
        )
    print(
        json.dumps(
            {
                "split": result["split"],
                "string": result["string"]["image_presence"],
                "detector": result["detector_on_accepted_subset"],
                "metrics_path": result["metrics_path"],
                "prediction_sheet": result["prediction_sheet"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
