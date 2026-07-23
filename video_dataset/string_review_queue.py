"""Rank pending 1A string labels for efficient visual review.

The queue is deliberately separate from annotation JSON. It is an active-learning
hint, never training truth: reviewers still approve or reject each component in
the workbench.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import TRACKING_CONFIG
from video_dataset.split_policy import parse_source_groups

REVIEWED = {"approved", "reviewed", "rejected", "unresolved"}
PRELABEL_FAILURES = {"no_mask", "too_many_components", "mask_area_too_large"}
BAD_CASE_WEIGHTS = {
    "motion_blur": 2.5,
    "string_ambiguous": 2.5,
    "yoyo_not_visible": 2.0,
    "hands_occluded": 1.5,
    "yoyo_edge_clipped": 1.25,
    "multiple_yoyo": 1.5,
    "non_trick_scene": 1.0,
}
AGREEMENT_EXCLUDED_BAD_CASES = {"motion_blur", "string_ambiguous", "multiple_yoyo", "hands_occluded"}


def load_prediction_polylines(dataset_dir: str | Path, label_path: str | Path) -> list[list[list[float]]]:
    """Load review-only semantic geometry for one label without editing it."""
    queue_path = Path(dataset_dir) / "string_review_queue.json"
    if not queue_path.exists():
        return []
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    resolved = str(Path(label_path).resolve())
    row = next(
        (
            item
            for item in payload.get("rows", [])
            if str(Path(str(item.get("label_path", ""))).resolve()) == resolved
        ),
        None,
    )
    value = ((row or {}).get("model") or {}).get("prediction_polylines") or []
    strokes = []
    for stroke in value:
        if not isinstance(stroke, list):
            continue
        points = [
            [float(point[0]), float(point[1])]
            for point in stroke
            if isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(number, (int, float)) for number in point)
        ]
        if len(points) >= 2:
            strokes.append(points)
    return strokes


def _polylines(data: dict[str, Any]) -> list[list[list[float]]]:
    value = data.get("string_polylines_pixel")
    if not value and data.get("string_polyline_pixel"):
        value = [data["string_polyline_pixel"]]
    return [stroke for stroke in (value or []) if isinstance(stroke, list) and len(stroke) >= 2]


def _metadata_score(data: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    qa = data.get("qa") or {}
    if qa.get("priority") == "high":
        score += 4.0
        reasons.append("qa_high_priority")
    warnings = qa.get("warnings") or []
    if warnings:
        score += min(3.0, 0.75 * len(warnings))
        reasons.append(f"qa_warnings:{len(warnings)}")
    prelabel = data.get("string_prelabel") or {}
    proposal_status = str(prelabel.get("status", "missing"))
    if proposal_status in PRELABEL_FAILURES:
        score += 4.0
        reasons.append(f"color_proposal_{proposal_status}")
    elif proposal_status == "missing":
        score += 2.0
        reasons.append("no_color_proposal")
    elif proposal_status == "auto_color_mask_needs_review":
        score += 0.75
    visibility = str(data.get("string_visibility", "uncertain"))
    if visibility == "uncertain":
        score += 3.0
        reasons.append("string_visibility_uncertain")
    elif visibility == "partial":
        score += 2.0
        reasons.append("partial_string")
    elif visibility == "visible":
        score += 0.5
    if visibility in {"visible", "partial"} and not _polylines(data):
        score += 4.0
        reasons.append("visible_without_polyline")
    for bad_case in data.get("bad_case") or []:
        weight = BAD_CASE_WEIGHTS.get(str(bad_case), 0.0)
        if weight:
            score += weight
            reasons.append(f"bad_case:{bad_case}")
    if data.get("visibility") in {"occluded", "uncertain"}:
        score += 1.0
        reasons.append("yoyo_visibility_uncertain")
    return round(score, 4), reasons


def _agreement_candidate(data: dict[str, Any]) -> bool:
    if str(data.get("string_visibility", "uncertain")) not in {"visible", "partial"}:
        return False
    if not _polylines(data) and not (data.get("string_mask_polygons_pixel") or []):
        return False
    if (data.get("qa") or {}).get("warnings"):
        return False
    if set(str(value) for value in (data.get("bad_case") or [])) & AGREEMENT_EXCLUDED_BAD_CASES:
        return False
    if str((data.get("string_prelabel") or {}).get("status", "")) in PRELABEL_FAILURES:
        return False
    return True


def _annotation_agreement(binary, data: dict[str, Any], meta) -> dict[str, Any] | None:
    """Compare model pixels with draft geometry as a review hint, never truth."""
    import cv2
    import numpy as np

    target = np.zeros(binary.shape, dtype=np.uint8)
    scale_x = float(meta.resized_width) / max(1.0, float(meta.original_width))
    scale_y = float(meta.resized_height) / max(1.0, float(meta.original_height))
    strokes = _polylines(data)
    if strokes:
        width = max(1, int(round(8.0 * min(scale_x, scale_y))))
        for stroke in strokes:
            points = np.asarray(
                [[float(point[0]) * scale_x, float(point[1]) * scale_y] for point in stroke],
                dtype=np.float32,
            ).round().astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(target, [points], False, 1, width, cv2.LINE_AA)
    else:
        for polygon in data.get("string_mask_polygons_pixel") or []:
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            points = np.asarray(
                [[float(point[0]) * scale_x, float(point[1]) * scale_y] for point in polygon],
                dtype=np.float32,
            ).round().astype(np.int32)
            cv2.fillPoly(target, [points], 1)
    target = target > 0
    prediction = np.asarray(binary, dtype=bool)
    target_pixels = int(target.sum())
    prediction_pixels = int(prediction.sum())
    if not target_pixels:
        return None
    intersection = int(np.logical_and(target, prediction).sum())
    exact_dice = (2.0 * intersection / (target_pixels + prediction_pixels)) if prediction_pixels else 0.0
    tolerance_original_px = 12
    tolerance_input_px = max(1, int(round(tolerance_original_px * min(scale_x, scale_y))))
    kernel = np.ones((tolerance_input_px * 2 + 1, tolerance_input_px * 2 + 1), dtype=np.uint8)
    target_dilated = cv2.dilate(target.astype(np.uint8), kernel, iterations=1) > 0
    prediction_dilated = cv2.dilate(prediction.astype(np.uint8), kernel, iterations=1) > 0
    precision = float(np.logical_and(prediction, target_dilated).sum() / prediction_pixels) if prediction_pixels else 0.0
    recall = float(np.logical_and(target, prediction_dilated).sum() / target_pixels)
    tolerant_f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "status": "review_hint_only",
        "exact_dice": round(exact_dice, 6),
        "tolerant_precision": round(precision, 6),
        "tolerant_recall": round(recall, 6),
        "tolerant_f1": round(tolerant_f1, 6),
        "target_pixels": target_pixels,
        "prediction_pixels": prediction_pixels,
        "tolerance_original_px": tolerance_original_px,
        "tolerance_input_px": tolerance_input_px,
    }


def _model_signal(
    data: dict[str, Any],
    weights: str | Path,
    device: str,
    cache: dict[str, Any] | None = None,
    preview_path: Path | None = None,
) -> tuple[float, list[str], dict[str, Any]]:
    """Run optional semantic inference and return review-only uncertainty."""
    import cv2
    import numpy as np
    from string_segmentation.semantic_model import (
        load_checkpoint,
        predict_letterboxed,
        semantic_mask_observation,
    )

    source = Path(str(data.get("source_image", "")))
    image = cv2.imread(str(source))
    if image is None:
        return 3.0, ["model_source_missing"], {"status": "source_missing"}
    model_device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    cache_key = f"{Path(weights).resolve()}::{model_device}"
    if cache is not None and cache_key in cache:
        model, checkpoint = cache[cache_key]
    else:
        model, checkpoint = load_checkpoint(weights, model_device)
        if cache is not None:
            cache[cache_key] = (model, checkpoint)
    config = checkpoint["model_config"]
    probability, meta = predict_letterboxed(
        model, image, int(config["input_width"]), int(config["input_height"]), model_device
    )
    threshold = float(checkpoint.get("threshold", 0.5))
    content_probability = probability[
        meta.pad_y : meta.pad_y + meta.resized_height,
        meta.pad_x : meta.pad_x + meta.resized_width,
    ]
    binary = content_probability >= threshold
    agreement = _annotation_agreement(binary, data, meta)
    predicted_pixels = int(binary.sum())
    predicted_fraction = float(predicted_pixels / binary.size) if binary.size else 0.0
    mean_probability = float(content_probability.mean()) if content_probability.size else 0.0
    max_probability = float(content_probability.max()) if content_probability.size else 0.0
    score = 0.0
    reasons: list[str] = []
    visibility = str(data.get("string_visibility", "uncertain"))
    if visibility in {"visible", "partial"} and not predicted_pixels:
        score += 4.0
        reasons.append("model_misses_visible_string")
    if visibility == "not_visible" and predicted_pixels:
        score += 4.0
        reasons.append("model_predicts_on_negative")
    ambiguity = max(0.0, 1.0 - abs(mean_probability - 0.5) * 2.0)
    if ambiguity > 0.2:
        score += 2.0 * ambiguity
        reasons.append("model_probability_ambiguous")
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype("uint8"), 8)
    components = max(0, int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= 8 for i in range(1, component_count))))
    if components > 1:
        score += min(2.0, 0.5 * components)
        reasons.append(f"model_components:{components}")
    bbox = data.get("yoyo_bbox_pixel")
    yoyo = None
    if isinstance(bbox, list) and len(bbox) == 4:
        yoyo = {
            "bbox": [float(value) for value in bbox],
            "center": [
                (float(bbox[0]) + float(bbox[2])) * 0.5,
                (float(bbox[1]) + float(bbox[3])) * 0.5,
            ],
        }
    action_group = str(data.get("action_group", "1A"))
    observation = semantic_mask_observation(
        probability,
        meta,
        threshold,
        yoyo=yoyo,
        attachment_class="hand_and_yoyo_attached" if action_group == "1A" else "unknown",
        min_component_pixels=8,
    )
    if preview_path is not None:
        _save_model_preview(image, probability, meta, threshold, preview_path)
    return round(score, 4), reasons, {
        "status": "ok",
        "threshold": round(threshold, 4),
        "predicted_pixels": predicted_pixels,
        "predicted_fraction": round(predicted_fraction, 8),
        "mean_probability": round(mean_probability, 6),
        "max_probability": round(max_probability, 6),
        "components": components,
        "prediction_preview": str(preview_path.resolve()) if preview_path is not None else None,
        "prediction_polylines": (observation or {}).get("polylines", []),
        "prediction_confidence": (observation or {}).get("confidence"),
        "prediction_anchored_to_yoyo": (observation or {}).get("anchored_to_yoyo"),
        "prediction_spatially_ambiguous": (observation or {}).get("spatially_ambiguous"),
        "editor_import_filter": "1A yoyo-spatial anchor when yoyo bbox is available",
        "annotation_agreement": agreement,
    }


def _save_model_preview(
    image_bgr,
    probability,
    meta,
    threshold: float,
    output: Path,
) -> Path:
    """Save a thresholded semantic overlay without changing annotation truth."""
    import cv2
    import numpy as np

    content = probability[
        meta.pad_y : meta.pad_y + meta.resized_height,
        meta.pad_x : meta.pad_x + meta.resized_width,
    ]
    restored = cv2.resize(
        content,
        (meta.original_width, meta.original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = restored >= float(threshold)
    overlay = image_bgr.copy()
    if np.any(mask):
        magenta = np.zeros_like(overlay)
        magenta[:, :] = (230, 35, 230)
        overlay[mask] = cv2.addWeighted(overlay, 0.30, magenta, 0.70, 0)[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), max(2, image_bgr.shape[1] // 1500))
    text = f"SEMANTIC RAW MASK / REVIEW ONLY  threshold={float(threshold):.4f}  pixels={int(mask.sum())}"
    cv2.putText(
        overlay,
        text,
        (20, max(36, image_bgr.shape[0] // 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, image_bgr.shape[1] / 3200),
        (255, 255, 255),
        max(2, image_bgr.shape[1] // 1600),
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise OSError(f"Could not write semantic prediction preview: {output}")
    return output


def _diverse_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("source_group") or row.get("video_id") or "unknown")].append(row)
    for values in buckets.values():
        values.sort(key=lambda item: (-float(item["priority_score"]), int(item.get("frame_index", 0))))
    ordered: list[dict[str, Any]] = []
    while buckets:
        heads = sorted(
            ((values[0]["priority_score"], group) for group, values in buckets.items() if values),
            key=lambda item: (-float(item[0]), item[1]),
        )
        for _, group in heads:
            values = buckets.get(group)
            if not values:
                continue
            ordered.append(values.pop(0))
            if not values:
                buckets.pop(group, None)
    for index, row in enumerate(ordered, 1):
        row["queue_rank"] = index
        row["batch_index"] = (index - 1) // 16 + 1
    return ordered


def _queue_sheet(root: Path, rows: list[dict[str, Any]], columns: int = 4, thumb_width: int = 480) -> Path | None:
    if not rows:
        return None
    import cv2
    import numpy as np

    thumb_height = round(thumb_width * 9 / 16)
    cell_height = thumb_height + 78
    visible_rows = rows[:64]
    canvas = np.full(
        (math.ceil(len(visible_rows) / columns) * cell_height, columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    labels_root = root / "annotations" / "labels"
    for index, row in enumerate(visible_rows):
        label_path = Path(row["label_path"])
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        relative = label_path.resolve().relative_to(labels_root.resolve())
        visualization = root / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")
        source_text = str(annotation.get("source_image", "")).strip()
        source = visualization if visualization.exists() else Path(source_text) if source_text else None
        image = cv2.imread(str(source)) if source is not None and source.is_file() else None
        if image is None:
            continue
        image = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = image
        title = f"#{row['queue_rank']} score={float(row['priority_score']):.2f} {row.get('video_id','?')[:8]} f{row.get('frame_index','?')}"
        reasons = ", ".join(row.get("reasons", [])[:3]) or "manual visual review"
        cv2.putText(canvas, title, (x + 6, y + thumb_height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, reasons[:76], (x + 6, y + thumb_height + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 80, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(row.get("string_visibility", "uncertain")), (x + 6, y + thumb_height + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 110, 30), 1, cv2.LINE_AA)
    output = root / "review_sheets" / "string_review_queue.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output


def build_queue(
    dataset_dir: str | Path,
    split: str = "all",
    limit: int = 0,
    with_model: bool = False,
    weights: str | Path | None = None,
    device: str = "",
    exclude_source_groups: set[str] | str | None = None,
    strategy: str = "uncertainty",
) -> dict[str, Any]:
    root = Path(dataset_dir)
    if strategy not in {"uncertainty", "agreement"}:
        raise ValueError(f"Unsupported review queue strategy: {strategy}")
    if strategy == "agreement" and not with_model:
        raise ValueError("strategy=agreement requires with_model=True")
    excluded_groups = (
        parse_source_groups(exclude_source_groups)
        if isinstance(exclude_source_groups, str)
        else {str(value).strip() for value in (exclude_source_groups or set()) if str(value).strip()}
    )
    labels_root = root / "annotations" / "labels"
    paths = sorted(labels_root.rglob("*.json")) if labels_root.exists() else []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if split != "all" and data.get("split") != split:
            continue
        source_group = str(data.get("source_group") or data.get("video_id") or "").strip()
        if source_group in excluded_groups:
            continue
        if str(data.get("string_review_status", "auto_labeled_needs_review")) in REVIEWED:
            continue
        if strategy == "agreement" and not _agreement_candidate(data):
            continue
        score, reasons = _metadata_score(data)
        rows.append({
            "label_path": str(path.resolve()),
            "video_id": data.get("video_id"),
            "source_group": data.get("source_group"),
            "action_group": data.get("action_group", "1A"),
            "split": data.get("split"),
            "frame_index": int(data.get("frame_index", 0)),
            "timestamp_s": data.get("timestamp_s"),
            "string_visibility": data.get("string_visibility", "uncertain"),
            "priority_score": round(score, 4),
            "reasons": reasons,
            "model": None,
            "_annotation": data,
        })
    rows = _diverse_order(rows) if strategy == "uncertainty" else rows
    if strategy == "uncertainty" and limit > 0:
        rows = rows[: int(limit)]
    if with_model:
        if not weights:
            raise ValueError("weights is required with_model=True")
        model_cache: dict[str, Any] = {}
        for row in rows:
            relative = Path(row["label_path"]).resolve().relative_to(labels_root.resolve())
            preview = root / "review_sheets" / "string_predictions" / relative.with_suffix(".jpg")
            model_score, model_reasons, model_details = _model_signal(
                row["_annotation"], weights, device, model_cache, preview
            )
            if strategy == "agreement":
                agreement = model_details.get("annotation_agreement") or {}
                tolerant_f1 = float(agreement.get("tolerant_f1", 0.0))
                confidence = float(model_details.get("prediction_confidence") or 0.0)
                components = int(model_details.get("components", 0))
                row["priority_score"] = round(10.0 * tolerant_f1 + 2.0 * confidence - 0.15 * max(0, components - 1), 4)
                row["reasons"] = [
                    f"annotation_model_tolerant_f1:{tolerant_f1:.4f}",
                    f"model_confidence:{confidence:.4f}",
                    f"model_components:{components}",
                ]
            else:
                row["priority_score"] = round(float(row["priority_score"]) + model_score, 4)
                row["reasons"].extend(model_reasons)
            row["model"] = model_details
        rows = _diverse_order(rows)
        if strategy == "agreement" and limit > 0:
            rows = rows[: int(limit)]
    for index, row in enumerate(rows, 1):
        row["queue_rank"] = index
        row["batch_index"] = (index - 1) // 16 + 1
        row.pop("_annotation", None)
    sheet = _queue_sheet(root, rows)
    output_json = root / "string_review_queue.json"
    output_csv = root / "string_review_queue.csv"
    payload = {
        "schema_version": "yoyo_string_review_queue_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "action_group": "1A",
        "split": split,
        "strategy": strategy,
        "exclude_source_groups": sorted(excluded_groups),
        "with_model": bool(with_model),
        "weights": str(Path(weights).resolve()) if weights else None,
        "count": len(rows),
        "rows": rows,
        "policy": (
            "annotation/model agreement is a review hint only; every item requires visual review"
            if strategy == "agreement"
            else "metadata/model uncertainty only; every item requires visual review"
        ),
        "review_sheet": str(sheet.resolve()) if sheet else None,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["queue_rank", "batch_index", "label_path", "video_id", "split", "frame_index", "timestamp_s", "priority_score", "reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: ";".join(row["reasons"]) if field == "reasons" else row.get(field)
                for field in fields
            })
    return {
        "queue": str(output_json.resolve()),
        "csv": str(output_csv.resolve()),
        "review_sheet": str(sheet.resolve()) if sheet else None,
        "count": len(rows),
        "with_model": bool(with_model),
        "strategy": strategy,
        "top": rows[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank pending 1A string annotations for visual review.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.string_weights_path))
    parser.add_argument("--device", default="")
    parser.add_argument("--exclude-source-groups", default="", help="Comma-separated source groups excluded before ranking/inference.")
    parser.add_argument("--strategy", choices=["uncertainty", "agreement"], default="uncertainty")
    args = parser.parse_args()
    result = build_queue(
        args.dataset_dir,
        args.split,
        args.limit,
        args.with_model,
        args.weights,
        args.device,
        args.exclude_source_groups,
        args.strategy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
