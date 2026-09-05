"""Model-assisted draft annotations for the Workbench editor.

This module deliberately owns the batch workflow.  It writes canonical v5
labels as review-only drafts and leaves the video tracking pipeline unchanged.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from common.files import atomic_write_text
from config import DETECTION_CONFIG, ORIENTATION_CONFIG, STRING_TRACKING_CONFIG, TRACKING_CONFIG
from video_tracking.orientation import load_orientation_model, predict_orientation
from yoyo_detection.inference import load_detector
from string_tracking.inference import load_runtime_string_model, predict_runtime_string_model
from workbench import dataset_annotation as base

# Public path-specific APIs; aliases keep the draft builder's injectable seam
# stable for Workbench integrations.
_load_string_model = load_runtime_string_model
_predict_string_model = predict_runtime_string_model


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backup_path(dataset: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return dataset.parent / f"{dataset.name}__preannotated_backup_{stamp}_{uuid.uuid4().hex[:8]}"


def _yoyo_record(
    bbox: list[float] | None,
    visibility: str,
    *,
    orientation: str = "normal",
    presentation: str = "frontal",
) -> dict[str, Any]:
    return {
        "visibility": visibility,
        "not_visible_reason": None,
        "trick_orientation": orientation,
        "presentation_orientation": presentation,
        "bbox_pixel": bbox,
        "bbox_2d": None,
        "bbox_review_status": "needs_review",
    }


def _orientation_fields(prediction: dict[str, Any] | None) -> tuple[str, str]:
    if not prediction:
        return "normal", "frontal"
    trick = str(prediction.get("label") or "normal")
    presentation = str(prediction.get("presentation_label") or "")
    if trick == "horizontal":
        return trick, "edge_horizontal"
    if trick == "not_applicable":
        return trick, "unknown"
    return trick, presentation if presentation in {"frontal", "edge_vertical"} else "frontal"


def _image_path(label_path: Path, labels_root: Path, images_root: Path, document: dict[str, Any]) -> Path:
    return base._resolve_source_image(label_path, labels_root, images_root, document)


def _draft_document(
    document: dict[str, Any],
    image: Any,
    detections: list[dict[str, Any]],
    string_model: Any,
    orientation_model: Any,
    device: str,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    yoyo_detections = [
        item for item in detections
        if str(item.get("class_name") or "").lower() in {"yoyo", "yo-yo", "yoyo_body"}
    ]
    yoyo_detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    active_detection = yoyo_detections[0] if yoyo_detections else None
    orientation_by_detection: dict[int, tuple[str, str]] = {}
    if orientation_model is not None:
        for index, detection in enumerate(yoyo_detections):
            prediction = predict_orientation(
                orientation_model,
                image,
                detection,
                ORIENTATION_CONFIG.imgsz,
                device,
                direct_inference=True,
            )
            orientation_by_detection[index] = _orientation_fields(prediction)
    trick, presentation = orientation_by_detection.get(
        0,
        ("not_applicable", "unknown") if active_detection is None else ("normal", "frontal"),
    )
    active = _yoyo_record(
        [float(value) for value in active_detection["bbox"]] if active_detection else None,
        "visible" if active_detection else "uncertain",
        orientation=trick,
        presentation=presentation,
    )
    backups = []
    for index, item in enumerate(yoyo_detections[1:], start=1):
        backup_trick, backup_presentation = orientation_by_detection.get(
            index, ("horizontal", "edge_horizontal")
        )
        backups.append(
            _yoyo_record(
                [float(value) for value in item["bbox"]],
                "visible",
                orientation=backup_trick,
                presentation=backup_presentation,
            )
        )
    string = None
    if string_model is not None:
        string = _predict_string_model(
            string_model,
            image,
            active_detection,
            STRING_TRACKING_CONFIG.confidence,
            STRING_TRACKING_CONFIG.imgsz,
            device,
            TRACKING_CONFIG.yoyo_division,
            TRACKING_CONFIG.string_inference_scale,
            [],
            TRACKING_CONFIG.string_color_probability_augment,
            TRACKING_CONFIG.string_color_probability_min_mean,
            TRACKING_CONFIG.string_color_probability_min_fraction,
            TRACKING_CONFIG.string_color_semantic_prefilter,
            TRACKING_CONFIG.string_bright_line_augment,
            TRACKING_CONFIG.string_bright_line_min_mean,
        )
    polylines = (string or {}).get("polylines") or ([] if string is None else ([string["points"]] if string.get("points") else []))
    result = dict(document)
    result.pop("preannotation", None)
    result["active_yoyo"] = active
    result["backup_yoyos"] = backups
    result["string_visibility"] = "partial" if polylines else "uncertain"
    result["string_polylines_pixel"] = polylines
    result["string_review_status"] = "needs_review"
    result["review_status"] = "needs_review"
    result["reviewed_at_utc"] = None
    result["reviewer"] = None
    result["updated_at_utc"] = _now()
    # Reuse the canonical editor's normalizer to derive all coordinate fields.
    normalized_active = base._normalize_yoyo_record(active, width, height)
    normalized_backups = [base._normalize_yoyo_record(item, width, height, default_orientation="horizontal") for item in backups]
    result["active_yoyo"] = normalized_active
    result["backup_yoyos"] = normalized_backups
    result["string_polylines_pixel"] = [[base._point(point, width, height) for point in line] for line in polylines]
    result["string_polylines_pixel"] = [line for line in result["string_polylines_pixel"] if len(line) >= 2]
    result["string_polylines_2d"] = [base._to_2d(line, width, height) for line in result["string_polylines_pixel"]]
    result["string_polyline_pixel"] = result["string_polylines_pixel"][0] if result["string_polylines_pixel"] else None
    result["string_polyline_2d"] = result["string_polylines_2d"][0] if result["string_polylines_2d"] else None
    result["string_mask_polygons_pixel"] = None
    string_path = result.get("string_path")
    if isinstance(string_path, dict):
        string_path = dict(string_path)
        string_path["paths"] = [
            {
                "path_id": f"model-line-{index + 1}",
                "start_anchor": "unknown",
                "end_anchor": "unknown",
                "points_pixel": line,
                "points_2d": result["string_polylines_2d"][index],
                "edges": [
                    {"from": point_index, "to": point_index + 1, "evidence": "inferred", "confidence": 0.0}
                    for point_index in range(len(line) - 1)
                ],
            }
            for index, line in enumerate(result["string_polylines_pixel"])
        ]
        if not result["string_polylines_pixel"]:
            string_path["topology"] = "uncertain"
            string_path["reconstruction_status"] = "not_applicable"
        result["string_path"] = string_path
    quality = result.get("quality")
    if isinstance(quality, dict):
        quality = dict(quality)
        quality["reviews"] = []
        result["quality"] = quality
    result.setdefault("workbench_edits", []).append(
        {
            "created_at_utc": result["updated_at_utc"],
            "actor": "workbench-preannotator",
            "before_sha256": base._content_digest(document),
            "fields": [
                "active_yoyo",
                "backup_yoyos",
                "string_visibility",
                "string_polylines_pixel",
                "string_review_status",
            ],
        }
    )
    return result


def preannotate_dataset(dataset_path: str | Path, device: str | None = None) -> dict[str, Any]:
    """Back up and overwrite every canonical label with a review-only draft."""
    dataset = base._managed_dataset_path(dataset_path)
    annotation_root, labels_root, images_root = base._annotation_roots(dataset)
    labels = sorted(labels_root.rglob("*.json"))
    if not labels:
        raise ValueError(f"no JSON labels found in {labels_root}")
    backup = _backup_path(dataset)
    shutil.copytree(dataset, backup)
    runtime_device = str(device if device is not None else TRACKING_CONFIG.device)
    detector = load_detector(DETECTION_CONFIG.weights_path, runtime_device)
    class_names = detector.class_names
    string_model, string_status = _load_string_model(STRING_TRACKING_CONFIG.weights_path, True, runtime_device)
    orientation_model, orientation_status = load_orientation_model(ORIENTATION_CONFIG.weights_path, True)
    processed = 0
    failures: list[dict[str, str]] = []
    for label_path in labels:
        try:
            document = base._read_document(label_path)
            image_path = _image_path(label_path, labels_root, images_root, document)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not read image: {image_path}")
            detections = detector.predict(
                image,
                confidence=DETECTION_CONFIG.confidence,
                iou=DETECTION_CONFIG.iou,
                imgsz=DETECTION_CONFIG.imgsz,
            )
            draft = _draft_document(document, image, detections, string_model, orientation_model, runtime_device)
            draft["image_size"] = [int(image.shape[1]), int(image.shape[0])]
            atomic_write_text(label_path, json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
            processed += 1
        except Exception as exc:
            failures.append({"label": str(label_path), "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        # Keep the backup usable and make partial completion explicit.
        status = "partial"
    else:
        status = "completed"
    return {
        "status": status,
        "dataset_path": str(dataset),
        "backup_path": str(backup),
        "label_count": len(labels),
        "processed_count": processed,
        "failure_count": len(failures),
        "failures": failures[:20],
        "models": {
            "yoyo": str(DETECTION_CONFIG.weights_path),
            "string": string_status,
            "orientation": orientation_status,
        },
    }


def ui_preannotate_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("preannotation request must be an object")
    return preannotate_dataset(str(payload.get("dataset_path") or ""), payload.get("device"))
