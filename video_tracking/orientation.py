"""ROI-based coarse trick-orientation inference for tracked video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ORIENTATION_CLASSES = {"horizontal", "normal", "not_applicable"}


def orientation_crop_box(
    width: int,
    height: int,
    yoyo: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    """Match the yoyo-only square crop used by the orientation training view."""
    bbox = (yoyo or {}).get("bbox")
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and float(bbox[2]) > float(bbox[0])
        and float(bbox[3]) > float(bbox[1])
    ):
        x1, y1, x2, y2 = (float(value) for value in bbox)
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        span = max(x2 - x1, y2 - y1)
        side = min(float(min(width, height)), max(span * 3.0, min(width, height) * 0.12))
    else:
        side = float(min(width, height)) * 0.28
        center_x, center_y = width / 2.0, height / 2.0
    left = max(0.0, min(center_x - side / 2.0, width - side))
    top = max(0.0, min(center_y - side / 2.0, height - side))
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def load_orientation_model(
    weights_path: str | Path | None,
    enabled: bool,
) -> tuple[Any | None, str]:
    if not enabled:
        return None, "disabled"
    path = Path(weights_path or "")
    if not path.is_file():
        return None, f"missing: {path}"
    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        names = {int(key): str(value) for key, value in dict(getattr(model, "names", {}) or {}).items()}
        if set(names.values()) != ORIENTATION_CLASSES:
            return None, f"incompatible_classes: {names}"
        return model, str(path)
    except Exception as exc:
        return None, f"error: {exc}"


def predict_orientation(
    model: Any,
    frame: np.ndarray,
    yoyo: dict[str, Any] | None,
    imgsz: int,
    device: str,
) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    left, top, right, bottom = orientation_crop_box(width, height, yoyo)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    kwargs: dict[str, Any] = {"source": crop, "imgsz": int(imgsz), "verbose": False}
    if str(device).strip():
        kwargs["device"] = str(device).strip()
    result = model.predict(**kwargs)[0]
    probs = getattr(result, "probs", None)
    if probs is None or getattr(probs, "data", None) is None:
        return None
    values = [float(value) for value in probs.data.detach().cpu().tolist()]
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    top1 = int(probs.top1)
    return {
        "label": names[top1],
        "confidence": round(float(probs.top1conf.detach().cpu().item()), 6),
        "probabilities": {names[index]: round(value, 6) for index, value in enumerate(values)},
        "crop_box_pixel": [left, top, right, bottom],
        "crop_policy": "yoyo_bbox_square_3p0_min_12pct; no_yoyo_center_square_28pct",
        "inference_status": "ran",
        "age_frames": 0,
    }


def carry_orientation(previous: dict[str, Any] | None, age_frames: int) -> dict[str, Any] | None:
    if previous is None:
        return None
    carried = dict(previous)
    carried["inference_status"] = "carried"
    carried["age_frames"] = int(age_frames)
    return carried
