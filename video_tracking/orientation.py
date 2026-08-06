"""ROI-based coarse trick-orientation inference for tracked video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np


ORIENTATION_CLASSES = {"horizontal", "normal", "not_applicable"}
ORIENTATION_CLASS_ORDER = ("horizontal", "normal", "not_applicable")


@dataclass
class OrientationTemporalFilter:
    """Causal EMA and hysteresis for sparse orientation observations."""

    ema_alpha: float = 0.4
    switch_margin: float = 0.05
    switch_confirmations: int = 3
    strong_switch_confidence: float = 0.9
    strong_switch_margin: float = 0.2
    _ema: dict[str, float] | None = field(default=None, init=False, repr=False)
    _label: str | None = field(default=None, init=False, repr=False)
    _pending_label: str | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= float(self.switch_margin) <= 1.0:
            raise ValueError("switch_margin must be between 0 and 1")
        if int(self.switch_confirmations) < 1:
            raise ValueError("switch_confirmations must be positive")
        if not 0.0 <= float(self.strong_switch_confidence) <= 1.0:
            raise ValueError("strong_switch_confidence must be between 0 and 1")
        if not 0.0 <= float(self.strong_switch_margin) <= 1.0:
            raise ValueError("strong_switch_margin must be between 0 and 1")

    def update(self, prediction: dict[str, Any]) -> dict[str, Any]:
        raw_probabilities = prediction.get("probabilities") or {}
        if set(raw_probabilities) != ORIENTATION_CLASSES:
            raise ValueError(f"orientation probabilities have incompatible classes: {raw_probabilities}")
        probabilities = {name: float(raw_probabilities[name]) for name in ORIENTATION_CLASS_ORDER}
        if not all(math.isfinite(value) and value >= 0.0 for value in probabilities.values()):
            raise ValueError("orientation probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0.0:
            raise ValueError("orientation probabilities must have positive mass")
        probabilities = {name: value / total for name, value in probabilities.items()}
        raw_label = max(ORIENTATION_CLASS_ORDER, key=probabilities.__getitem__)

        status = "stable"
        if self._ema is None:
            self._ema = dict(probabilities)
            self._label = raw_label
            status = "initialized"
        else:
            alpha = float(self.ema_alpha)
            self._ema = {
                name: (1.0 - alpha) * self._ema[name] + alpha * probabilities[name]
                for name in ORIENTATION_CLASS_ORDER
            }
            candidate = max(ORIENTATION_CLASS_ORDER, key=self._ema.__getitem__)
            if candidate == self._label:
                self._pending_label = None
                self._pending_count = 0
            else:
                if candidate == self._pending_label:
                    self._pending_count += 1
                else:
                    self._pending_label = candidate
                    self._pending_count = 1
                current_label = str(self._label)
                confirmed = (
                    self._pending_count >= int(self.switch_confirmations)
                    and self._ema[candidate] - self._ema[current_label] >= float(self.switch_margin)
                )
                strong = (
                    raw_label == candidate
                    and probabilities[candidate] >= float(self.strong_switch_confidence)
                    and probabilities[candidate] - probabilities[current_label]
                    >= float(self.strong_switch_margin)
                )
                if confirmed or strong:
                    self._label = candidate
                    self._pending_label = None
                    self._pending_count = 0
                    status = "strong_switched" if strong else "confirmed_switched"
                else:
                    status = "pending"

        label = str(self._label)
        result = dict(prediction)
        result.update({
            "label": label,
            "confidence": round(float(self._ema[label]), 6),
            "probabilities": {
                name: round(float(self._ema[name]), 6) for name in ORIENTATION_CLASS_ORDER
            },
            "raw_label": raw_label,
            "raw_confidence": round(float(probabilities[raw_label]), 6),
            "raw_probabilities": {
                name: round(float(probabilities[name]), 6) for name in ORIENTATION_CLASS_ORDER
            },
            "temporal_filter": {
                "status": status,
                "pending_label": self._pending_label,
                "pending_count": int(self._pending_count),
                "ema_alpha": float(self.ema_alpha),
                "switch_margin": float(self.switch_margin),
                "switch_confirmations": int(self.switch_confirmations),
                "strong_switch_confidence": float(self.strong_switch_confidence),
                "strong_switch_margin": float(self.strong_switch_margin),
            },
        })
        return result


def orientation_observation_is_unstable(
    prediction: dict[str, Any] | None,
    min_confidence: float,
    inference_error: bool = False,
) -> bool:
    """Return whether the next orientation observation should use burst cadence."""
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if prediction is None or inference_error:
        return True
    label = str(prediction.get("label") or "")
    raw_label = str(prediction.get("raw_label") or label)
    raw_confidence = float(prediction.get("raw_confidence", prediction.get("confidence", 0.0)))
    filter_status = str((prediction.get("temporal_filter") or {}).get("status") or "stable")
    return bool(
        raw_label != label
        or raw_confidence < float(min_confidence)
        or filter_status != "stable"
    )


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
