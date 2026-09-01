"""ROI-based coarse trick-orientation inference for tracked video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from common.orientation import (
    PRESENTATION_ORIENTATION_CLASSES,
    PRESENTATION_ORIENTATION_CLASS_ORDER,
    PRESENTATION_TO_TRICK,
    TRICK_ORIENTATION_CLASSES,
    TRICK_ORIENTATION_CLASS_ORDER,
    validate_orientation_names,
)
from video_tracking.orientation_four import decode as decode_four_class
from video_tracking.orientation_three import decode as decode_three_class


# Backward-compatible names used by tracking and evaluation callers.
ORIENTATION_CLASSES = TRICK_ORIENTATION_CLASSES
ORIENTATION_CLASS_ORDER = TRICK_ORIENTATION_CLASS_ORDER


@dataclass
class OrientationTemporalFilter:
    """Causal EMA and hysteresis for sparse orientation observations."""

    ema_alpha: float = 0.4
    switch_margin: float = 0.05
    switch_confirmations: int = 3
    strong_switch_confidence: float = 0.9
    strong_switch_margin: float = 0.2
    switch_confirmation_seconds: float | None = None
    ema_time_constant_seconds: float | None = None
    _ema: dict[str, float] | None = field(default=None, init=False, repr=False)
    _label: str | None = field(default=None, init=False, repr=False)
    _pending_label: str | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)
    _pending_since_s: float | None = field(default=None, init=False, repr=False)
    _last_timestamp_s: float | None = field(default=None, init=False, repr=False)

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
        if self.switch_confirmation_seconds is not None and float(self.switch_confirmation_seconds) <= 0.0:
            raise ValueError("switch_confirmation_seconds must be positive when set")
        if self.ema_time_constant_seconds is not None and float(self.ema_time_constant_seconds) <= 0.0:
            raise ValueError("ema_time_constant_seconds must be positive when set")

    def update(self, prediction: dict[str, Any], timestamp_s: float | None = None) -> dict[str, Any]:
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

        timestamp = None if timestamp_s is None else float(timestamp_s)
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite when set")
        status = "stable"
        if self._ema is None:
            self._ema = dict(probabilities)
            self._label = raw_label
            status = "initialized"
        else:
            if self.ema_time_constant_seconds is not None and timestamp is not None and self._last_timestamp_s is not None:
                dt = max(0.0, timestamp - self._last_timestamp_s)
                alpha = 1.0 - math.exp(-dt / float(self.ema_time_constant_seconds))
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
                self._pending_since_s = None
            else:
                if candidate == self._pending_label:
                    self._pending_count += 1
                else:
                    self._pending_label = candidate
                    self._pending_count = 1
                    self._pending_since_s = timestamp
                current_label = str(self._label)
                margin_ok = self._ema[candidate] - self._ema[current_label] >= float(self.switch_margin)
                if self.switch_confirmation_seconds is not None and timestamp is not None and self._pending_since_s is not None:
                    confirmation_ok = timestamp - self._pending_since_s >= float(self.switch_confirmation_seconds)
                else:
                    confirmation_ok = self._pending_count >= int(self.switch_confirmations)
                confirmed = confirmation_ok and margin_ok
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
                    self._pending_since_s = None
                    status = "strong_switched" if strong else "confirmed_switched"
                else:
                    status = "pending"

        self._last_timestamp_s = timestamp if timestamp is not None else self._last_timestamp_s
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
                "pending_duration_s": (
                    round(max(0.0, timestamp - self._pending_since_s), 6)
                    if timestamp is not None and self._pending_since_s is not None else None
                ),
                "ema_alpha": float(self.ema_alpha),
                "switch_margin": float(self.switch_margin),
                "switch_confirmations": int(self.switch_confirmations),
                "strong_switch_confidence": float(self.strong_switch_confidence),
                "strong_switch_margin": float(self.strong_switch_margin),
                "switch_confirmation_seconds": self.switch_confirmation_seconds,
                "ema_time_constant_seconds": self.ema_time_constant_seconds,
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
        try:
            validate_orientation_names(names)
        except ValueError:
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
    direct_inference: bool = False,
) -> dict[str, Any] | None:
    height, width = frame.shape[:2]
    left, top, right, bottom = orientation_crop_box(width, height, yoyo)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    if direct_inference:
        import cv2
        import torch
        from PIL import Image
        from ultralytics.data.augment import classify_transforms
        from ultralytics.utils.torch_utils import select_device

        runtime_key = (int(imgsz), str(device).strip())
        runtime = getattr(model, "_yoyo_orientation_runtime", None)
        if runtime is None or runtime["key"] != runtime_key:
            network = model.model.fuse(verbose=False)
            selected_device = select_device(str(device).strip(), verbose=False)
            network = network.to(selected_device).eval()
            transforms = network.transforms
            first_transform = getattr(transforms, "transforms", [None])[0]
            if getattr(first_transform, "size", None) != int(imgsz):
                transforms = classify_transforms((int(imgsz), int(imgsz)))
            runtime = {
                "key": runtime_key,
                "network": network,
                "device": selected_device,
                "transforms": transforms,
            }
            setattr(model, "_yoyo_orientation_runtime", runtime)
        tensor = runtime["transforms"](
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(runtime["device"])
        with torch.inference_mode():
            output = runtime["network"](tensor)
            probabilities = output[0] if isinstance(output, (list, tuple)) else output
            values = [float(value) for value in probabilities[0].detach().cpu().tolist()]
        top1 = max(range(len(values)), key=values.__getitem__)
        top1_confidence = values[top1]
    else:
        kwargs: dict[str, Any] = {"source": crop, "imgsz": int(imgsz), "verbose": False}
        if str(device).strip():
            kwargs["device"] = str(device).strip()
        result = model.predict(**kwargs)[0]
        probs = getattr(result, "probs", None)
        if probs is None or getattr(probs, "data", None) is None:
            return None
        values = [float(value) for value in probs.data.detach().cpu().tolist()]
        top1 = int(probs.top1)
        top1_confidence = float(probs.top1conf.detach().cpu().item())
    try:
        variant = validate_orientation_names(names)
    except ValueError:
        raise ValueError(f"orientation model has incompatible classes: {names}")
    if variant == "four":
        coarse_probabilities, label, presentation_label, presentation_probabilities = decode_four_class(
            names, values, top1
        )
    else:
        coarse_probabilities, label, presentation_label, presentation_probabilities = decode_three_class(
            names, values, top1
        )
    confidence = coarse_probabilities[label]
    return {
        "label": label,
        "confidence": round(confidence, 6),
        "probabilities": {name: round(coarse_probabilities[name], 6) for name in ORIENTATION_CLASS_ORDER},
        "presentation_label": presentation_label,
        "presentation_confidence": round(top1_confidence, 6),
        "presentation_probabilities": (
            {name: round(presentation_probabilities[name], 6) for name in PRESENTATION_ORIENTATION_CLASS_ORDER}
            if presentation_probabilities is not None else None
        ),
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
