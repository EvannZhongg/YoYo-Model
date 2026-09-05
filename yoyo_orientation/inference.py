"""Orientation-only runtime API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_tracking.orientation import (
    OrientationTemporalFilter,
    carry_orientation,
    orientation_crop_box,
    orientation_observation_is_unstable,
    predict_orientation,
)


def load_orientation_model(weights: str | Path, enabled: bool = True):
    from video_tracking.orientation import load_orientation_model as _load

    return _load(weights, enabled)


class OrientationRecognizer:
    def __init__(self, weights: str | Path, device: str = "cpu", imgsz: int = 320) -> None:
        model, status = load_orientation_model(weights, True)
        if model is None:
            raise RuntimeError(f"Unable to load orientation model: {status}")
        self.model, self.status, self.device, self.imgsz = model, status, str(device), int(imgsz)

    def predict(self, frame: np.ndarray, yoyo: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return predict_orientation(self.model, frame, yoyo, self.imgsz, self.device, direct_inference=True)


__all__ = [
    "OrientationRecognizer", "OrientationTemporalFilter", "carry_orientation",
    "orientation_crop_box", "orientation_observation_is_unstable", "predict_orientation",
    "load_orientation_model",
]
