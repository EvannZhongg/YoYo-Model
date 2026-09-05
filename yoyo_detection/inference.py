"""Detector-only runtime API.

The API deliberately knows nothing about string or orientation models.  A
caller may pass its detections to another pipeline through a frame context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def extract_detections(result: Any, class_names: dict[int, str] | None = None) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    names = class_names or {}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    detections: list[dict[str, Any]] = []
    for bbox, confidence, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "confidence": float(confidence),
            "class_id": int(class_id),
            "class_name": names.get(int(class_id), str(class_id)),
        })
    return detections


class Detector:
    def __init__(self, weights: str | Path, device: str = "") -> None:
        from ultralytics import YOLO

        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.device = str(device)
        self.class_names = {int(key): str(value) for key, value in dict(getattr(self.model, "names", {}) or {}).items()}

    def predict(self, frame: np.ndarray, *, confidence: float = 0.15, iou: float = 0.7, imgsz: int = 1024) -> list[dict[str, Any]]:
        result = self.model.predict(frame, conf=float(confidence), iou=float(iou), imgsz=int(imgsz), device=self.device or None, verbose=False)[0]
        return extract_detections(result, self.class_names)


def load_detector(weights: str | Path, device: str = "") -> Detector:
    return Detector(weights, device)


__all__ = ["Detector", "extract_detections", "load_detector"]
