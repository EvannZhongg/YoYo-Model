"""Project-local RTMPose-m WholeBody inference backend.

The model files are always resolved from explicit paths. RTMLib's URL-based
cache is deliberately not used so inference never writes weights to a user
profile directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import BASE_DIR


COCO_WHOLEBODY_KEYPOINT_COUNT = 133
COCO_BODY_KEYPOINT_COUNT = 17
LEFT_HAND_RANGE = range(91, 112)
RIGHT_HAND_RANGE = range(112, 133)
DEFAULT_DETECTOR_PATH = BASE_DIR / "models" / "rtmpose" / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
DEFAULT_POSE_PATH = BASE_DIR / "models" / "rtmpose" / "rtmpose-m-wholebody-256x192.onnx"


def _rtmlib_device(device: str) -> str:
    value = str(device or "").strip().lower()
    if not value or value == "cpu":
        return "cpu"
    if value.isdigit() or value.startswith("cuda") or value == "gpu":
        return "cuda"
    raise ValueError(f"Unsupported RTMPose device: {device}")


@dataclass(frozen=True)
class WholebodyPrediction:
    keypoints: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray


class RTMPoseWholebody:
    """RTMPose-m 133-keypoint model with a project-local person detector."""

    backend_name = "rtmpose-m_wholebody_onnx"
    keypoint_schema = "coco_wholebody_133"

    def __init__(
        self,
        pose_path: str | Path = DEFAULT_POSE_PATH,
        detector_path: str | Path = DEFAULT_DETECTOR_PATH,
        device: str = "",
    ) -> None:
        self.pose_path = Path(pose_path).resolve()
        self.detector_path = Path(detector_path).resolve()
        missing = [path for path in (self.pose_path, self.detector_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "RTMPose model files are missing: "
                + ", ".join(str(path) for path in missing)
                + ". Run cli.models.download_rtmpose inside the project virtual environment."
            )

        # Importing torch first makes its bundled CUDA/cuDNN DLLs available to
        # ONNX Runtime on Windows. CPU-only environments do not need it.
        resolved_device = _rtmlib_device(device)
        if resolved_device == "cuda":
            import torch  # noqa: F401

            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        from rtmlib import Wholebody

        self.device = resolved_device
        self._model = Wholebody(
            det=str(self.detector_path),
            det_input_size=(640, 640),
            pose=str(self.pose_path),
            pose_input_size=(192, 256),
            to_openpose=False,
            backend="onnxruntime",
            device=resolved_device,
        )

    def predict(self, frame: np.ndarray) -> WholebodyPrediction:
        boxes = np.asarray(self._model.det_model(frame), dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return WholebodyPrediction(
                np.empty((0, COCO_WHOLEBODY_KEYPOINT_COUNT, 2), dtype=np.float32),
                np.empty((0, COCO_WHOLEBODY_KEYPOINT_COUNT), dtype=np.float32),
                boxes,
            )
        keypoints, scores = self._model.pose_model(frame, bboxes=boxes)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (COCO_WHOLEBODY_KEYPOINT_COUNT, 2):
            raise ValueError(f"Unexpected RTMPose keypoint shape: {keypoints.shape}")
        if scores.shape != keypoints.shape[:2]:
            raise ValueError(f"Unexpected RTMPose score shape: {scores.shape}")
        return WholebodyPrediction(keypoints, scores, boxes)


def hand_landmarks(
    points: np.ndarray,
    scores: np.ndarray,
    side: str,
    threshold: float = 0.20,
) -> list[dict[str, Any]]:
    indexes = LEFT_HAND_RANGE if side == "left" else RIGHT_HAND_RANGE
    result: list[dict[str, Any]] = []
    for local_index, global_index in enumerate(indexes):
        confidence = float(scores[global_index])
        if confidence < threshold:
            continue
        result.append(
            {
                "index": local_index,
                "global_index": global_index,
                "x": float(points[global_index][0]),
                "y": float(points[global_index][1]),
                "confidence": confidence,
            }
        )
    return result
