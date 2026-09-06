"""Frame-level string recognition API.

Only the optional detector context (a yoyo bbox and hand points) is accepted;
the recognizer can therefore be run independently on a video or image stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from string_segmentation.semantic_model import (
    is_semantic_checkpoint,
    load_checkpoint,
    predict_prepared_probability,
    prepare_letterboxed_input,
    semantic_mask_observation,
)


class StringRecognizer:
    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        import torch

        path = Path(weights).resolve()
        if not path.is_file() or not is_semantic_checkpoint(path):
            raise ValueError(f"Unsupported string checkpoint: {path}")
        self.model, self.checkpoint = load_checkpoint(path, torch.device(device))
        self.device = torch.device(device)
        self.weights = path

    def predict(
        self,
        frame: np.ndarray,
        *,
        yoyo: dict[str, Any] | None = None,
        wrists: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        max_components: int = 32,
    ) -> dict[str, Any] | None:
        config = self.checkpoint["model_config"]
        tensor, meta = prepare_letterboxed_input(
            frame, int(config["input_width"]), int(config["input_height"]), self.device
        )
        probability = predict_prepared_probability(self.model, tensor)
        threshold = max(float(self.checkpoint.get("threshold", 0.5)), float(confidence or 0.0))
        return semantic_mask_observation(
            probability,
            meta,
            threshold=threshold,
            yoyo=yoyo,
            hand_points=[[float(item["x"]), float(item["y"])] for item in (wrists or []) if "x" in item and "y" in item],
            max_components=max(1, int(max_components)),
        )


def load_string_model(weights: str | Path, device: str = "cpu") -> StringRecognizer:
    return StringRecognizer(weights, device)


def load_runtime_string_model(*args, **kwargs):
    """Load the production semantic backend for the compositor."""
    from video_tracking.tracker import _load_string_model

    return _load_string_model(*args, **kwargs)


def predict_runtime_string_model(*args, **kwargs):
    """Run the production string post-processing backend."""
    from video_tracking.tracker import _predict_string_model

    return _predict_string_model(*args, **kwargs)


__all__ = ["StringRecognizer", "load_string_model", "load_runtime_string_model", "predict_runtime_string_model"]
