from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from string_segmentation.semantic_model import prepare_letterboxed_input
from .evaluate import decode_centerline, load_model
from .train import fuse_geometry


class CenterlineFusionRecognizer:
    """Runtime adapter exposing the same ``predict`` shape as StringRecognizer."""

    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        self.weights = Path(weights).resolve()
        self.device = torch.device(device)
        self.model, self.checkpoint = load_model(self.weights, self.device)

    @torch.inference_mode()
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
        output = self.model(tensor)
        fused = fuse_geometry(output)[0, 0].cpu().numpy()
        tangent = torch.tanh(output[0, 2:]).cpu().numpy()
        threshold = float(self.checkpoint.get("threshold", 0.25) if confidence is None else confidence)
        binary = decode_centerline(fused, tangent, threshold)
        from string_segmentation.semantic_model import _skeleton_cover_paths, _skeletonize, restore_coordinates

        paths = [restore_coordinates(path, meta) for path in _skeleton_cover_paths(_skeletonize(binary), 8, 256) if len(path) >= 2]
        if not paths:
            return None
        paths = paths[: max(1, int(max_components))]
        return {
            "points": paths[0],
            "polylines": paths,
            "confidence": float(np.max(fused[binary > 0])) if np.any(binary) else 0.0,
            "method": "centerline_geometry_fusion",
            "probability_threshold": threshold,
            "component_count": len(paths),
            "polyline_count": len(paths),
            "tangent_field": True,
            "anchored_to_yoyo": yoyo is not None,
        }


def is_centerline_checkpoint(path: str | Path) -> bool:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(checkpoint, dict) and checkpoint.get("format") == "yoyo_centerline_fusion_v1"
