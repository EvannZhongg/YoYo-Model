"""Runtime device selection shared by semantic training and evaluation."""

from __future__ import annotations

import torch


def resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.isdigit():
        requested = f"cuda:{requested}"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
