"""Multi-task training pipeline for yoyo detection, strings, and orientation."""

from .prepare_dataset import build_training_dataset

__all__ = ["build_training_dataset"]
