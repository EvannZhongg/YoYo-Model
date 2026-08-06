"""Pose-excluded, hand-independent multi-task training pipeline."""

from .prepare_dataset import build_training_dataset

__all__ = ["build_training_dataset"]
