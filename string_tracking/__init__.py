"""Independent string recognition/tracking training and inference pipeline."""

from .inference import StringRecognizer, load_string_model

__all__ = ["StringRecognizer", "load_string_model"]
