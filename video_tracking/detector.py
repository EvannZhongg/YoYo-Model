"""Detector stage used by the video compositor."""

from yoyo_detection.inference import Detector, extract_detections, load_detector

__all__ = ["Detector", "extract_detections", "load_detector"]
