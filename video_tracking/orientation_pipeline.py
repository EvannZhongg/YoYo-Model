"""Orientation stage facade with an explicit detector context contract."""

from yoyo_orientation.inference import (
    OrientationRecognizer,
    OrientationTemporalFilter,
    load_orientation_model,
    predict_orientation,
)


def run_orientation(recognizer: OrientationRecognizer, frame, context: dict | None = None):
    return recognizer.predict(frame, (context or {}).get("yoyo"))


__all__ = ["OrientationRecognizer", "OrientationTemporalFilter", "load_orientation_model", "predict_orientation", "run_orientation"]
