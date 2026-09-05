"""String stage facade with an explicit frame/context contract."""

from string_tracking.inference import StringRecognizer, load_string_model


def run_string(recognizer: StringRecognizer, frame, context: dict | None = None):
    context = context or {}
    return recognizer.predict(frame, yoyo=context.get("yoyo"), wrists=context.get("wrists"))


__all__ = ["StringRecognizer", "load_string_model", "run_string"]
