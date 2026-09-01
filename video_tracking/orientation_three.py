"""Three-class trick-orientation decoding."""

from __future__ import annotations


def decode(names: dict[int, str], values: list[float], top1: int) -> tuple[dict[str, float], str, None, None]:
    probabilities = {names[index]: float(value) for index, value in enumerate(values)}
    return probabilities, names[top1], None, None

