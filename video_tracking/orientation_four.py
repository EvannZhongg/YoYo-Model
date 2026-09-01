"""Four-class presentation-orientation decoding."""

from __future__ import annotations

from common.orientation import PRESENTATION_TO_TRICK, to_trick_probabilities


def decode(
    names: dict[int, str],
    values: list[float],
    top1: int,
) -> tuple[dict[str, float], str, str, dict[str, float]]:
    probabilities = {names[index]: float(value) for index, value in enumerate(values)}
    coarse = to_trick_probabilities(probabilities)
    presentation_label = names[top1]
    return coarse, PRESENTATION_TO_TRICK[presentation_label], presentation_label, probabilities

