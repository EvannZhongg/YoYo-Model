"""Shared label contracts for the two orientation model variants."""

from __future__ import annotations

TRICK_ORIENTATION_CLASSES = frozenset({"horizontal", "normal", "not_applicable"})
TRICK_ORIENTATION_CLASS_ORDER = ("horizontal", "normal", "not_applicable")
PRESENTATION_ORIENTATION_CLASSES = frozenset({"frontal", "edge_horizontal", "edge_vertical", "unknown"})
PRESENTATION_ORIENTATION_CLASS_ORDER = ("frontal", "edge_horizontal", "edge_vertical", "unknown")
PRESENTATION_TO_TRICK = {
    "frontal": "normal",
    "edge_vertical": "normal",
    "edge_horizontal": "horizontal",
    "unknown": "not_applicable",
}


def orientation_variant(classes: set[str] | frozenset[str]) -> str:
    """Return the explicit model variant for a complete class set."""
    if classes == TRICK_ORIENTATION_CLASSES:
        return "three"
    if classes == PRESENTATION_ORIENTATION_CLASSES:
        return "four"
    raise ValueError(f"incompatible orientation classes: {sorted(classes)}")


def to_trick_probabilities(
    probabilities: dict[str, float],
) -> dict[str, float]:
    """Project either model's probabilities to the supported trick labels."""
    variant = orientation_variant(frozenset(probabilities))
    if variant == "three":
        return {name: float(probabilities[name]) for name in TRICK_ORIENTATION_CLASS_ORDER}
    projected = {name: 0.0 for name in TRICK_ORIENTATION_CLASS_ORDER}
    for presentation, value in probabilities.items():
        projected[PRESENTATION_TO_TRICK[presentation]] += float(value)
    return projected


def validate_orientation_names(names: dict[int, str]) -> str:
    """Validate model names and return ``three`` or ``four`` variant."""
    return orientation_variant(frozenset(names.values()))
