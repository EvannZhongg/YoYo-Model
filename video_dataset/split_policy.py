"""Explicit derived-split policy for creating a fresh source holdout."""

from __future__ import annotations


VALID_SPLITS = {"train", "val", "test"}
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def parse_source_groups(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def derived_split(
    original_split: str,
    source_group: str,
    holdout_source_groups: set[str] | None = None,
    exclude_original_test: bool = False,
) -> tuple[str | None, str | None]:
    """Route whole source groups without mutating canonical annotation metadata."""
    original = str(original_split).strip().lower()
    group = str(source_group).strip()
    holdouts = set(holdout_source_groups or set())
    if original not in VALID_SPLITS:
        return None, f"invalid_split={original}"
    if group in holdouts:
        return "test", None
    if exclude_original_test and original == "test":
        return None, "original_test_excluded_for_fresh_holdout"
    return original, None


def remove_cross_split_duplicate_hashes(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Keep duplicate content only in the highest-priority evaluation split."""
    owners: dict[str, str] = {}
    for record in records:
        digest = str(record["image_sha256"])
        split = str(record["split"])
        current = owners.get(digest)
        if current is None or SPLIT_PRIORITY[split] < SPLIT_PRIORITY[current]:
            owners[digest] = split
    kept, dropped = [], []
    for record in records:
        owner = owners[str(record["image_sha256"])]
        if str(record["split"]) == owner:
            kept.append(record)
        else:
            dropped.append({**record, "duplicate_owner_split": owner})
    return kept, dropped
