"""Shared run-manifest helpers for the independent model pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.files import sha256_file
from config import BASE_DIR


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def resolve_weights(path: str | Path, default_name: str, models_dir: str | Path | None = None) -> Path:
    """Resolve an explicit checkpoint or a project-local foundation weight."""
    candidate = Path(path) if str(path).strip() else Path(models_dir or BASE_DIR / "models") / default_name
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Initial weights not found: {candidate}")
    return candidate.resolve()


def initialization_lineage(initial_weights: Path, source_groups: dict[str, Any]) -> dict[str, Any]:
    parent_manifest_path = initial_weights.parent.parent / "run_manifest.json"
    lineage: dict[str, Any] = {
        "kind": "versioned_run" if parent_manifest_path.exists() else "foundation_pretrained",
        "parent_run_manifest": str(parent_manifest_path.resolve()) if parent_manifest_path.exists() else "",
        "evaluation_source_overlap": {"val": [], "test": []},
        "promotion_eligible": True,
    }
    if not parent_manifest_path.exists():
        return lineage
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_train = set((parent.get("source_groups") or {}).get("train", []))
    overlap = {
        split: sorted(parent_train & set(source_groups.get(split, [])))
        for split in ("val", "test")
    }
    lineage["evaluation_source_overlap"] = overlap
    lineage["promotion_eligible"] = not any(overlap.values())
    return lineage


def write_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


__all__ = ["json_value", "resolve_weights", "initialization_lineage", "write_manifest", "sha256_file"]
