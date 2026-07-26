"""Build a reproducible index of trained yoyo and string model versions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.files import sha256_file
from config import BASE_DIR, TRACKING_CONFIG


def _resolve(value: str | Path | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    values = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
    result: dict[str, Any] = {}
    if "map50" in values:
        result.update({"map50": values.get("map50"), "map50_95": values.get("map50_95")})
    if isinstance(values.get("seg"), dict):
        result.update({"mask_map50": values["seg"].get("map50"), "mask_map50_95": values["seg"].get("map50_95")})
    if isinstance(values.get("pixel"), dict):
        result["pixel_dice"] = values["pixel"].get("dice")
    if isinstance(values.get("tolerant"), dict):
        result["tolerant_f1"] = values["tolerant"].get("f1")
    if isinstance(values.get("image_presence"), dict):
        result["image_presence_f1"] = values["image_presence"].get("f1")
    if "negative_mean_false_positive_pixels" in values:
        result["negative_mean_false_positive_pixels"] = values.get("negative_mean_false_positive_pixels")
    ultralytics_keys = {
        "metrics/precision(B)": "box_precision",
        "metrics/recall(B)": "box_recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "metrics/precision(M)": "mask_precision",
        "metrics/recall(M)": "mask_recall",
        "metrics/mAP50(M)": "mask_map50",
        "metrics/mAP50-95(M)": "mask_map50_95",
        "metrics/accuracy_top1": "top1_accuracy",
        "metrics/accuracy_top5": "top5_accuracy",
    }
    for source, target in ultralytics_keys.items():
        if source in values:
            result[target] = values[source]
    if "macro_recall" in values:
        result["macro_recall"] = values["macro_recall"]
    if "per_class_recall" in values:
        result["per_class_recall"] = values["per_class_recall"]
    result["split"] = metrics.get("split")
    return {key: value for key, value in result.items() if value is not None}


def build_registry(base_dir: Path = BASE_DIR, runs_dir: Path | None = None) -> dict[str, Any]:
    runs_dir = runs_dir or base_dir / "runs"
    default_yoyo = TRACKING_CONFIG.weights_path.resolve()
    default_string = TRACKING_CONFIG.string_weights_path.resolve()
    default_orientation = TRACKING_CONFIG.orientation_weights_path.resolve()
    entries: list[dict[str, Any]] = []
    represented_weights: set[Path] = set()
    metric_names = ("test_metrics_current.json", "test_metrics.json", "test_segmentation_metrics.json", "test_semantic_metrics.json")
    for manifest_path in sorted(runs_dir.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = (manifest.get("artifacts") or {}).get("best")
        weights = _resolve(artifact, base_dir) or run_dir / "weights" / "best.pt"
        weights = weights.resolve()
        represented_weights.add(weights)
        dataset_manifest = _resolve(manifest.get("dataset_manifest"), base_dir)
        recorded_dataset_sha = str(manifest.get("dataset_manifest_sha256", ""))
        current_dataset_sha = sha256_file(dataset_manifest) if dataset_manifest and dataset_manifest.exists() else ""
        metrics_path = next((run_dir / name for name in metric_names if (run_dir / name).exists()), None)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else {}
        warnings = []
        if not weights.exists():
            warnings.append("best_weights_missing")
        if dataset_manifest is None or not dataset_manifest.exists():
            warnings.append("dataset_manifest_missing")
        elif recorded_dataset_sha and current_dataset_sha != recorded_dataset_sha:
            warnings.append("dataset_manifest_sha256_drift")
        if not metrics_path:
            warnings.append("independent_test_metrics_missing")
        roles = []
        if weights == default_yoyo:
            roles.append("default_yoyo_detector")
        if weights == default_string:
            roles.append("default_string_model")
        if weights == default_orientation:
            roles.append("default_orientation_model")
        entries.append(
            {
                "model_id": run_dir.relative_to(runs_dir).as_posix(),
                "task": manifest.get("task", "unknown"),
                "created_at_utc": manifest.get("created_at_utc"),
                "run_manifest": str(manifest_path.resolve()),
                "run_manifest_sha256": sha256_file(manifest_path),
                "weights": str(weights),
                "weights_sha256": sha256_file(weights) if weights.exists() else "",
                "dataset_manifest": str(dataset_manifest.resolve()) if dataset_manifest and dataset_manifest.exists() else str(dataset_manifest or ""),
                "dataset_manifest_sha256_recorded": recorded_dataset_sha,
                "dataset_manifest_sha256_current": current_dataset_sha,
                "metrics_file": str(metrics_path.resolve()) if metrics_path else "",
                "metrics": _metric_summary(metrics),
                "roles": roles,
                "warnings": warnings,
                "complete": not warnings,
            }
        )

    for weights in sorted(path.resolve() for path in runs_dir.rglob("weights/best.pt")):
        if weights in represented_weights:
            continue
        run_dir = weights.parent.parent
        roles = []
        if weights == default_yoyo:
            roles.append("default_yoyo_detector")
        if weights == default_string:
            roles.append("default_string_model")
        if weights == default_orientation:
            roles.append("default_orientation_model")
        entries.append(
            {
                "model_id": run_dir.relative_to(runs_dir).as_posix(),
                "task": "unknown",
                "run_manifest": "",
                "weights": str(weights),
                "weights_sha256": sha256_file(weights),
                "roles": roles,
                "warnings": ["run_manifest_missing"],
                "complete": False,
            }
        )
    entries.sort(key=lambda item: item["model_id"])
    return {
        "schema_version": "yoyo_model_registry_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs_dir.resolve()),
        "default_yoyo_weights": str(default_yoyo),
        "default_string_weights": str(default_string),
        "default_orientation_weights": str(default_orientation),
        "model_count": len(entries),
        "complete_model_count": sum(bool(item["complete"]) for item in entries),
        "models": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hash-verified model version registry.")
    parser.add_argument("--runs-dir", default=str(BASE_DIR / "runs"))
    parser.add_argument("--output", default=str(BASE_DIR / "runs" / "model_registry.json"))
    args = parser.parse_args()
    registry = build_registry(BASE_DIR, Path(args.runs_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"models": registry["model_count"], "complete": registry["complete_model_count"], "output": str(output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
