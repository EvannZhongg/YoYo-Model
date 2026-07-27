"""Evaluate one versioned v2/v3 model on the untouched source-group test split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.files import sha256_file


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _detection_recall_from_confusion(
    matrix: list[list[float]],
    class_names: list[str],
) -> tuple[dict[str, float], list[list[float]]]:
    class_count = len(class_names)
    matrix_size = min(len(matrix), class_count + 1)
    full_matrix = [
        [float(value) for value in row[:matrix_size]]
        for row in matrix[:matrix_size]
    ]
    per_class_recall: dict[str, float] = {}
    for true_index, name in enumerate(class_names):
        support = sum(full_matrix[predicted][true_index] for predicted in range(matrix_size))
        per_class_recall[name] = full_matrix[true_index][true_index] / support if support else 0.0
    return per_class_recall, full_matrix


def evaluate_run(run_dir: Path, device: str = "0", workers: int = 0) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest_path = Path(run_manifest["dataset_manifest"])
    current_dataset_hash = sha256_file(dataset_manifest_path)
    if current_dataset_hash != run_manifest["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed after training; refusing incomparable test evaluation")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    task = str(run_manifest["task"])
    weights = Path(run_manifest["artifacts"]["best"])
    if sha256_file(weights) != run_manifest["artifacts"]["best_sha256"]:
        raise RuntimeError("Best weights hash does not match the training run manifest")
    from ultralytics import YOLO

    parameters = run_manifest["parameters"]
    model = YOLO(str(weights))
    task_data = dataset_manifest.get("data") or (dataset_manifest.get("tasks") or {}).get(task, {}).get("data")
    if not task_data:
        raise RuntimeError(f"Dataset manifest does not define data for task={task}")
    source_groups = dataset_manifest.get("source_groups") or dataset_manifest["split_policy"]["source_groups"]
    test_counts = dataset_manifest["counts"]["test"]
    sample_count = int(test_counts.get("samples", test_counts.get("total", 0)))
    val_kwargs: dict[str, Any] = {
        "data": str(task_data),
        "split": "test",
        "imgsz": int(parameters["imgsz"]),
        "batch": parameters["batch"],
        "workers": workers,
        "project": str(run_dir),
        "name": "test_evaluation",
        "exist_ok": True,
    }
    if device:
        val_kwargs["device"] = device
    results = model.val(**val_kwargs)
    metrics = _json_value(getattr(results, "results_dict", {}))
    confusion = getattr(getattr(results, "confusion_matrix", None), "matrix", None)
    if confusion is not None:
        matrix = _json_value(confusion)
        class_names = [str(model.names[index]) for index in sorted(model.names)]
        per_class_recall, full_matrix = _detection_recall_from_confusion(matrix, class_names)
        metrics["confusion_matrix_predicted_by_true"] = full_matrix
        metrics["per_class_recall"] = per_class_recall
        metrics["macro_recall"] = sum(per_class_recall.values()) / max(1, len(per_class_recall))
    result = {
        "schema_version": "yoyo_test_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "split": "test",
        "dataset_id": run_manifest["dataset_id"],
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": current_dataset_hash,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "source_groups": source_groups["test"],
        "sample_count": sample_count,
        "parameters": val_kwargs,
        "metrics": metrics,
        "artifacts": {"evaluation_dir": str(run_dir / "test_evaluation")},
    }
    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    promotion_eligible = bool((run_manifest.get("initialization_lineage") or {}).get("promotion_eligible", True))
    run_manifest["promotion"] = {
        "status": "test_evaluated_candidate" if promotion_eligible else "test_evaluated_ineligible_source_overlap",
        "test_metrics": str(metrics_path),
        "test_metrics_sha256": sha256_file(metrics_path),
        "note": (
            "Candidate is recorded; deployment defaults are changed only by an explicit promotion step."
            if promotion_eligible
            else "Evaluation is recorded for analysis only; initialization lineage overlaps evaluation sources."
        ),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a versioned model on its untouched test split.")
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_run(Path(args.run_dir), args.device, args.workers)
    print(json.dumps({"task": result["task"], "sample_count": result["sample_count"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
