"""Command orchestration for the dataset and model workbench.

This module deliberately has no Gradio dependency. The UI can bind these
operations, while tests and future clients can exercise the workflow directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import (
    BASE_DIR,
    DATASET_CONFIG,
    SEMANTIC_STRING_CONFIG,
    STRING_SEGMENTATION_CONFIG,
)


def _run_workbench_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        return f"Command failed to start: {exc}"
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return f"Exit code: {completed.returncode}\n{output}" if output else f"Exit code: {completed.returncode}"


def _append_value(args: list[str], option: str, value: object) -> None:
    normalized = str(value).strip()
    if normalized:
        args.extend([option, normalized])


def workbench_build(videos_dir: str, dataset_dir: str, sample_fps: float, max_frames_per_video: int) -> str:
    return _run_workbench_command([
        "-m", "video_dataset.build", "--videos-dir", videos_dir, "--output-dir", dataset_dir,
        "--sample-fps", str(sample_fps), "--max-frames-per-video", str(int(max_frames_per_video)),
        "--action-group", DATASET_CONFIG.current_action_group,
    ])


def workbench_audit(dataset_dir: str, strict: bool) -> str:
    args = ["-m", "video_dataset.audit", "--dataset-dir", dataset_dir]
    if strict:
        args.append("--strict")
    return _run_workbench_command(args)


def workbench_model_registry() -> str:
    return _run_workbench_command(["model_registry.py"])


def workbench_candidates(
    dataset_dir: str,
    weights: str,
    sample_fps: float,
    confidence: float,
    max_candidates: int,
    exclude_source_groups: str = "",
) -> str:
    args = [
        "-m", "video_dataset.select_candidates", "--dataset-dir", dataset_dir, "--weights", weights,
        "--sample-fps", str(sample_fps), "--confidence", str(confidence),
        "--max-candidates-per-video", str(int(max_candidates)),
    ]
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    return _run_workbench_command(args)


def workbench_vlm(
    dataset_dir: str,
    split: str,
    limit: int,
    workers: int,
    candidates_only: bool,
    exclude_source_groups: str = "",
) -> str:
    args = [
        "-m", "annotation.video_frame_annotator", "--dataset-dir", dataset_dir,
        "--split", split, "--workers", str(int(workers)),
    ]
    if int(limit) > 0:
        args.extend(["--limit", str(int(limit))])
    if candidates_only:
        args.append("--candidates-only")
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    return _run_workbench_command(args)


def _version_output_guard(output_dir: str | Path, replace_existing: bool, label: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    if (path.is_file() or any(path.iterdir())) and not replace_existing:
        return (
            f"Refused: {label} already exists and is immutable by default: {path}. "
            "Choose a new versioned path or explicitly enable replacement."
        )
    return None


def _append_derived_split_args(
    args: list[str],
    holdout_source_groups: str,
    exclude_original_test: bool,
) -> None:
    _append_value(args, "--holdout-source-groups", holdout_source_groups)
    if exclude_original_test:
        args.append("--exclude-original-test")


def _workbench_holdout_defaults() -> tuple[str, bool]:
    manifest_path = SEMANTIC_STRING_CONFIG.dataset_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        policy = manifest.get("derived_split_policy") or {}
        groups = ",".join(str(item) for item in policy.get("holdout_source_groups") or [])
        return groups, bool(policy.get("exclude_original_test", False))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return "", False


def workbench_qa_export(
    dataset_dir: str,
    yolo_dir: str,
    replace_existing: bool = False,
    holdout_source_groups: str = "",
    exclude_original_test: bool = False,
) -> str:
    guard = _version_output_guard(yolo_dir, replace_existing, "YOLO dataset version")
    if guard:
        return guard
    qa_output = _run_workbench_command(["-m", "annotation.qa", "--dataset-dir", dataset_dir])
    export_args = [
        "-m", "yolo_training.prepare_dataset", "--annotations-dir", f"{dataset_dir}/annotations",
        "--output-dir", yolo_dir,
    ]
    _append_derived_split_args(export_args, holdout_source_groups, exclude_original_test)
    if replace_existing:
        export_args.append("--clear")
    export_output = _run_workbench_command(export_args)
    return f"QA\n{qa_output}\n\nYOLO export\n{export_output}"


def workbench_prepare_string(
    dataset_dir: str,
    output_dir: str,
    replace_existing: bool = False,
    holdout_source_groups: str = "",
    exclude_original_test: bool = False,
) -> str:
    guard = _version_output_guard(output_dir, replace_existing, "string dataset version")
    if guard:
        return guard
    args = [
        "-m", "string_segmentation.prepare_dataset",
        "--annotations-dir", f"{dataset_dir}/annotations",
        "--output-dir", output_dir,
    ]
    _append_derived_split_args(args, holdout_source_groups, exclude_original_test)
    if replace_existing:
        args.append("--clear")
    return _run_workbench_command(args)


def workbench_train_string(
    dataset_dir: str,
    output_dir: str,
    epochs: int,
    device: str,
    run_name: str = "",
) -> str:
    selected_name = run_name.strip() or f"{STRING_SEGMENTATION_CONFIG.run_name}_candidate"
    run_dir = Path(STRING_SEGMENTATION_CONFIG.project) / selected_name
    guard = _version_output_guard(run_dir, False, "YOLO string model run")
    if guard:
        return guard
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.is_file():
        return (
            f"Refused: string dataset manifest is missing: {manifest_path}. "
            "Export and audit a versioned dataset before training."
        )
    args = [
        "-m", "string_segmentation.train",
        "--annotations-dir", f"{dataset_dir}/annotations",
        "--dataset-dir", output_dir,
        "--project", str(STRING_SEGMENTATION_CONFIG.project),
        "--name", selected_name,
        "--epochs", str(int(epochs)),
        "--auto-download",
        "--no-prepare",
    ]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_train_semantic(
    string_dataset_dir: str,
    output_dir: str,
    run_name: str,
    epochs: int,
    device: str,
    architecture: str,
    pretrained_backbone: bool,
    initial_weights: str,
    learning_rate: float,
    hard_negative_weight: float,
    early_stopping_patience: int,
    early_stopping_min_epochs: int,
) -> str:
    selected_name = run_name.strip() or SEMANTIC_STRING_CONFIG.run_name
    guard = _version_output_guard(Path(output_dir) / selected_name, False, "semantic model run")
    if guard:
        return guard
    args = [
        "-m", "string_segmentation.train_semantic",
        "--dataset-dir", string_dataset_dir,
        "--project", output_dir,
        "--name", selected_name,
        "--epochs", str(int(epochs)),
        "--architecture", architecture,
        "--lr", str(float(learning_rate)),
        "--hard-negative-weight", str(float(hard_negative_weight)),
        "--early-stopping-patience", str(int(early_stopping_patience)),
        "--early-stopping-min-epochs", str(int(early_stopping_min_epochs)),
    ]
    if pretrained_backbone:
        args.append("--pretrained-backbone")
    _append_value(args, "--initial-weights", initial_weights)
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_evaluate_semantic(weights: str, string_dataset_dir: str, device: str) -> str:
    args = [
        "-m", "string_segmentation.evaluate_semantic",
        "--weights", weights,
        "--dataset-dir", string_dataset_dir,
        "--split", "test",
    ]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_evaluate_pipeline(
    dataset_dir: str,
    string_dataset_dir: str,
    detector_weights: str,
    string_weights: str,
    string_inference_scale: float,
    split: str,
    output_dir: str,
    device: str,
    confirm_test: bool,
) -> tuple[str, str | None]:
    if split not in {"val", "test"}:
        return f"Unsupported pipeline evaluation split: {split}", None
    if split == "test" and not confirm_test:
        return "Refused: frozen test evaluation requires explicit final-evaluation confirmation.", None
    args = [
        "-m", "video_tracking.evaluate_pipeline",
        "--detector-weights", detector_weights,
        "--string-weights", string_weights,
        "--string-inference-scale", str(float(string_inference_scale)),
        "--dataset-dir", string_dataset_dir,
        "--annotations-dir", str(Path(dataset_dir) / "annotations"),
        "--split", split,
        "--output-dir", output_dir,
    ]
    _append_value(args, "--device", device)
    output = _run_workbench_command(args)
    sheet = Path(output_dir) / f"{split}_tracking_pipeline_predictions.jpg"
    return output, str(sheet) if sheet.exists() else None


def workbench_prelabel_strings(
    dataset_dir: str,
    split: str,
    limit: int,
    exclude_source_groups: str = "",
) -> str:
    args = ["-m", "annotation.string_prelabel", "--dataset-dir", dataset_dir, "--split", split]
    if int(limit) > 0:
        args.extend(["--limit", str(int(limit))])
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    return _run_workbench_command(args)


def workbench_string_review_queue(
    dataset_dir: str,
    split: str,
    limit: int,
    with_model: bool,
    weights: str,
    device: str,
    exclude_source_groups: str = "",
    strategy: str = "uncertainty",
) -> tuple[str, str | None]:
    if strategy not in {"uncertainty", "agreement"}:
        return f"Unsupported string review strategy: {strategy}", None
    if strategy == "agreement" and not with_model:
        return "Agreement-first review requires the current semantic model.", None
    args = [
        "-m", "video_dataset.string_review_queue",
        "--dataset-dir", dataset_dir,
        "--split", split,
        "--limit", str(int(limit)),
        "--strategy", strategy,
    ]
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    if with_model:
        args.extend(["--with-model", "--weights", weights])
        _append_value(args, "--device", device)
    output = _run_workbench_command(args)
    sheet = Path(dataset_dir) / "review_sheets" / "string_review_queue.jpg"
    return output, str(sheet) if sheet.exists() else None


def workbench_hard_negative_queue(
    dataset_dir: str,
    weights: str,
    device: str,
    output_name: str,
    exclude_source_groups: str = "",
) -> tuple[str, str | None, str]:
    output_name = str(output_name).strip() or "string_hard_negative_queue"
    args = [
        "-m", "video_dataset.hard_negative_queue",
        "--dataset-dir", dataset_dir,
        "--weights", weights,
        "--output-name", output_name,
    ]
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    _append_value(args, "--device", device)
    output = _run_workbench_command(args)
    root = Path(dataset_dir)
    sheet = root / "review_sheets" / f"{output_name}.jpg"
    queue = root / f"{output_name}.json"
    return output, str(sheet) if sheet.exists() else None, str(queue)


def workbench_hard_negative_neighbors(
    dataset_dir: str,
    queue_path: str,
    offsets: str,
    top_anchors: int,
    limit: int,
    include_yoyo_visible: bool,
    include_clean_anchors: bool,
    output_name: str,
    exclude_source_groups: str = "",
) -> tuple[str, str | None, str]:
    output_name = str(output_name).strip() or "hard_negative_neighbor_candidates"
    args = [
        "-m", "video_dataset.hard_negative_candidates",
        "--dataset-dir", dataset_dir,
        "--queue", queue_path,
        "--offset-seconds", str(offsets).strip(),
        "--top-anchors", str(int(top_anchors)),
        "--limit", str(int(limit)),
        "--output-name", output_name,
    ]
    _append_value(args, "--exclude-source-groups", exclude_source_groups)
    if include_yoyo_visible:
        args.append("--include-yoyo-visible")
    if include_clean_anchors:
        args.append("--include-clean-anchors")
    output = _run_workbench_command(args)
    root = Path(dataset_dir)
    sheet = root / "review_sheets" / f"{output_name}.jpg"
    candidates = root / f"{output_name}.json"
    return output, str(sheet) if sheet.exists() else None, str(candidates)
