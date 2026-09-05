"""Command orchestration for Workbench training and evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR, SEMANTIC_STRING_CONFIG


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


def workbench_train_v2v3(
    dataset_dir: str,
    project_dir: str,
    task: str,
    epochs: int,
    device: str,
) -> str:
    supported = {"detection", "string_tracking", "orientation"}
    if task not in supported:
        return f"Unsupported training path: {task}"

    dataset_root = Path(dataset_dir)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return f"Refused: dataset manifest is missing: {manifest_path}"

    if task == "string_tracking":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        args = [
            "-m", "string_tracking.train",
            "--dataset-dir", str(dataset_root / "string_segmentation"),
            "--project", project_dir,
            "--name", f"{manifest['dataset_id']}_semantic_string",
            "--epochs", str(int(epochs)),
            "--architecture", SEMANTIC_STRING_CONFIG.architecture,
            "--input-width", str(SEMANTIC_STRING_CONFIG.input_width),
            "--input-height", str(SEMANTIC_STRING_CONFIG.input_height),
            "--batch", str(SEMANTIC_STRING_CONFIG.batch),
            "--workers", str(SEMANTIC_STRING_CONFIG.workers),
            "--lr", str(SEMANTIC_STRING_CONFIG.learning_rate),
            "--base-channels", str(SEMANTIC_STRING_CONFIG.base_channels),
            "--min-mask-width-px", str(SEMANTIC_STRING_CONFIG.min_mask_width_px),
            "--freeze-backbone-epochs", str(SEMANTIC_STRING_CONFIG.freeze_backbone_epochs),
            "--backbone-lr-multiplier", str(SEMANTIC_STRING_CONFIG.backbone_lr_multiplier),
            "--hard-negative-weight", str(SEMANTIC_STRING_CONFIG.hard_negative_weight),
            "--negative-sample-weight", str(SEMANTIC_STRING_CONFIG.negative_sample_weight),
            "--early-stopping-patience", str(SEMANTIC_STRING_CONFIG.early_stopping_patience),
            "--early-stopping-min-epochs", str(SEMANTIC_STRING_CONFIG.early_stopping_min_epochs),
            "--seed", str(SEMANTIC_STRING_CONFIG.seed),
        ]
        if SEMANTIC_STRING_CONFIG.pretrained_backbone:
            args.append("--pretrained-backbone")
    elif task == "orientation":
        args = [
            "-m", "yoyo_orientation.train",
            "--view-manifest", str(dataset_root / "orientation_roi" / "manifest.json"),
            "--project-dir", project_dir,
            "--epochs", str(int(epochs)),
        ]
    else:
        args = [
            "-m", "yoyo_detection.train",
            "--dataset-dir", dataset_dir,
            "--project-dir", project_dir,
            "--epochs", str(int(epochs)),
        ]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_evaluate_v2v3(run_dir: str, device: str) -> str:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return f"Refused: model run manifest is missing: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(manifest.get("task") or "")
    evaluator = {
        "detection": "yoyo_detection.evaluate",
        "orientation": "yoyo_orientation.evaluate",
        "binary_semantic_segmentation": "string_tracking.evaluate",
    }.get(task)
    if evaluator is None:
        return f"Refused: unsupported run task: {task}"
    args = ["-m", evaluator, run_dir]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_train_detection(dataset_dir: str, project_dir: str, epochs: int, device: str) -> str:
    return workbench_train_v2v3(dataset_dir, project_dir, "detection", epochs, device)


def workbench_train_string(dataset_dir: str, project_dir: str, epochs: int, device: str) -> str:
    return workbench_train_v2v3(dataset_dir, project_dir, "string_tracking", epochs, device)


def workbench_train_orientation(dataset_dir: str, project_dir: str, epochs: int, device: str) -> str:
    return workbench_train_v2v3(dataset_dir, project_dir, "orientation", epochs, device)
