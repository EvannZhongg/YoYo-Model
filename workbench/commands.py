"""Command orchestration for the unified v2/v3 training workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import BASE_DIR


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
    supported = {"all", "detection", "string_segmentation", "semantic_string", "orientation", "orientation_roi"}
    if task not in supported:
        return f"Unsupported unified training task: {task}"

    dataset_root = Path(dataset_dir)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return f"Refused: unified dataset manifest is missing: {manifest_path}"

    if task == "semantic_string":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        args = [
            "-m", "string_segmentation.train_semantic",
            "--dataset-dir", str(dataset_root / "string_segmentation"),
            "--project", project_dir,
            "--name", f"{manifest['dataset_id']}_semantic_string",
            "--epochs", str(int(epochs)),
            "--architecture", "lraspp_mobilenet_v3",
            "--pretrained-backbone",
            "--input-width", "960",
            "--input-height", "544",
            "--batch", "2",
            "--hard-negative-weight", "0.1",
            "--negative-sample-weight", "4.0",
            "--early-stopping-patience", "10",
            "--early-stopping-min-epochs", "15",
        ]
    elif task == "orientation_roi":
        args = [
            "-m", "training_v3.train_orientation",
            "--view-manifest", str(dataset_root / "orientation_roi" / "manifest.json"),
            "--project-dir", project_dir,
            "--epochs", str(int(epochs)),
        ]
    else:
        args = [
            "-m", "training_v3.train",
            "--dataset-dir", dataset_dir,
            "--project-dir", project_dir,
            "--task", task,
            "--epochs", str(int(epochs)),
            "--auto-download",
        ]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)


def workbench_evaluate_v2v3(run_dir: str, device: str) -> str:
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return f"Refused: model run manifest is missing: {manifest_path}"
    args = ["-m", "training_v3.evaluate", run_dir]
    _append_value(args, "--device", device)
    return _run_workbench_command(args)
