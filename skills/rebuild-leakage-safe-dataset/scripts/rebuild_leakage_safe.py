#!/usr/bin/env python3
"""Wrap and verify leakage-safe incremental dataset rebuilds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SPLITS = ("train", "val", "test")
MODES = ("append-isolated", "strict-eval")
BASELINE_TOKEN = "{baseline_manifest}"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """Raised when a manifest violates the skill contract."""


@dataclass(frozen=True)
class ManifestView:
    path: Path
    raw: dict[str, Any]
    assignment: dict[str, str]
    records_by_hash: dict[str, dict[str, str]]
    counts: dict[str, int]
    target_ratios: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest root must be an object: {path}")
    return value


def _target_ratios(raw: dict[str, Any]) -> dict[str, float]:
    configured = ((raw.get("split_policy") or {}).get("target_sample_ratios") or {})
    defaults = {"train": 0.70, "val": 0.15, "test": 0.15}
    ratios: dict[str, float] = {}
    for split in SPLITS:
        try:
            value = float(configured.get(split, defaults[split]))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"target ratio for {split} is not numeric") from exc
        if not 0.0 <= value <= 1.0:
            raise ContractError(f"target ratio for {split} must be between 0 and 1")
        ratios[split] = value
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ContractError("target sample ratios must sum to 1")
    return ratios


def load_manifest(path_value: str | Path) -> ManifestView:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ContractError(f"manifest does not exist: {path}")
    raw = _read_json(path)
    source_groups = ((raw.get("split_policy") or {}).get("source_groups") or {})
    if not isinstance(source_groups, dict):
        raise ContractError("split_policy.source_groups must be an object")

    assignment: dict[str, str] = {}
    for split in SPLITS:
        groups = source_groups.get(split)
        if not isinstance(groups, list):
            raise ContractError(f"split_policy.source_groups.{split} must be an array")
        for group_value in groups:
            group = str(group_value).strip()
            if not group:
                raise ContractError(f"empty source group in split {split}")
            previous = assignment.get(group)
            if previous is not None:
                raise ContractError(
                    f"source group {group!r} appears in both {previous} and {split}"
                )
            assignment[group] = split

    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("records must be a non-empty array")
    records_by_hash: dict[str, dict[str, str]] = {}
    counts = {split: 0 for split in SPLITS}
    for index, record_value in enumerate(records):
        if not isinstance(record_value, dict):
            raise ContractError(f"records[{index}] must be an object")
        group = str(record_value.get("source_group", "")).strip()
        split = str(record_value.get("split", "")).strip()
        image_hash = str(record_value.get("image_sha256", "")).strip()
        if group not in assignment:
            raise ContractError(f"records[{index}] has unassigned source_group={group!r}")
        if split not in SPLITS:
            raise ContractError(f"records[{index}] has invalid split={split!r}")
        if assignment[group] != split:
            raise ContractError(
                f"records[{index}] split={split} disagrees with source group {group!r} "
                f"assignment={assignment[group]}"
            )
        if not SHA256_RE.fullmatch(image_hash):
            raise ContractError(f"records[{index}] image_sha256 is not a lowercase SHA-256")
        if image_hash in records_by_hash:
            previous = records_by_hash[image_hash]
            raise ContractError(
                f"duplicate image_sha256={image_hash} in {previous['split']} and {split}"
            )
        records_by_hash[image_hash] = {
            "split": split,
            "source_group": group,
        }
        counts[split] += 1

    declared_leakage = ((raw.get("split_policy") or {}).get("leakage") or {})
    for key in ("source_group_overlap_count", "image_sha256_overlap_count"):
        if key in declared_leakage:
            try:
                value = int(declared_leakage[key])
            except (TypeError, ValueError) as exc:
                raise ContractError(f"split_policy.leakage.{key} must be an integer") from exc
            if value != 0:
                raise ContractError(f"split_policy.leakage.{key}={value}; expected 0")

    return ManifestView(
        path=path,
        raw=raw,
        assignment=assignment,
        records_by_hash=records_by_hash,
        counts=counts,
        target_ratios=_target_ratios(raw),
    )


def verify_manifests(
    baseline: ManifestView,
    rebuilt: ManifestView,
    mode: str,
    max_ratio_deviation: float,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    errors: list[str] = []

    moved_groups = sorted(
        group
        for group, old_split in baseline.assignment.items()
        if group in rebuilt.assignment and rebuilt.assignment[group] != old_split
    )
    missing_groups = sorted(set(baseline.assignment) - set(rebuilt.assignment))
    for group in moved_groups:
        errors.append(
            f"existing source group moved: {group}: "
            f"{baseline.assignment[group]} -> {rebuilt.assignment[group]}"
        )
    for group in missing_groups:
        errors.append(f"existing source group is missing: {group}")

    missing_hashes: list[str] = []
    moved_hashes: list[str] = []
    for image_hash, old_record in baseline.records_by_hash.items():
        new_record = rebuilt.records_by_hash.get(image_hash)
        if new_record is None:
            missing_hashes.append(image_hash)
        elif new_record["split"] != old_record["split"]:
            moved_hashes.append(image_hash)
    if missing_hashes:
        errors.append(f"{len(missing_hashes)} existing image hashes are missing")
    if moved_hashes:
        errors.append(f"{len(moved_hashes)} existing image hashes changed split")

    new_groups = sorted(set(rebuilt.assignment) - set(baseline.assignment))
    new_groups_by_split = {
        split: sorted(group for group in new_groups if rebuilt.assignment[group] == split)
        for split in SPLITS
    }
    new_hashes = sorted(set(rebuilt.records_by_hash) - set(baseline.records_by_hash))
    new_hashes_by_split = {
        split: sorted(
            image_hash
            for image_hash in new_hashes
            if rebuilt.records_by_hash[image_hash]["split"] == split
        )
        for split in SPLITS
    }
    evaluation_added = new_hashes_by_split["val"] + new_hashes_by_split["test"]
    if mode == "strict-eval":
        non_train_groups = new_groups_by_split["val"] + new_groups_by_split["test"]
        if non_train_groups:
            errors.append(
                f"strict-eval requires new groups to be train-only; found {len(non_train_groups)}"
            )
        if evaluation_added:
            errors.append(
                f"strict-eval forbids evaluation expansion; found {len(evaluation_added)} new hashes"
            )

    total = sum(rebuilt.counts.values())
    actual_ratios = {
        split: (rebuilt.counts[split] / total if total else 0.0) for split in SPLITS
    }
    ratio_deviation = {
        split: abs(actual_ratios[split] - rebuilt.target_ratios[split]) for split in SPLITS
    }
    excessive = {
        split: value
        for split, value in ratio_deviation.items()
        if value > float(max_ratio_deviation)
    }
    if excessive:
        detail = ", ".join(f"{split}={value:.4f}" for split, value in excessive.items())
        errors.append(
            f"final split ratio deviation exceeds {max_ratio_deviation:.4f}: {detail}"
        )

    return {
        "ok": not errors,
        "mode": mode,
        "errors": errors,
        "baseline": {
            "manifest": str(baseline.path),
            "source_group_count": len(baseline.assignment),
            "sample_count": len(baseline.records_by_hash),
            "counts": baseline.counts,
        },
        "rebuilt": {
            "manifest": str(rebuilt.path),
            "source_group_count": len(rebuilt.assignment),
            "sample_count": len(rebuilt.records_by_hash),
            "counts": rebuilt.counts,
            "target_ratios": rebuilt.target_ratios,
            "actual_ratios": {key: round(value, 8) for key, value in actual_ratios.items()},
            "ratio_deviation": {key: round(value, 8) for key, value in ratio_deviation.items()},
        },
        "lineage": {
            "moved_existing_groups": moved_groups,
            "missing_existing_groups": missing_groups,
            "missing_existing_hash_count": len(missing_hashes),
            "moved_existing_hash_count": len(moved_hashes),
            "new_groups_by_split": new_groups_by_split,
            "new_image_count_by_split": {
                split: len(values) for split, values in new_hashes_by_split.items()
            },
            "evaluation_expanded": bool(evaluation_added),
            "evaluation_added_image_count": len(evaluation_added),
        },
        "leakage": {
            "source_group_overlap_count": 0,
            "image_sha256_overlap_count": 0,
            "duplicate_image_sha256_count": 0,
        },
    }


def _write_json(path_value: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path_value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _snapshot_manifest(source: Path, destination: Path, overwrite: bool) -> None:
    if destination == source:
        raise ContractError("snapshot path must differ from the active manifest")
    if destination.is_relative_to(source.parent):
        raise ContractError("snapshot must be outside the dataset output directory")
    if destination.exists() and not overwrite:
        raise ContractError(f"snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)


def _command_args(values: Sequence[str], baseline_path: Path, allow_no_token: bool) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ContractError("run requires an external builder command after --")
    contains_token = any(BASELINE_TOKEN in value for value in command)
    if not contains_token and not allow_no_token:
        raise ContractError(
            f"builder command must contain {BASELINE_TOKEN}; use --allow-command-without-baseline "
            "only when the builder reads the active manifest before clearing output"
        )
    return [value.replace(BASELINE_TOKEN, str(baseline_path)) for value in command]


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_verify(args: argparse.Namespace) -> int:
    try:
        baseline = load_manifest(args.baseline)
        rebuilt = load_manifest(args.rebuilt)
        result = verify_manifests(
            baseline,
            rebuilt,
            mode=args.mode,
            max_ratio_deviation=args.max_ratio_deviation,
        )
    except ContractError as exc:
        result = {"ok": False, "mode": args.mode, "errors": [str(exc)]}
    if args.report:
        report_path = Path(args.report).resolve()
        result["report"] = str(report_path)
        _write_json(report_path, result)
    _print(result)
    return 0 if result["ok"] else 4


def run_build(args: argparse.Namespace) -> int:
    try:
        manifest_path = Path(args.manifest).resolve()
        baseline = load_manifest(manifest_path)
        snapshot_path = Path(args.snapshot_out).resolve()
        if snapshot_path.is_relative_to(manifest_path.parent):
            raise ContractError("snapshot must be outside the dataset output directory")
        command = _command_args(
            args.command,
            snapshot_path,
            allow_no_token=args.allow_command_without_baseline,
        )
        cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve()
        if not cwd.is_dir():
            raise ContractError(f"working directory does not exist: {cwd}")
        preflight = {
            "ok": True,
            "dry_run": bool(args.dry_run),
            "mode": args.mode,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "snapshot_out": str(snapshot_path),
            "cwd": str(cwd),
            "command": command,
            "baseline_source_group_count": len(baseline.assignment),
            "baseline_sample_count": len(baseline.records_by_hash),
        }
        if args.dry_run:
            _print(preflight)
            return 0

        _snapshot_manifest(manifest_path, snapshot_path, args.overwrite_snapshot)
        baseline_snapshot = load_manifest(snapshot_path)
        completed = subprocess.run(command, cwd=cwd, shell=False, check=False)
        if completed.returncode != 0:
            result = {
                **preflight,
                "ok": False,
                "dry_run": False,
                "builder_returncode": completed.returncode,
                "errors": ["external builder command failed; rebuild verification was not run"],
            }
            if args.report:
                report_path = Path(args.report).resolve()
                result["report"] = str(report_path)
                _write_json(report_path, result)
            _print(result)
            return 3

        rebuilt = load_manifest(manifest_path)
        verification = verify_manifests(
            baseline_snapshot,
            rebuilt,
            mode=args.mode,
            max_ratio_deviation=args.max_ratio_deviation,
        )
        result = {
            **preflight,
            **verification,
            "dry_run": False,
            "builder_returncode": completed.returncode,
            "baseline_manifest": str(snapshot_path),
            "baseline_manifest_sha256": sha256_file(snapshot_path),
            "rebuilt_manifest_sha256": sha256_file(manifest_path),
        }
    except ContractError as exc:
        result = {"ok": False, "mode": args.mode, "errors": [str(exc)]}
    if args.report:
        report_path = Path(args.report).resolve()
        result["report"] = str(report_path)
        _write_json(report_path, result)
    _print(result)
    return 0 if result["ok"] else 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or verify a leakage-safe incremental dataset rebuild."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    verify = subparsers.add_parser("verify", help="Compare an existing baseline and rebuild.")
    verify.add_argument("--baseline", required=True)
    verify.add_argument("--rebuilt", required=True)
    verify.add_argument("--mode", choices=MODES, default="append-isolated")
    verify.add_argument("--max-ratio-deviation", type=float, default=0.20)
    verify.add_argument("--report", default="")
    verify.set_defaults(handler=run_verify)

    run = subparsers.add_parser("run", help="Snapshot, invoke a builder, then verify.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--snapshot-out", required=True)
    run.add_argument("--report", default="")
    run.add_argument("--mode", choices=MODES, default="append-isolated")
    run.add_argument("--max-ratio-deviation", type=float, default=0.20)
    run.add_argument("--cwd", default="")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--overwrite-snapshot", action="store_true")
    run.add_argument("--allow-command-without-baseline", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=run_build)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= float(args.max_ratio_deviation) <= 1.0:
        print("max-ratio-deviation must be between 0 and 1", file=sys.stderr)
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
