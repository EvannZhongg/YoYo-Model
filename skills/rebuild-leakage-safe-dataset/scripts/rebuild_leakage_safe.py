#!/usr/bin/env python3
"""Wrap and verify leakage-safe incremental dataset rebuilds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SPLITS = ("train", "val", "test")
MODES = ("append-isolated", "strict-eval")
BASELINE_TOKEN = "{baseline_manifest}"
PROTECTED_CANONICAL_TOKEN = "{protected_canonical}"
REVIEW_SCHEMA_VERSION = "yoyo_dataset_review_v3"
PLAN_SCHEMA_VERSION = "leakage_safe_incremental_plan_v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
VALID_ORIENTATIONS = ("normal", "horizontal", "not_applicable")
VALID_STRING_VISIBILITY = ("visible", "partial", "not_visible")
FEATURES = (
    "samples",
    "yoyo_positive",
    "yoyo_negative",
    "string_positive",
    "string_negative",
    *(f"orientation:{value}" for value in VALID_ORIENTATIONS),
    *(f"string_visibility:{value}" for value in VALID_STRING_VISIBILITY),
)
NON_TASK_FIELDS = {
    "hands",
    "hands_pixel",
    "hands_2d",
    "hands_normalized",
    "hand_landmarks_pixel",
    "hand_pose",
    "pose",
    "pose_person",
}


class ContractError(ValueError):
    """Raised when a manifest violates the skill contract."""


@dataclass(frozen=True)
class ManifestView:
    path: Path
    raw: dict[str, Any]
    assignment: dict[str, str]
    records_by_hash: dict[str, dict[str, Any]]
    counts: dict[str, int]
    target_ratios: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_revision(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


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


def _task_labels(record: dict[str, Any], index: int) -> dict[str, Any]:
    orientation = record.get("trick_orientation")
    if orientation not in VALID_ORIENTATIONS:
        raise ContractError(
            f"records[{index}].trick_orientation must be one of {VALID_ORIENTATIONS}"
        )
    string_visibility = record.get("string_visibility")
    if string_visibility not in VALID_STRING_VISIBILITY:
        raise ContractError(
            f"records[{index}].string_visibility must be one of {VALID_STRING_VISIBILITY}"
        )
    yoyo_positive = record.get("yoyo_positive")
    string_positive = record.get("string_positive")
    if not isinstance(yoyo_positive, bool):
        raise ContractError(f"records[{index}].yoyo_positive must be boolean")
    if not isinstance(string_positive, bool):
        raise ContractError(f"records[{index}].string_positive must be boolean")
    expected_string_positive = string_visibility in {"visible", "partial"}
    if string_positive != expected_string_positive:
        raise ContractError(
            f"records[{index}].string_positive disagrees with string_visibility"
        )
    return {
        "trick_orientation": orientation,
        "yoyo_positive": yoyo_positive,
        "string_positive": string_positive,
        "string_visibility": string_visibility,
    }


def _manifest_view(path: Path, raw: dict[str, Any]) -> ManifestView:
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
    records_by_hash: dict[str, dict[str, Any]] = {}
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
            **_task_labels(record_value, index),
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


def load_manifest(path_value: str | Path) -> ManifestView:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ContractError(f"manifest does not exist: {path}")
    return _manifest_view(path, _read_json(path))


def _record_features(record: dict[str, Any]) -> Counter[str]:
    return Counter(
        {
            "samples": 1,
            "yoyo_positive": int(record["yoyo_positive"]),
            "yoyo_negative": int(not record["yoyo_positive"]),
            "string_positive": int(record["string_positive"]),
            "string_negative": int(not record["string_positive"]),
            f"orientation:{record['trick_orientation']}": 1,
            f"string_visibility:{record['string_visibility']}": 1,
        }
    )


def _group_features(manifest: ManifestView) -> dict[str, Counter[str]]:
    result = {group: Counter() for group in manifest.assignment}
    for record in manifest.records_by_hash.values():
        result[record["source_group"]].update(_record_features(record))
    return result


def _assignment_score(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
    ratios: dict[str, float],
) -> float:
    totals = sum(group_features.values(), Counter())
    split_counts = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        split_counts[split].update(group_features[group])
    score = 0.0
    for split in SPLITS:
        for feature in FEATURES:
            target = totals[feature] * ratios[split]
            weight = 8.0 if feature == "samples" else 1.0
            score += weight * ((split_counts[split][feature] - target) ** 2) / max(1.0, target)
    return score


def _coverage_gap_count(
    assignment: dict[str, str],
    group_features: dict[str, Counter[str]],
    min_support_groups: int,
) -> int:
    split_counts = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        split_counts[split].update(group_features[group])
    return sum(
        1
        for feature in FEATURES[1:]
        if sum(bool(values[feature]) for values in group_features.values()) >= min_support_groups
        for split in SPLITS
        if not split_counts[split][feature]
    )


def _label_balance_summary(
    manifest: ManifestView,
    min_support_groups: int,
) -> dict[str, Any]:
    group_features = _group_features(manifest)
    totals = sum(group_features.values(), Counter())
    split_counts = {split: Counter() for split in SPLITS}
    for group, split in manifest.assignment.items():
        split_counts[split].update(group_features[group])
    features: dict[str, Any] = {}
    coverage_gaps: list[dict[str, Any]] = []
    for feature in FEATURES[1:]:
        counts = {split: split_counts[split][feature] for split in SPLITS}
        support = {
            split: sum(
                bool(group_features[group][feature])
                for group, assigned_split in manifest.assignment.items()
                if assigned_split == split
            )
            for split in SPLITS
        }
        supporting_groups = sum(support.values())
        missing_splits = [
            split
            for split in SPLITS
            if supporting_groups >= min_support_groups and support[split] == 0
        ]
        if missing_splits:
            coverage_gaps.append(
                {
                    "feature": feature,
                    "supporting_group_count": supporting_groups,
                    "missing_splits": missing_splits,
                }
            )
        features[feature] = {
            "total_samples": totals[feature],
            "sample_counts": counts,
            "supporting_group_count": supporting_groups,
            "supporting_group_counts": support,
            "split_ratio_deviation": {
                split: round(
                    abs((counts[split] / totals[feature]) - manifest.target_ratios[split])
                    if totals[feature]
                    else 0.0,
                    8,
                )
                for split in SPLITS
            },
        }
    return {
        "strategy": "atomic_source_groups_multitask_label_stratification",
        "sample_weight": 8.0,
        "label_feature_weight": 1.0,
        "weighted_assignment_score": round(
            _assignment_score(
                manifest.assignment,
                group_features,
                manifest.target_ratios,
            ),
            8,
        ),
        "minimum_support_groups_for_coverage": min_support_groups,
        "features": features,
        "coverage_gaps": coverage_gaps,
    }


def build_incremental_plan(
    baseline: ManifestView,
    candidate: ManifestView,
    seed: int,
    attempts: int,
    min_support_groups: int = len(SPLITS),
) -> tuple[dict[str, Any], dict[str, str]]:
    missing_groups = sorted(set(baseline.assignment) - set(candidate.assignment))
    missing_hashes = sorted(set(baseline.records_by_hash) - set(candidate.records_by_hash))
    if missing_groups:
        raise ContractError(f"candidate is missing existing source groups: {missing_groups}")
    if missing_hashes:
        raise ContractError(f"candidate is missing {len(missing_hashes)} existing image hashes")

    group_features = _group_features(candidate)
    new_groups = sorted(set(candidate.assignment) - set(baseline.assignment))
    fixed_assignment = {
        group: baseline.assignment.get(group, "") for group in candidate.assignment
    }

    if new_groups:
        combinations = 3 ** len(new_groups)
        if combinations <= attempts:
            choices = itertools.product(SPLITS, repeat=len(new_groups))
        else:
            rng = random.Random(seed)
            choices = itertools.chain(
                [tuple(candidate.assignment[group] for group in new_groups)],
                (
                    tuple(
                        rng.choices(
                            SPLITS,
                            weights=[candidate.target_ratios[split] for split in SPLITS],
                            k=len(new_groups),
                        )
                    )
                    for _ in range(attempts)
                )
            )
        best: tuple[int, float, tuple[str, ...]] | None = None
        for choice in choices:
            assignment = dict(fixed_assignment)
            assignment.update(zip(new_groups, choice, strict=True))
            gap_count = _coverage_gap_count(
                assignment,
                group_features,
                min_support_groups,
            )
            score = _assignment_score(
                assignment,
                group_features,
                candidate.target_ratios,
            )
            item = (gap_count, score, tuple(choice))
            if best is None or item < best:
                best = item
        assert best is not None
        fixed_assignment.update(zip(new_groups, best[2], strict=True))

    raw = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "dataset_id": candidate.raw.get("dataset_id", ""),
        "split_policy": {
            "strategy": "stable_existing_groups_incremental_multitask_stratified",
            "target_sample_ratios": candidate.target_ratios,
            "source_groups": {
                split: sorted(
                    group for group, value in fixed_assignment.items() if value == split
                )
                for split in SPLITS
            },
            "leakage": {
                "source_group_overlap_count": 0,
                "image_sha256_overlap_count": 0,
            },
            "incremental_plan": {
                "baseline_manifest": str(baseline.path),
                "baseline_manifest_sha256": sha256_file(baseline.path),
                "candidate_manifest": str(candidate.path),
                "candidate_manifest_sha256": sha256_file(candidate.path),
                "existing_source_group_count": len(baseline.assignment),
                "new_source_groups": {
                    split: sorted(
                        group for group in new_groups if fixed_assignment[group] == split
                    )
                    for split in SPLITS
                },
                "seed": seed,
                "attempts": attempts,
                "minimum_support_groups_for_coverage": min_support_groups,
                "balanced_features": list(FEATURES),
            },
        },
        "records": [
            {
                "source_group": record["source_group"],
                "split": fixed_assignment[record["source_group"]],
                "image_sha256": image_hash,
                "trick_orientation": record["trick_orientation"],
                "yoyo_positive": record["yoyo_positive"],
                "string_positive": record["string_positive"],
                "string_visibility": record["string_visibility"],
            }
            for image_hash, record in sorted(candidate.records_by_hash.items())
        ],
    }
    return raw, fixed_assignment


def verify_manifests(
    baseline: ManifestView,
    rebuilt: ManifestView,
    mode: str,
    max_ratio_deviation: float,
    min_support_groups: int = len(SPLITS),
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
    regrouped_hashes: list[str] = []
    relabeled_hashes: list[str] = []
    for image_hash, old_record in baseline.records_by_hash.items():
        new_record = rebuilt.records_by_hash.get(image_hash)
        if new_record is None:
            missing_hashes.append(image_hash)
        else:
            if new_record["split"] != old_record["split"]:
                moved_hashes.append(image_hash)
            if new_record["source_group"] != old_record["source_group"]:
                regrouped_hashes.append(image_hash)
            if any(
                new_record[field] != old_record[field]
                for field in (
                    "trick_orientation",
                    "yoyo_positive",
                    "string_positive",
                    "string_visibility",
                )
            ):
                relabeled_hashes.append(image_hash)
    if missing_hashes:
        errors.append(f"{len(missing_hashes)} existing image hashes are missing")
    if moved_hashes:
        errors.append(f"{len(moved_hashes)} existing image hashes changed split")
    if regrouped_hashes:
        errors.append(f"{len(regrouped_hashes)} existing image hashes changed source group")
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

    label_balance = _label_balance_summary(rebuilt, min_support_groups)
    if mode == "append-isolated" and label_balance["coverage_gaps"]:
        detail = ", ".join(
            f"{gap['feature']} missing {gap['missing_splits']}"
            for gap in label_balance["coverage_gaps"]
        )
        errors.append(f"label coverage gate failed: {detail}")

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
            "regrouped_existing_hash_count": len(regrouped_hashes),
            "relabeled_existing_hash_count": len(relabeled_hashes),
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
        "label_balance": label_balance,
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


def _snapshot_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ContractError(f"snapshot source does not exist: {source}")
    if destination.exists():
        raise ContractError(f"snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)


def _is_nested(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _record_label_paths(
    manifest: ManifestView,
    dataset_root: Path,
    recorded_dataset_root: Path,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, raw_record in enumerate(manifest.raw["records"]):
        image_hash = str(raw_record.get("image_sha256", "")).strip()
        raw_label = str(raw_record.get("canonical_label", "")).strip()
        if not raw_label:
            raise ContractError(f"records[{index}] lacks canonical_label required for protected rebuild")
        label_path = Path(raw_label)
        if label_path.is_absolute():
            try:
                relative = label_path.resolve().relative_to(recorded_dataset_root)
            except ValueError as exc:
                raise ContractError(
                    f"records[{index}].canonical_label is outside the active dataset: {label_path}"
                ) from exc
        else:
            relative = label_path
        resolved = (dataset_root / relative).resolve()
        if not _is_nested(resolved, dataset_root):
            raise ContractError(f"records[{index}].canonical_label escapes the dataset")
        if not resolved.is_file():
            raise ContractError(f"canonical label does not exist: {resolved}")
        result[image_hash] = resolved
    return result


def _stable_label_value(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _present_non_task_fields(path: Path) -> set[str]:
    value = _read_json(path)
    return NON_TASK_FIELDS.intersection(value) | ({"dataset_management"} if "dataset_management" in value else set())


def _label_key(path: Path, dataset_root: Path) -> str:
    labels_root = (dataset_root / "canonical" / "labels").resolve()
    try:
        return path.resolve().relative_to(labels_root).as_posix()
    except ValueError as exc:
        raise ContractError(f"canonical label is outside canonical/labels: {path}") from exc


def _load_protected_reviews(
    review_map_path: Path,
    dataset_key: str,
    labels_by_hash: dict[str, Path],
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _read_json(review_map_path)
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ContractError(f"unsupported review map schema: {document.get('schema_version')!r}")
    datasets = document.get("datasets")
    if not isinstance(datasets, dict):
        raise ContractError("review map datasets must be an object")
    dataset = datasets.get(dataset_key)
    if dataset is None:
        return document, {}
    if not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), dict):
        raise ContractError(f"review map dataset entry is invalid: {dataset_key}")
    hashes_by_key = {
        _label_key(label_path, dataset_root): image_hash
        for image_hash, label_path in labels_by_hash.items()
    }
    reviews_by_hash: dict[str, dict[str, Any]] = {}
    for key, raw_review in dataset["samples"].items():
        if not isinstance(raw_review, dict):
            raise ContractError(f"review entry must be an object: {key}")
        image_hash = hashes_by_key.get(str(key))
        if image_hash is None:
            raise ContractError(f"review entry has no manifest record: {key}")
        label_path = labels_by_hash[image_hash]
        label_size_bytes, label_mtime_ns = file_revision(label_path)
        if (
            raw_review.get("label_size_bytes") != label_size_bytes
            or raw_review.get("label_mtime_ns") != label_mtime_ns
        ):
            raise ContractError(f"review entry is stale before rebuild: {key}")
        reviews_by_hash[image_hash] = dict(raw_review)
    return document, reviews_by_hash


def _verify_protected_labels(
    old_labels: dict[str, Path],
    new_labels: dict[str, Path],
) -> dict[str, int]:
    changed = [
        image_hash
        for image_hash, old_path in old_labels.items()
        if image_hash not in new_labels
        or _stable_label_value(old_path) != _stable_label_value(new_labels[image_hash])
    ]
    if changed:
        preview = ", ".join(changed[:5])
        raise ContractError(
            f"protected canonical label content changed for {len(changed)} existing images: {preview}"
        )
    residual = {
        image_hash: sorted(_present_non_task_fields(path))
        for image_hash, path in new_labels.items()
        if _present_non_task_fields(path)
    }
    if residual:
        preview = ", ".join(f"{key}:{'/'.join(value)}" for key, value in list(residual.items())[:5])
        raise ContractError(
            f"rebuilt canonical labels retain unsupported non-task fields for {len(residual)} images: {preview}"
        )
    return {
        "non_task_fields_removed_label_count": sum(
            bool(_present_non_task_fields(path)) for path in old_labels.values()
        ),
        "non_task_field_residual_count": 0,
    }


def _rebind_reviews(
    document: dict[str, Any],
    dataset_key: str,
    reviews_by_hash: dict[str, dict[str, Any]],
    new_labels: dict[str, Path],
    dataset_root: Path,
) -> dict[str, Any]:
    rebound: dict[str, dict[str, Any]] = {}
    for image_hash, review in reviews_by_hash.items():
        label_path = new_labels.get(image_hash)
        if label_path is None:
            raise ContractError(f"reviewed image is missing after rebuild: {image_hash}")
        updated = dict(review)
        updated["label_size_bytes"], updated["label_mtime_ns"] = file_revision(label_path)
        rebound[_label_key(label_path, dataset_root)] = updated
    datasets = document.setdefault("datasets", {})
    if reviews_by_hash:
        current = datasets.get(dataset_key)
        if not isinstance(current, dict):
            current = {}
        current["samples"] = rebound
        datasets[dataset_key] = current
    else:
        datasets.pop(dataset_key, None)
    return document


def _restore_dataset(active_root: Path, backup_root: Path) -> None:
    if active_root.exists():
        if active_root.is_dir():
            shutil.rmtree(active_root)
        else:
            active_root.unlink()
    os.replace(backup_root, active_root)


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


def _protected_command_args(
    values: Sequence[str],
    baseline_path: Path,
    protected_canonical: Path,
    allow_no_baseline_token: bool,
) -> list[str]:
    command = _command_args(values, baseline_path, allow_no_baseline_token)
    if not any(PROTECTED_CANONICAL_TOKEN in value for value in command):
        raise ContractError(
            f"protected builder command must contain {PROTECTED_CANONICAL_TOKEN} as a --source value"
        )
    return [
        value.replace(PROTECTED_CANONICAL_TOKEN, str(protected_canonical))
        for value in command
    ]


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
            min_support_groups=args.min_support_groups,
        )
    except ContractError as exc:
        result = {"ok": False, "mode": args.mode, "errors": [str(exc)]}
    if args.report:
        report_path = Path(args.report).resolve()
        result["report"] = str(report_path)
        _write_json(report_path, result)
    _print(result)
    return 0 if result["ok"] else 4


def run_plan(args: argparse.Namespace) -> int:
    try:
        baseline = load_manifest(args.baseline)
        candidate = load_manifest(args.candidate)
        output_path = Path(args.output).resolve()
        if output_path in {baseline.path, candidate.path}:
            raise ContractError("plan output must differ from baseline and candidate manifests")
        raw, _ = build_incremental_plan(
            baseline,
            candidate,
            seed=args.seed,
            attempts=args.attempts,
            min_support_groups=args.min_support_groups,
        )
        planned = _manifest_view(output_path, raw)
        result = verify_manifests(
            baseline,
            planned,
            mode="append-isolated",
            max_ratio_deviation=args.max_ratio_deviation,
            min_support_groups=args.min_support_groups,
        )
        result["candidate_manifest"] = str(candidate.path)
        result["candidate_manifest_sha256"] = sha256_file(candidate.path)
        if result["ok"]:
            _write_json(output_path, raw)
            result["plan_manifest"] = str(output_path)
            result["plan_manifest_sha256"] = sha256_file(output_path)
    except ContractError as exc:
        result = {"ok": False, "mode": "append-isolated", "errors": [str(exc)]}
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
        if any(str(record.get("canonical_label", "")).strip() for record in baseline.raw["records"]):
            raise ContractError(
                "active manifest contains canonical labels; use protected-run to preserve workbench state"
            )
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
            min_support_groups=args.min_support_groups,
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


def run_protected_build(args: argparse.Namespace) -> int:
    moved = False
    active_root: Path | None = None
    backup_root: Path | None = None
    preflight: dict[str, Any] = {"ok": False, "mode": args.mode}
    try:
        manifest_path = Path(args.manifest).resolve()
        baseline = load_manifest(manifest_path)
        active_root = manifest_path.parent
        backup_root = Path(args.backup_dir).resolve()
        review_map_path = Path(args.review_map).resolve()
        review_snapshot_path = Path(args.review_snapshot_out).resolve()
        if backup_root.exists():
            raise ContractError(f"dataset backup already exists: {backup_root}")
        if review_snapshot_path.exists():
            raise ContractError(f"review snapshot already exists: {review_snapshot_path}")
        if active_root.anchor.lower() != backup_root.anchor.lower():
            raise ContractError("dataset backup must be on the same filesystem as the active dataset")
        if _is_nested(backup_root, active_root) or _is_nested(active_root, backup_root):
            raise ContractError("dataset backup and active dataset must not contain each other")
        if _is_nested(review_snapshot_path, active_root) or _is_nested(review_snapshot_path, backup_root):
            raise ContractError("review snapshot must be outside the active dataset and dataset backup")
        if _is_nested(review_map_path, active_root):
            raise ContractError("review map must be outside the active dataset")
        old_labels = _record_label_paths(baseline, active_root, active_root)
        review_document, reviews_by_hash = _load_protected_reviews(
            review_map_path,
            args.review_dataset_key,
            old_labels,
            active_root,
        )
        protected_manifest = backup_root / manifest_path.name
        protected_canonical = backup_root / "canonical"
        command = _protected_command_args(
            args.command,
            protected_manifest,
            protected_canonical,
            args.allow_command_without_baseline,
        )
        cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd().resolve()
        if not cwd.is_dir():
            raise ContractError(f"working directory does not exist: {cwd}")
        preflight = {
            "ok": True,
            "dry_run": bool(args.dry_run),
            "protected": True,
            "mode": args.mode,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "active_dataset": str(active_root),
            "dataset_backup": str(backup_root),
            "review_map": str(review_map_path),
            "review_snapshot": str(review_snapshot_path),
            "review_dataset_key": args.review_dataset_key,
            "review_entry_count": len(reviews_by_hash),
            "protected_label_count": len(old_labels),
            "cwd": str(cwd),
            "command": command,
        }
        if args.dry_run:
            _print(preflight)
            return 0

        _snapshot_file(review_map_path, review_snapshot_path)
        backup_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active_root, backup_root)
        moved = True
        completed = subprocess.run(command, cwd=cwd, shell=False, check=False)
        if completed.returncode != 0:
            raise ContractError(f"external builder command failed with exit code {completed.returncode}")
        rebuilt = load_manifest(manifest_path)
        verification = verify_manifests(
            baseline,
            rebuilt,
            mode=args.mode,
            max_ratio_deviation=args.max_ratio_deviation,
            min_support_groups=args.min_support_groups,
        )
        if not verification["ok"]:
            raise ContractError("; ".join(verification["errors"]))
        old_labels = _record_label_paths(baseline, backup_root, active_root)
        new_labels = _record_label_paths(rebuilt, active_root, active_root)
        contract_cleanup = _verify_protected_labels(old_labels, new_labels)
        rebound_document = _rebind_reviews(
            review_document,
            args.review_dataset_key,
            reviews_by_hash,
            new_labels,
            active_root,
        )
        _write_json(review_map_path, rebound_document)
        result = {
            **preflight,
            **verification,
            "ok": True,
            "dry_run": False,
            "protected": True,
            "dataset_backup_retained": True,
            "protected_label_count": len(old_labels),
            **contract_cleanup,
            "review_entry_count_rebound": len(reviews_by_hash),
            "rebuilt_manifest_sha256": sha256_file(manifest_path),
        }
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        rollback_error = ""
        if moved and active_root is not None and backup_root is not None and backup_root.exists():
            try:
                _restore_dataset(active_root, backup_root)
            except OSError as rollback_exc:
                rollback_error = str(rollback_exc)
        errors = [str(exc)]
        if rollback_error:
            errors.append(f"automatic rollback failed: {rollback_error}")
        result = {
            **preflight,
            "ok": False,
            "dry_run": False,
            "protected": True,
            "rolled_back": bool(moved and not rollback_error),
            "errors": errors,
        }
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

    plan = subparsers.add_parser(
        "plan",
        help="Preserve old assignments and balance new source groups from a candidate manifest.",
    )
    plan.add_argument("--baseline", required=True)
    plan.add_argument("--candidate", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--attempts", type=int, default=10000)
    plan.add_argument("--max-ratio-deviation", type=float, default=0.20)
    plan.add_argument("--min-support-groups", type=int, default=len(SPLITS))
    plan.add_argument("--report", default="")
    plan.set_defaults(handler=run_plan)

    verify = subparsers.add_parser("verify", help="Compare an existing baseline and rebuild.")
    verify.add_argument("--baseline", required=True)
    verify.add_argument("--rebuilt", required=True)
    verify.add_argument("--mode", choices=MODES, default="append-isolated")
    verify.add_argument("--max-ratio-deviation", type=float, default=0.20)
    verify.add_argument("--min-support-groups", type=int, default=len(SPLITS))
    verify.add_argument("--report", default="")
    verify.set_defaults(handler=run_verify)

    run = subparsers.add_parser("run", help="Snapshot, invoke a builder, then verify.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--snapshot-out", required=True)
    run.add_argument("--report", default="")
    run.add_argument("--mode", choices=MODES, default="append-isolated")
    run.add_argument("--max-ratio-deviation", type=float, default=0.20)
    run.add_argument("--min-support-groups", type=int, default=len(SPLITS))
    run.add_argument("--cwd", default="")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--overwrite-snapshot", action="store_true")
    run.add_argument("--allow-command-without-baseline", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=run_build)

    protected = subparsers.add_parser(
        "protected-run",
        help="Rebuild transactionally while preserving canonical edits and review mappings.",
    )
    protected.add_argument("--manifest", required=True)
    protected.add_argument("--backup-dir", required=True)
    protected.add_argument("--review-map", required=True)
    protected.add_argument("--review-snapshot-out", required=True)
    protected.add_argument("--review-dataset-key", required=True)
    protected.add_argument("--report", default="")
    protected.add_argument("--mode", choices=MODES, default="append-isolated")
    protected.add_argument("--max-ratio-deviation", type=float, default=0.20)
    protected.add_argument("--min-support-groups", type=int, default=len(SPLITS))
    protected.add_argument("--cwd", default="")
    protected.add_argument("--dry-run", action="store_true")
    protected.add_argument("--allow-command-without-baseline", action="store_true")
    protected.add_argument("command", nargs=argparse.REMAINDER)
    protected.set_defaults(handler=run_protected_build)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= float(args.max_ratio_deviation) <= 1.0:
        print("max-ratio-deviation must be between 0 and 1", file=sys.stderr)
        return 2
    if hasattr(args, "attempts") and args.attempts < 1:
        print("attempts must be positive", file=sys.stderr)
        return 2
    if args.min_support_groups < len(SPLITS):
        print(f"min-support-groups must be at least {len(SPLITS)}", file=sys.stderr)
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
