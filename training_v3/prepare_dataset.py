"""Build aligned yoyo detection, string segmentation, and orientation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image

from common.files import sha256_file
from config import BASE_DIR


SCHEMA_VERSION = "yoyo_multitask_dataset_v6"
ANNOTATION_SCHEMA_VERSION = "agent_yoyo_string_annotation_v5"
SOURCE_POLICY = "quality_approved; image_sha256_deduplicated; source_group_isolated; annotation_schema_v5"
VALID_ORIENTATIONS = ("normal", "horizontal", "not_applicable")
VALID_STRING_VISIBILITY = {"visible", "partial", "not_visible"}
SPLITS = ("train", "val", "test")
EXCLUDED_ANNOTATION_DIRS = {"score_annotations"}


@dataclass(frozen=True)
class Sample:
    dataset: str
    label_path: Path
    image_path: Path
    image_sha256: str
    annotation: dict[str, Any]

    @property
    def source_group(self) -> str:
        return str(self.annotation["source_group"])

    @property
    def orientation(self) -> str:
        return str(self.annotation["trick_orientation"])

    @property
    def has_yoyo(self) -> bool:
        return _valid_bbox(self.annotation.get("yoyo_bbox_pixel")) is not None

    @property
    def has_string(self) -> bool:
        return str(self.annotation.get("string_visibility")) in {"visible", "partial"}


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _approved_scopes(annotation: dict[str, Any]) -> set[str]:
    scopes: set[str] = set()
    for review in (annotation.get("quality") or {}).get("reviews") or []:
        if str(review.get("decision", "")).lower() == "approve":
            scopes.update(str(value) for value in review.get("review_scope") or [])
    return scopes


def _resolve_image(label_path: Path, dataset_root: Path, annotation: dict[str, Any]) -> Path | None:
    source = str(annotation.get("source_image") or "").strip()
    if source:
        candidate = Path(source)
        candidates = [candidate] if candidate.is_absolute() else [label_path.parent / candidate, dataset_root / candidate]
        for value in candidates:
            resolved = value.resolve()
            if resolved.is_file():
                return resolved
    relative = label_path.relative_to(dataset_root / "labels")
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = dataset_root / "images" / relative.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_samples(source_roots: Iterable[Path]) -> tuple[list[Sample], list[dict[str, str]]]:
    samples_by_hash: dict[str, Sample] = {}
    excluded: list[dict[str, str]] = []
    for root in source_roots:
        root = root.resolve()
        if root.name == "video_v1":
            raise ValueError("video_v1 is explicitly forbidden in the fresh training pipeline")
        labels_root = root / "labels"
        if not labels_root.is_dir():
            raise FileNotFoundError(f"labels directory not found: {labels_root}")
        for label_path in sorted(labels_root.rglob("*.json")):
            try:
                annotation = json.loads(label_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                excluded.append({"label": str(label_path), "reason": f"invalid_json: {exc}"})
                continue
            group = str(annotation.get("source_group") or "").strip()
            orientation = str(annotation.get("trick_orientation") or "").strip()
            string_visibility = str(annotation.get("string_visibility") or "").strip()
            scopes = _approved_scopes(annotation)
            reason = ""
            if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
                reason = f"unsupported_schema_version={annotation.get('schema_version')}"
            elif "dataset_management" in annotation:
                reason = "unsupported_dataset_management_field"
            elif not group:
                reason = "missing_source_group"
            elif orientation not in VALID_ORIENTATIONS:
                reason = f"invalid_trick_orientation={orientation}"
            elif str(annotation.get("string_review_status", "")).lower() not in {"approved", "reviewed"}:
                reason = f"string_review_status={annotation.get('string_review_status')}"
            elif string_visibility not in VALID_STRING_VISIBILITY:
                reason = f"invalid_string_visibility={string_visibility}"
            elif not {"visible_geometry", "yoyo_bbox"}.issubset(scopes):
                reason = "missing_geometry_or_yoyo_quality_approval"
            image_path = _resolve_image(label_path, root, annotation)
            if not reason and image_path is None:
                reason = "source_image_missing"
            if reason:
                excluded.append({"label": str(label_path), "reason": reason})
                continue
            assert image_path is not None
            image_digest = sha256_file(image_path)
            declared_digest = str(annotation.get("image_sha256") or "").strip()
            if declared_digest and declared_digest != image_digest:
                excluded.append({"label": str(label_path), "reason": "image_sha256_mismatch"})
                continue
            sample = Sample(root.name, label_path.resolve(), image_path, image_digest, annotation)
            previous = samples_by_hash.get(image_digest)
            if previous is not None:
                previous_rank = (str(previous.annotation.get("updated_at_utc") or ""), str(previous.label_path))
                sample_rank = (str(sample.annotation.get("updated_at_utc") or ""), str(sample.label_path))
                if sample_rank <= previous_rank:
                    excluded.append({"label": str(label_path), "reason": f"duplicate_image_superseded_by={previous.label_path}"})
                    continue
                excluded.append({"label": str(previous.label_path), "reason": f"duplicate_image_replaced_by={label_path}"})
            samples_by_hash[image_digest] = sample
    return sorted(samples_by_hash.values(), key=lambda item: (item.source_group, item.dataset, str(item.label_path))), excluded


def discover_annotation_sources(annotations_dir: Path = BASE_DIR / "annotations") -> list[Path]:
    """Return every direct annotation export containing labels, excluding task-specific stores."""
    root = annotations_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"annotations directory not found: {root}")
    return sorted(
        (path.resolve() for path in root.iterdir() if path.is_dir() and path.name not in EXCLUDED_ANNOTATION_DIRS and (path / "labels").is_dir()),
        key=lambda path: path.name,
    )


def _split_quotas(group_count: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if group_count < 3:
        raise ValueError("At least three source groups are required for train/val/test isolation")
    val = max(1, int(round(group_count * val_ratio)))
    test = max(1, int(round(group_count * test_ratio)))
    if val + test >= group_count:
        overflow = val + test - group_count + 1
        if val >= test:
            val -= overflow
        else:
            test -= overflow
    return {"train": group_count - val - test, "val": val, "test": test}


def _group_features(samples: list[Sample]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        values = result[sample.source_group]
        values["samples"] += 1
        values[f"orientation:{sample.orientation}"] += 1
        values["yoyo_positive"] += int(sample.has_yoyo)
        values["yoyo_negative"] += int(not sample.has_yoyo)
        values["string_positive"] += int(sample.has_string)
        values["string_negative"] += int(not sample.has_string)
        values[f"string_visibility:{sample.annotation['string_visibility']}"] += 1
    return dict(result)


def _assignment_score(
    assignment: dict[str, str],
    features: dict[str, Counter[str]],
    ratios: dict[str, float],
) -> float:
    totals = sum(features.values(), Counter())
    keys = [
        "samples", "yoyo_positive", "yoyo_negative", "string_positive", "string_negative",
        *(f"orientation:{value}" for value in VALID_ORIENTATIONS),
        *(f"string_visibility:{value}" for value in sorted(VALID_STRING_VISIBILITY)),
    ]
    split_counts = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        split_counts[split].update(features[group])
    score = 0.0
    for split in SPLITS:
        for key in keys:
            target = totals[key] * ratios[split]
            weight = 8.0 if key == "samples" else 1.0
            score += weight * ((split_counts[split][key] - target) ** 2) / max(1.0, target)
        for key in keys[1:]:
            supporting_groups = sum(bool(values[key]) for values in features.values())
            if supporting_groups >= len(SPLITS) and not split_counts[split][key]:
                score += 100.0
    return score


def assign_source_splits(
    samples: list[Sample],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    attempts: int = 6000,
    frozen_assignment: dict[str, str] | None = None,
) -> dict[str, str]:
    features = _group_features(samples)
    groups = sorted(features)
    _split_quotas(len(groups), val_ratio, test_ratio)
    if frozen_assignment is not None:
        invalid = sorted(group for group, split in frozen_assignment.items() if split not in SPLITS)
        if invalid:
            raise ValueError(f"Frozen split assignment contains invalid split values: {invalid}")
        missing = sorted(set(frozen_assignment) - set(groups))
        if missing:
            raise ValueError(f"Frozen split source groups are missing from the current annotations: {missing}")
        assignment = {
            group: frozen_assignment.get(group, "train")
            for group in groups
        }
        if set(assignment.values()) != set(SPLITS):
            raise ValueError("Frozen split assignment must preserve non-empty train, val, and test splits")
        return assignment
    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...], dict[str, str]] | None = None
    for _ in range(max(1, attempts)):
        candidate_slots = rng.choices(SPLITS, weights=[ratios[split] for split in SPLITS], k=len(groups))
        if set(candidate_slots) != set(SPLITS):
            continue
        assignment = dict(zip(groups, candidate_slots, strict=True))
        score = _assignment_score(assignment, features, ratios)
        signature = tuple(assignment[group] for group in groups)
        item = (score, signature, assignment)
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None
    return best[2]


def _load_frozen_split_assignment(manifest_path: Path) -> tuple[dict[str, str], str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_groups = (manifest.get("split_policy") or {}).get("source_groups")
    if not isinstance(source_groups, dict):
        raise ValueError(f"Frozen split manifest does not define split_policy.source_groups: {manifest_path}")
    assignment: dict[str, str] = {}
    duplicates: set[str] = set()
    for split in SPLITS:
        values = source_groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Frozen split manifest requires a non-empty {split} source group list")
        for raw_group in values:
            group = str(raw_group)
            if group in assignment:
                duplicates.add(group)
            assignment[group] = split
    if duplicates:
        raise ValueError(f"Frozen split manifest assigns source groups more than once: {sorted(duplicates)}")
    return assignment, sha256_file(manifest_path)


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip("-_") or "unknown"


def _relative_stem(sample: Sample) -> Path:
    stem = _safe_name(sample.image_path.stem)
    digest_suffix = f"-{sample.image_sha256[:8]}"
    while stem.lower().endswith(digest_suffix):
        stem = stem[: -len(digest_suffix)]
    return Path(_safe_name(sample.source_group)) / f"{stem}{digest_suffix}"


def _bbox_line(sample: Sample) -> str:
    bbox = _valid_bbox(sample.annotation.get("yoyo_bbox_pixel"))
    if bbox is None:
        return ""
    width, height = (int(value) for value in sample.annotation["image_size"])
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if x2 <= x1 or y2 <= y1:
        return ""
    return f"0 {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n"


def _normalized_polygons(annotation: dict[str, Any], line_width_px: int) -> list[list[tuple[float, float]]]:
    width, height = (int(value) for value in annotation["image_size"])
    polygons: list[list[tuple[float, float]]] = []
    raw_masks = annotation.get("string_mask_polygons_pixel") or []
    for polygon in raw_masks:
        points = [(max(0.0, min(float(x), width - 1)) / width, max(0.0, min(float(y), height - 1)) / height) for x, y in polygon]
        if len(points) >= 3:
            polygons.append(points)
    if polygons:
        return polygons
    adaptive_width = max(int(line_width_px), int(round(np.hypot(width, height) * 0.0015)))
    for polyline in annotation.get("string_polylines_pixel") or []:
        if len(polyline) < 2:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.asarray(polyline, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.polylines(mask, [points.round().astype(np.int32)], False, 255, adaptive_width, cv2.LINE_8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = max(0.5, adaptive_width * 0.12)
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) >= 3:
                polygons.append([(float(x) / width, float(y) / height) for x, y in simplified])
    return polygons


def _string_lines(sample: Sample, line_width_px: int) -> str:
    visibility = str(sample.annotation["string_visibility"])
    if visibility == "not_visible":
        return ""
    polygons = _normalized_polygons(sample.annotation, line_width_px)
    if not polygons:
        raise ValueError(f"Reviewed {visibility} string has no usable geometry: {sample.label_path}")
    return "".join(f"0 {' '.join(f'{value:.6f}' for point in polygon for value in point)}\n" for polygon in polygons)


def _write_yaml(task_root: Path, name: str) -> Path:
    path = task_root / "data.yaml"
    data = {
        "path": str(task_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": [name],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _annotation_digest(samples: list[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.dataset.encode("utf-8"))
        digest.update(sample.label_path.read_bytes())
        digest.update(sample.image_sha256.encode("ascii"))
    return digest.hexdigest()


def build_training_dataset(
    source_roots: Iterable[Path],
    output_dir: Path,
    seed: int = 20260726,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    line_width_px: int = 8,
    clear: bool = False,
    freeze_splits_from: Path | None = None,
) -> dict[str, Any]:
    roots = sorted({Path(value).resolve() for value in source_roots}, key=lambda path: path.name)
    if not roots:
        raise ValueError("At least one annotation source is required")
    forbidden = [root for root in roots if root.name in EXCLUDED_ANNOTATION_DIRS]
    if forbidden:
        raise ValueError(f"Task-specific annotation stores cannot be training sources: {forbidden}")
    missing_labels = [root for root in roots if not (root / "labels").is_dir()]
    if missing_labels:
        raise FileNotFoundError(f"labels directory not found for annotation sources: {missing_labels}")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    samples, excluded = _load_samples(roots)
    if not samples:
        raise ValueError("No quality-approved samples were found")
    frozen_assignment: dict[str, str] | None = None
    frozen_manifest_sha256 = ""
    if freeze_splits_from is not None:
        frozen_assignment, frozen_manifest_sha256 = _load_frozen_split_assignment(Path(freeze_splits_from))
    assignment = assign_source_splits(
        samples,
        seed,
        val_ratio,
        test_ratio,
        frozen_assignment=frozen_assignment,
    )
    new_train_source_groups = sorted(set(assignment) - set(frozen_assignment or {}))
    annotation_sha256 = _annotation_digest(samples)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "sources": [root.name for root in roots],
        "annotation_sha256": annotation_sha256,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "line_width_px": line_width_px,
        "assignment": assignment,
    }
    dataset_id = f"yoyo_unified_{hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:12]}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not clear:
            raise FileExistsError(f"Output already exists; pass --clear to rebuild: {output_dir}")
        shutil.rmtree(output_dir)
    canonical_root = output_dir / "canonical"
    detection_root = output_dir / "detection"
    string_root = output_dir / "string_segmentation"
    orientation_root = output_dir / "orientation"
    transfer_modes: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    overall_distribution: Counter[str] = Counter()
    source_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict[str, Any]] = []
    orientation_train_images: dict[str, list[Path]] = defaultdict(list)
    for sample in samples:
        split = assignment[sample.source_group]
        relative = _relative_stem(sample)
        extension = sample.image_path.suffix.lower()
        canonical_image = canonical_root / "images" / relative.with_suffix(extension)
        canonical_label = canonical_root / "labels" / relative.with_suffix(".json")
        detection_image = detection_root / "images" / split / relative.with_suffix(extension)
        string_image = string_root / "images" / split / relative.with_suffix(extension)
        orientation_name = f"{_safe_name(sample.source_group)}__{relative.name}{extension}"
        orientation_image = orientation_root / split / sample.orientation / orientation_name
        transfer_modes[f"canonical_{_link_or_copy(sample.image_path, canonical_image)}"] += 1
        transfer_modes[f"task_{_link_or_copy(canonical_image, detection_image)}"] += 1
        transfer_modes[f"task_{_link_or_copy(canonical_image, string_image)}"] += 1
        transfer_modes[f"task_{_link_or_copy(canonical_image, orientation_image)}"] += 1
        if split == "train":
            orientation_train_images[sample.orientation].append(orientation_image)
        canonical_label.parent.mkdir(parents=True, exist_ok=True)
        canonical_annotation = dict(sample.annotation)
        canonical_label.write_text(json.dumps(canonical_annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        detection_label = detection_root / "labels" / split / relative.with_suffix(".txt")
        string_label = string_root / "labels" / split / relative.with_suffix(".txt")
        detection_label.parent.mkdir(parents=True, exist_ok=True)
        string_label.parent.mkdir(parents=True, exist_ok=True)
        detection_label.write_text(_bbox_line(sample), encoding="utf-8")
        string_label.write_text(_string_lines(sample, line_width_px), encoding="utf-8")
        counts[split]["samples"] += 1
        counts[split][f"orientation:{sample.orientation}"] += 1
        counts[split]["yoyo_positive"] += int(sample.has_yoyo)
        counts[split]["yoyo_negative"] += int(not sample.has_yoyo)
        counts[split]["string_positive"] += int(sample.has_string)
        counts[split]["string_negative"] += int(not sample.has_string)
        counts[split][f"string_visibility:{sample.annotation['string_visibility']}"] += 1
        counts[split][f"source_dataset:{sample.dataset}"] += 1
        distribution_values = {
            "samples": 1,
            f"orientation:{sample.orientation}": 1,
            "yoyo_positive": int(sample.has_yoyo),
            "yoyo_negative": int(not sample.has_yoyo),
            "string_positive": int(sample.has_string),
            "string_negative": int(not sample.has_string),
            f"string_visibility:{sample.annotation['string_visibility']}": 1,
        }
        overall_distribution.update(distribution_values)
        source_distribution[sample.dataset].update(distribution_values)
        records.append(
            {
                "dataset": sample.dataset,
                "label": str(sample.label_path),
                "image": str(sample.image_path),
                "image_sha256": sample.image_sha256,
                "canonical_image": str(canonical_image),
                "canonical_label": str(canonical_label),
                "source_group": sample.source_group,
                "split": split,
                "trick_orientation": sample.orientation,
                "yoyo_positive": sample.has_yoyo,
                "string_positive": sample.has_string,
                "string_visibility": str(sample.annotation["string_visibility"]),
            }
        )
    orientation_train_counts = {name: len(paths) for name, paths in sorted(orientation_train_images.items())}
    orientation_balance_target = max(orientation_train_counts.values())
    orientation_repeat_count = 0
    for orientation, paths in sorted(orientation_train_images.items()):
        for repeat_index in range(orientation_balance_target - len(paths)):
            source = paths[repeat_index % len(paths)]
            target = source.with_name(f"{source.stem}__repeat_{repeat_index + 1:03d}{source.suffix}")
            transfer_modes[f"orientation_balance_{_link_or_copy(source, target)}"] += 1
            orientation_repeat_count += 1
    detection_yaml = _write_yaml(detection_root, "yoyo")
    string_yaml = _write_yaml(string_root, "string")
    source_groups = {split: sorted(group for group, value in assignment.items() if value == split) for split in SPLITS}
    included_by_source = Counter(sample.dataset for sample in samples)
    excluded_by_source = Counter()
    for item in excluded:
        label = Path(item["label"])
        for root in roots:
            if label.is_relative_to(root):
                excluded_by_source[root.name] += 1
                break
    source_inventory = {
        root.name: {
            "root": str(root),
            "labels_discovered": len(list((root / "labels").rglob("*.json"))),
            "samples_included": included_by_source[root.name],
            "records_excluded": excluded_by_source[root.name],
        }
        for root in roots
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": SOURCE_POLICY,
        "import_sources": [str(root) for root in roots],
        "source_inventory": source_inventory,
        "source_annotation_sha256": annotation_sha256,
        "output_dir": str(output_dir),
        "split_policy": {
            "strategy": (
                "frozen_source_groups_new_sources_train"
                if frozen_assignment is not None
                else "source_group_stratified_random_search"
            ),
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "target_sample_ratios": {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio},
            "actual_sample_ratios": {
                split: counts[split]["samples"] / len(samples) for split in SPLITS
            },
            "source_groups": source_groups,
            "frozen_from_manifest": str(Path(freeze_splits_from).resolve()) if freeze_splits_from is not None else "",
            "frozen_from_manifest_sha256": frozen_manifest_sha256,
            "frozen_source_group_count": len(frozen_assignment or {}),
            "new_train_source_groups": new_train_source_groups,
            "leakage": {
                "source_group_overlap_count": 0,
                "image_sha256_overlap_count": 0,
                "guarantee": "each source_group and deduplicated image_sha256 belongs to exactly one split",
            },
        },
        "tasks": {
            "detection": {"data": str(detection_yaml), "classes": ["yoyo"]},
            "string_segmentation": {"data": str(string_yaml), "classes": ["string"]},
            "orientation": {
                "data": str(orientation_root),
                "classes": list(VALID_ORIENTATIONS),
                "train_balance": {
                    "strategy": "repeat_minority_samples_with_runtime_augmentation",
                    "original_counts": orientation_train_counts,
                    "target_per_class": orientation_balance_target,
                    "repeated_image_count": orientation_repeat_count,
                    "balanced_train_count": orientation_balance_target * len(VALID_ORIENTATIONS),
                    "val_and_test_unchanged": True,
                },
            },
        },
        "counts": {split: dict(sorted(value.items())) for split, value in counts.items()},
        "distributions": {
            "overall": dict(sorted(overall_distribution.items())),
            "by_source_dataset": {
                source: dict(sorted(values.items())) for source, values in sorted(source_distribution.items())
            },
        },
        "sample_count": len(samples),
        "source_group_count": len(assignment),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "image_materialization": dict(transfer_modes),
        "records": records,
        "label_semantics": {
            "detection": "quality-approved yoyo_bbox_pixel; empty label is a reviewed negative",
            "string_segmentation": "approved masks or buffered visible centerlines; not_visible is a reviewed negative",
            "orientation": "three-way trick_orientation including not_applicable",
        },
        "task_input_dependencies": {
            "detection": ["image", "yoyo_bbox_pixel"],
            "string_segmentation": ["image", "string_mask_or_polyline"],
            "orientation": ["image", "yoyo_bbox_pixel_or_fixed_negative_crop", "trick_orientation"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (canonical_root / "index.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    from training_v3.semantic_view import write_semantic_view_manifest

    write_semantic_view_manifest(output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified multitask dataset from all non-score annotation exports.")
    parser.add_argument("--source", action="append", default=[], help="Annotation root; repeat to override automatic annotations/ discovery.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "datasets" / "1Ayoyo_dataset"))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--line-width-px", type=int, default=8)
    parser.add_argument("--clear", action="store_true")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--freeze-splits-from",
        default="",
        help="Preserve source-group splits from this dataset manifest; unseen groups are assigned to train.",
    )
    split_group.add_argument(
        "--resplit",
        action="store_true",
        help="Explicitly allow a fresh split instead of preserving an existing output manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [Path(value) for value in args.source] or discover_annotation_sources()
    output_dir = Path(args.output_dir)
    existing_manifest = output_dir / "manifest.json"
    freeze_splits_from = (
        Path(args.freeze_splits_from)
        if args.freeze_splits_from
        else existing_manifest if args.clear and existing_manifest.is_file() and not args.resplit
        else None
    )
    manifest = build_training_dataset(
        sources,
        output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        line_width_px=args.line_width_px,
        clear=args.clear,
        freeze_splits_from=freeze_splits_from,
    )
    print(json.dumps({key: manifest[key] for key in ("dataset_id", "sample_count", "source_group_count", "counts", "output_dir")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
