"""Export human-confirmed Workbench annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
APPROVAL_SCOPES = ("visible_geometry", "yoyo_bbox", "trick_orientation")
REVIEW_SCHEMA_VERSION = "yoyo_dataset_review_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_revision(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_image(canonical_root: Path, label_path: Path, annotation: dict[str, Any]) -> Path:
    source = str(annotation.get("source_image") or "").strip()
    if source:
        candidate = Path(source)
        candidates = [candidate] if candidate.is_absolute() else [label_path.parent / candidate, canonical_root / candidate]
        for value in candidates:
            resolved = value.resolve()
            if resolved.is_file():
                return resolved
    relative = label_path.relative_to(canonical_root / "labels")
    for suffix in IMAGE_SUFFIXES:
        candidate = canonical_root / "images" / relative.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"image not found for label: {label_path}")


def validate_sample_key(raw_key: str) -> Path:
    key = Path(raw_key.replace("\\", "/"))
    if key.is_absolute() or ".." in key.parts or key.suffix.lower() != ".json":
        raise ValueError(f"invalid review sample key: {raw_key!r}")
    return key


def approved_annotation(
    annotation: dict[str, Any],
    *,
    dataset_key: str,
    sample_key: str,
    review: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(annotation))
    confirmed_at = str(review.get("confirmed_at_utc") or datetime.now(timezone.utc).isoformat())
    reviewer = str(review.get("reviewer") or "workbench-reviewer")
    original_string_status = str(value.get("string_review_status") or "")
    original_string_visibility = str(value.get("string_visibility") or "")

    value["updated_at_utc"] = confirmed_at
    value["review_status"] = "reviewed"
    value["bbox_review_status"] = "reviewed"
    if original_string_status.lower() not in {"approved", "reviewed"}:
        value["string_review_status"] = "reviewed"

    quality = value.get("quality")
    if not isinstance(quality, dict):
        quality = {"revision": 0, "min_model_approvals": 1, "history": [], "reviews": []}
        value["quality"] = quality
    reviews = quality.get("reviews")
    if not isinstance(reviews, list):
        reviews = []
        quality["reviews"] = reviews
    reviews.append(
        {
            "reviewer_id": reviewer,
            "model": "human-workbench",
            "decision": "approve",
            "review_scope": list(APPROVAL_SCOPES),
            "notes": f"Imported from Workbench confirmation for {dataset_key}",
            "created_at_utc": confirmed_at,
        }
    )
    value["workbench_review_import"] = {
        "source_dataset": dataset_key,
        "source_sample_key": sample_key,
        "confirmed": True,
        "confirmed_at_utc": confirmed_at,
        "reviewer": reviewer,
        "original_string_visibility": original_string_visibility,
        "original_string_review_status": original_string_status,
    }
    return value


def export_reviewed(args: argparse.Namespace) -> dict[str, Any]:
    source_dataset = args.source_dataset.resolve()
    canonical_root = source_dataset / "canonical"
    labels_root = canonical_root / "labels"
    source_manifest = source_dataset / "manifest.json"
    if not labels_root.is_dir() or not source_manifest.is_file():
        raise FileNotFoundError(f"invalid source dataset: {source_dataset}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")

    review_map = read_json(args.review_map)
    if review_map.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"unsupported review map schema: {review_map.get('schema_version')!r}")
    datasets = review_map.get("datasets")
    if not isinstance(datasets, dict) or args.dataset_key not in datasets:
        raise KeyError(f"dataset not present in review map: {args.dataset_key}")
    dataset_reviews = datasets[args.dataset_key]
    samples = dataset_reviews.get("samples") if isinstance(dataset_reviews, dict) else None
    if not isinstance(samples, dict):
        raise ValueError(f"review map dataset has no samples object: {args.dataset_key}")

    all_labels = sorted(labels_root.rglob("*.json"))
    source_keys = {path.relative_to(labels_root).as_posix(): path for path in all_labels}
    unknown_review_keys = sorted(set(samples) - set(source_keys))
    if unknown_review_keys:
        raise ValueError(f"review map contains {len(unknown_review_keys)} missing source labels")

    confirmed = {key: value for key, value in samples.items() if isinstance(value, dict) and value.get("confirmed") is True}
    if not confirmed:
        raise ValueError("no confirmed review entries found")

    staging_parent = args.output.resolve().parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=staging_parent))
    selected: list[dict[str, Any]] = []
    try:
        for raw_key in sorted(confirmed):
            relative = validate_sample_key(raw_key)
            normalized_key = relative.as_posix()
            label_path = source_keys.get(normalized_key)
            if label_path is None:
                raise FileNotFoundError(f"confirmed label not found: {normalized_key}")
            review = confirmed[raw_key]
            label_size_bytes, label_mtime_ns = file_revision(label_path)
            if (
                review.get("label_size_bytes") != label_size_bytes
                or review.get("label_mtime_ns") != label_mtime_ns
            ):
                raise ValueError(f"review label revision mismatch: {normalized_key}")

            annotation = read_json(label_path)
            image_path = resolve_image(canonical_root, label_path, annotation)
            image_digest = sha256_file(image_path)
            declared_image_digest = str(annotation.get("image_sha256") or "").lower()
            if image_digest != declared_image_digest:
                raise ValueError(f"annotation image SHA mismatch: {normalized_key}")

            target_image = staging / "images" / relative.with_suffix(image_path.suffix.lower())
            target_label = staging / "labels" / relative
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(image_path, target_image)
                image_materialization = "hardlink"
            except OSError:
                shutil.copy2(image_path, target_image)
                image_materialization = "copy"

            exported = approved_annotation(
                annotation,
                dataset_key=args.dataset_key,
                sample_key=normalized_key,
                review=review,
            )
            exported["source_image"] = (Path("images") / relative.with_suffix(image_path.suffix.lower())).as_posix()
            write_json(target_label, exported)
            selected.append(
                {
                    "source_sample_key": normalized_key,
                    "source_group": str(annotation.get("source_group") or ""),
                    "image_sha256": image_digest,
                    "exported_label": str((args.output / "labels" / relative).resolve()),
                    "image_materialization": image_materialization,
                    "reviewer": str(review.get("reviewer") or "workbench-reviewer"),
                    "confirmed_at_utc": str(review.get("confirmed_at_utc") or ""),
                    "original_string_visibility": str(annotation.get("string_visibility") or ""),
                    "original_string_review_status": str(annotation.get("string_review_status") or ""),
                    "exported_string_review_status": str(exported.get("string_review_status") or ""),
                }
            )
        staging.replace(args.output.resolve())
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    excluded = [
        {"source_sample_key": key, "reason": "not_confirmed_in_review_map"}
        for key in sorted(set(source_keys) - set(confirmed))
    ]
    report = {
        "schema_version": "reviewed_annotation_selection_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset),
        "source_dataset_key": args.dataset_key,
        "export_root": str(args.output.resolve()),
        "source_sample_count": len(source_keys),
        "confirmed_review_count": len(selected),
        "excluded_unreviewed_count": len(excluded),
        "selected": selected,
        "excluded": excluded,
    }
    write_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--review-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_reviewed(args)
    print(json.dumps({key: report[key] for key in ("source_sample_count", "confirmed_review_count", "excluded_unreviewed_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
