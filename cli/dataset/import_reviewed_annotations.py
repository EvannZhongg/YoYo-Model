"""Import reviewed-selection entries into a rebuilt dataset review map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_reviews(args: argparse.Namespace) -> dict[str, Any]:
    review_map_path = args.review_map.resolve()
    manifest_path = args.manifest.resolve()
    selection_path = args.selection_report.resolve()
    snapshot_path = args.snapshot.resolve()
    report_path = args.report.resolve()
    dataset_root = manifest_path.parent
    labels_root = (dataset_root / "canonical" / "labels").resolve()
    if snapshot_path.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot_path}")
    if report_path.exists():
        raise FileExistsError(f"report already exists: {report_path}")

    review_document = read_json(review_map_path)
    manifest = read_json(manifest_path)
    selection = read_json(selection_path)
    selected = selection.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection report has no selected records")

    records_by_hash: dict[str, dict[str, Any]] = {}
    for record in manifest.get("records") or []:
        image_hash = str(record.get("image_sha256") or "").lower()
        if not image_hash or image_hash in records_by_hash:
            raise ValueError("rebuilt manifest has missing or duplicate image hashes")
        records_by_hash[image_hash] = record

    datasets = review_document.setdefault("datasets", {})
    dataset = datasets.setdefault(args.dataset_key, {})
    samples = dataset.setdefault("samples", {})
    if not isinstance(samples, dict):
        raise ValueError(f"review map samples is not an object: {args.dataset_key}")
    review_count_before = len(samples)

    manifest_keys: dict[str, str] = {}
    for image_hash, record in records_by_hash.items():
        label_path = Path(str(record.get("canonical_label") or "")).resolve()
        try:
            key = label_path.relative_to(labels_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"canonical label is outside rebuilt dataset: {label_path}") from exc
        if not label_path.is_file():
            raise FileNotFoundError(f"canonical label not found: {label_path}")
        manifest_keys[key] = image_hash

    for key, review in samples.items():
        image_hash = manifest_keys.get(str(key))
        if image_hash is None:
            raise ValueError(f"existing review entry has no rebuilt manifest record: {key}")
        label_path = Path(str(records_by_hash[image_hash]["canonical_label"]))
        if sha256_file(label_path) != str(review.get("label_sha256") or "").lower():
            raise ValueError(f"existing review entry is stale: {key}")

    selection_sha = sha256_file(selection_path)
    imported: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in selected:
        image_hash = str(item.get("image_sha256") or "").lower()
        if not image_hash or image_hash in seen_hashes:
            raise ValueError("selection report has missing or duplicate image hashes")
        seen_hashes.add(image_hash)
        record = records_by_hash.get(image_hash)
        if record is None:
            raise ValueError(f"selected image is missing from rebuilt manifest: {image_hash}")
        label_path = Path(str(record["canonical_label"])).resolve()
        key = label_path.relative_to(labels_root).as_posix()
        if key in samples:
            raise ValueError(f"selected review would overwrite an existing review entry: {key}")
        annotation = read_json(label_path)
        review_import = annotation.get("workbench_review_import")
        source_label_sha = str(item.get("source_label_sha256") or "").lower()
        if not isinstance(review_import, dict) or str(review_import.get("source_label_sha256") or "").lower() != source_label_sha:
            raise ValueError(f"selected provenance mismatch in rebuilt label: {key}")
        target_label_sha = sha256_file(label_path)
        samples[key] = {
            "confirmed": True,
            "confirmed_at_utc": str(item.get("confirmed_at_utc") or ""),
            "reviewer": str(item.get("reviewer") or "workbench-reviewer"),
            "label_sha256": target_label_sha,
            "source_dataset": str(selection.get("source_dataset") or ""),
            "source_sample_key": str(item.get("source_sample_key") or ""),
            "source_label_sha256": source_label_sha,
            "selection_report_sha256": selection_sha,
        }
        imported.append(
            {
                "image_sha256": image_hash,
                "split": str(record.get("split") or ""),
                "source_group": str(record.get("source_group") or ""),
                "target_key": key,
                "target_label_sha256": target_label_sha,
                "source_sample_key": str(item.get("source_sample_key") or ""),
                "source_label_sha256": source_label_sha,
                "reviewer": str(item.get("reviewer") or "workbench-reviewer"),
                "confirmed_at_utc": str(item.get("confirmed_at_utc") or ""),
            }
        )

    review_document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_map_path, snapshot_path)
    before_sha = sha256_file(review_map_path)
    write_json_atomic(review_map_path, review_document)
    result = {
        "schema_version": "yoyo_review_import_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_map": str(review_map_path),
        "review_snapshot": str(snapshot_path),
        "review_map_before_sha256": before_sha,
        "review_map_after_sha256": sha256_file(review_map_path),
        "selection_report": str(selection_path),
        "selection_report_sha256": selection_sha,
        "rebuilt_manifest": str(manifest_path),
        "rebuilt_manifest_sha256": sha256_file(manifest_path),
        "review_count_before": review_count_before,
        "imported_review_count": len(imported),
        "review_count_after": len(samples),
        "imported": imported,
    }
    write_json_atomic(report_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-map", required=True, type=Path)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    result = import_reviews(parse_args())
    print(json.dumps({key: result[key] for key in ("review_count_before", "imported_review_count", "review_count_after")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
