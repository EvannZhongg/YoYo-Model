"""Build a yoyo orientation view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from common.files import sha256_file
from config import BASE_DIR


PRESENTATION_ORIENTATIONS = ("frontal", "edge_horizontal", "edge_vertical", "unknown")
COARSE_ORIENTATIONS = ("horizontal", "normal", "not_applicable")
PRESENTATION_TO_TRICK = {
    "frontal": "normal",
    "edge_vertical": "normal",
    "edge_horizontal": "horizontal",
    "unknown": "not_applicable",
}


def _yoyo_bbox(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    active = annotation.get("active_yoyo") or {"bbox_pixel": annotation.get("yoyo_bbox_pixel")}
    bbox = active.get("bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return None


def _crop_box(
    annotation: dict[str, Any],
    width: int,
    height: int,
    bbox_override: tuple[float, float, float, float] | None = None,
) -> tuple[int, int, int, int]:
    bbox = bbox_override or _yoyo_bbox(annotation)
    if bbox is None:
        # Reviewed not-applicable frames may not contain a visible yoyo.
        side = float(min(width, height)) * 0.28
        center_x, center_y = width / 2.0, height / 2.0
    else:
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        span = max(x2 - x1, y2 - y1)
        side = min(float(min(width, height)), max(span * 3.0, min(width, height) * 0.12))
    left = max(0.0, min(center_x - side / 2.0, width - side))
    top = max(0.0, min(center_y - side / 2.0, height - side))
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def _link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _training_class(presentation_orientation: str, coarse_classes: bool) -> str:
    if not coarse_classes:
        return presentation_orientation
    return PRESENTATION_TO_TRICK[presentation_orientation]


def build_orientation_view(
    dataset_dir: Path,
    clear: bool = False,
    include_backup_yoyos_train: bool = False,
    output_name: str = "orientation_roi",
    coarse_classes: bool = False,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    parent_path = dataset_dir / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    output = dataset_dir / output_name
    if output.exists():
        if not clear:
            raise FileExistsError(f"Orientation ROI view already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    yoyo_visibility_counts: Counter[str] = Counter()
    train_paths: dict[str, list[Path]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for record in parent["records"]:
        if bool(record.get("yoyo_ignored", False)):
            continue
        annotation = json.loads(Path(record["canonical_label"]).read_text(encoding="utf-8"))
        active_yoyo = annotation.get("active_yoyo") or {
            "visibility": annotation.get("visibility"),
            "trick_orientation": record.get("trick_orientation") or annotation.get("trick_orientation"),
            "presentation_orientation": annotation.get("presentation_orientation"),
        }
        source = Path(record["canonical_image"])
        split = str(record["split"])
        annotation_presentation = str(active_yoyo.get("presentation_orientation") or "").strip()
        orientation = annotation_presentation or {
            "normal": "frontal",
            "horizontal": "edge_horizontal",
            "not_applicable": "unknown",
        }.get(str(active_yoyo.get("trick_orientation") or ""), "unknown")
        if orientation not in PRESENTATION_ORIENTATIONS:
            raise ValueError(f"invalid presentation orientation: {orientation}")
        yoyo_visibility_counts[str(active_yoyo.get("visibility") or "uncertain")] += 1
        name = f"{record['source_group']}__{source.stem}.jpg"
        training_class = _training_class(orientation, coarse_classes)
        target = output / split / training_class / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            crop_box = _crop_box(annotation, image.width, image.height)
            image.crop(crop_box).save(target, format="JPEG", quality=94, optimize=True)
        counts[split][training_class] += 1
        counts[split]["total"] += 1
        if split == "train":
            train_paths[training_class].append(target)
        records.append(
            {
                "source_group": record["source_group"],
                "split": split,
                "yoyo_role": "active",
                "training_class": training_class,
                "trick_orientation": str(active_yoyo.get("trick_orientation") or ""),
                "presentation_orientation": orientation,
                "source_image_sha256": record["image_sha256"],
                "crop_box_pixel": list(crop_box),
                "image": str(target),
                "image_sha256": sha256_file(target),
            }
        )
        if include_backup_yoyos_train and split == "train":
            for backup_index, backup in enumerate(annotation.get("backup_yoyos") or []):
                if str(backup.get("visibility") or "") not in {"visible", "partial"}:
                    continue
                backup_box = backup.get("bbox_pixel")
                if not isinstance(backup_box, list) or len(backup_box) != 4:
                    continue
                try:
                    backup_bbox = tuple(float(value) for value in backup_box)
                except (TypeError, ValueError):
                    continue
                if not (backup_bbox[2] > backup_bbox[0] and backup_bbox[3] > backup_bbox[1]):
                    continue
                backup_orientation = str(backup.get("presentation_orientation") or "").strip()
                if not backup_orientation:
                    backup_orientation = {
                        "normal": "frontal",
                        "horizontal": "edge_horizontal",
                        "not_applicable": "unknown",
                    }.get(str(backup.get("trick_orientation") or ""), "unknown")
                if backup_orientation not in PRESENTATION_ORIENTATIONS:
                    raise ValueError(f"invalid backup presentation orientation: {backup_orientation}")
                backup_name = f"{record['source_group']}__{source.stem}__backup_{backup_index + 1:02d}.jpg"
                backup_class = _training_class(backup_orientation, coarse_classes)
                backup_target = output / split / backup_class / backup_name
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    backup_crop_box = _crop_box(annotation, image.width, image.height, backup_bbox)
                    image.crop(backup_crop_box).save(backup_target, format="JPEG", quality=94, optimize=True)
                counts[split][backup_class] += 1
                counts[split]["total"] += 1
                train_paths[backup_class].append(backup_target)
                records.append(
                    {
                        "source_group": record["source_group"],
                        "split": split,
                        "yoyo_role": "backup",
                        "backup_index": backup_index,
                        "training_class": backup_class,
                        "trick_orientation": str(backup.get("trick_orientation") or ""),
                        "presentation_orientation": backup_orientation,
                        "source_image_sha256": record["image_sha256"],
                        "crop_box_pixel": list(backup_crop_box),
                        "image": str(backup_target),
                        "image_sha256": sha256_file(backup_target),
                    }
                )
    target_per_class = max(len(paths) for paths in train_paths.values())
    repeated = 0
    for orientation, paths in sorted(train_paths.items()):
        for index in range(target_per_class - len(paths)):
            source = paths[index % len(paths)]
            target = source.with_name(f"{source.stem}__repeat_{index + 1:03d}{source.suffix}")
            _link(source, target)
            repeated += 1
    identity = {
        "schema": "yoyo_orientation_roi_view_v2",
        "parent_dataset_id": parent["dataset_id"],
        "parent_manifest_sha256": sha256_file(parent_path),
        "crop_policy": "yoyo_bbox_square_3p0_min_12pct; no_yoyo_center_square_28pct",
        "include_backup_yoyos_train": bool(include_backup_yoyos_train),
        "coarse_classes": bool(coarse_classes),
    }
    view_id = f"orientation_roi_{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]}"
    manifest = {
        "schema_version": "yoyo_orientation_roi_view_v2",
        "task": "orientation",
        "view_id": view_id,
        "dataset_id": parent["dataset_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(output),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": sha256_file(parent_path),
        "source_policy": parent["source_policy"],
        "source_groups": parent["split_policy"]["source_groups"],
        "counts": {split: dict(values) for split, values in counts.items()},
        "classes": list(COARSE_ORIENTATIONS if coarse_classes else PRESENTATION_ORIENTATIONS),
        "label_field": "active_yoyo.presentation_orientation",
        "coarse_mapping": dict(PRESENTATION_TO_TRICK),
        "crop_policy": identity["crop_policy"],
        "include_backup_yoyos_train": bool(include_backup_yoyos_train),
        "coarse_classes": bool(coarse_classes),
        "input_dependencies": {
            "active_yoyo_bbox_pixel": True,
            "string_geometry": False,
            "no_yoyo_policy": "deterministic_center_crop",
        },
        "yoyo_visibility_counts": dict(yoyo_visibility_counts),
        "train_balance": {
            "original_counts": {name: len(paths) for name, paths in sorted(train_paths.items())},
            "target_per_class": target_per_class,
            "repeated_image_count": repeated,
            "val_and_test_unchanged": True,
        },
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the trick-region orientation classification view.")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "1Ayoyo_dataset"))
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--include-backup-yoyos-train", action="store_true")
    parser.add_argument("--output-name", default="orientation_roi")
    parser.add_argument("--coarse-classes", action="store_true")
    args = parser.parse_args()
    manifest = build_orientation_view(
        Path(args.dataset_dir),
        args.clear,
        args.include_backup_yoyos_train,
        args.output_name,
        args.coarse_classes,
    )
    print(json.dumps({"view_id": manifest["view_id"], "counts": manifest["counts"], "train_balance": manifest["train_balance"], "data": manifest["data"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
