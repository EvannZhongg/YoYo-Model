"""Build a crop-focused orientation view without mutating the canonical dataset."""

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


def _points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    bbox = annotation.get("yoyo_bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        result.extend([(x1, y1), (x2, y2)])
    for value in (annotation.get("hands_pixel") or {}).values():
        if isinstance(value, list) and len(value) == 2:
            result.append((float(value[0]), float(value[1])))
    for polyline in annotation.get("string_polylines_pixel") or []:
        for value in polyline:
            if isinstance(value, list) and len(value) == 2:
                result.append((float(value[0]), float(value[1])))
    return result


def _crop_box(annotation: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    points = _points(annotation)
    if not points:
        return 0, 0, width, height
    xs, ys = zip(*points)
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    side = min(float(min(width, height)), max(span * 1.6, min(width, height) * 0.28))
    left = max(0.0, min(center_x - side / 2.0, width - side))
    top = max(0.0, min(center_y - side / 2.0, height - side))
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def _link(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_orientation_view(dataset_dir: Path, clear: bool = False) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    parent_path = dataset_dir / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    output = dataset_dir / "orientation_roi"
    if output.exists():
        if not clear:
            raise FileExistsError(f"Orientation ROI view already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    train_paths: dict[str, list[Path]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for record in parent["records"]:
        annotation = json.loads(Path(record["canonical_label"]).read_text(encoding="utf-8"))
        source = Path(record["canonical_image"])
        split = str(record["split"])
        orientation = str(record["trick_orientation"])
        name = f"{record['source_group']}__{source.stem}.jpg"
        target = output / split / orientation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            crop_box = _crop_box(annotation, image.width, image.height)
            image.crop(crop_box).save(target, format="JPEG", quality=94, optimize=True)
        counts[split][orientation] += 1
        counts[split]["total"] += 1
        if split == "train":
            train_paths[orientation].append(target)
        records.append(
            {
                "source_group": record["source_group"],
                "split": split,
                "trick_orientation": orientation,
                "source_image_sha256": record["image_sha256"],
                "crop_box_pixel": list(crop_box),
                "image": str(target),
                "image_sha256": sha256_file(target),
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
        "schema": "yoyo_orientation_roi_view_v1",
        "parent_dataset_id": parent["dataset_id"],
        "parent_manifest_sha256": sha256_file(parent_path),
        "crop_policy": "square_union_hands_yoyo_string_1p6_min_28pct",
    }
    view_id = f"orientation_roi_{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]}"
    manifest = {
        "schema_version": "yoyo_orientation_roi_view_v1",
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
        "classes": ["horizontal", "normal", "not_applicable"],
        "crop_policy": identity["crop_policy"],
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
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "yoyo_dataset"))
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    manifest = build_orientation_view(Path(args.dataset_dir), args.clear)
    print(json.dumps({"view_id": manifest["view_id"], "counts": manifest["counts"], "train_balance": manifest["train_balance"], "data": manifest["data"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
