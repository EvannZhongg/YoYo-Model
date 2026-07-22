import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from config import BASE_DIR, DATASET_CONFIG, YOLO_CONFIG


LOG_FILE = BASE_DIR / "prepare_yolo_dataset.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert auto-annotation JSON files into an Ultralytics YOLO dataset.")
    parser.add_argument("--annotations-dir", default=str(DATASET_CONFIG.annotation_output_dir), help="Annotation root directory.")
    parser.add_argument("--output-dir", default=str(YOLO_CONFIG.dataset_dir), help="YOLO dataset output directory.")
    parser.add_argument("--train-split", type=float, default=YOLO_CONFIG.train_split, help="Train split ratio.")
    parser.add_argument("--allow-image-level-split", action="store_true", help="Allow random image-level split; unsafe for video frames.")
    parser.add_argument("--include-unreviewed", action="store_true", help="Include auto/unreviewed labels. Not recommended.")
    parser.add_argument("--seed", type=int, default=YOLO_CONFIG.seed, help="Shuffle seed.")
    parser.add_argument("--clear", action="store_true", help="Clear output directory before generating the dataset.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_for_annotation(annotation: dict[str, Any], label_path: Path, annotations_dir: Path) -> Path | None:
    for field in ("saved_image", "source_image"):
        value = annotation.get(field)
        if value:
            path = Path(value)
            if path.exists():
                return path

    rel_stem = label_path.relative_to(annotations_dir / "labels").with_suffix("")
    images_dir = annotations_dir / "images"
    for ext in DATASET_CONFIG.image_extensions:
        candidate = images_dir / rel_stem.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


def bbox_to_yolo_line(bbox: dict[str, Any], image_width: int, image_height: int, class_index: int) -> str | None:
    coords = bbox.get("bbox_pixel")
    if not coords and bbox.get("bbox_2d"):
        x1n, y1n, x2n, y2n = [float(value) for value in bbox["bbox_2d"]]
        coords = [
            x1n / 999.0 * image_width,
            y1n / 999.0 * image_height,
            x2n / 999.0 * image_width,
            y2n / 999.0 * image_height,
        ]

    if not coords or len(coords) != 4:
        return None

    x1, y1, x2, y2 = [float(value) for value in coords]
    x1 = max(0.0, min(x1, image_width))
    x2 = max(0.0, min(x2, image_width))
    y1 = max(0.0, min(y1, image_height))
    y2 = max(0.0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def class_index_for_bbox(bbox: dict[str, Any], class_names: tuple[str, ...]) -> int | None:
    label = str(bbox.get("label", "")).strip().lower()
    normalized_names = [name.lower() for name in class_names]

    if label in normalized_names:
        return normalized_names.index(label)
    if len(class_names) == 1:
        return 0
    return None


def source_group(annotation: dict[str, Any], label_path: Path) -> str:
    for field in ("source_group", "source_video", "video_id", "subject_id"):
        value = annotation.get(field)
        if value:
            return str(value)
    provenance = annotation.get("provenance")
    if isinstance(provenance, dict):
        for field in ("source_group", "source_video", "video_id", "subject_id"):
            if provenance.get(field):
                return str(provenance[field])
    # Legacy still images have no video provenance; keep each image isolated.
    return f"legacy:{label_path.as_posix()}"


def split_items(items: list[tuple[Path, Path, dict[str, Any]]], train_split: float, seed: int, group_by_source: bool = True):
    explicit = [str(item[2].get("split", "")).lower() for item in items]
    if items and all(value in {"train", "val", "test"} for value in explicit):
        split_map = {name: [item for item in items if str(item[2].get("split")).lower() == name] for name in ("train", "val", "test")}
        groups = {name: sorted({source_group(item[2], item[0]) for item in values}) for name, values in split_map.items()}
        return split_map["train"], split_map["val"], split_map["test"], groups
    rng = random.Random(seed)
    if not group_by_source:
        shuffled = list(items)
        rng.shuffle(shuffled)
        split_index = int(len(shuffled) * train_split)
        if len(shuffled) > 1:
            split_index = max(1, min(split_index, len(shuffled) - 1))
        return shuffled[:split_index], shuffled[split_index:], [], {"train": ["legacy:image_level"], "val": ["legacy:image_level"], "test": []}

    groups: dict[str, list[tuple[Path, Path, dict[str, Any]]]] = defaultdict(list)
    for item in items:
        groups[source_group(item[2], item[0])].append(item)
    group_names = list(groups)
    rng.shuffle(group_names)
    target = max(1, int(round(len(items) * train_split))) if len(items) > 1 else len(items)
    train_items: list[tuple[Path, Path, dict[str, Any]]] = []
    train_groups: list[str] = []
    val_items: list[tuple[Path, Path, dict[str, Any]]] = []
    val_groups: list[str] = []
    for name in group_names:
        group_items = groups[name]
        if train_items and len(train_items) + len(group_items) > target:
            val_items.extend(group_items)
            val_groups.append(name)
        else:
            train_items.extend(group_items)
            train_groups.append(name)
    if not val_items and len(group_names) > 1:
        moved = train_groups.pop()
        val_items.extend(groups[moved])
        train_items = [item for item in train_items if source_group(item[2], item[0]) != moved]
        val_groups.append(moved)
    split_groups = {"train": train_groups, "val": val_groups}
    return train_items, val_items, [], {"train": train_groups, "val": val_groups, "test": []}


def write_data_yaml(output_dir: Path, class_names: tuple[str, ...], has_test: bool = False) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": list(class_names),
    }
    if has_test:
        data["test"] = "images/test"
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return data_yaml_path


def prepare_yolo_dataset(
    annotations_dir: Path,
    output_dir: Path,
    train_split: float,
    seed: int,
    clear: bool = False,
    group_by_source: bool = True,
    include_unreviewed: bool = False,
) -> dict[str, Any]:
    labels_dir = annotations_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Annotation labels directory does not exist: {labels_dir}")

    if clear and output_dir.exists():
        shutil.rmtree(output_dir)

    label_paths = sorted(labels_dir.rglob("*.json"))
    items = []
    skipped = []

    for label_path in label_paths:
        annotation = load_json(label_path)
        review_status = str(annotation.get("bbox_review_status", annotation.get("review_status", "approved"))).lower()
        if not include_unreviewed and review_status not in {"approved", "reviewed"}:
            skipped.append((label_path, f"bbox_review_status={review_status}"))
            continue
        image_path = find_image_for_annotation(annotation, label_path, annotations_dir)
        if image_path is None:
            skipped.append((label_path, "image not found"))
            continue
        items.append((label_path, image_path, annotation))

    train_items, val_items, test_items, split_groups = split_items(items, train_split, seed, group_by_source=group_by_source)
    split_map = {"train": train_items, "val": val_items, "test": test_items}
    written_images = 0
    written_boxes = 0

    for split_name, split_items_list in split_map.items():
        image_out_dir = output_dir / "images" / split_name
        label_out_dir = output_dir / "labels" / split_name
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for label_path, image_path, annotation in split_items_list:
            rel_stem = label_path.relative_to(labels_dir).with_suffix("")
            image_out_path = image_out_dir / rel_stem.with_suffix(image_path.suffix)
            label_out_path = label_out_dir / rel_stem.with_suffix(".txt")
            image_out_path.parent.mkdir(parents=True, exist_ok=True)
            label_out_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, image_out_path)

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            yolo_lines = []
            for bbox in annotation.get("bbox", []):
                class_index = class_index_for_bbox(bbox, YOLO_CONFIG.class_names)
                if class_index is None:
                    continue
                line = bbox_to_yolo_line(bbox, image_width, image_height, class_index)
                if line:
                    yolo_lines.append(line)

            label_out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            written_images += 1
            written_boxes += len(yolo_lines)

    data_yaml_path = write_data_yaml(output_dir, YOLO_CONFIG.class_names, has_test=bool(test_items))
    manifest = {
        "schema_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "data_yaml": str(data_yaml_path),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "written_images": written_images,
        "written_boxes": written_boxes,
        "skipped": [{"label": str(path), "reason": reason} for path, reason in skipped],
        "split_strategy": "source_group" if group_by_source else "image_level_legacy",
        "source_groups": split_groups,
        "class_names": list(YOLO_CONFIG.class_names),
        "seed": seed,
        "train_split": train_split,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare_yolo_dataset(
        annotations_dir=Path(args.annotations_dir),
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        seed=args.seed,
        clear=args.clear,
        group_by_source=not args.allow_image_level_split,
        include_unreviewed=args.include_unreviewed,
    )
    logger.info("YOLO dataset prepared: %s", manifest["output_dir"])
    logger.info("data.yaml: %s", manifest["data_yaml"])
    logger.info("train=%s val=%s boxes=%s skipped=%s", manifest["train_count"], manifest["val_count"], manifest["written_boxes"], len(manifest["skipped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
