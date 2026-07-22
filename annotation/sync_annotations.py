import argparse
import logging
from pathlib import Path

from annotation.annotator import annotation_output_paths
from config import BASE_DIR, DATASET_CONFIG


LOG_FILE = BASE_DIR / "sync_annotations.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset image directory does not exist: {input_dir}")

    pattern = "**/*" if DATASET_CONFIG.recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in DATASET_CONFIG.image_extensions
    )


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return

    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def key_for_label(path: Path, labels_dir: Path) -> str:
    return path.relative_to(labels_dir).with_suffix("").as_posix()


def key_for_image(path: Path, images_dir: Path) -> str:
    return path.relative_to(images_dir).with_suffix("").as_posix()


def key_for_visualization(path: Path, visualizations_dir: Path) -> str:
    rel_path = path.relative_to(visualizations_dir)
    stem = rel_path.stem
    if stem.endswith("_vis"):
        stem = stem[:-4]
    return rel_path.with_name(stem).with_suffix("").as_posix()


def index_output_files(output_dir: Path) -> dict[str, dict[str, list[Path]]]:
    groups: dict[str, dict[str, list[Path]]] = {}
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    visualizations_dir = output_dir / "visualizations"

    for path in labels_dir.rglob("*.json") if labels_dir.exists() else []:
        key = key_for_label(path, labels_dir)
        groups.setdefault(key, {}).setdefault("label", []).append(path)

    for path in images_dir.rglob("*") if images_dir.exists() else []:
        if path.is_file() and path.suffix.lower() in DATASET_CONFIG.image_extensions:
            key = key_for_image(path, images_dir)
            groups.setdefault(key, {}).setdefault("image", []).append(path)

    for path in visualizations_dir.rglob("*_vis.png") if visualizations_dir.exists() else []:
        key = key_for_visualization(path, visualizations_dir)
        groups.setdefault(key, {}).setdefault("visualization", []).append(path)

    return groups


def expected_source_keys(input_dir: Path, output_dir: Path) -> set[str]:
    keys = set()
    for image_path in collect_images(input_dir):
        paths = annotation_output_paths(image_path, input_dir, output_dir)
        keys.add(paths["label"].relative_to(output_dir / "labels").with_suffix("").as_posix())
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize annotation outputs after manual review. "
            "Any annotation group missing labels/images/visualizations will be removed together."
        )
    )
    parser.add_argument("--input-dir", default=str(DATASET_CONFIG.image_input_dir), help="Original image directory.")
    parser.add_argument("--output-dir", default=str(DATASET_CONFIG.annotation_output_dir), help="Annotation output directory.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this flag, only prints a dry-run.")
    parser.add_argument("--prune-empty-dirs", action="store_true", help="Remove empty annotation subdirectories after deletion.")
    parser.add_argument(
        "--delete-orphans",
        action="store_true",
        help="Also delete complete annotation groups that are absent from the current input directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    groups = index_output_files(output_dir)
    source_keys = expected_source_keys(input_dir, output_dir)

    required_parts = ["label", "visualization"]
    if DATASET_CONFIG.keep_source_images:
        required_parts.append("image")

    delete_candidates: list[Path] = []
    incomplete_count = 0

    for key in sorted(groups):
        group = groups[key]
        missing_parts = [part for part in required_parts if not group.get(part)]
        is_orphan = key not in source_keys

        if not missing_parts and (not args.delete_orphans or not is_orphan):
            continue

        incomplete_count += 1
        existing_paths = [path for paths in group.values() for path in paths]
        delete_candidates.extend(existing_paths)

        reason_parts = []
        if missing_parts:
            reason_parts.append(f"missing: {', '.join(missing_parts)}")
        if is_orphan and args.delete_orphans:
            reason_parts.append("source image not found in current input-dir")

        logger.info("Incomplete annotation group: %s (%s)", key, "; ".join(reason_parts))
        for part in required_parts:
            paths = group.get(part, [])
            if paths:
                for path in paths:
                    logger.info("  exists  %-13s %s", part, path)
            else:
                logger.info("  missing %-13s", part)

    unique_delete_candidates = sorted(set(delete_candidates))

    if not unique_delete_candidates:
        logger.info("No incomplete annotation groups found. Nothing to delete.")
        return 0

    logger.info("Found %s incomplete/orphan annotation group(s).", incomplete_count)
    logger.info("%s file(s) will be deleted%s:", len(unique_delete_candidates), "" if args.apply else " (dry-run)")
    for path in unique_delete_candidates:
        logger.info("  %s", path)

    if not args.apply:
        logger.info("Dry-run only. Re-run with --apply to delete these files.")
        return 0

    for path in unique_delete_candidates:
        path.unlink(missing_ok=True)

    if args.prune_empty_dirs:
        remove_empty_dirs(output_dir / "images")
        remove_empty_dirs(output_dir / "labels")
        remove_empty_dirs(output_dir / "visualizations")

    logger.info("Deleted %s file(s).", len(unique_delete_candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
