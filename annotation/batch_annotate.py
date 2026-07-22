import argparse
import logging
from pathlib import Path

from annotation.annotator import annotate_image_for_dataset, has_complete_annotation
from annotation.prompts import YOYO_DETECTION_PROMPT
from config import BASE_DIR, DATASET_CONFIG, MODEL_CONFIG


LOG_FILE = BASE_DIR / "batch_annotate.log"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch annotate yoyo images with the configured vision model.")
    parser.add_argument("--input-dir", default=str(DATASET_CONFIG.image_input_dir), help="Image directory to annotate.")
    parser.add_argument("--output-dir", default=str(DATASET_CONFIG.annotation_output_dir), help="Annotation output directory.")
    parser.add_argument("--model", default=MODEL_CONFIG.default_model, help="Model name.")
    parser.add_argument("--min-image-tokens", default=MODEL_CONFIG.min_image_tokens, help="Minimum image token count.")
    parser.add_argument("--max-image-tokens", default=MODEL_CONFIG.max_image_tokens, help="Maximum image token count.")
    parser.add_argument("--limit", type=int, default=0, help="Annotate only the first N images. 0 means no limit.")
    parser.add_argument("--force", action="store_true", help="Re-annotate images even if complete outputs already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_paths = collect_images(input_dir)

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        logger.warning("No images found in %s", input_dir)
        return 0

    total_input_count = len(image_paths)
    if args.force:
        skipped_count = 0
    else:
        pending_paths = []
        skipped_count = 0
        for image_path in image_paths:
            if has_complete_annotation(image_path, input_dir, output_dir):
                skipped_count += 1
            else:
                pending_paths.append(image_path)
        image_paths = pending_paths

    logger.info("Found %s image(s)", total_input_count)
    logger.info("Skipped already annotated image(s): %s", skipped_count)
    logger.info("Pending image(s): %s", len(image_paths))
    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Model: %s", args.model)
    logger.info("Image transport: %s", MODEL_CONFIG.image_transport)

    if not image_paths:
        logger.info("Nothing to annotate. Use --force to re-run completed images.")
        return 0

    success_count = 0
    failed_count = 0
    total_boxes = 0

    for index, image_path in enumerate(image_paths, start=1):
        logger.info("[%s/%s] Annotating %s", index, len(image_paths), image_path)
        try:
            result = annotate_image_for_dataset(
                image_path=image_path,
                input_dir=input_dir,
                output_dir=output_dir,
                prompt=YOYO_DETECTION_PROMPT,
                model=args.model,
                min_pixels_str=args.min_image_tokens,
                max_pixels_str=args.max_image_tokens,
            )
        except Exception:
            failed_count += 1
            logger.exception("[%s/%s] Failed: %s", index, len(image_paths), image_path)
            continue

        success_count += 1
        total_boxes += result["bbox_count"]
        logger.info(
            "[%s/%s] Saved %s bbox(es): %s",
            index,
            len(image_paths),
            result["bbox_count"],
            result["label_path"],
        )

    logger.info(
        "Done. Success: %s, Failed: %s, Total boxes: %s, Output: %s",
        success_count,
        failed_count,
        total_boxes,
        output_dir,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
