import argparse
import logging
import urllib.request
from pathlib import Path

from config import BASE_DIR, YOLO_CONFIG


LOG_FILE = BASE_DIR / "download_yolo_model.log"
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
    parser = argparse.ArgumentParser(description="Download a YOLO11 model checkpoint into the configured models directory.")
    parser.add_argument("--model", default=YOLO_CONFIG.model_name, help="Model filename, e.g. yolo11n.pt.")
    parser.add_argument("--models-dir", default=str(YOLO_CONFIG.models_dir), help="Directory to save model checkpoints.")
    parser.add_argument("--url", default="", help="Explicit checkpoint URL. Overrides config yolo.model_url_template.")
    parser.add_argument("--force", action="store_true", help="Download even if the target file already exists.")
    return parser.parse_args()


def download_model(model_name: str, models_dir: Path, url: str = "", force: bool = False) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / model_name

    if target_path.exists() and not force:
        logger.info("Model already exists: %s", target_path)
        return target_path

    download_url = url or YOLO_CONFIG.model_url_template.format(model_name=model_name)
    logger.info("Downloading %s", download_url)
    logger.info("Saving to %s", target_path)
    urllib.request.urlretrieve(download_url, target_path)
    logger.info("Download complete: %s", target_path)
    return target_path


def main() -> int:
    args = parse_args()
    download_model(
        model_name=args.model,
        models_dir=Path(args.models_dir),
        url=args.url,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
