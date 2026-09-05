"""String recognition training entry point."""

from string_segmentation.train_semantic import main, parse_args, train

__all__ = ["main", "parse_args", "train"]

if __name__ == "__main__":
    raise SystemExit(main())
