"""Compatibility wrapper for ``python -m video_dataset.audit``."""

from video_dataset.audit import main


if __name__ == "__main__":
    raise SystemExit(main())
