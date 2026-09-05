"""ROI orientation training entry point."""

from yoyo_orientation._training import main, train_orientation

__all__ = ["main", "train_orientation"]

if __name__ == "__main__":
    raise SystemExit(main())
