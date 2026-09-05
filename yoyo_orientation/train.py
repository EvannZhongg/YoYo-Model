"""ROI orientation training entry point."""

from training_v3.train_orientation import main, train_orientation

__all__ = ["main", "train_orientation"]

if __name__ == "__main__":
    raise SystemExit(main())
