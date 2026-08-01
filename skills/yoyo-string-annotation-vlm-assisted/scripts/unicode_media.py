"""Unicode-safe OpenCV image I/O for Windows paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def imread(path: str | Path, flags: int | None = None) -> Any:
    """Read an image without passing a Unicode path through cv2.imread."""
    import cv2
    import numpy as np

    source = Path(path)
    raw = np.fromfile(source, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"image file is empty: {source}")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR if flags is None else flags)
    if image is None:
        raise ValueError(f"OpenCV could not decode image bytes: {source}")
    return image


def imwrite(path: str | Path, image: Any, params: list[int] | None = None) -> Path:
    """Write an image without passing a Unicode path through cv2.imwrite."""
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image, params or [])
    if not ok:
        raise ValueError(f"OpenCV could not encode image as {extension}: {target}")
    encoded.tofile(target)
    return target
