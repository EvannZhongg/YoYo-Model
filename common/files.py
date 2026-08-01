"""File-system helpers shared across dataset, training, and review flows."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Collection
from pathlib import Path


def collect_files(root: Path, extensions: Collection[str], *, recursive: bool = True) -> list[Path]:
    """Return files under ``root`` whose lowercase suffix is allowed."""
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    allowed = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in allowed
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file digest without loading the complete artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    """Write text through a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
