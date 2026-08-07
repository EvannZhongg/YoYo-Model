"""File-system helpers shared across dataset, training, and review flows."""

from __future__ import annotations

import hashlib
import os
import secrets
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
    descriptor: int | None = None
    try:
        mode = 0o666 if os.name == "nt" else 0o600
        for _ in range(128):
            candidate = path.parent / f".{path.stem}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
                temporary_path = candidate
                break
            except FileExistsError:
                continue
        if descriptor is None or temporary_path is None:
            raise FileExistsError(f"could not allocate a temporary file beside {path}")
        with os.fdopen(descriptor, mode="w", encoding=encoding) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
