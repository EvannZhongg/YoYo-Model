"""Three-class trick-orientation view entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.orientation import TRICK_ORIENTATION_CLASS_ORDER
from training_v3.orientation_view import _build_orientation_view


def build_orientation_view(
    dataset_dir: Path,
    clear: bool = False,
    include_backup_yoyos_train: bool = False,
    output_name: str = "orientation_roi",
) -> dict[str, Any]:
    return _build_orientation_view(
        dataset_dir,
        clear,
        include_backup_yoyos_train,
        output_name,
        TRICK_ORIENTATION_CLASS_ORDER,
    )

