"""Write the compatibility manifest for binary semantic string training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.files import sha256_file
from config import BASE_DIR


def write_semantic_view_manifest(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    root_manifest_path = dataset_dir / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    string_root = dataset_dir / "string_segmentation"
    if not (string_root / "data.yaml").is_file():
        raise FileNotFoundError(f"String training view is missing: {string_root}")
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        source = root_manifest["counts"][split]
        labels = list((string_root / "labels" / split).rglob("*.txt"))
        instance_count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in labels)
        positive = int(source["string_positive"])
        total = int(source["samples"])
        counts[split] = {
            "total": total,
            "positive": positive,
            "negative": total - positive,
            "instances": instance_count,
        }
    manifest = {
        "schema_version": "yoyo_semantic_string_view_v1",
        "task": "binary_semantic_segmentation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": root_manifest["dataset_id"],
        "parent_manifest": str(root_manifest_path),
        "parent_manifest_sha256": sha256_file(root_manifest_path),
        "output_dir": str(string_root),
        "data_yaml": str((string_root / "data.yaml").resolve()),
        "counts": counts,
        "source_groups": root_manifest["split_policy"]["source_groups"],
        "source_policy": root_manifest["source_policy"],
        "label_semantics": root_manifest["label_semantics"]["string_segmentation"],
    }
    (string_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the unified dataset for semantic string training.")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "1Ayoyo_dataset"))
    args = parser.parse_args()
    manifest = write_semantic_view_manifest(Path(args.dataset_dir))
    print(json.dumps({"dataset_id": manifest["dataset_id"], "counts": manifest["counts"], "output_dir": manifest["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
