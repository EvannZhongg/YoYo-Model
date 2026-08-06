"""Download RTMPose-m WholeBody model files directly into the project."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from common.files import sha256_file
from config import BASE_DIR


DETECTOR_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip"


def _download_onnx(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_value:
        temp = Path(temp_value)
        archive = temp / "model.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            members = [member for member in bundle.namelist() if member.lower().endswith(".onnx")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one ONNX model in {url}; found {members}")
            extracted = Path(bundle.extract(members[0], temp)).resolve()
        shutil.move(str(extracted), destination)


def download_models(output_dir: Path, force: bool = False) -> dict[str, object]:
    output_dir = output_dir.resolve()
    models = {
        "detector": (DETECTOR_URL, output_dir / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"),
        "pose": (POSE_URL, output_dir / "rtmpose-m-wholebody-256x192.onnx"),
    }
    records = {}
    for name, (url, path) in models.items():
        if force or not path.is_file():
            _download_onnx(url, path)
        records[name] = {"url": url, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "model_family": "RTMPose-m WholeBody",
        "keypoint_schema": "COCO-WholeBody 133",
        "storage_policy": "project_local_only",
        "models": records,
    }
    (output_dir / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "models" / "rtmpose"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_models(Path(args.output_dir), args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
