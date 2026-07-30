"""Build a validation-calibrated semantic checkpoint by interpolating compatible models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from common.files import sha256_file
from config import BASE_DIR
from string_segmentation.device import resolve_device
from string_segmentation.semantic_metrics import collect_probabilities, select_threshold
from string_segmentation.semantic_model import ReviewedStringDataset, build_string_model, save_checkpoint


def interpolate_state_dicts(
    baseline: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Return `(1 - alpha) * baseline + alpha * candidate` for floating tensors."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if baseline.keys() != candidate.keys():
        raise ValueError("checkpoint state_dict keys do not match")
    result: dict[str, torch.Tensor] = {}
    for name, left in baseline.items():
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        if torch.is_floating_point(left):
            result[name] = torch.lerp(left.float(), right.float(), float(alpha)).to(left.dtype)
        else:
            result[name] = right.clone() if alpha >= 0.5 else left.clone()
    return result


def _parent_lineage(weights: Path, evaluation_groups: dict[str, set[str]]) -> dict[str, Any]:
    manifest_path = weights.parent.parent / "run_manifest.json"
    if not manifest_path.is_file():
        return {
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "run_manifest": "",
            "evaluation_source_overlap": {"val": [], "test": []},
            "promotion_eligible": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_groups = set((manifest.get("source_groups") or {}).get("train", []))
    overlap = {
        split: sorted(train_groups & evaluation_groups[split])
        for split in ("val", "test")
    }
    return {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "run_manifest": str(manifest_path.resolve()),
        "evaluation_source_overlap": overlap,
        "promotion_eligible": not any(overlap.values()),
    }


def build_model_soup(
    baseline_weights: Path,
    candidate_weights: Path,
    dataset_dir: Path,
    project_dir: Path,
    name: str,
    alpha: float,
    device_value: str,
) -> dict[str, Any]:
    baseline_weights = baseline_weights.resolve()
    candidate_weights = candidate_weights.resolve()
    dataset_dir = dataset_dir.resolve()
    run_dir = (project_dir / name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Model-soup run already exists: {run_dir}")

    baseline = torch.load(baseline_weights, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_weights, map_location="cpu", weights_only=True)
    baseline_config = dict(baseline.get("model_config") or {})
    candidate_config = dict(candidate.get("model_config") or {})
    if baseline.get("format") != candidate.get("format") or baseline_config != candidate_config:
        raise ValueError("semantic checkpoints use incompatible formats or model configurations")

    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    source_groups = {
        split: set(values)
        for split, values in (dataset_manifest.get("source_groups") or {}).items()
    }
    if not all(split in source_groups for split in ("train", "val", "test")):
        raise ValueError("semantic dataset manifest is missing source groups")
    parents = [
        _parent_lineage(baseline_weights, source_groups),
        _parent_lineage(candidate_weights, source_groups),
    ]
    promotion_eligible = all(parent["promotion_eligible"] for parent in parents)

    device = resolve_device(device_value)
    model = build_string_model(
        architecture=str(baseline_config.get("architecture", "tiny_unet")),
        base_channels=int(baseline_config.get("base_channels", 16)),
        pretrained_backbone=False,
    )
    state_dict = interpolate_state_dicts(baseline["state_dict"], candidate["state_dict"], alpha)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    validation = ReviewedStringDataset(
        dataset_dir,
        "val",
        int(baseline_config["input_width"]),
        int(baseline_config["input_height"]),
        int(baseline_config.get("min_mask_width_px", 2)),
        augment=False,
    )
    loader = DataLoader(validation, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    samples = collect_probabilities(model, loader, device)
    threshold, metrics, _ = select_threshold(samples)

    weights_path = run_dir / "weights" / "best.pt"
    manifest_hash = sha256_file(dataset_manifest_path)
    save_checkpoint(weights_path, model, baseline_config, threshold, 0, metrics, manifest_hash)
    manifest = {
        "schema_version": "yoyo_semantic_model_soup_v1",
        "task": "binary_semantic_segmentation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_counts": dataset_manifest.get("counts", {}),
        "source_groups": {split: sorted(values) for split, values in source_groups.items()},
        "initialization": {
            "mode": "linear_weight_interpolation",
            "baseline_weight": 1.0 - float(alpha),
            "candidate_weight": float(alpha),
            "parents": parents,
            "promotion_eligible": promotion_eligible,
        },
        "selection": {
            "split": "val",
            "metric": "harmonic_tolerant_presence_then_presence_then_negative_fp_then_tolerant_then_pixel_dice",
            "threshold": threshold,
            "metrics": metrics,
        },
        "artifacts": {
            "best": str(weights_path),
            "best_sha256": sha256_file(weights_path),
        },
        "promotion": {
            "status": "candidate" if promotion_eligible else "ineligible_source_overlap_or_missing_lineage",
            "rule": "Promote only after semantic evaluation on the untouched test split.",
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpolate two semantic string checkpoints and calibrate on validation.")
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--candidate-weights", required=True)
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "datasets" / "yoyo_dataset" / "string_segmentation"))
    parser.add_argument("--project", default=str(BASE_DIR / "runs" / "candidates"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Candidate model weight in [0, 1].")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build_model_soup(
        Path(args.baseline_weights),
        Path(args.candidate_weights),
        Path(args.dataset_dir),
        Path(args.project),
        args.name,
        args.alpha,
        args.device,
    )
    print(json.dumps({"run_dir": result["run_dir"], "selection": result["selection"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
