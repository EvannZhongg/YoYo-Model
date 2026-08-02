"""Create a versioned linear weight soup from compatible YOLO runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_models(baseline: torch.nn.Module, candidate: torch.nn.Module, alpha: float) -> torch.nn.Module:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if baseline_state.keys() != candidate_state.keys():
        raise ValueError("YOLO checkpoints do not have matching model state keys")
    averaged = copy.deepcopy(baseline)
    output_state: dict[str, torch.Tensor] = {}
    for key, baseline_value in baseline_state.items():
        candidate_value = candidate_state[key]
        if baseline_value.shape != candidate_value.shape:
            raise ValueError(f"YOLO checkpoint tensor shape mismatch: {key}")
        if baseline_value.is_floating_point() or baseline_value.is_complex():
            value = torch.lerp(baseline_value.float(), candidate_value.float(), alpha)
            output_state[key] = value.to(dtype=baseline_value.dtype)
        else:
            if not torch.equal(baseline_value, candidate_value):
                raise ValueError(f"non-floating YOLO checkpoint tensor differs: {key}")
            output_state[key] = baseline_value
    averaged.load_state_dict(output_state, strict=True)
    return averaged


def build_soup(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    baseline_weights = args.baseline_weights.resolve()
    candidate_run = args.candidate_run.resolve()
    candidate_manifest_path = candidate_run / "run_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_weights = Path(candidate_manifest["artifacts"]["best"]).resolve()
    output_dir = (args.project / args.name).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output run already exists: {output_dir}")
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True)

    baseline_checkpoint = torch.load(baseline_weights, map_location="cpu", weights_only=False)
    candidate_checkpoint = torch.load(candidate_weights, map_location="cpu", weights_only=False)
    baseline_model = baseline_checkpoint.get("model")
    candidate_model = candidate_checkpoint.get("model")
    if not isinstance(baseline_model, torch.nn.Module) or not isinstance(candidate_model, torch.nn.Module):
        raise ValueError("YOLO checkpoint does not contain a model module")
    soup_model = average_models(baseline_model, candidate_model, float(args.alpha))

    output_weights = weights_dir / "best.pt"
    output_checkpoint = copy.deepcopy(candidate_checkpoint)
    output_checkpoint["date"] = datetime.now(timezone.utc).isoformat()
    output_checkpoint["model"] = soup_model
    output_checkpoint["ema"] = None
    output_checkpoint["model_soup"] = {
        "baseline_weights": str(baseline_weights),
        "baseline_weights_sha256": sha256_file(baseline_weights),
        "candidate_weights": str(candidate_weights),
        "candidate_weights_sha256": sha256_file(candidate_weights),
        "candidate_alpha": float(args.alpha),
    }
    torch.save(output_checkpoint, output_weights)

    run_manifest = copy.deepcopy(candidate_manifest)
    run_manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": args.name,
            "run_dir": str(output_dir),
            "initialization_lineage": {
                "kind": "validation_selected_model_soup",
                "parent_run_manifest": str(candidate_manifest_path),
                "evaluation_source_overlap": (candidate_manifest.get("initialization_lineage") or {}).get(
                    "evaluation_source_overlap", {"val": [], "test": []}
                ),
                "promotion_eligible": bool(
                    (candidate_manifest.get("initialization_lineage") or {}).get("promotion_eligible", False)
                ),
            },
            "model_soup": output_checkpoint["model_soup"],
            "metrics": {},
            "artifacts": {
                "best": str(output_weights),
                "best_sha256": sha256_file(output_weights),
                "last": "",
                "last_sha256": "",
            },
            "promotion": {
                "status": "candidate",
                "rule": "Select alpha on validation only; evaluate test once after selection.",
            },
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-weights", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--alpha", required=True, type=float, help="Candidate-model interpolation weight.")
    return parser.parse_args()


def main() -> int:
    result = build_soup(parse_args())
    print(json.dumps({"run_dir": result["run_dir"], "weights": result["artifacts"]["best"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
