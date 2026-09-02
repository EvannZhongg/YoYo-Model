from __future__ import annotations

import argparse, json, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from common.files import sha256_file
from string_segmentation.device import resolve_device
from string_segmentation.semantic_model import load_dataset_manifest
from training_v4.data import CenterlineDataset
from training_v4.model import build_model


def loss_fn(output: torch.Tensor, target: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    heat_logit, direction = output[:, :1], torch.tanh(output[:, 1:])
    heat_target = target[:, :1]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(heat_logit, heat_target)
    prob = torch.sigmoid(heat_logit); inter = (prob * heat_target).sum((1, 2, 3)); den = prob.sum((1, 2, 3)) + heat_target.sum((1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    dir_loss = torch.nn.functional.smooth_l1_loss(direction * context, target[:, 1:] * context)
    loss = bce + dice + 0.5 * dir_loss
    return loss, {"heat_bce": float(bce.detach()), "heat_dice": float(dice.detach()), "direction": float(dir_loss.detach())}


@torch.inference_mode()
def validate(model, loader, device, threshold: float = 0.5) -> dict[str, float]:
    model.eval(); probabilities = []; targets = []
    for batch in loader:
        probabilities.extend(torch.sigmoid(model(batch["image"].to(device))[:, :1]).cpu().numpy()[:, 0]); targets.extend((batch["target"][:, 0].numpy() >= 0.15))
    best = None
    for candidate_threshold in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        tp = fp = fn = center_pred = center_target = center_pred_hits = center_target_hits = 0
        for probability, target in zip(probabilities, targets):
            pred = probability >= candidate_threshold; tp += int(np.logical_and(pred, target).sum()); fp += int(np.logical_and(pred, ~target).sum()); fn += int(np.logical_and(~pred, target).sum())
            # Validation uses the narrow heatmap support as a fast centerline
            # proxy; exact skeleton/path metrics are computed by evaluate.py.
            predicted_center = pred.astype(np.uint8); expected_center = target.astype(np.uint8); center_pred += int(predicted_center.sum()); center_target += int(expected_center.sum())
            if predicted_center.any() and expected_center.any():
                dist_expected = cv2.distanceTransform((1 - expected_center).astype(np.uint8), cv2.DIST_L2, 5); dist_predicted = cv2.distanceTransform((1 - predicted_center).astype(np.uint8), cv2.DIST_L2, 5); center_pred_hits += int((dist_expected[predicted_center > 0] <= 8).sum()); center_target_hits += int((dist_predicted[expected_center > 0] <= 8).sum())
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); center_precision = center_pred_hits / max(1, center_pred); center_recall = center_target_hits / max(1, center_target); metrics = {"threshold": candidate_threshold, "heatmap_precision": precision, "heatmap_recall": recall, "heatmap_f1": 2 * precision * recall / max(1e-9, precision + recall), "centerline_precision": center_precision, "centerline_recall": center_recall, "centerline_f1_at_8": 2 * center_precision * center_recall / max(1e-9, center_precision + center_recall)}
        if best is None or metrics["centerline_f1_at_8"] > best["centerline_f1_at_8"]: best = metrics
    return best or {"threshold": threshold, "centerline_f1_at_8": 0.0}


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device); dataset_dir = Path(args.dataset_dir); manifest = load_dataset_manifest(dataset_dir)
    run_dir = Path(args.project) / args.name; run_dir.mkdir(parents=True, exist_ok=args.exist_ok); (run_dir / "weights").mkdir(exist_ok=True)
    train_ds = CenterlineDataset(dataset_dir, "train", args.input_width, args.input_height, augment=True, radius=args.context_radius)
    val_ds = CenterlineDataset(dataset_dir, "val", args.input_width, args.input_height, augment=False, radius=args.context_radius)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers); val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model = build_model(args.architecture, args.base_channels, args.pretrained_backbone).to(device)
    initialization = {"mode": "imagenet_backbone" if args.pretrained_backbone else "random", "weights": "", "weights_sha256": "", "loaded_parameter_count": 0}
    if str(args.initial_weights).strip():
        initial_path = Path(args.initial_weights).resolve()
        checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
        source_state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(source_state, dict):
            raise RuntimeError(f"Initial checkpoint has no state_dict: {initial_path}")
        target_state = model.state_dict(); compatible = {key: value for key, value in source_state.items() if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)}
        model.load_state_dict(compatible, strict=False)
        initialization = {"mode": "compatible_warm_start", "weights": str(initial_path), "weights_sha256": sha256_file(initial_path), "loaded_parameter_count": len(compatible), "source_format": checkpoint.get("format", "")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best = {}; best_path = run_dir / "weights" / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = {"loss": 0.0, "heat_bce": 0.0, "heat_dice": 0.0, "direction": 0.0}
        for batch in train_loader:
            images, target, context = batch["image"].to(device), batch["target"].to(device), batch["context"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"): loss, parts = loss_fn(model(images), target, context)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); totals["loss"] += float(loss.detach());
            for key, value in parts.items(): totals[key] += value
        metrics = validate(model, val_loader, device, args.threshold); row = {"epoch": epoch, **{k: v / max(1, len(train_loader)) for k, v in totals.items()}, "validation": metrics}; (run_dir / "train_history.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        if not best or metrics["centerline_f1_at_8"] > best["centerline_f1_at_8"]: best = metrics; torch.save({"format": "yoyo_centerline_direction_v1", "model_config": {"architecture": args.architecture, "base_channels": args.base_channels, "input_width": args.input_width, "input_height": args.input_height, "context_radius": args.context_radius}, "state_dict": model.state_dict(), "threshold": metrics.get("threshold", args.threshold), "epoch": epoch, "validation_metrics": metrics, "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json")}, best_path)
    manifest_out = {"schema_version": "yoyo_training_v4_run_v1", "task": "centerline_heatmap_direction", "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()), "dataset_manifest": str((dataset_dir / "manifest.json").resolve()), "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"), "source_groups": manifest.get("source_groups", {}), "initialization": initialization, "parameters": vars(args), "best_validation": best, "artifacts": {"best": str(best_path.resolve()), "best_sha256": sha256_file(best_path)}}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"); return manifest_out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset-dir", default="datasets/1Ayoyo_dataset/string_segmentation"); p.add_argument("--project", default="runs/experiments"); p.add_argument("--name", default="centerline_v4"); p.add_argument("--epochs", type=int, default=12); p.add_argument("--input-width", type=int, default=960); p.add_argument("--input-height", type=int, default=544); p.add_argument("--batch", type=int, default=4); p.add_argument("--workers", type=int, default=0); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--architecture", choices=["mobilenet_v3_fpn", "tiny_unet"], default="mobilenet_v3_fpn"); p.add_argument("--base-channels", type=int, default=16); p.add_argument("--context-radius", type=float, default=6.0); p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--pretrained-backbone", action="store_true"); p.add_argument("--initial-weights", default=""); p.add_argument("--seed", type=int, default=20260902); p.add_argument("--device", default="cuda"); p.add_argument("--exist-ok", action="store_true"); args = p.parse_args(); document = json.dumps(train(args), ensure_ascii=False, indent=2); encoding = sys.stdout.encoding or "utf-8"; print(document.encode(encoding, errors="backslashreplace").decode(encoding)); return 0


if __name__ == "__main__": raise SystemExit(main())
