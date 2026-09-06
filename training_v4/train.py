from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.files import sha256_file
from string_segmentation.device import resolve_device
from string_segmentation.semantic_metrics import validation_is_better
from string_segmentation.semantic_model import focal_dice_loss, load_dataset_manifest
from .data import CenterlineDataset
from .model import build_model

CHECKPOINT_FORMAT = "yoyo_centerline_fusion_v1"


def geometry_loss(output: torch.Tensor, target: torch.Tensor, context: torch.Tensor, tangent_valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mask_logits, heat_logits = output[:, :1], output[:, 1:2]
    tangent = torch.tanh(output[:, 2:])
    mask_loss, mask_parts = focal_dice_loss(mask_logits, target[:, :1], hard_negative_weight=0.2)
    heat_target = target[:, 1:2]
    heat_bce = torch.nn.functional.binary_cross_entropy_with_logits(heat_logits, heat_target)
    heat_prob = torch.sigmoid(heat_logits)
    intersection = (heat_prob * heat_target).sum((1, 2, 3))
    denominator = heat_prob.sum((1, 2, 3)) + heat_target.sum((1, 2, 3))
    heat_dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    valid = tangent_valid.expand_as(tangent) > 0.5
    target_tangent = target[:, 2:]
    predicted_norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(1e-6)
    predicted_tangent = tangent / predicted_norm
    cosine = (predicted_tangent * target_tangent).sum(dim=1, keepdim=True).abs()
    tangent_loss = (1.0 - cosine)[valid[:, :1]].mean() if valid.any() else output.new_zeros(())
    loss = mask_loss + 0.8 * heat_bce + 0.8 * heat_dice + 0.35 * tangent_loss
    return loss, {
        "mask": float(mask_loss.detach()),
        "heat_bce": float(heat_bce.detach()),
        "heat_dice": float(heat_dice.detach()),
        "tangent": float(tangent_loss.detach()),
        **{f"mask_{key}": value for key, value in mask_parts.items() if key in {"focal", "dice_loss", "hard_negative"}},
    }


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    true_positive = false_positive = false_negative = 0
    tangent_total = tangent_count = 0.0
    for batch in loader:
        output = model(batch["image"].to(device))
        fused = fuse_geometry(output).cpu().numpy()
        target = batch["target"][:, 1].numpy()
        for prediction, expected in zip(fused, target):
            predicted = prediction >= 0.25
            target_mask = expected >= 0.15
            true_positive += int(np.logical_and(predicted, target_mask).sum())
            false_positive += int(np.logical_and(predicted, ~target_mask).sum())
            false_negative += int(np.logical_and(~predicted, target_mask).sum())
        valid = batch["tangent_valid"].to(device) > 0.5
        if valid.any():
            predicted_tangent = torch.nn.functional.normalize(torch.tanh(output[:, 2:]), dim=1)
            target_tangent = batch["target"][:, 2:].to(device)
            tangent_total += float((predicted_tangent * target_tangent).sum(dim=1).abs()[valid[:, 0]].sum())
            tangent_count += float(valid[:, 0].sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "fused_precision": precision,
        "fused_recall": recall,
        "fused_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        "tangent_cosine_abs": tangent_total / max(1.0, tangent_count),
    }


def fuse_geometry(output: torch.Tensor) -> torch.Tensor:
    """Fuse semantic mask confidence with soft centerline confidence."""
    mask_probability = torch.sigmoid(output[:, 0:1])
    centerline_probability = torch.sigmoid(output[:, 1:2])
    return mask_probability * (0.35 + 0.65 * centerline_probability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task yoyo centerline fusion model.")
    parser.add_argument("--dataset-dir", default="datasets/1Ayoyo_dataset/string_segmentation")
    parser.add_argument("--project", default="runs/experiments")
    parser.add_argument("--name", default="centerline_fusion_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--input-height", type=int, default=544)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--architecture", choices=["mobilenet_v3_fpn", "tiny_unet"], default="mobilenet_v3_fpn")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--context-radius", type=float, default=6.0)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--initial-weights", default="")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.input_width % 16 or args.input_height % 16:
        raise ValueError("epochs must be positive and input dimensions divisible by 16")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest = load_dataset_manifest(dataset_dir)
    run_dir = Path(args.project).resolve() / str(args.name)
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "train_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    train_dataset = CenterlineDataset(dataset_dir, "train", args.input_width, args.input_height, True, args.context_radius, args.heatmap_sigma)
    val_dataset = CenterlineDataset(dataset_dir, "val", args.input_width, args.input_height, False, args.context_radius, args.heatmap_sigma)
    train_loader = DataLoader(train_dataset, batch_size=max(1, args.batch), shuffle=True, num_workers=max(0, args.workers), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch), shuffle=False, num_workers=max(0, args.workers), pin_memory=device.type == "cuda")
    model = build_model(args.architecture, args.base_channels, args.pretrained_backbone).to(device)
    initialization: dict[str, Any] = {"mode": "imagenet_backbone" if args.pretrained_backbone else "random"}
    if str(args.initial_weights).strip():
        initial_path = Path(args.initial_weights).resolve()
        checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
        source = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(source, dict):
            raise RuntimeError(f"Initial checkpoint has no state_dict: {initial_path}")
        target = model.state_dict()
        compatible = {key: value for key, value in source.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
        model.load_state_dict(compatible, strict=False)
        initialization = {"mode": "compatible_warm_start", "weights": str(initial_path), "weights_sha256": sha256_file(initial_path), "loaded_parameter_count": len(compatible), "source_format": checkpoint.get("format", "")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=float(args.lr) * 0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    manifest_hash = sha256_file(dataset_dir / "manifest.json")
    best_validation: dict[str, float] = {}
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            tangent_valid = batch["tangent_valid"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, parts = geometry_loss(model(images), target, context, tangent_valid)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in {"loss": float(loss.detach()), **parts}.items():
                totals[key] = totals.get(key, 0.0) + value
        scheduler.step()
        validation = validate(model, val_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": {key: value / max(1, len(train_loader)) for key, value in totals.items()}, "validation": validation}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"epoch={epoch}/{args.epochs} loss={row['train']['loss']:.4f} fused_f1={validation['fused_f1']:.4f} tangent={validation['tangent_cosine_abs']:.4f}", flush=True)
        if not best_validation or validation["fused_f1"] > best_validation["fused_f1"]:
            best_validation = validation
            best_epoch = epoch
            torch.save({"format": CHECKPOINT_FORMAT, "model_config": {"architecture": args.architecture, "base_channels": args.base_channels, "input_width": args.input_width, "input_height": args.input_height, "context_radius": args.context_radius, "heatmap_sigma": args.heatmap_sigma}, "state_dict": model.state_dict(), "threshold": 0.25, "epoch": epoch, "validation_metrics": validation, "dataset_manifest_sha256": manifest_hash}, weights_dir / "best.pt")
    if not (weights_dir / "best.pt").exists():
        raise RuntimeError("Training produced no checkpoint")
    torch.save({"format": CHECKPOINT_FORMAT, "model_config": {"architecture": args.architecture, "base_channels": args.base_channels, "input_width": args.input_width, "input_height": args.input_height, "context_radius": args.context_radius, "heatmap_sigma": args.heatmap_sigma}, "state_dict": model.state_dict(), "threshold": 0.25, "epoch": args.epochs, "validation_metrics": best_validation, "dataset_manifest_sha256": manifest_hash}, weights_dir / "last.pt")
    run_manifest = {
        "schema_version": "yoyo_training_v4_run_v2",
        "task": "mask_centerline_tangent_fusion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_manifest": str((dataset_dir / "manifest.json")),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_counts": manifest.get("counts", {}),
        "source_groups": manifest.get("source_groups", {}),
        "initialization": initialization,
        "parameters": vars(args),
        "selection": {"split": "val", "metric": "fused_heatmap_proxy_f1", "best_epoch": best_epoch, "validation": best_validation},
        "artifacts": {"best": str((weights_dir / "best.pt")), "best_sha256": sha256_file(weights_dir / "best.pt"), "last": str((weights_dir / "last.pt")), "last_sha256": sha256_file(weights_dir / "last.pt"), "history": str(history_path), "history_sha256": sha256_file(history_path)},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_manifest


def main() -> int:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
