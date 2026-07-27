"""Train and version a review-gated semantic string segmentation model."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.files import sha256_file
from config import SEMANTIC_STRING_CONFIG
from string_segmentation.device import resolve_device
from string_segmentation.semantic_metrics import balanced_validation_key, collect_probabilities, select_threshold
from string_segmentation.semantic_model import (
    ReviewedStringDataset,
    build_string_model,
    focal_dice_loss,
    load_dataset_manifest,
    load_checkpoint,
    save_checkpoint,
)


def _reviewed_sample_weights(dataset: ReviewedStringDataset, negative_sample_weight: float) -> list[float]:
    return [
        float(negative_sample_weight) if not label_path.read_text(encoding="utf-8").strip() else 1.0
        for _, label_path in dataset.pairs
    ]


def _loader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool,
    negative_sample_weight: float = 1.0,
) -> DataLoader:
    sampler = None
    if shuffle and float(negative_sample_weight) != 1.0:
        sampler = WeightedRandomSampler(
            _reviewed_sample_weights(dataset, negative_sample_weight),
            num_samples=len(dataset),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch)),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=max(0, int(workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _initialization_lineage(
    initial_weights: Path | None,
    current_groups: dict[str, set[str]],
) -> dict[str, Any]:
    if initial_weights is None:
        return {
            "kind": "foundation_or_scratch",
            "parent_run_manifest": "",
            "evaluation_source_overlap": {"val": [], "test": []},
            "promotion_eligible": True,
        }

    parent_manifest_path = initial_weights.resolve().parent.parent / "run_manifest.json"
    if not parent_manifest_path.exists():
        return {
            "kind": "unversioned_checkpoint",
            "parent_run_manifest": "",
            "evaluation_source_overlap": {"val": [], "test": []},
            "promotion_eligible": False,
        }

    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_train = set((parent.get("source_groups") or {}).get("train", []))
    overlap = {
        split: sorted(parent_train & current_groups.get(split, set()))
        for split in ("val", "test")
    }
    return {
        "kind": "versioned_run",
        "parent_run_manifest": str(parent_manifest_path),
        "evaluation_source_overlap": overlap,
        "promotion_eligible": not any(overlap.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reviewed semantic yoyo-string model.")
    parser.add_argument("--dataset-dir", default=str(SEMANTIC_STRING_CONFIG.dataset_dir))
    parser.add_argument("--project", default=str(SEMANTIC_STRING_CONFIG.project))
    parser.add_argument("--name", default=SEMANTIC_STRING_CONFIG.run_name)
    parser.add_argument("--epochs", type=int, default=SEMANTIC_STRING_CONFIG.epochs)
    parser.add_argument("--input-width", type=int, default=SEMANTIC_STRING_CONFIG.input_width)
    parser.add_argument("--input-height", type=int, default=SEMANTIC_STRING_CONFIG.input_height)
    parser.add_argument("--batch", type=int, default=SEMANTIC_STRING_CONFIG.batch)
    parser.add_argument("--workers", type=int, default=SEMANTIC_STRING_CONFIG.workers)
    parser.add_argument("--lr", type=float, default=SEMANTIC_STRING_CONFIG.learning_rate)
    parser.add_argument("--base-channels", type=int, default=SEMANTIC_STRING_CONFIG.base_channels)
    parser.add_argument(
        "--architecture",
        choices=["tiny_unet", "lraspp_mobilenet_v3"],
        default="tiny_unet",
        help="Semantic model architecture. LR-ASPP is recommended for the small reviewed dataset.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize the LR-ASPP MobileNetV3 backbone from ImageNet weights.",
    )
    parser.add_argument("--min-mask-width-px", type=int, default=SEMANTIC_STRING_CONFIG.min_mask_width_px)
    parser.add_argument("--hard-negative-weight", type=float, default=0.05)
    parser.add_argument(
        "--negative-sample-weight",
        type=float,
        default=4.0,
        help="Relative sampling weight for reviewed empty-mask train images; 1 disables rebalancing.",
    )
    parser.add_argument("--seed", type=int, default=SEMANTIC_STRING_CONFIG.seed)
    parser.add_argument("--device", default=SEMANTIC_STRING_CONFIG.device)
    parser.add_argument("--initial-weights", default="")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.input_width < 32 or args.input_height < 32:
        raise ValueError("epochs, input dimensions, and base channels must be positive")
    if args.architecture == "tiny_unet" and args.base_channels < 4:
        raise ValueError("tiny_unet base channels must be at least 4")
    if args.pretrained_backbone and args.architecture != "lraspp_mobilenet_v3":
        raise ValueError("--pretrained-backbone is only supported by lraspp_mobilenet_v3")
    if args.hard_negative_weight < 0:
        raise ValueError("hard-negative weight must be non-negative")
    if args.negative_sample_weight <= 0:
        raise ValueError("negative sample weight must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 1:
        raise ValueError("early-stopping patience must be non-negative and minimum epochs must be positive")
    if args.input_width % 16 or args.input_height % 16:
        raise ValueError("Semantic input width and height must be divisible by 16")
    dataset_dir = Path(args.dataset_dir)
    dataset_manifest_path = dataset_dir / "manifest.json"
    manifest = load_dataset_manifest(dataset_dir)
    counts = manifest.get("counts") or {}
    for split in ("train", "val", "test"):
        if int((counts.get(split) or {}).get("total", 0)) < 1:
            raise RuntimeError(f"Semantic training requires a non-empty reviewed {split} split")
    groups = {name: set(values) for name, values in (manifest.get("source_groups") or {}).items()}
    if groups.get("train", set()) & (groups.get("val", set()) | groups.get("test", set())):
        raise RuntimeError("source_group leakage detected in semantic dataset manifest")
    if groups.get("val", set()) & groups.get("test", set()):
        raise RuntimeError("source_group leakage detected between val and test")

    run_dir = Path(args.project) / str(args.name)
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Semantic run already exists: {run_dir}. Choose a new --name or pass --exist-ok.")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "train_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    train_dataset = ReviewedStringDataset(
        dataset_dir,
        "train",
        args.input_width,
        args.input_height,
        args.min_mask_width_px,
        augment=True,
    )
    val_dataset = ReviewedStringDataset(
        dataset_dir,
        "val",
        args.input_width,
        args.input_height,
        args.min_mask_width_px,
        augment=False,
    )
    train_loader = _loader(
        train_dataset,
        args.batch,
        args.workers,
        True,
        args.negative_sample_weight,
    )
    val_loader = _loader(val_dataset, args.batch, args.workers, False)
    model_config = {
        "architecture": str(args.architecture),
        "base_channels": int(args.base_channels),
        "input_width": int(args.input_width),
        "input_height": int(args.input_height),
        "min_mask_width_px": int(args.min_mask_width_px),
    }
    initialization: dict[str, Any] = {
        "mode": "imagenet_backbone" if args.pretrained_backbone else "random",
    }
    initial_weights_path: Path | None = None
    if str(args.initial_weights).strip():
        initial_weights_path = Path(args.initial_weights)
        model, initial_checkpoint = load_checkpoint(initial_weights_path, device)
        initial_config = initial_checkpoint["model_config"]
        expected = {
            "architecture": str(args.architecture),
            "base_channels": int(args.base_channels),
            "input_width": int(args.input_width),
            "input_height": int(args.input_height),
        }
        mismatches = {
            key: {"checkpoint": initial_config.get(key), "requested": value}
            for key, value in expected.items()
            if (
                str(initial_config.get(key, "tiny_unet")) != value
                if key == "architecture"
                else int(initial_config.get(key, -1)) != value
            )
        }
        if mismatches:
            raise RuntimeError(f"Initial semantic checkpoint is incompatible: {mismatches}")
        initialization = {
            "mode": "warm_start",
            "weights": str(initial_weights_path.resolve()),
            "weights_sha256": sha256_file(initial_weights_path),
            "checkpoint_epoch": int(initial_checkpoint.get("epoch", 0)),
            "dataset_manifest_sha256": str(initial_checkpoint.get("dataset_manifest_sha256", "")),
        }
    else:
        model = build_string_model(
            architecture=args.architecture,
            base_channels=args.base_channels,
            pretrained_backbone=args.pretrained_backbone,
        ).to(device)
    initialization["lineage"] = _initialization_lineage(initial_weights_path, groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=float(args.lr) * 0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    manifest_hash = sha256_file(dataset_manifest_path)
    best_key = (-1.0, -1.0, float("-inf"), -1.0, -1.0)
    best_epoch = 0
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = focal_total = dice_total = hard_negative_total = 0.0
        batch_count = 0
        for batch_data in train_loader:
            images = batch_data["image"].to(device, non_blocking=True)
            masks = batch_data["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss, parts = focal_dice_loss(logits, masks, args.hard_negative_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach().cpu())
            focal_total += parts["focal"]
            dice_total += parts["dice_loss"]
            hard_negative_total += parts["hard_negative"]
            batch_count += 1
        scheduler.step()
        validation_samples = collect_probabilities(model, val_loader, device)
        threshold, validation_metrics, threshold_sweep = select_threshold(validation_samples)
        key = balanced_validation_key(validation_metrics)
        improved = key > best_key
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_total / max(1, batch_count),
            "train_focal": focal_total / max(1, batch_count),
            "train_dice_loss": dice_total / max(1, batch_count),
            "train_hard_negative": hard_negative_total / max(1, batch_count),
            "is_best": improved,
            "validation": validation_metrics,
            "threshold_sweep": threshold_sweep,
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} "
            f"threshold={threshold:.2f} val_balanced_f1={key[0]:.4f} "
            f"val_tol_f1={validation_metrics['tolerant']['f1']:.4f} "
            f"val_presence_f1={validation_metrics['image_presence']['f1']:.4f} "
            f"val_dice={validation_metrics['pixel']['dice']:.4f}",
            flush=True,
        )
        save_checkpoint(
            weights_dir / "last.pt",
            model,
            model_config,
            threshold,
            epoch,
            validation_metrics,
            manifest_hash,
        )
        completed_epochs = epoch
        if improved:
            best_key = key
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                weights_dir / "best.pt",
                model,
                model_config,
                threshold,
                epoch,
                validation_metrics,
                manifest_hash,
            )
        else:
            epochs_without_improvement += 1
        if (
            args.early_stopping_patience > 0
            and epoch >= args.early_stopping_min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"patience={args.early_stopping_patience}",
                flush=True,
            )
            break

    run_manifest = {
        "schema_version": "1.0",
        "task": "binary_semantic_segmentation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "dataset_manifest": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_counts": counts,
        "source_groups": {name: sorted(values) for name, values in groups.items()},
        "initialization": initialization,
        "parameters": {
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "workers": int(args.workers),
            "learning_rate": float(args.lr),
            "device": str(device),
            "seed": int(args.seed),
            "hard_negative_weight": float(args.hard_negative_weight),
            "negative_sample_weight": float(args.negative_sample_weight),
            "early_stopping_patience": int(args.early_stopping_patience),
            "early_stopping_min_epochs": int(args.early_stopping_min_epochs),
            **model_config,
        },
        "training": {
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "stop_reason": "validation_patience_exhausted" if stopped_early else "requested_epochs_completed",
        },
        "selection": {
            "split": "val",
            "metric": "harmonic_tolerant_presence_then_presence_then_negative_fp_then_tolerant_then_pixel_dice",
            "best_epoch": best_epoch,
            "threshold": best_threshold,
            "metrics": best_metrics,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "artifacts": {
            "best": str((weights_dir / "best.pt").resolve()),
            "best_sha256": sha256_file(weights_dir / "best.pt"),
            "last": str((weights_dir / "last.pt").resolve()),
            "last_sha256": sha256_file(weights_dir / "last.pt"),
            "history": str(history_path.resolve()),
            "history_sha256": sha256_file(history_path),
        },
        "promotion": {
            "status": (
                "candidate"
                if initialization["lineage"]["promotion_eligible"]
                else "ineligible_source_overlap_or_unversioned_parent"
            ),
            "rule": (
                "Promote only after semantic evaluation on the untouched test split."
                if initialization["lineage"]["promotion_eligible"]
                else "Do not promote: warm-start lineage does not preserve independent evaluation sources."
            ),
        },
        "limitations": [
            "The reviewed dataset is very small; validation threshold selection is high variance.",
            "Tolerant metrics allow small localization offsets and must be reported with exact pixel metrics.",
            "Only the independent test split may be used for promotion decisions.",
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_manifest


def main() -> int:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
