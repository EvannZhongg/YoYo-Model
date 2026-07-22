"""Pixel and tolerance-aware metrics for reviewed thin-string masks."""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
import torch


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_pixels <= 1 or not np.any(binary):
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_pixels:
            output[labels == index] = 1
    return output


@torch.inference_mode()
def collect_probabilities(model, loader, device: str | torch.device) -> list[dict[str, Any]]:
    model.eval()
    samples: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(images)).detach().cpu().numpy()[:, 0]
        targets = batch["mask"].numpy()[:, 0]
        paths = list(batch["image_path"])
        for probability, target, path in zip(probabilities, targets, paths):
            samples.append(
                {
                    "probability": probability.astype(np.float32),
                    "target": (target > 0.5).astype(np.uint8),
                    "image_path": str(path),
                }
            )
    return samples


def metrics_at_threshold(
    samples: Iterable[dict[str, Any]],
    threshold: float,
    tolerance_px: int = 3,
    min_component_pixels: int = 8,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    tolerant_prediction_match = tolerant_target_match = 0
    prediction_pixels = target_pixels = 0
    image_tp = image_fp = image_fn = image_tn = 0
    negative_false_positive_pixels: list[int] = []
    kernel_size = max(1, tolerance_px * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_rows = []
    for sample in samples:
        target = (sample["target"] > 0).astype(np.uint8)
        prediction = remove_small_components(sample["probability"] >= threshold, min_component_pixels)
        tp = int(np.logical_and(prediction, target).sum())
        fp = int(np.logical_and(prediction, np.logical_not(target)).sum())
        fn = int(np.logical_and(np.logical_not(prediction), target).sum())
        true_positive += tp
        false_positive += fp
        false_negative += fn
        pred_count = int(prediction.sum())
        target_count = int(target.sum())
        prediction_pixels += pred_count
        target_pixels += target_count
        if pred_count and target_count:
            target_dilated = cv2.dilate(target, kernel, iterations=1)
            prediction_dilated = cv2.dilate(prediction, kernel, iterations=1)
            tolerant_prediction_match += int(np.logical_and(prediction, target_dilated).sum())
            tolerant_target_match += int(np.logical_and(target, prediction_dilated).sum())
        target_present = target_count > 0
        prediction_present = pred_count > 0
        if target_present and prediction_present:
            image_tp += 1
        elif prediction_present:
            image_fp += 1
        elif target_present:
            image_fn += 1
        else:
            image_tn += 1
        if not target_present:
            negative_false_positive_pixels.append(pred_count)
        image_rows.append(
            {
                "image_path": sample["image_path"],
                "target_present": target_present,
                "prediction_present": prediction_present,
                "prediction_pixels": pred_count,
                "target_pixels": target_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    dice = ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = ratio(true_positive, true_positive + false_positive + false_negative)
    tolerant_precision = ratio(tolerant_prediction_match, prediction_pixels)
    tolerant_recall = ratio(tolerant_target_match, target_pixels)
    tolerant_f1 = ratio(2 * tolerant_precision * tolerant_recall, tolerant_precision + tolerant_recall)
    image_precision = ratio(image_tp, image_tp + image_fp)
    image_recall = ratio(image_tp, image_tp + image_fn)
    image_f1 = ratio(2 * image_precision * image_recall, image_precision + image_recall)
    return {
        "threshold": round(float(threshold), 4),
        "pixel": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "iou": round(iou, 6),
            "dice": round(dice, 6),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        },
        "tolerance_px": int(tolerance_px),
        "tolerant": {
            "precision": round(tolerant_precision, 6),
            "recall": round(tolerant_recall, 6),
            "f1": round(tolerant_f1, 6),
        },
        "image_presence": {
            "precision": round(image_precision, 6),
            "recall": round(image_recall, 6),
            "f1": round(image_f1, 6),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "tn": image_tn,
        },
        "negative_mean_false_positive_pixels": round(
            float(np.mean(negative_false_positive_pixels)) if negative_false_positive_pixels else 0.0,
            3,
        ),
        "sample_count": len(image_rows),
        "positive_images": image_tp + image_fn,
        "negative_images": image_fp + image_tn,
        "images": image_rows,
    }


def select_threshold(
    samples: list[dict[str, Any]],
    thresholds: Iterable[float] | None = None,
    tolerance_px: int = 3,
    min_component_pixels: int = 8,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    # Thin-string models often need a high operating threshold: the negative
    # class is visually dominated by dark fabric and stage graphics, so a
    # lower threshold can produce large false-positive components.
    values = list(thresholds or np.linspace(0.15, 0.995, 35).tolist())
    results = [
        metrics_at_threshold(samples, value, tolerance_px, min_component_pixels)
        for value in values
    ]
    best = max(
        results,
        key=lambda item: (
            item["tolerant"]["f1"],
            item["image_presence"]["f1"],
            item["pixel"]["dice"],
            -item["negative_mean_false_positive_pixels"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), best, results
