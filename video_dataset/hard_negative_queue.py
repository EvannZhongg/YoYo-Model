"""Rank reviewed train negatives by semantic false-positive response.

This diagnostic writes reports and previews only. It never changes annotation
JSON or promotes model output to training truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import TRACKING_CONFIG
from video_dataset.split_policy import parse_source_groups
from video_dataset.string_review_queue import _model_signal


ACCEPTED_REVIEW = {"approved", "reviewed"}


def _false_positive_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    model = row["model"]
    return (
        -float(model.get("predicted_pixels", 0)),
        -float(model.get("components", 0)),
        -float(model.get("max_probability", 0.0)),
        str(row["label_path"]),
    )


def _contact_sheet(
    root: Path,
    rows: list[dict[str, Any]],
    output_name: str,
    columns: int = 4,
    thumb_width: int = 480,
) -> Path | None:
    if not rows:
        return None
    import cv2
    import numpy as np

    thumb_height = round(thumb_width * 9 / 16)
    label_height = 92
    cell_height = thumb_height + label_height
    visible_rows = rows[:64]
    canvas = np.full(
        (math.ceil(len(visible_rows) / columns) * cell_height, columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, row in enumerate(visible_rows):
        model = row["model"]
        preview_text = str(model.get("prediction_preview") or "").strip()
        preview = Path(preview_text) if preview_text else None
        source_text = str(row.get("source_image") or "").strip()
        source = preview if preview is not None and preview.is_file() else Path(source_text) if source_text else None
        image = cv2.imread(str(source)) if source is not None and source.is_file() else None
        if image is None:
            continue
        image = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = image
        title = (
            f"#{row['queue_rank']} px={int(model.get('predicted_pixels', 0))} "
            f"comp={int(model.get('components', 0))} max={float(model.get('max_probability', 0.0)):.3f}"
        )
        identity = f"{str(row.get('video_id') or '?')[:12]} f{row.get('frame_index', '?')}"
        context = (
            f"yoyo={row.get('yoyo_visibility', 'uncertain')} "
            f"scene={row.get('scene_label') or 'unlabeled'}"
        )
        cv2.putText(canvas, title, (x + 6, y + thumb_height + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, identity, (x + 6, y + thumb_height + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 80, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, context[:76], (x + 6, y + thumb_height + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 110, 30), 1, cv2.LINE_AA)
    output = root / "review_sheets" / f"{output_name}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise OSError(f"Could not write hard-negative contact sheet: {output}")
    return output


def build_hard_negative_queue(
    dataset_dir: str | Path,
    weights: str | Path,
    device: str = "",
    limit: int = 0,
    output_name: str = "string_hard_negative_queue",
    exclude_source_groups: set[str] | str | None = None,
) -> dict[str, Any]:
    """Score trusted train negatives without mutating their annotations."""
    root = Path(dataset_dir)
    excluded_groups = (
        parse_source_groups(exclude_source_groups)
        if isinstance(exclude_source_groups, str)
        else {str(value).strip() for value in (exclude_source_groups or set()) if str(value).strip()}
    )
    output_name = str(output_name).strip()
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be a non-empty filename stem")
    weights_path = Path(weights)
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    labels_root = root / "annotations" / "labels"
    paths = sorted(labels_root.rglob("*.json")) if labels_root.exists() else []
    model_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    inconsistent_annotations = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("split")) != "train":
            continue
        source_group = str(data.get("source_group") or data.get("video_id") or "").strip()
        if source_group in excluded_groups:
            continue
        if str(data.get("string_review_status", "auto_labeled_needs_review")) not in ACCEPTED_REVIEW:
            continue
        if str(data.get("string_visibility", "uncertain")) != "not_visible":
            continue
        polylines = data.get("string_polylines_pixel") or []
        if not polylines and data.get("string_polyline_pixel"):
            polylines = [data["string_polyline_pixel"]]
        polygons = data.get("string_mask_polygons_pixel") or []
        annotation_consistent = not bool(polylines or polygons)
        if not annotation_consistent:
            inconsistent_annotations += 1
        relative = path.resolve().relative_to(labels_root.resolve())
        preview = (
            root / "review_sheets" / "hard_negative_predictions" / output_name / relative.with_suffix(".jpg")
        )
        _, _, model_details = _model_signal(data, weights, device, model_cache, preview)
        rows.append({
            "label_path": str(path.resolve()),
            "source_image": str(data.get("source_image") or ""),
            "video_id": data.get("video_id"),
            "source_group": data.get("source_group"),
            "split": "train",
            "frame_index": int(data.get("frame_index", 0)),
            "timestamp_s": data.get("timestamp_s"),
            "string_review_status": data.get("string_review_status"),
            "string_visibility": "not_visible",
            "yoyo_visibility": data.get("visibility", "uncertain"),
            "scene_label": data.get("scene_label"),
            "bad_case": data.get("bad_case") or [],
            "annotation_consistent": annotation_consistent,
            "model": model_details,
        })
    rows.sort(key=_false_positive_sort_key)
    if limit > 0:
        rows = rows[: int(limit)]
    for index, row in enumerate(rows, 1):
        row["queue_rank"] = index
        row["false_positive"] = int(row["model"].get("predicted_pixels", 0)) > 0

    sheet = _contact_sheet(root, rows, output_name)
    output_json = root / f"{output_name}.json"
    output_csv = root / f"{output_name}.csv"
    payload = {
        "schema_version": "yoyo_string_hard_negative_queue_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "split": "train",
        "exclude_source_groups": sorted(excluded_groups),
        "selection": {
            "string_review_status": sorted(ACCEPTED_REVIEW),
            "string_visibility": "not_visible",
        },
        "weights": str(Path(weights).resolve()),
        "weights_sha256": weights_sha256,
        "count": len(rows),
        "false_positive_count": sum(bool(row["false_positive"]) for row in rows),
        "inconsistent_annotation_count": inconsistent_annotations,
        "sort": ["predicted_pixels_desc", "components_desc", "max_probability_desc"],
        "rows": rows,
        "policy": "REVIEW ONLY; model output is diagnostic and annotation JSON is never modified",
        "review_sheet": str(sheet.resolve()) if sheet else None,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "queue_rank", "label_path", "video_id", "source_group", "frame_index", "timestamp_s",
        "yoyo_visibility", "scene_label", "annotation_consistent", "false_positive",
        "predicted_pixels", "predicted_fraction", "components", "max_probability", "mean_probability",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            model = row["model"]
            writer.writerow({
                field: model.get(field) if field in model else row.get(field)
                for field in fields
            })
    return {
        "queue": str(output_json.resolve()),
        "csv": str(output_csv.resolve()),
        "review_sheet": str(sheet.resolve()) if sheet else None,
        "count": len(rows),
        "false_positive_count": payload["false_positive_count"],
        "top": rows[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank reviewed train negatives by semantic false positives.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--weights", default=str(TRACKING_CONFIG.string_weights_path))
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-name", default="string_hard_negative_queue")
    parser.add_argument("--exclude-source-groups", default="", help="Comma-separated source groups excluded before inference.")
    args = parser.parse_args()
    result = build_hard_negative_queue(
        args.dataset_dir,
        args.weights,
        args.device,
        args.limit,
        args.output_name,
        args.exclude_source_groups,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
