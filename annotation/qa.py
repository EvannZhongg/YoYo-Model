"""Cross-check VLM annotations against bootstrap detections and geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def qa_annotation(annotation: dict[str, Any], frame_record: dict[str, Any] | None) -> dict[str, Any]:
    warnings: list[str] = []
    bootstrap = (frame_record or {}).get("bootstrap_detections", [])
    bbox = annotation.get("yoyo_bbox_pixel")
    max_iou = None
    if bbox and bootstrap:
        max_iou = max(iou(bbox, item["bbox_pixel"]) for item in bootstrap)
        if max_iou < 0.10:
            warnings.append("vlm_bootstrap_bbox_disagree")
    if annotation.get("visibility") in {"visible", "partially_visible"} and not bbox:
        warnings.append("visible_without_bbox")
    if annotation.get("visibility") in {"absent", "out_of_frame"} and bootstrap:
        warnings.append("absent_but_bootstrap_detected")
    polylines = annotation.get("string_polylines_pixel")
    if not polylines and annotation.get("string_polyline_pixel"):
        polylines = [annotation["string_polyline_pixel"]]
    polylines = [stroke for stroke in (polylines or []) if isinstance(stroke, list) and len(stroke) >= 2]
    if not polylines and annotation.get("string_visibility") in {"visible", "partial"}:
        warnings.append("visible_string_without_polyline")
    prelabel = annotation.get("string_prelabel") or {}
    if isinstance(prelabel, dict) and prelabel.get("status") in {"no_mask", "too_many_components", "mask_area_too_large"}:
        warnings.append(f"string_color_proposal_{prelabel['status']}")
    if annotation.get("visibility") in {"uncertain", "occluded"}:
        warnings.append("yoyo_visibility_requires_review")
    priority = "high" if warnings else "normal"
    return {
        "priority": priority,
        "warnings": warnings,
        "bootstrap_bbox_iou": round(max_iou, 4) if max_iou is not None else None,
        "bootstrap_detection_count": len(bootstrap),
        "requires_visual_review": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run consistency QA over VLM video-frame annotations.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    frame_records = {}
    for line in (dataset_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["video_id"], int(record["frame_index"]))
        if record.get("bootstrap_detections") or key not in frame_records:
            frame_records[key] = record
    labels_root = dataset_dir / "annotations" / "labels"
    report_rows = []
    for label_path in sorted(labels_root.rglob("*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        # Older labels only had one overall status. Preserve an existing bbox
        # decision, but never infer string approval from it: string geometry
        # requires its own visual check.
        overall_status = data.get("review_status", "auto_labeled_needs_review")
        data.setdefault("bbox_review_status", overall_status)
        data.setdefault(
            "string_review_status",
            "rejected" if overall_status == "rejected" else "auto_labeled_needs_review",
        )
        qa = qa_annotation(data, frame_records.get((data["video_id"], int(data["frame_index"]))))
        data["qa"] = qa
        label_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report_rows.append({
            "label_path": str(label_path),
            "split": data.get("split"),
            "video_id": data.get("video_id"),
            "frame_index": data.get("frame_index"),
            "review_status": data.get("review_status"),
            "qa_priority": qa["priority"],
            "qa_warnings": ",".join(qa["warnings"]),
            "bootstrap_bbox_iou": qa["bootstrap_bbox_iou"],
        })
    report_path = dataset_dir / "annotation_qa.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]) if report_rows else ["label_path"])
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "annotation_count": len(report_rows),
        "high_priority_count": sum(row["qa_priority"] == "high" for row in report_rows),
        "report": str(report_path.resolve()),
    }
    (dataset_dir / "annotation_qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
