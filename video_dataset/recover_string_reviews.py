"""Recover reviewed string masks after an annotation metadata overwrite.

The prepared string dataset is review-gated and contains the exact polygons
that were accepted at preparation time. This utility merges those polygons
back into annotation JSON files without replacing existing reviewed geometry.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


SPECIAL_CASES = {
    ("train", "ab03bb7118b0", 1000): {
        "string_polyline_pixel": [[1864, 1178], [2153, 1016], [2306, 930], [2498, 800], [2671, 681]],
        "string_visibility": "visible",
        "string_review_status": "reviewed",
        "string_attachment_class": "hand_and_yoyo_attached",
        "review_notes": "Restored reviewed centerline; noisy automatic color mask remains excluded.",
    },
    ("train", "ab03bb7118b0", 1050): {
        "string_polyline_pixel": [[2345, 616], [2537, 497], [2710, 378]],
        "string_visibility": "visible",
        "string_review_status": "reviewed",
        "string_attachment_class": "hand_and_yoyo_attached",
        "review_notes": "Restored reviewed centerline; noisy automatic color mask remains excluded.",
    },
    ("train", "ee40e9221fc4", 500): {
        "string_polyline_pixel": [[1814, 843], [1837, 919], [1864, 984], [1883, 1027], [1922, 1059], [1914, 1114], [1930, 1178]],
        "string_visibility": "visible",
        "string_review_status": "auto_labeled_needs_review",
        "string_attachment_class": "hand_and_yoyo_attached",
        "review_notes": "Restored complex 1A centerline proposal; multi-stroke manual editing is still required.",
    },
    ("train", "c43840972091", 200): {
        "string_visibility": "visible",
        "string_review_status": "rejected",
        "string_attachment_class": "hand_and_yoyo_attached",
        "review_notes": "Restored prior rejected string-review decision.",
    },
    ("train", "9a86c6fcc304", 100): {
        "string_visibility": "uncertain",
        "string_review_status": "reviewed",
        "string_attachment_class": "hand_and_yoyo_attached",
        "review_notes": "Restored prior uncertain visibility; excluded from training.",
    },
}


def _load_frames(dataset_dir: Path) -> dict[tuple[str, str, int], dict]:
    records: dict[tuple[str, str, int], dict] = {}
    for line in (dataset_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["split"], record["video_id"], int(record["frame_index"]))
        records[key] = record
    return records


def _polygons(label_path: Path, width: int, height: int) -> list[list[list[float]]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            continue
        coords = [float(value) for value in fields[1:]]
        polygon = [
            [round(coords[index] * width, 2), round(coords[index + 1] * height, 2)]
            for index in range(0, len(coords), 2)
        ]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def recover(dataset_dir: Path, prepared_dir: Path) -> dict[str, int]:
    frames = _load_frames(dataset_dir)
    annotations_root = dataset_dir / "annotations" / "labels"
    images_root = prepared_dir / "images"
    labels_root = prepared_dir / "labels"
    counts = {"recovered_positive": 0, "recovered_negative": 0, "preserved": 0, "special": 0, "missing": 0}

    for prepared_label in sorted(labels_root.rglob("*.txt")):
        relative = prepared_label.relative_to(labels_root)
        if len(relative.parts) < 3:
            continue
        split, video_id = relative.parts[0], relative.parts[1]
        frame_index = int(prepared_label.stem.removeprefix("frame_"))
        annotation_path = annotations_root / relative.with_suffix(".json")
        if not annotation_path.exists():
            counts["missing"] += 1
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("string_review_status") in {"approved", "reviewed"}:
            counts["preserved"] += 1
            continue

        image_path = images_root / relative.with_suffix(".jpg")
        with Image.open(image_path) as image:
            width, height = image.size
        polygons = _polygons(prepared_label, width, height)
        frame = frames.get((split, video_id, frame_index), {})

        annotation.update(
            {
                "schema_version": "1.0",
                "source_image": frame.get("frame_path", str(image_path.resolve())),
                "source_video": frame.get("source_video"),
                "source_video_sha256": frame.get("source_video_sha256"),
                "source_group": frame.get("source_group", video_id),
                "action_group": frame.get("action_group", "1A"),
                "video_id": video_id,
                "split": split,
                "frame_index": frame_index,
                "timestamp_s": frame.get("timestamp_s"),
                "image_size": [width, height],
                "bbox_review_status": "approved",
                "string_review_status": "reviewed",
                "review_status": "reviewed",
                "string_attachment_class": annotation.get("string_attachment_class", "unknown"),
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": "recovered_from_review_gated_string_dataset",
                "review_notes": "Recovered from the previously prepared review-gated string dataset.",
            }
        )
        if polygons:
            annotation["string_visibility"] = "visible"
            annotation["string_mask_polygons_pixel"] = polygons
            counts["recovered_positive"] += 1
        else:
            annotation["string_visibility"] = "not_visible"
            annotation.pop("string_mask_polygons_pixel", None)
            annotation.pop("string_polyline_pixel", None)
            annotation.pop("string_polylines_pixel", None)
            annotation["bad_case"] = sorted(set(annotation.get("bad_case", []) + ["string_not_visible"]))
            counts["recovered_negative"] += 1
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, values in SPECIAL_CASES.items():
        split, video_id, frame_index = key
        annotation_path = annotations_root / split / video_id / f"frame_{frame_index:08d}.json"
        if not annotation_path.exists():
            counts["missing"] += 1
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = frames.get(key, {})
        annotation.update(
            {
                "schema_version": "1.0",
                "source_image": frame.get("frame_path", annotation.get("source_image")),
                "source_video": frame.get("source_video"),
                "source_video_sha256": frame.get("source_video_sha256"),
                "source_group": frame.get("source_group", video_id),
                "action_group": frame.get("action_group", "1A"),
                "video_id": video_id,
                "split": split,
                "frame_index": frame_index,
                "timestamp_s": frame.get("timestamp_s"),
                "bbox_review_status": "approved",
                "string_mask_polygons_pixel": None,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": "recovered_from_review_history",
                **values,
            }
        )
        annotation.pop("string_mask_polygons_pixel", None)
        string_status = annotation["string_review_status"]
        annotation["review_status"] = "reviewed" if string_status == "reviewed" else "partially_reviewed"
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        counts["special"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover reviewed string annotations from a prepared dataset.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--prepared-dir", default="datasets/video_v1/string_seg")
    args = parser.parse_args()
    print(json.dumps(recover(Path(args.dataset_dir), Path(args.prepared_dir)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
