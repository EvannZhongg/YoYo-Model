"""Extract unreviewed train neighbors around semantic hard-negative anchors.

The resulting frames are review-only diagnostics. This module never appends to
``frames.jsonl`` and never creates or changes annotation truth.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_dataset.split_policy import parse_source_groups


def _parse_offsets(value: str) -> list[float]:
    offsets = sorted({float(item.strip()) for item in str(value).split(",") if item.strip()})
    offsets = [item for item in offsets if item != 0.0]
    if not offsets:
        raise ValueError("At least one non-zero offset is required")
    return offsets


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _existing_frame_keys(dataset_dir: Path) -> set[tuple[str, int]]:
    path = dataset_dir / "frames.jsonl"
    if not path.exists():
        return set()
    return {
        (str(row.get("video_id")), int(row.get("frame_index", -1)))
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _contact_sheet(root: Path, rows: list[dict[str, Any]], output_name: str) -> Path | None:
    if not rows:
        return None
    columns = 4
    thumb_width = 480
    thumb_height = round(thumb_width * 9 / 16)
    label_height = 72
    cell_height = thumb_height + label_height
    canvas = np.full(
        (math.ceil(len(rows[:64]) / columns) * cell_height, columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, row in enumerate(rows[:64]):
        image = cv2.imread(str(row["candidate_image"]))
        if image is None:
            continue
        image = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        x = (index % columns) * thumb_width
        y = (index // columns) * cell_height
        canvas[y : y + thumb_height, x : x + thumb_width] = image
        title = f"#{index + 1} {row['video_id'][:12]} f{row['frame_index']} t={row['timestamp_s']:.2f}s"
        context = (
            f"anchor#{row['anchor_rank']} f{row['anchor_frame_index']} "
            f"offset={row['offset_seconds']:+.2f}s anchor_px={row['anchor_predicted_pixels']}"
        )
        cv2.putText(canvas, title, (x + 6, y + thumb_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, context, (x + 6, y + thumb_height + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 80, 180), 1, cv2.LINE_AA)
    output = root / "review_sheets" / f"{output_name}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise OSError(f"Could not write hard-negative candidate sheet: {output}")
    return output


def build_neighbor_candidates(
    dataset_dir: str | Path,
    queue_path: str | Path,
    offsets_seconds: list[float],
    top_anchors: int = 8,
    limit: int = 32,
    output_name: str = "hard_negative_neighbor_candidates",
    require_yoyo_absent: bool = True,
    require_false_positive: bool = True,
    exclude_source_groups: set[str] | str | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    excluded_groups = (
        parse_source_groups(exclude_source_groups)
        if isinstance(exclude_source_groups, str)
        else {str(value).strip() for value in (exclude_source_groups or set()) if str(value).strip()}
    )
    output_name = str(output_name).strip()
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be a non-empty filename stem")
    queue_file = Path(queue_path)
    queue = json.loads(queue_file.read_text(encoding="utf-8"))
    source_manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    sources = {
        str(item["video_id"]): item
        for item in source_manifest.get("sources", [])
        if item.get("video_id")
    }
    existing = _existing_frame_keys(root)
    labels_root = root / "annotations" / "labels"
    output_root = root / "review_sheets" / "hard_negative_candidates" / output_name
    anchors = []
    for row in queue.get("rows", []):
        if str(row.get("split")) != "train":
            continue
        source_group = str(row.get("source_group") or row.get("video_id") or "").strip()
        if source_group in excluded_groups:
            continue
        if require_false_positive and not bool(row.get("false_positive")):
            continue
        if require_yoyo_absent and str(row.get("yoyo_visibility")) not in {"absent", "out_of_frame"}:
            continue
        anchors.append(row)
        if top_anchors > 0 and len(anchors) >= int(top_anchors):
            break

    rows: list[dict[str, Any]] = []
    candidate_keys: set[tuple[str, int]] = set()
    for anchor in anchors:
        video_id = str(anchor.get("video_id") or "")
        source = sources.get(video_id)
        if source is None or str(source.get("split")) != "train":
            continue
        if str(source.get("source_group") or video_id).strip() in excluded_groups:
            continue
        fps = float(source.get("fps") or 0.0)
        if fps <= 0.0:
            continue
        anchor_frame = int(anchor.get("frame_index", 0))
        for offset in offsets_seconds:
            frame_index = anchor_frame + int(round(float(offset) * fps))
            key = (video_id, frame_index)
            if frame_index < 0 or key in existing or key in candidate_keys:
                continue
            label_path = labels_root / "train" / video_id / f"frame_{frame_index:08d}.json"
            if label_path.exists():
                continue
            frame = _read_video_frame(Path(str(source["path"])), frame_index)
            if frame is None:
                continue
            image_path = output_root / "train" / video_id / f"frame_{frame_index:08d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise OSError(f"Could not write candidate frame: {image_path}")
            candidate_keys.add(key)
            rows.append({
                "candidate_image": str(image_path.resolve()),
                "source_video": str(Path(str(source["path"])).resolve()),
                "source_video_sha256": source.get("sha256"),
                "video_id": video_id,
                "source_group": source.get("source_group", video_id),
                "action_group": source.get("action_group", source_manifest.get("current_action_group", "1A")),
                "split": "train",
                "frame_index": frame_index,
                "timestamp_s": round(frame_index / fps, 4),
                "anchor_rank": int(anchor.get("queue_rank", 0)),
                "anchor_frame_index": anchor_frame,
                "anchor_predicted_pixels": int((anchor.get("model") or {}).get("predicted_pixels", 0)),
                "offset_seconds": float(offset),
                "review_status": "unreviewed_review_only",
            })
            if limit > 0 and len(rows) >= int(limit):
                break
        if limit > 0 and len(rows) >= int(limit):
            break

    sheet = _contact_sheet(root, rows, output_name)
    output_json = root / f"{output_name}.json"
    payload = {
        "schema_version": "yoyo_hard_negative_neighbor_candidates_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root.resolve()),
        "source_queue": str(queue_file.resolve()),
        "selection": {
            "split": "train",
            "require_yoyo_absent": bool(require_yoyo_absent),
            "require_false_positive": bool(require_false_positive),
            "exclude_source_groups": sorted(excluded_groups),
            "top_anchors": int(top_anchors),
            "offsets_seconds": [float(value) for value in offsets_seconds],
        },
        "count": len(rows),
        "rows": rows,
        "policy": "REVIEW ONLY; candidates are not in frames.jsonl and are not annotation truth",
        "review_sheet": str(sheet.resolve()) if sheet else None,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "candidates": str(output_json.resolve()),
        "review_sheet": str(sheet.resolve()) if sheet else None,
        "count": len(rows),
    }


def _parse_approval_keys(values: list[str]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Approval key must be video_id:frame_index: {item}")
            video_id, frame_text = item.rsplit(":", 1)
            keys.add((video_id.strip(), int(frame_text)))
    if not keys:
        raise ValueError("At least one approval key is required")
    return keys


def promote_reviewed_negatives(
    dataset_dir: str | Path,
    candidates_path: str | Path,
    approved_keys: set[tuple[str, int]],
    reviewer: str = "manual",
    reason: str = "Visually confirmed no string; added as a train hard negative.",
    repair_missing: bool = False,
) -> dict[str, Any]:
    """Promote an explicit, manually selected subset into reviewed train negatives."""
    from annotation.review import update_annotation_status
    from annotation.video_frame_annotator import draw_visualization

    root = Path(dataset_dir)
    candidate_file = Path(candidates_path)
    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    selected = [
        row
        for row in rows
        if (str(row.get("video_id")), int(row.get("frame_index", -1))) in approved_keys
    ]
    found_keys = {(str(row.get("video_id")), int(row.get("frame_index", -1))) for row in selected}
    missing = sorted(approved_keys - found_keys)
    if missing:
        raise ValueError(f"Approval keys are not present in candidate JSON: {missing}")
    if any(str(row.get("split")) != "train" for row in selected):
        raise ValueError("Only train candidates can be promoted")
    source_manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    sources = {
        str(item["video_id"]): item
        for item in source_manifest.get("sources", [])
        if item.get("video_id")
    }
    frames_path = root / "frames.jsonl"
    frame_rows = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if frames_path.exists() else []
    frames_by_key = {
        (str(row.get("video_id")), int(row.get("frame_index", -1))): row
        for row in frame_rows
    }
    labels_root = root / "annotations" / "labels"
    for row in selected:
        key = (str(row["video_id"]), int(row["frame_index"]))
        label_path = labels_root / "train" / key[0] / f"frame_{key[1]:08d}.json"
        frame_exists = key in frames_by_key
        label_exists = label_path.exists()
        if not repair_missing and (frame_exists or label_exists):
            raise ValueError(f"Candidate already exists in dataset: {key}")
        if repair_missing and frame_exists != label_exists and label_exists:
            raise ValueError(f"Cannot repair label without a matching frame record: {key}")
        if not Path(str(row.get("candidate_image", ""))).is_file():
            raise FileNotFoundError(f"Candidate image is missing: {row.get('candidate_image')}")
        if key[0] not in sources:
            raise ValueError(f"Candidate source is missing from sources.json: {key[0]}")

    promoted: list[dict[str, Any]] = []
    already_present: list[dict[str, Any]] = []
    for row in selected:
        video_id = str(row["video_id"])
        frame_index = int(row["frame_index"])
        key = (video_id, frame_index)
        source = sources[video_id]
        label_path = labels_root / "train" / video_id / f"frame_{frame_index:08d}.json"
        if key in frames_by_key:
            target_image = Path(str(frames_by_key[key]["frame_path"]))
        else:
            target_image = root / "candidate_frames" / "train" / video_id / f"frame_{frame_index:08d}.jpg"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(row["candidate_image"]), str(target_image))
            frame_record = {
                "schema_version": "1.0",
                "frame_path": str(target_image.resolve()),
                "source_video": source["path"],
                "source_video_sha256": source.get("sha256"),
                "video_id": video_id,
                "source_group": source.get("source_group", video_id),
                "action_group": source.get("action_group", source_manifest.get("current_action_group", "1A")),
                "subject_id": source.get("subject_id"),
                "split": "train",
                "frame_index": frame_index,
                "timestamp_s": row.get("timestamp_s"),
                "annotation_status": "partially_reviewed",
                "candidate_only": False,
                "visibility": "uncertain",
                "yoyo_bbox": None,
                "string_visibility": "not_visible",
                "string_polyline": None,
                "hands": None,
                "pose": None,
                "bad_case": ["hard_negative_neighbor", "string_not_visible"],
                "review_notes": reason,
            }
            frame_rows.append(frame_record)
            frames_by_key[key] = frame_record
        if label_path.exists():
            already_present.append({"video_id": video_id, "frame_index": frame_index, "label_path": str(label_path.resolve())})
            continue
        image = cv2.imread(str(target_image))
        if image is None:
            raise RuntimeError(f"Could not read promoted candidate image: {target_image}")
        width, height = image.shape[1::-1]
        annotation = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "review_status": "partially_reviewed",
            "bbox_review_status": "auto_labeled_needs_review",
            "string_review_status": "auto_labeled_needs_review",
            "string_attachment_class": "unknown",
            "model": "manual_hard_negative_neighbor",
            "source_image": str(target_image.resolve()),
            "source_video": source["path"],
            "source_video_sha256": source.get("sha256"),
            "source_group": source.get("source_group", video_id),
            "action_group": source.get("action_group", source_manifest.get("current_action_group", "1A")),
            "video_id": video_id,
            "split": "train",
            "frame_index": frame_index,
            "timestamp_s": row.get("timestamp_s"),
            "image_size": [width, height],
            "bbox": [],
            "visibility": "uncertain",
            "string_visibility": "not_visible",
            "yoyo_bbox_2d": None,
            "yoyo_bbox_pixel": None,
            "bad_case": ["hard_negative_neighbor", "string_not_visible"],
            "scene_label": "non_trick",
            "notes": reason,
            "qa": {"priority": "high", "warnings": ["manual_hard_negative_neighbor"], "requires_visual_review": False},
        }
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        update_annotation_status(
            label_path,
            "reviewed",
            reviewer=reviewer,
            notes=reason,
            component="string",
            string_visibility="not_visible",
            scene_label="non_trick",
        )
        visualization = root / "annotations" / "visualizations" / "train" / video_id / f"frame_{frame_index:08d}_vis.jpg"
        draw_visualization(target_image, json.loads(label_path.read_text(encoding="utf-8")), visualization)
        promoted.append({"video_id": video_id, "frame_index": frame_index, "label_path": str(label_path.resolve())})

    frame_rows.sort(key=lambda item: (str(item.get("split")), str(item.get("video_id")), int(item.get("frame_index", 0))))
    with frames_path.open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        key = (str(row.get("video_id")), int(row.get("frame_index", -1)))
        if key in approved_keys:
            row["review_status"] = "approved_negative"
            row["reviewer"] = reviewer
            row["review_notes"] = reason
    candidate_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "promoted": promoted,
        "already_present": already_present,
        "count": len(promoted),
        "candidates": str(candidate_file.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract review-only neighbors around train hard negatives.")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--offset-seconds", default="-1,-0.5,0.5,1")
    parser.add_argument("--top-anchors", type=int, default=8)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output-name", default="hard_negative_neighbor_candidates")
    parser.add_argument("--include-yoyo-visible", action="store_true")
    parser.add_argument(
        "--include-clean-anchors",
        action="store_true",
        help="Include reviewed train negatives that the model already classifies correctly.",
    )
    parser.add_argument("--exclude-source-groups", default="", help="Comma-separated source groups excluded before anchor selection.")
    parser.add_argument("--approve", action="append", default=[], help="Explicit video_id:frame_index key; repeat or comma-separate.")
    parser.add_argument("--reviewer", default="manual")
    parser.add_argument("--reason", default="Visually confirmed no string; added as a train hard negative.")
    parser.add_argument("--repair-missing", action="store_true", help="Create missing labels for existing promoted frame records.")
    args = parser.parse_args()
    if args.approve:
        result = promote_reviewed_negatives(
            args.dataset_dir,
            args.queue,
            _parse_approval_keys(args.approve),
            args.reviewer,
            args.reason,
            args.repair_missing,
        )
    else:
        result = build_neighbor_candidates(
            args.dataset_dir,
            args.queue,
            _parse_offsets(args.offset_seconds),
            args.top_anchors,
            args.limit,
            args.output_name,
            not args.include_yoyo_visible,
            not args.include_clean_anchors,
            args.exclude_source_groups,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
