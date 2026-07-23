"""Read-only annotation review queries used by UI clients."""

from __future__ import annotations

import json
from pathlib import Path

from video_dataset.split_policy import parse_source_groups


def review_label_paths(
    dataset_dir: str,
    status: str,
    split: str,
    component: str = "all",
    exclude_source_groups: str = "",
) -> list[Path]:
    root = Path(dataset_dir)
    labels_root = root / "annotations" / "labels"
    if not labels_root.exists():
        return []
    results = []
    excluded_groups = parse_source_groups(exclude_source_groups)
    for path in sorted(labels_root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status_field = {
            "bbox": "bbox_review_status",
            "string": "string_review_status",
        }.get(component, "review_status")
        current_status = data.get(status_field)
        if current_status is None:
            current_status = data.get("review_status") if component == "all" else "auto_labeled_needs_review"
        if status and current_status != status:
            continue
        if split != "all" and data.get("split") != split:
            continue
        source_group = str(data.get("source_group") or data.get("video_id") or "").strip()
        if source_group not in excluded_groups:
            results.append(path)

    queue_rank: dict[str, int] = {}
    queue_path = root / "string_review_queue.json"
    if queue_path.exists():
        try:
            payload = json.loads(queue_path.read_text(encoding="utf-8"))
            queue_rank = {
                str(Path(row["label_path"]).resolve()): int(row.get("queue_rank", 10**9))
                for row in payload.get("rows", [])
                if row.get("label_path")
            }
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
    return sorted(results, key=lambda path: (queue_rank.get(str(path.resolve()), 10**9), str(path)))


def review_visualization_path(label_path: Path, dataset_dir: str) -> Path:
    labels_root = Path(dataset_dir) / "annotations" / "labels"
    relative = label_path.relative_to(labels_root)
    return Path(dataset_dir) / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")


def review_label_preview(label_path: str | None, dataset_dir: str) -> tuple[str | None, str, str]:
    if not label_path:
        return None, "", ""
    path = Path(label_path)
    if not path.exists():
        return None, "Label not found", label_path
    data = json.loads(path.read_text(encoding="utf-8"))
    preview = review_visualization_path(path, dataset_dir)
    queue_path = Path(dataset_dir) / "string_review_queue.json"
    if queue_path.exists():
        try:
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            selected = next(
                (
                    row
                    for row in queue.get("rows", [])
                    if str(Path(str(row.get("label_path", ""))).resolve()) == str(path.resolve())
                ),
                None,
            )
            if selected:
                data = {**data, "review_queue": selected}
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
    summary = json.dumps(data, ensure_ascii=False, indent=2)
    return str(preview) if preview.exists() else None, summary, str(path)


def workbench_stats(dataset_dir: str) -> str:
    """Return compact, refreshable counts for the visual workbench."""
    root = Path(dataset_dir)
    labels_root = root / "annotations" / "labels"
    labels = sorted(labels_root.rglob("*.json")) if labels_root.exists() else []
    counts = {
        "labels": len(labels),
        "bbox_pending": 0,
        "bbox_approved": 0,
        "bbox_unresolved": 0,
        "string_pending": 0,
        "string_approved": 0,
        "string_unresolved": 0,
        "rejected": 0,
        "trick": 0,
        "transition": 0,
        "non_trick": 0,
        "scene_unknown": 0,
    }
    terminal_statuses = {"approved", "reviewed", "rejected", "unresolved"}
    for path in labels:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bbox_status = data.get("bbox_review_status", data.get("review_status", "auto_labeled_needs_review"))
        string_status = data.get("string_review_status", "auto_labeled_needs_review")
        counts["bbox_pending"] += bbox_status not in terminal_statuses
        counts["bbox_approved"] += bbox_status in {"approved", "reviewed"}
        counts["bbox_unresolved"] += bbox_status == "unresolved"
        counts["string_pending"] += string_status not in terminal_statuses
        counts["string_approved"] += string_status in {"approved", "reviewed"}
        counts["string_unresolved"] += string_status == "unresolved"
        counts["rejected"] += data.get("review_status") == "rejected"
        scene = str(data.get("scene_label", "unknown"))
        key = scene if scene in {"trick", "transition", "non_trick"} else "scene_unknown"
        counts[key] += 1

    frames_path = root / "frames.jsonl"
    frame_count = (
        sum(1 for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if frames_path.exists()
        else 0
    )
    return (
        f"Labels: {counts['labels']} | Frame records: {frame_count}\n"
        f"BBox pending: {counts['bbox_pending']} | approved: {counts['bbox_approved']} | unresolved: {counts['bbox_unresolved']}\n"
        f"String pending: {counts['string_pending']} | approved: {counts['string_approved']} | unresolved: {counts['string_unresolved']}\n"
        f"Scenes - trick: {counts['trick']} | transition: {counts['transition']} | "
        f"non-trick: {counts['non_trick']} | unknown: {counts['scene_unknown']}\n"
        f"Rejected frames: {counts['rejected']}"
    )
