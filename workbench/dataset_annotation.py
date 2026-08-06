"""Interactive editor for canonical yoyo and string annotations."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from common.files import atomic_write_text
from config import BASE_DIR


DATASETS_DIR = BASE_DIR / "datasets"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VALID_YOYO_VISIBILITY = {
    "visible", "partially_visible", "out_of_frame", "absent", "not_visible", "uncertain",
}
VALID_TRICK_ORIENTATIONS = {"normal", "horizontal", "not_applicable"}
VALID_STRING_VISIBILITY = {"visible", "partial", "not_visible", "uncertain"}
VALID_REVIEW_STATUS = {"approved", "reviewed", "unresolved", "needs_review"}
ANNOTATION_SCHEMA_VERSION = "agent_yoyo_string_annotation_v5"
SUPPORTED_ANNOTATION_SCHEMA_VERSIONS = {
    "agent_yoyo_string_annotation_v4",
    ANNOTATION_SCHEMA_VERSION,
}
REVIEW_SCHEMA_VERSION = "yoyo_dataset_review_v2"
REVIEW_MAP_FILENAME = "dataset_review_status.json"
REVIEW_MAP_PATH = BASE_DIR / "workbench_state" / REVIEW_MAP_FILENAME
_STORAGE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _managed_dataset_path(value: str | Path) -> Path:
    raw = Path(str(value or "").strip())
    if not str(raw):
        raise ValueError("dataset path is required")
    path = (raw if raw.is_absolute() else BASE_DIR / raw).resolve()
    root = DATASETS_DIR.resolve()
    if path != root and not path.is_relative_to(root):
        raise ValueError("dataset path must be inside the repository datasets directory")
    if not path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {path}")
    return path


def _annotation_roots(dataset_path: Path) -> tuple[Path, Path, Path]:
    candidates = (dataset_path / "canonical", dataset_path)
    for root in candidates:
        labels = root / "labels"
        images = root / "images"
        if labels.is_dir() and images.is_dir():
            return root, labels, images
    raise ValueError("dataset must contain canonical/labels + canonical/images or labels + images")


def _resolve_source_image(label_path: Path, labels_root: Path, images_root: Path, document: dict[str, Any]) -> Path:
    # Prefer the dataset-owned image. Gradio exposes this managed directory,
    # while source_image may point back to an external annotation archive.
    relative = label_path.relative_to(labels_root).with_suffix("")
    stems = [relative]
    if "-" in relative.name:
        stems.append(relative.with_name(relative.name.rsplit("-", 1)[0]))
    for stem in stems:
        for suffix in IMAGE_SUFFIXES:
            candidate = (images_root / stem).with_suffix(suffix)
            if candidate.is_file():
                return candidate.resolve()

    source = str(document.get("source_image") or "").strip()
    if source:
        candidate = Path(source)
        options = [candidate] if candidate.is_absolute() else [label_path.parent / candidate, labels_root.parent / candidate]
        for option in options:
            resolved = option.resolve()
            if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
                return resolved
    raise FileNotFoundError(f"image not found for label: {label_path}")


def _read_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid annotation JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"annotation must be a JSON object: {path}")
    if document.get("schema_version") not in SUPPORTED_ANNOTATION_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported annotation schema: {document.get('schema_version')!r}")
    return document


def _review_dataset_key(dataset_path: Path) -> str:
    return dataset_path.resolve().relative_to(DATASETS_DIR.resolve()).as_posix()


def _read_review_map() -> dict[str, Any]:
    path = REVIEW_MAP_PATH
    if not path.is_file():
        return {"schema_version": REVIEW_SCHEMA_VERSION, "datasets": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid review status JSON: {path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"unsupported review status schema: {path}")
    if not isinstance(document.get("datasets"), dict):
        raise ValueError(f"review status datasets must be an object: {path}")
    return document


def _dataset_reviews(document: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    dataset = document["datasets"].get(_review_dataset_key(dataset_path))
    if dataset is None:
        return {}
    if not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), dict):
        raise ValueError(f"review status dataset entry is invalid: {REVIEW_MAP_PATH}")
    return dataset["samples"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _review_summary(label_path: Path, review: object) -> dict[str, Any]:
    valid = (
        isinstance(review, dict)
        and review.get("label_sha256") == _file_sha256(label_path)
        and bool(review.get("confirmed"))
    )
    return {
        "reviewed": valid,
        "reviewed_at_utc": str(review.get("confirmed_at_utc") or "") if valid else "",
        "reviewer": str(review.get("reviewer") or "") if valid else "",
    }


def _sample_summary(
    index: int,
    label_path: Path,
    labels_root: Path,
    images_root: Path,
    review: object = None,
) -> dict[str, Any]:
    document = _read_document(label_path)
    image_path = _resolve_source_image(label_path, labels_root, images_root, document)
    relative = label_path.relative_to(labels_root).as_posix()
    bbox = document.get("yoyo_bbox_pixel")
    polylines = document.get("string_polylines_pixel") or []
    return {
        "index": index,
        "key": relative,
        "name": image_path.name,
        "group": str(document.get("source_group") or label_path.parent.name),
        "frame_index": document.get("frame_index"),
        "yoyo_visibility": str(document.get("visibility") or ("visible" if bbox else "uncertain")),
        "has_yoyo": isinstance(bbox, list) and len(bbox) == 4,
        "string_visibility": str(document.get("string_visibility") or "uncertain"),
        "string_count": len(polylines) if isinstance(polylines, list) else 0,
        "review_status": str(document.get("string_review_status") or "unresolved"),
        **_review_summary(label_path, review),
    }


def list_annotation_datasets(*, include_consecutive: bool = False) -> list[dict[str, str]]:
    """List editable dataset roots, optionally including mapped frame sequences."""
    if not DATASETS_DIR.is_dir():
        return []
    results: list[dict[str, str]] = []
    seen: set[Path] = set()
    for labels in sorted(DATASETS_DIR.rglob("labels")):
        if not labels.is_dir() or not (labels.parent / "images").is_dir():
            continue
        # Consecutive-frame datasets are owned by the consecutive annotation
        # page, which validates and consumes their explicit group mapping.
        dataset_root = labels.parent.parent if labels.parent.name == "canonical" else labels.parent
        if not include_consecutive and (dataset_root / "consecutive_groups.json").is_file():
            continue
        first_label = next(labels.rglob("*.json"), None)
        if first_label is None:
            continue
        try:
            if json.loads(first_label.read_text(encoding="utf-8")).get("schema_version") not in SUPPORTED_ANNOTATION_SCHEMA_VERSIONS:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        root = dataset_root
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        results.append({"name": root.name, "path": str(root)})
    return results


def open_annotation_dataset(dataset_path: str) -> dict[str, Any]:
    """Open a managed dataset and return the complete, lightweight sample index."""
    path = _managed_dataset_path(dataset_path)
    annotation_root, labels_root, images_root = _annotation_roots(path)
    label_paths = sorted(labels_root.rglob("*.json"))
    if not label_paths:
        raise ValueError(f"no JSON labels found in {labels_root}")
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    review_map = _read_review_map()
    reviews = _dataset_reviews(review_map, path)
    for index, label_path in enumerate(label_paths):
        try:
            key = label_path.relative_to(labels_root).as_posix()
            samples.append(_sample_summary(index, label_path, labels_root, images_root, reviews.get(key)))
        except (OSError, ValueError) as exc:
            errors.append({"label": str(label_path), "error": str(exc)})
    if not samples:
        raise ValueError("no editable image/label pairs were found")
    return {
        "dataset_path": str(path),
        "annotation_root": str(annotation_root),
        "sample_count": len(samples),
        "reviewed_count": sum(1 for sample in samples if sample["reviewed"]),
        "error_count": len(errors),
        "samples": samples,
        "errors": errors[:20],
    }


def _managed_label(dataset_path: str, sample_key: str) -> tuple[Path, Path, Path, Path]:
    path = _managed_dataset_path(dataset_path)
    annotation_root, labels_root, images_root = _annotation_roots(path)
    key = Path(str(sample_key or ""))
    if key.is_absolute() or ".." in key.parts or key.suffix.lower() != ".json":
        raise ValueError("invalid sample key")
    label_path = (labels_root / key).resolve()
    if not label_path.is_relative_to(labels_root.resolve()) or not label_path.is_file():
        raise FileNotFoundError("annotation sample does not exist")
    return annotation_root, labels_root, images_root, label_path


def load_annotation_sample(dataset_path: str, sample_key: str) -> dict[str, Any]:
    """Load one full label document and its managed image URL source."""
    _, labels_root, images_root, label_path = _managed_label(dataset_path, sample_key)
    path = _managed_dataset_path(dataset_path)
    document = _read_document(label_path)
    image_path = _resolve_source_image(label_path, labels_root, images_root, document)
    gr.set_static_paths(paths=[images_root])
    return {
        "key": label_path.relative_to(labels_root).as_posix(),
        "label_path": str(label_path),
        "image_path": str(image_path),
        "annotation": document,
        **_review_summary(
            label_path,
            _dataset_reviews(_read_review_map(), path).get(label_path.relative_to(labels_root).as_posix()),
        ),
    }


def set_annotation_sample_reviewed(
    dataset_path: str,
    sample_key: str,
    reviewer: str,
    confirmed: bool = True,
) -> dict[str, Any]:
    """Persist manual verification separately from canonical annotation JSON."""
    path = _managed_dataset_path(dataset_path)
    _, labels_root, _, label_path = _managed_label(dataset_path, sample_key)
    key = label_path.relative_to(labels_root).as_posix()
    reviewer_name = str(reviewer or "workbench-reviewer").strip() or "workbench-reviewer"
    with _STORAGE_LOCK:
        document = _read_review_map()
        dataset_key = _review_dataset_key(path)
        dataset = document["datasets"].get(dataset_key)
        if confirmed:
            if dataset is None:
                dataset = {"samples": {}}
                document["datasets"][dataset_key] = dataset
            elif not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), dict):
                raise ValueError(f"review status dataset entry is invalid: {REVIEW_MAP_PATH}")
            dataset["samples"][key] = {
                "confirmed": True,
                "confirmed_at_utc": _utc_now(),
                "reviewer": reviewer_name,
                "label_sha256": _file_sha256(label_path),
            }
        elif dataset is not None:
            if not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), dict):
                raise ValueError(f"review status dataset entry is invalid: {REVIEW_MAP_PATH}")
            dataset["samples"].pop(key, None)
            if not dataset["samples"]:
                document["datasets"].pop(dataset_key)
        document["updated_at_utc"] = _utc_now()
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(REVIEW_MAP_PATH, payload)
    return {"key": key, **_review_summary(label_path, _dataset_reviews(document, path).get(key))}


def set_all_annotation_samples_reviewed(
    dataset_path: str,
    reviewer: str = "workbench-reviewer",
) -> dict[str, Any]:
    """Bind manual verification to every current label using one atomic map write."""
    path = _managed_dataset_path(dataset_path)
    _, labels_root, _ = _annotation_roots(path)
    label_paths = sorted(labels_root.rglob("*.json"))
    if not label_paths:
        raise ValueError(f"no JSON labels found in {labels_root}")
    reviewer_name = str(reviewer or "workbench-reviewer").strip() or "workbench-reviewer"
    confirmed_at = _utc_now()
    with _STORAGE_LOCK:
        document = _read_review_map()
        dataset_key = _review_dataset_key(path)
        dataset = document["datasets"].get(dataset_key)
        if dataset is None:
            dataset = {"samples": {}}
            document["datasets"][dataset_key] = dataset
        elif not isinstance(dataset, dict) or not isinstance(dataset.get("samples"), dict):
            raise ValueError(f"review status dataset entry is invalid: {REVIEW_MAP_PATH}")
        samples = dataset["samples"]
        current_keys = {label_path.relative_to(labels_root).as_posix() for label_path in label_paths}
        removed_orphan_count = sum(key not in current_keys for key in samples)
        for key in list(samples):
            if key not in current_keys:
                samples.pop(key)
        updated_count = 0
        for label_path in label_paths:
            key = label_path.relative_to(labels_root).as_posix()
            label_sha256 = _file_sha256(label_path)
            existing = samples.get(key)
            if (
                isinstance(existing, dict)
                and existing.get("confirmed") is True
                and existing.get("label_sha256") == label_sha256
                and existing.get("reviewer") == reviewer_name
            ):
                continue
            samples[key] = {
                "confirmed": True,
                "confirmed_at_utc": confirmed_at,
                "reviewer": reviewer_name,
                "label_sha256": label_sha256,
            }
            updated_count += 1
        if updated_count or removed_orphan_count:
            document["updated_at_utc"] = confirmed_at
            payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
            atomic_write_text(REVIEW_MAP_PATH, payload)
    return {
        "dataset": dataset_key,
        "label_count": len(label_paths),
        "reviewed_count": len(samples),
        "updated_count": updated_count,
        "removed_orphan_count": removed_orphan_count,
        "reviewer": reviewer_name,
    }


def _point(value: Any, width: int, height: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("every string point must contain x and y")
    x, y = float(value[0]), float(value[1])
    if not 0 <= x <= width or not 0 <= y <= height:
        raise ValueError("string point is outside the image")
    return [round(x, 3), round(y, 3)]


def _bbox(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("yoyo bbox must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = (float(item) for item in value)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("yoyo bbox is invalid or outside the image")
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _to_2d(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[round(x / width * 1000, 3), round(y / height * 1000, 3)] for x, y in points]


def _content_digest(document: dict[str, Any]) -> str:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_annotation_sample(
    dataset_path: str,
    sample_key: str,
    edit_json: str | dict[str, Any],
) -> dict[str, Any]:
    """Validate and atomically write geometry and label corrections for one sample."""
    dataset_root = _managed_dataset_path(dataset_path)
    _, labels_root, images_root, label_path = _managed_label(dataset_path, sample_key)
    edit = json.loads(edit_json) if isinstance(edit_json, str) else edit_json
    if not isinstance(edit, dict):
        raise ValueError("annotation edit must be an object")
    document = _read_document(label_path)
    size = document.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        image_path = _resolve_source_image(label_path, labels_root, images_root, document)
        from PIL import Image

        with Image.open(image_path) as image:
            size = list(image.size)
        document["image_size"] = size
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("invalid image_size in annotation")

    yoyo_visibility = str(edit.get("yoyo_visibility") or "uncertain")
    trick_orientation = str(edit.get("trick_orientation") or "")
    string_visibility = str(edit.get("string_visibility") or "uncertain")
    review_status = str(edit.get("string_review_status") or "unresolved")
    if yoyo_visibility not in VALID_YOYO_VISIBILITY:
        raise ValueError("invalid yoyo visibility")
    if trick_orientation not in VALID_TRICK_ORIENTATIONS:
        raise ValueError("invalid trick orientation")
    if string_visibility not in VALID_STRING_VISIBILITY:
        raise ValueError("invalid string visibility")
    if review_status not in VALID_REVIEW_STATUS:
        raise ValueError("invalid string review status")

    bbox = _bbox(edit.get("yoyo_bbox_pixel"), width, height)
    raw_polylines = edit.get("string_polylines_pixel") or []
    if not isinstance(raw_polylines, list):
        raise ValueError("string polylines must be a list")
    polylines = [[_point(point, width, height) for point in line] for line in raw_polylines]
    polylines = [line for line in polylines if len(line) >= 2]
    if yoyo_visibility in {"not_visible", "absent", "out_of_frame"}:
        bbox = None
    elif yoyo_visibility == "visible" and bbox is None:
        raise ValueError("visible yoyo requires a bounding box")
    if string_visibility == "not_visible":
        polylines = []
    elif string_visibility in {"visible", "partial"} and not polylines:
        raise ValueError("visible or partial string requires at least one polyline")

    before_digest = _content_digest(document)
    bbox_2d = None
    if bbox is not None:
        bbox_2d = [
            round(bbox[0] / width * 1000, 3), round(bbox[1] / height * 1000, 3),
            round(bbox[2] / width * 1000, 3), round(bbox[3] / height * 1000, 3),
        ]
    polylines_2d = [_to_2d(line, width, height) for line in polylines]
    document["visibility"] = yoyo_visibility
    document["trick_orientation"] = trick_orientation
    document["yoyo_bbox_pixel"] = bbox
    document["yoyo_bbox_2d"] = bbox_2d
    document["bbox"] = [] if bbox is None else [{
        "label": "yoyo", "sub_label": "visible yoyo body", "bbox_pixel": bbox, "bbox_2d": bbox_2d,
    }]
    document["string_visibility"] = string_visibility
    document["string_polylines_pixel"] = polylines
    document["string_polylines_2d"] = polylines_2d
    document["string_polyline_pixel"] = polylines[0] if polylines else None
    document["string_polyline_2d"] = polylines_2d[0] if polylines_2d else None
    document["string_mask_polygons_pixel"] = None
    document["string_review_status"] = review_status
    document["bbox_review_status"] = str(edit.get("bbox_review_status") or "reviewed")
    document["notes"] = str(edit.get("notes") or "").strip()
    document["updated_at_utc"] = _utc_now()

    string_path = document.get("string_path")
    if isinstance(string_path, dict):
        previous_paths = string_path.get("paths") if isinstance(string_path.get("paths"), list) else []
        updated_paths = []
        for index, line in enumerate(polylines):
            previous = previous_paths[index] if index < len(previous_paths) and isinstance(previous_paths[index], dict) else {}
            path = dict(previous)
            path["path_id"] = str(path.get("path_id") or f"workbench-line-{index + 1}")
            path["points_pixel"] = line
            path["points_2d"] = polylines_2d[index]
            path["edges"] = [
                {"from": point_index, "to": point_index + 1, "evidence": "reviewed", "confidence": 1.0}
                for point_index in range(len(line) - 1)
            ]
            updated_paths.append(path)
        string_path["paths"] = updated_paths
        if not polylines:
            string_path["topology"] = "not_visible"
            string_path["reconstruction_status"] = "not_visible"

    edit_event = {
        "created_at_utc": document["updated_at_utc"],
        "actor": str(edit.get("reviewer") or "workbench-reviewer").strip() or "workbench-reviewer",
        "before_sha256": before_digest,
        "fields": [
            "visibility", "trick_orientation", "yoyo_bbox_pixel", "string_visibility", "string_polylines_pixel",
            "string_review_status", "bbox_review_status", "notes",
        ],
    }
    document.setdefault("workbench_edits", []).append(edit_event)
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    with _STORAGE_LOCK:
        atomic_write_text(label_path, payload)
    if REVIEW_MAP_PATH.is_file():
        set_annotation_sample_reviewed(str(dataset_root), sample_key, "", confirmed=False)
    return {
        "saved": True,
        "path": str(label_path),
        "key": label_path.relative_to(labels_root).as_posix(),
        "annotation": document,
        "summary": _sample_summary(0, label_path, labels_root, images_root),
    }


def ui_list_annotation_datasets(_payload: object = None) -> list[dict[str, str]]:
    """Single-payload adapter for Gradio HTML server functions."""
    return list_annotation_datasets()


def ui_open_annotation_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("dataset request must be an object")
    return open_annotation_dataset(str(payload.get("dataset_path") or ""))


def ui_load_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("sample request must be an object")
    return load_annotation_sample(
        str(payload.get("dataset_path") or ""),
        str(payload.get("sample_key") or ""),
    )


def ui_save_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("save request must be an object")
    return save_annotation_sample(
        str(payload.get("dataset_path") or ""),
        str(payload.get("sample_key") or ""),
        payload.get("edit") or {},
    )


def ui_set_annotation_sample_reviewed(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("review request must be an object")
    return set_annotation_sample_reviewed(
        str(payload.get("dataset_path") or ""),
        str(payload.get("sample_key") or ""),
        str(payload.get("reviewer") or ""),
        bool(payload.get("confirmed", True)),
    )


DATASET_ANNOTATION_HTML = r"""
<div class="yda" data-yoyo-dataset-annotation>
  <header class="yda__header">
    <div><h2>数据标注</h2><p id="yda-status">选择 datasets 下的数据集开始审阅</p></div>
    <div class="yda__dataset-picker">
      <label>数据集<select id="yda-dataset-select"><option value="">扫描中...</option></select></label>
      <label>本地路径<input id="yda-dataset-path" type="text" spellcheck="false"></label>
      <button class="yda__button yda__button--primary" id="yda-open" type="button">打开</button>
    </div>
  </header>
  <main class="yda__workspace">
    <aside class="yda__sidebar">
      <div class="yda__list-tools">
        <input id="yda-search" type="search" placeholder="搜索文件或分组">
        <select id="yda-filter" aria-label="筛选标注状态">
          <option value="all">全部数据</option><option value="needs_yoyo">缺少悠悠球框</option>
          <option value="needs_string">缺少绳线</option><option value="unresolved">待审阅</option>
          <option value="unreviewed">未核验</option><option value="reviewed">已核验</option>
        </select>
      </div>
      <div class="yda__list-summary" id="yda-list-summary">0 条数据</div>
      <ol class="yda__sample-list" id="yda-sample-list"></ol>
    </aside>
    <section class="yda__stage-panel">
      <div class="yda__toolbar" role="toolbar" aria-label="几何标注工具">
        <div class="yda__segmented" id="yda-tools">
          <button type="button" data-tool="select" class="is-active" title="选择和拖动标注">选择</button>
          <button type="button" data-tool="box" title="拖动绘制悠悠球框">悠悠球框</button>
          <button type="button" data-tool="string" title="逐点绘制绳线">绳线</button>
        </div>
        <button type="button" id="yda-finish-line" disabled>结束当前绳线</button>
        <button type="button" id="yda-undo" title="撤销最近一次几何修改">撤销</button>
        <button type="button" id="yda-reset" title="恢复到本次加载或上次保存的标注" disabled>重置</button>
        <button type="button" id="yda-delete" title="删除选中的绳线">删除绳线</button>
        <button type="button" id="yda-toggle-annotations" title="显示或隐藏标注（H）" aria-pressed="false">隐藏标注</button>
        <label class="yda__zoom">缩放<input id="yda-zoom" type="range" min="25" max="200" value="100" step="25"><output id="yda-zoom-value">100%</output></label>
      </div>
      <div class="yda__viewport" id="yda-viewport">
        <div class="yda__empty" id="yda-empty">尚未打开数据集</div>
        <div class="yda__canvas-layer">
          <canvas id="yda-canvas" tabindex="0"></canvas>
        </div>
      </div>
      <div class="yda__navigation">
        <button type="button" id="yda-prev" title="上一条">上一条</button>
        <output id="yda-position">0 / 0</output>
        <button type="button" id="yda-next" title="下一条">下一条</button>
      </div>
    </section>
    <aside class="yda__editor">
      <fieldset id="yda-fields" disabled>
        <div class="yda__editor-scroll">
          <legend>悠悠球识别</legend>
          <label>可见状态<select id="yda-yoyo-visibility"><option value="visible">可见</option><option value="partially_visible">部分可见</option><option value="out_of_frame">画面外</option><option value="absent">不存在</option><option value="not_visible">不可见</option><option value="uncertain">不确定</option></select></label>
          <label>方向<select id="yda-trick-orientation"><option value="normal">常规（normal）</option><option value="horizontal">水平（horizontal）</option><option value="not_applicable">不适用（not_applicable）</option></select></label>
          <div class="yda__coords"><label>X1<input id="yda-x1" type="number" min="0" step="1"></label><label>Y1<input id="yda-y1" type="number" min="0" step="1"></label><label>X2<input id="yda-x2" type="number" min="0" step="1"></label><label>Y2<input id="yda-y2" type="number" min="0" step="1"></label></div>
          <button type="button" id="yda-clear-box">清除悠悠球框</button>
          <legend>绳线识别</legend>
          <label>可见状态<select id="yda-string-visibility"><option value="visible">完整可见</option><option value="partial">部分可见</option><option value="not_visible">不可见</option><option value="uncertain">不确定</option></select></label>
          <label>审阅状态<select id="yda-review-status"><option value="approved">已批准</option><option value="reviewed">已审阅</option><option value="needs_review">需要审阅</option><option value="unresolved">未解决</option></select></label>
          <div class="yda__line-list" id="yda-line-list"></div>
          <div class="yda__line-actions"><button type="button" id="yda-add-line">新增绳线</button><button type="button" id="yda-redraw-lines">重绘绳线</button><button type="button" id="yda-clear-lines">标记为不可见</button></div>
        </div>
        <div class="yda__record">
          <span id="yda-dirty" role="status" aria-live="polite"></span>
          <legend>记录</legend>
          <label>审阅者<input id="yda-reviewer" type="text" value="workbench-reviewer" maxlength="80"></label>
          <label>备注<textarea id="yda-notes" rows="3" maxlength="2000"></textarea></label>
          <div class="yda__validation" id="yda-validation" role="status"></div>
          <div class="yda__record-actions">
            <button class="yda__button yda__review" id="yda-review" type="button">核验完成</button>
            <button class="yda__button yda__button--primary yda__save" id="yda-save" type="button">保存标注</button>
          </div>
        </div>
      </fieldset>
    </aside>
  </main>
  <div class="yda__toast" id="yda-toast" role="status" aria-live="polite"></div>
</div>
"""


DATASET_ANNOTATION_CSS = r"""
.yda { --ink:#202420; --muted:#687068; --line:#d7ddd7; --surface:#fff; --soft:#f4f6f3; --accent:#176b55; --danger:#b63b3b; background:var(--surface); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; width:100%; }
.yda * { box-sizing:border-box; }
.yda button,.yda input,.yda select,.yda textarea { font:inherit; letter-spacing:0; }
.yda h2,.yda h3,.yda p { margin:0; color:var(--ink); }
.yda h2 { font-size:22px; }
.yda h3 { font-size:14px; overflow-wrap:anywhere; }
.yda__header { align-items:end; border-bottom:1px solid var(--line); display:flex; gap:24px; justify-content:space-between; padding:16px 4px; }
.yda__header p { color:var(--muted); font-size:12px; margin-top:4px; }
.yda__dataset-picker { align-items:end; display:grid; gap:8px; grid-template-columns:minmax(150px,220px) minmax(260px,420px) auto; width:min(760px,70%); }
.yda label { color:#505750; display:grid; font-size:11px; font-weight:650; gap:5px; }
.yda input,.yda select,.yda textarea { background:#fff; border:1px solid #c9d0c9; border-radius:5px; color:var(--ink); min-width:0; padding:7px 9px; width:100%; }
.yda input,.yda select { height:36px; }
.yda textarea { resize:vertical; }
.yda button,.yda__button { background:#fff; border:1px solid #c7cec7; border-radius:6px; color:#303630; cursor:pointer; font-size:12px; font-weight:650; min-height:34px; padding:7px 11px; }
.yda button:hover { background:var(--soft); }
.yda button:disabled { cursor:not-allowed; opacity:.42; }
.yda__button--primary { background:var(--accent)!important; border-color:var(--accent)!important; color:#fff!important; }
.yda__workspace { display:grid; grid-template-columns:240px minmax(0,1fr) 330px; height:clamp(700px,calc(100dvh - 100px),960px); min-height:0; overflow:hidden; }
.yda__sidebar { border-right:1px solid var(--line); display:grid; grid-template-rows:auto auto minmax(0,1fr); min-height:0; min-width:0; overflow:hidden; }
.yda__list-tools { display:grid; gap:7px; padding:12px; }
.yda__list-summary { border-bottom:1px solid var(--line); color:var(--muted); font-size:11px; padding:0 12px 9px; }
.yda__sample-list { list-style:none; margin:0; min-height:0; overflow:auto; overscroll-behavior:contain; padding:0; scrollbar-gutter:stable; }
.yda__sample-list button { align-items:flex-start; border:0; border-bottom:1px solid #e6e9e5; border-radius:0; display:grid; gap:3px; min-height:62px; padding:9px 12px; text-align:left; width:100%; }
.yda__sample-list button.is-active { background:#e5f3ed; box-shadow:inset 3px 0 var(--accent); }
.yda__sample-title { display:block; font-size:11px; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; width:100%; }
.yda__sample-meta { color:var(--muted); display:flex; font-size:10px; gap:7px; width:100%; }
.yda__badges { display:flex; gap:4px; }
.yda__badge { background:#edf0ed; border-radius:3px; color:#4f5750; font-size:9px; padding:2px 4px; }
.yda__badge--warn { background:#f8eadc; color:#8b531a; }
.yda__badge--ok { background:#dceee7; color:#0c5a44; }
.yda__stage-panel { background:#202320; display:grid; grid-template-rows:auto minmax(0,1fr) auto; min-width:0; overflow:hidden; }
.yda__toolbar,.yda__navigation { align-items:center; background:#f5f7f4; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:7px; min-height:50px; padding:8px 10px; }
.yda__segmented { display:flex; }
.yda__segmented button { border-radius:0; }
.yda__segmented button:first-child { border-radius:5px 0 0 5px; }
.yda__segmented button:last-child { border-radius:0 5px 5px 0; }
.yda__segmented button + button { border-left:0; }
.yda__segmented button.is-active { background:#dceee7; color:#0c5a44; }
.yda__zoom { align-items:center; display:flex; gap:6px; grid-auto-flow:column; margin-left:auto; }
.yda__zoom input { accent-color:var(--accent); height:auto; padding:0; width:100px; }
.yda__zoom output { min-width:38px; }
.yda__viewport { contain:size layout paint; min-height:0; min-width:0; overflow:auto; overscroll-behavior:contain; position:relative; }
.yda__canvas-layer { display:grid; min-height:100%; min-width:100%; place-items:center; width:max-content; }
.yda__viewport canvas { background:#101210; cursor:default; display:none; touch-action:none; }
.yda__viewport canvas.is-ready { display:block; }
.yda__viewport canvas[data-tool="box"] { cursor:crosshair; }
.yda__viewport canvas[data-tool="string"] { cursor:copy; }
.yda__empty { color:#b9c0ba; font-size:13px; position:absolute; }
.yda__navigation { border-bottom:0; border-top:1px solid var(--line); justify-content:center; }
.yda__navigation output { color:#4e554f; font-size:12px; min-width:90px; text-align:center; }
.yda__editor { border-left:1px solid var(--line); display:grid; grid-template-rows:minmax(0,1fr); min-height:0; overflow:hidden; padding:12px; }
#yda-dirty { color:#9a5e20; font-size:10px; position:absolute; right:0; top:0; }
.yda fieldset { border:0; display:grid; grid-template-rows:minmax(0,1fr) auto; height:100%; margin:0; min-height:0; padding:0; }
.yda fieldset:disabled { opacity:.48; }
.yda legend { border-top:1px solid var(--line); font-size:12px; font-weight:750; margin-top:12px; padding-top:11px; width:100%; }
.yda fieldset label { margin-top:7px; }
.yda__editor-scroll { min-height:0; overflow:auto; padding-right:5px; scrollbar-gutter:stable; }
.yda__editor-scroll > legend:first-child { border-top:0; margin-top:0; }
.yda__record { background:var(--surface); border-top:1px solid var(--line); padding-top:9px; position:relative; }
.yda__record legend { border-top:0; margin-top:0; padding-top:0; }
.yda__record textarea { min-height:48px; resize:none; }
.yda__record-actions { display:grid; gap:6px; grid-template-columns:1fr 1fr; }
.yda__review.is-reviewed { background:#dceee7; border-color:#84b9a7; color:#0c5a44; }
.yda__coords { display:grid; gap:6px; grid-template-columns:1fr 1fr; margin:9px 0; }
.yda__line-list { display:grid; gap:5px; margin:9px 0; }
.yda__line-actions { display:grid; gap:6px; grid-template-columns:1fr 1fr; }
.yda__line-actions #yda-clear-lines { grid-column:1/-1; }
.yda__line-row { align-items:center; background:var(--soft); display:flex; font-size:10px; justify-content:space-between; padding:6px 8px; }
.yda__line-row.is-active { box-shadow:inset 3px 0 #d68b27; }
.yda__validation { color:var(--danger); font-size:11px; min-height:22px; padding-top:5px; }
.yda__save { width:100%; }
.yda__toast { background:#202420; border-radius:5px; bottom:24px; color:#fff; font-size:12px; left:50%; opacity:0; padding:9px 13px; pointer-events:none; position:fixed; transform:translate(-50%,8px); transition:.16s; z-index:50; }
.yda__toast.is-visible { opacity:1; transform:translate(-50%,0); }
@media (max-width:1120px) { .yda__workspace { grid-template-columns:210px minmax(0,1fr) 300px; } }
@media (max-width:820px) { .yda__header { align-items:stretch; flex-direction:column; } .yda__dataset-picker { grid-template-columns:1fr; width:100%; } .yda__workspace { display:flex; flex-direction:column; height:auto; } .yda__sidebar { border-bottom:1px solid var(--line); border-right:0; flex:none; height:270px; } .yda__stage-panel { min-height:520px; } .yda__editor { border-left:0; border-top:1px solid var(--line); max-height:none; min-height:0; overflow:visible; } .yda__editor fieldset { display:block; height:auto; } .yda__editor-scroll { overflow:visible; padding-right:0; } .yda__record { margin-top:12px; } }
"""


DATASET_ANNOTATION_JS = r"""
if (element.dataset.initialized === "true") return;
element.dataset.initialized = "true";
const $ = selector => element.querySelector(selector);
const canvas = $("#yda-canvas");
const ctx = canvas.getContext("2d");
const state = {dataset:null,samples:[],filtered:[],current:-1,sample:null,image:null,tool:"select",zoom:1,annotationsVisible:true,bbox:null,lines:[],baseline:null,activeLine:null,selectedLine:null,selectedPoint:null,drag:null,history:[],redrawPending:false,loadSerial:0,dirty:false};
let toastTimer = null;
const toast = message => { const node=$("#yda-toast"); node.textContent=message; node.classList.add("is-visible"); clearTimeout(toastTimer); toastTimer=setTimeout(()=>node.classList.remove("is-visible"),1800); };
const escapeHtml = value => String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fileUrl = path => { const config=window.gradio_config||{}; const root=String(config.root||window.location.origin).replace(/\/$/,""); const prefix=`/${String(config.api_prefix||"gradio_api").replace(/^\/+|\/+$/g,"")}`; return `${root}${prefix}/file=${encodeURIComponent(path)}`; };
const cloneGeometry = () => ({bbox:state.bbox?state.bbox.slice():null,lines:state.lines.map(line=>line.map(point=>point.slice()))});
const editorSnapshot = () => ({...cloneGeometry(),yoyoVisibility:$("#yda-yoyo-visibility").value,trickOrientation:$("#yda-trick-orientation").value,stringVisibility:$("#yda-string-visibility").value,reviewStatus:$("#yda-review-status").value,notes:$("#yda-notes").value});
function pushHistory(){ state.history.push(cloneGeometry()); if(state.history.length>60)state.history.shift(); }
function syncReviewButton(){ const button=$("#yda-review"),reviewed=Boolean(state.sample?.reviewed);button.disabled=!state.sample||state.dirty;button.classList.toggle("is-reviewed",reviewed);button.textContent=reviewed?"取消核验":"核验完成"; }
function syncResetButton(){ $("#yda-reset").disabled=!state.sample||!state.dirty; }
function syncAnnotationVisibility(){ const button=$("#yda-toggle-annotations");button.textContent=state.annotationsVisible?"隐藏标注":"显示标注";button.setAttribute("aria-pressed",String(!state.annotationsVisible)); }
function setAnnotationsVisible(visible){ state.annotationsVisible=Boolean(visible);syncAnnotationVisibility();renderCanvas(); }
function toggleAnnotations(){ if(state.annotationsVisible&&state.activeLine!==null)finishLine();if(state.annotationsVisible&&state.tool!=="select")setTool("select");setAnnotationsVisible(!state.annotationsVisible); }
function markDirty(){ state.dirty=true; $("#yda-dirty").textContent="未保存"; syncReviewButton();syncResetButton();renderCanvas(); renderLineList(); syncCoordinateInputs(); }
function applyGeometry(value){ state.bbox=value.bbox?value.bbox.slice():null; state.lines=(value.lines||[]).map(line=>line.map(point=>point.slice())); state.activeLine=null; state.selectedLine=null; state.selectedPoint=null; state.redrawPending=false; markDirty(); }
function setTool(tool){ if(tool!=="select"&&!state.annotationsVisible)setAnnotationsVisible(true);state.tool=tool; canvas.dataset.tool=tool; element.querySelectorAll("#yda-tools button").forEach(button=>button.classList.toggle("is-active",button.dataset.tool===tool)); $("#yda-finish-line").disabled=tool!=="string"||state.activeLine===null; }
function setZoom(value){ const percent=Math.max(25,Math.min(200,Math.round(Number(value)/25)*25));state.zoom=percent/100;$("#yda-zoom").value=String(percent);$("#yda-zoom-value").textContent=`${percent}%`;renderCanvas(); }
function canvasPoint(event){ const rect=canvas.getBoundingClientRect(); const sx=state.image.naturalWidth/rect.width, sy=state.image.naturalHeight/rect.height; return [Math.max(0,Math.min(state.image.naturalWidth,(event.clientX-rect.left)*sx)),Math.max(0,Math.min(state.image.naturalHeight,(event.clientY-rect.top)*sy))]; }
function lineDistance(point,a,b){ const dx=b[0]-a[0],dy=b[1]-a[1],l=dx*dx+dy*dy; if(!l)return Math.hypot(point[0]-a[0],point[1]-a[1]); const t=Math.max(0,Math.min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/l)); return Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy); }
function hitTest(point){ if(!state.annotationsVisible)return null;const radius=12/state.zoom; for(let li=0;li<state.lines.length;li++){ for(let pi=0;pi<state.lines[li].length;pi++){ if(Math.hypot(point[0]-state.lines[li][pi][0],point[1]-state.lines[li][pi][1])<=radius)return {type:"point",line:li,point:pi}; } } if(state.bbox){ const corners=[[state.bbox[0],state.bbox[1]],[state.bbox[2],state.bbox[1]],[state.bbox[2],state.bbox[3]],[state.bbox[0],state.bbox[3]]]; for(let i=0;i<4;i++)if(Math.hypot(point[0]-corners[i][0],point[1]-corners[i][1])<=radius)return {type:"corner",corner:i}; } for(let li=0;li<state.lines.length;li++){ const line=state.lines[li]; for(let i=1;i<line.length;i++)if(lineDistance(point,line[i-1],line[i])<=radius)return {type:"line",line:li,segment:i}; } return null; }
function renderCanvas(){ if(!state.image)return; const w=state.image.naturalWidth,h=state.image.naturalHeight; const maxWidth=Math.max(180,$("#yda-viewport").clientWidth-24), maxHeight=Math.max(180,$("#yda-viewport").clientHeight-24); const fit=Math.min(maxWidth/w,maxHeight/h,1); const cssW=Math.round(w*fit*state.zoom),cssH=Math.round(h*fit*state.zoom); const dpr=window.devicePixelRatio||1; canvas.width=Math.max(1,Math.round(cssW*dpr));canvas.height=Math.max(1,Math.round(cssH*dpr));canvas.style.width=`${cssW}px`;canvas.style.height=`${cssH}px`;ctx.setTransform(dpr*cssW/w,0,0,dpr*cssH/h,0,0);ctx.clearRect(0,0,w,h);ctx.drawImage(state.image,0,0,w,h);if(!state.annotationsVisible)return; const scale=w/cssW; if(state.bbox){ctx.strokeStyle="#35d39a";ctx.lineWidth=3*scale;ctx.strokeRect(state.bbox[0],state.bbox[1],state.bbox[2]-state.bbox[0],state.bbox[3]-state.bbox[1]); [[state.bbox[0],state.bbox[1]],[state.bbox[2],state.bbox[1]],[state.bbox[2],state.bbox[3]],[state.bbox[0],state.bbox[3]]].forEach(p=>{ctx.fillStyle="#fff";ctx.strokeStyle="#176b55";ctx.lineWidth=2*scale;ctx.beginPath();ctx.arc(p[0],p[1],5*scale,0,Math.PI*2);ctx.fill();ctx.stroke();});} state.lines.forEach((line,index)=>{if(line.length<1)return;const lineColor=index===state.selectedLine?"#ffae42":"#43b8ff";ctx.strokeStyle=lineColor;ctx.lineWidth=(index===state.selectedLine?4:3)*scale;ctx.beginPath();ctx.moveTo(line[0][0],line[0][1]);line.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.stroke();line.forEach((p,pointIndex)=>{const selected=state.selectedLine===index&&state.selectedPoint?.line===index&&state.selectedPoint?.point===pointIndex;ctx.fillStyle=selected?"#ff4d4f":lineColor;ctx.strokeStyle=selected?"#fff":lineColor;ctx.lineWidth=(selected?2:0)*scale;ctx.beginPath();ctx.arc(p[0],p[1],(selected?7:4.5)*scale,0,Math.PI*2);ctx.fill();if(selected)ctx.stroke();});}); }
function renderLineList(){ const node=$("#yda-line-list"); node.innerHTML=state.lines.length?state.lines.map((line,index)=>`<button type="button" class="yda__line-row ${index===state.selectedLine?'is-active':''}" data-line="${index}"><span>绳线 ${index+1}</span><span>${line.length} 个点</span></button>`).join(""):"<div class='yda__line-row'><span>没有绳线</span></div>"; node.querySelectorAll("button").forEach(button=>button.onclick=()=>{state.selectedLine=Number(button.dataset.line);state.selectedPoint=null;renderCanvas();renderLineList();}); }
function syncCoordinateInputs(){ const values=state.bbox||["","","",""]; ["x1","y1","x2","y2"].forEach((name,index)=>$("#yda-"+name).value=values[index]); }
function filteredSamples(){ const query=$("#yda-search").value.trim().toLowerCase(),filter=$("#yda-filter").value; return state.samples.filter(sample=>{ const text=`${sample.name} ${sample.group}`.toLowerCase(); if(query&&!text.includes(query))return false; if(filter==="needs_yoyo"&&sample.has_yoyo)return false; if(filter==="needs_string"&&sample.string_count>0)return false; if(filter==="unresolved"&&!['unresolved','needs_review'].includes(sample.review_status))return false; if(filter==="unreviewed"&&sample.reviewed)return false; if(filter==="reviewed"&&!sample.reviewed)return false; return true; }); }
function renderSamples(){ state.filtered=filteredSamples(); const reviewed=state.samples.filter(sample=>sample.reviewed).length;$("#yda-list-summary").textContent=`${state.filtered.length} / ${state.samples.length} 条 · 已核验 ${reviewed}`; $("#yda-sample-list").innerHTML=state.filtered.map(sample=>`<li><button type="button" data-key="${escapeHtml(sample.key)}" class="${state.sample?.key===sample.key?'is-active':''}"><span class="yda__sample-title">${escapeHtml(sample.name)}</span><span class="yda__sample-meta"><span>${escapeHtml(sample.group)}</span><span>f${sample.frame_index??'-'}</span></span><span class="yda__badges"><span class="yda__badge ${sample.has_yoyo?'':'yda__badge--warn'}">悠悠球${sample.has_yoyo?'有框':'无框'}</span><span class="yda__badge ${sample.string_count?'':'yda__badge--warn'}">绳线 ${sample.string_count}</span><span class="yda__badge ${sample.reviewed?'yda__badge--ok':'yda__badge--warn'}">${sample.reviewed?'已核验':'未核验'}</span></span></button></li>`).join(""); $("#yda-sample-list").querySelectorAll("button").forEach(button=>button.onclick=()=>selectSample(button.dataset.key)); }
const normalizedDatasetPath = value => String(value||"").replace(/\\/g,"/").replace(/\/+$/,"").toLowerCase();
const datasetNameFromPath = value => String(value||"").split(/[\\/]/).filter(Boolean).at(-1)||String(value||"");
function syncDatasetChoice(path){ const select=$("#yda-dataset-select"),normalized=normalizedDatasetPath(path);let option=Array.from(select.options).find(item=>normalizedDatasetPath(item.value)===normalized);if(!option&&path){option=new Option(datasetNameFromPath(path),path);select.add(option);}if(option)select.value=option.value; }
let datasetListRequest=0;
async function refreshDatasetOptions(preferredPath=""){ const request=++datasetListRequest,datasets=await server.ui_list_annotation_datasets();if(request!==datasetListRequest)return datasets;const select=$("#yda-dataset-select"),selectedPath=preferredPath||select.value||$("#yda-dataset-path").value.trim();select.replaceChildren();datasets.forEach(item=>select.add(new Option(item.name,item.path)));if(!datasets.length&&!selectedPath)select.add(new Option("未发现数据集",""));syncDatasetChoice(selectedPath);return datasets; }
async function selectSample(key){ if(state.dirty&&!window.confirm("当前修改尚未保存，确定切换数据吗？"))return; const request=++state.loadSerial;state.image=null;canvas.classList.remove("is-ready");$("#yda-empty").hidden=false;$("#yda-empty").textContent="正在加载图像...";$("#yda-status").textContent="正在加载标注..."; try{ const result=await server.ui_load_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:key});if(request!==state.loadSerial)return; state.sample=result; const annotation=result.annotation; state.bbox=Array.isArray(annotation.yoyo_bbox_pixel)?annotation.yoyo_bbox_pixel.map(Number):null; state.lines=(annotation.string_polylines_pixel||[]).map(line=>line.map(point=>point.map(Number))); state.activeLine=null;state.selectedLine=null;state.history=[];state.dirty=false; $("#yda-dirty").textContent=result.reviewed?"已核验":"已加载"; $("#yda-fields").disabled=false; $("#yda-yoyo-visibility").value=annotation.visibility|| (state.bbox?"visible":"uncertain"); $("#yda-trick-orientation").value=annotation.trick_orientation; $("#yda-string-visibility").value=annotation.string_visibility||"uncertain"; const status=annotation.string_review_status||"unresolved"; $("#yda-review-status").value=['approved','reviewed','needs_review','unresolved'].includes(status)?status:"unresolved"; $("#yda-notes").value=annotation.notes||""; $("#yda-validation").textContent="";state.baseline=editorSnapshot();syncReviewButton();syncResetButton(); const image=new Image(); image.onload=()=>{if(request!==state.loadSerial)return;state.image=image;canvas.classList.add("is-ready");$("#yda-empty").hidden=true;renderCanvas();}; image.onerror=()=>{if(request===state.loadSerial){$("#yda-empty").textContent="图像加载失败";toast("图像加载失败");}}; image.src=fileUrl(result.image_path); state.current=state.samples.findIndex(item=>item.key===key); $("#yda-position").textContent=`${state.current+1} / ${state.samples.length}`; $("#yda-prev").disabled=state.current<=0;$("#yda-next").disabled=state.current>=state.samples.length-1; renderSamples();renderLineList();syncCoordinateInputs(); $("#yda-status").textContent=result.label_path; }catch(error){if(request===state.loadSerial)toast(`加载失败：${error?.message||error}`);} }
async function openDataset(){ const path=$("#yda-dataset-path").value.trim(); if(!path)return toast("请输入或选择数据集路径");if(state.dirty&&!window.confirm("当前修改尚未保存，确定打开其他数据集吗？"))return; $("#yda-status").textContent="正在扫描数据集..."; try{ const result=await server.ui_open_annotation_dataset({dataset_path:path}); $("#yda-dataset-path").value=result.dataset_path;syncDatasetChoice(result.dataset_path);refreshDatasetOptions(result.dataset_path).catch(error=>toast(`刷新数据集失败：${error?.message||error}`));state.dataset=result;state.samples=result.samples;state.sample=null;state.dirty=false;renderSamples(); $("#yda-status").textContent=`已加载 ${result.sample_count} 条数据${result.error_count?`，${result.error_count} 条无法读取`:''}`; await selectSample(result.samples[0].key); }catch(error){$("#yda-status").textContent="数据集打开失败";toast(error?.message||error);} }
function finishLine(){ if(state.activeLine===null)return; if(state.lines[state.activeLine].length<2)state.lines.splice(state.activeLine,1); state.activeLine=null;$("#yda-finish-line").disabled=true;markDirty(); }
function startNewLine(recordHistory=true){ if(!state.image)return toast("请先加载图像");if(state.activeLine!==null)finishLine();if(recordHistory)pushHistory();state.lines.push([]);state.activeLine=state.lines.length-1;state.selectedLine=state.activeLine;state.redrawPending=false;setTool("string");$("#yda-finish-line").disabled=false;renderCanvas();renderLineList();toast("在图像上逐点绘制，双击或点击结束当前绳线"); }
function deleteSelectedPoint(){const selected=state.selectedPoint;if(!selected||state.selectedLine!==selected.line)return false;const line=state.lines[selected.line];if(!line||selected.point<0||selected.point>=line.length){state.selectedPoint=null;return false;}if(line.length<=2){toast("一条绳线至少保留两个点");return true;}pushHistory();line.splice(selected.point,1);state.selectedPoint=null;markDirty();toast("已删除标注点并连接相邻点");return true;}
function resetUnsavedChanges(){ if(!state.sample||!state.dirty||!state.baseline)return;if(!window.confirm("放弃当前图片的全部未保存修改并恢复原始标注吗？"))return;const baseline=state.baseline;state.bbox=baseline.bbox?baseline.bbox.slice():null;state.lines=baseline.lines.map(line=>line.map(point=>point.slice()));state.activeLine=null;state.selectedLine=null;state.drag=null;state.history=[];state.redrawPending=false;$("#yda-yoyo-visibility").value=baseline.yoyoVisibility;$("#yda-trick-orientation").value=baseline.trickOrientation;$("#yda-string-visibility").value=baseline.stringVisibility;$("#yda-review-status").value=baseline.reviewStatus;$("#yda-notes").value=baseline.notes;state.dirty=false;$("#yda-dirty").textContent=state.sample.reviewed?"已核验":"已加载";$("#yda-validation").textContent="";setTool("select");syncReviewButton();syncResetButton();renderCanvas();renderLineList();syncCoordinateInputs();toast("已恢复到保存前的原始标注"); }
canvas.addEventListener("pointerdown",event=>{if(!state.image)return;const point=canvasPoint(event);if(state.tool==="box"){pushHistory();state.drag={type:"drawbox",start:point};state.bbox=[point[0],point[1],point[0]+1,point[1]+1];canvas.setPointerCapture(event.pointerId);markDirty();return;}if(state.tool==="string"){if(state.activeLine===null){if(!state.redrawPending)pushHistory();state.redrawPending=false;state.lines.push([]);state.activeLine=state.lines.length-1;state.selectedLine=state.activeLine;}state.lines[state.activeLine].push(point);$("#yda-finish-line").disabled=false;markDirty();return;}const hit=hitTest(point);if(!hit){state.selectedLine=null;renderCanvas();renderLineList();return;}pushHistory();state.drag={...hit,start:point,original:cloneGeometry()};if(hit.line!==undefined)state.selectedLine=hit.line;canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{if(!state.drag||!state.image)return;const p=canvasPoint(event),d=state.drag;if(d.type==="drawbox"){state.bbox=[Math.min(d.start[0],p[0]),Math.min(d.start[1],p[1]),Math.max(d.start[0],p[0]),Math.max(d.start[1],p[1])];}else if(d.type==="point"){state.lines[d.line][d.point]=p;}else if(d.type==="corner"){const b=state.bbox.slice();if(d.corner===0||d.corner===3)b[0]=p[0];else b[2]=p[0];if(d.corner===0||d.corner===1)b[1]=p[1];else b[3]=p[1];state.bbox=[Math.min(b[0],b[2]),Math.min(b[1],b[3]),Math.max(b[0],b[2]),Math.max(b[1],b[3])];}else if(d.type==="line"){const dx=p[0]-d.start[0],dy=p[1]-d.start[1],w=state.image.naturalWidth,h=state.image.naturalHeight;state.lines[d.line]=d.original.lines[d.line].map(q=>[Math.max(0,Math.min(w,q[0]+dx)),Math.max(0,Math.min(h,q[1]+dy))]);}markDirty();});
canvas.addEventListener("pointerup",()=>{state.drag=null;}); canvas.addEventListener("dblclick",event=>{event.preventDefault();if(state.tool==="string"){const line=state.lines[state.activeLine];if(line?.length>1&&Math.hypot(line.at(-1)[0]-line.at(-2)[0],line.at(-1)[1]-line.at(-2)[1])<4)line.pop();finishLine();return;}if(state.tool==="select"){const point=canvasPoint(event),hit=hitTest(point);if(hit?.type==="line"){pushHistory();state.lines[hit.line].splice(hit.segment,0,point);state.selectedLine=hit.line;markDirty();}}});
canvas.addEventListener("contextmenu",event=>{if(state.tool!=="select"||!state.image)return;const hit=hitTest(canvasPoint(event));if(hit?.type!=="point")return;event.preventDefault();const line=state.lines[hit.line];if(line.length<=2)return toast("一条绳线至少保留两个点");pushHistory();line.splice(hit.point,1);state.selectedLine=hit.line;markDirty();});
canvas.addEventListener("pointerup",event=>{if(state.tool!=="select"||!state.image)return;const hit=hitTest(canvasPoint(event));state.selectedPoint=hit?.type==="point"?{line:hit.line,point:hit.point}:null;if(state.selectedPoint)state.selectedLine=state.selectedPoint.line;canvas.focus({preventScroll:true});renderCanvas();renderLineList();});
canvas.addEventListener("contextmenu",()=>{state.selectedPoint=null;renderCanvas();});
element.querySelectorAll("#yda-tools button").forEach(button=>button.onclick=()=>{if(state.activeLine!==null)finishLine();setTool(button.dataset.tool);});
$("#yda-finish-line").onclick=finishLine; $("#yda-undo").onclick=()=>{const value=state.history.pop();if(value)applyGeometry(value);};
$("#yda-reset").onclick=resetUnsavedChanges;
$("#yda-delete").onclick=()=>{if(state.selectedLine===null)return toast("请先选择一条绳线");pushHistory();state.lines.splice(state.selectedLine,1);state.selectedLine=null;markDirty();};
$("#yda-toggle-annotations").onclick=toggleAnnotations;
$("#yda-clear-box").onclick=()=>{pushHistory();state.bbox=null;$("#yda-yoyo-visibility").value="not_visible";markDirty();};
$("#yda-add-line").onclick=()=>startNewLine();
$("#yda-redraw-lines").onclick=()=>{if(state.lines.length&&!window.confirm("清空现有绳线并重新绘制吗？可使用撤销恢复。"))return;pushHistory();state.lines=[];state.activeLine=null;state.selectedLine=null;$("#yda-string-visibility").value="partial";startNewLine(false);markDirty();};
$("#yda-clear-lines").onclick=()=>{pushHistory();state.lines=[];state.activeLine=null;state.selectedLine=null;$("#yda-string-visibility").value="not_visible";markDirty();};
['x1','y1','x2','y2'].forEach(name=>$("#yda-"+name).addEventListener("change",()=>{const values=['x1','y1','x2','y2'].map(key=>Number($("#yda-"+key).value));if(values.every(Number.isFinite)){pushHistory();state.bbox=values;markDirty();}}));
['yoyo-visibility','trick-orientation','string-visibility','review-status','notes'].forEach(name=>$("#yda-"+name).addEventListener("change",()=>{state.dirty=true;$("#yda-dirty").textContent="未保存";syncReviewButton();syncResetButton();}));
$("#yda-save").onclick=async()=>{if(!state.sample)return; if(state.activeLine!==null)finishLine();const edit={yoyo_visibility:$("#yda-yoyo-visibility").value,trick_orientation:$("#yda-trick-orientation").value,yoyo_bbox_pixel:state.bbox,string_visibility:$("#yda-string-visibility").value,string_polylines_pixel:state.lines,string_review_status:$("#yda-review-status").value,bbox_review_status:"reviewed",reviewer:$("#yda-reviewer").value,notes:$("#yda-notes").value}; $("#yda-validation").textContent="正在保存...";try{const result=await server.ui_save_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:state.sample.key,edit});state.sample.annotation=result.annotation;state.sample.reviewed=false;state.dirty=false;state.baseline=editorSnapshot();$("#yda-dirty").textContent="已保存，待核验";$("#yda-validation").textContent="";const index=state.samples.findIndex(item=>item.key===state.sample.key);state.samples[index]={...state.samples[index],...result.summary,index};renderSamples();syncReviewButton();syncResetButton();toast("当前标注已保存");}catch(error){$("#yda-validation").textContent=error?.message||String(error);}};
$("#yda-review").onclick=async()=>{if(!state.sample||state.dirty)return;const confirmed=!state.sample.reviewed;$("#yda-review").disabled=true;try{const result=await server.ui_set_annotation_sample_reviewed({dataset_path:state.dataset.dataset_path,sample_key:state.sample.key,reviewer:$("#yda-reviewer").value,confirmed});state.sample={...state.sample,...result};const index=state.samples.findIndex(item=>item.key===state.sample.key);state.samples[index]={...state.samples[index],...result};$("#yda-dirty").textContent=result.reviewed?"已核验":"已取消核验";renderSamples();syncReviewButton();toast(result.reviewed?"已记录为核验完成":"已取消核验");if(result.reviewed&&index<state.samples.length-1)await selectSample(state.samples[index+1].key);}catch(error){toast(`核验状态保存失败：${error?.message||error}`);syncReviewButton();}};
$("#yda-prev").onclick=()=>{if(state.current>0)selectSample(state.samples[state.current-1].key);}; $("#yda-next").onclick=()=>{if(state.current<state.samples.length-1)selectSample(state.samples[state.current+1].key);};
$("#yda-search").addEventListener("input",renderSamples);$("#yda-filter").addEventListener("change",renderSamples);$("#yda-open").onclick=openDataset;
$("#yda-dataset-select").addEventListener("change",event=>{$("#yda-dataset-path").value=event.target.value;});
["pointerenter","focus","pointerdown"].forEach(name=>$("#yda-dataset-select").addEventListener(name,()=>refreshDatasetOptions($("#yda-dataset-select").value).catch(error=>toast(`刷新数据集失败：${error?.message||error}`))));
$("#yda-zoom").addEventListener("input",event=>setZoom(Number(event.target.value)));
$("#yda-viewport").addEventListener("wheel",event=>{if(!state.image)return;event.preventDefault();event.stopPropagation();setZoom(state.zoom*100+(event.deltaY<0?25:-25));},{passive:false,capture:true});
new ResizeObserver(()=>renderCanvas()).observe($("#yda-viewport"));
element.addEventListener("keydown",event=>{if(event.key!=="Delete"||event.target.matches("input,select,textarea"))return;if(deleteSelectedPoint())event.preventDefault();});
element.addEventListener("keydown",event=>{if(event.target.matches("input,select,textarea"))return;if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();$("#yda-save").click();return;}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();$("#yda-undo").click();return;}if(event.key.toLowerCase()==='h'){event.preventDefault();toggleAnnotations();}if(event.key==='ArrowLeft'){event.preventDefault();$("#yda-prev").click();}if(event.key==='ArrowRight'){event.preventDefault();$("#yda-next").click();}});
(async()=>{try{const datasets=await refreshDatasetOptions();if(datasets.length){$("#yda-dataset-path").value=datasets[0].path;syncDatasetChoice(datasets[0].path);await openDataset();}}catch(error){toast(`扫描失败：${error?.message||error}`);}})();
"""


def dataset_annotation_component_kwargs() -> dict[str, Any]:
    return {
        "value": DATASET_ANNOTATION_HTML,
        "css_template": DATASET_ANNOTATION_CSS,
        "js_on_load": DATASET_ANNOTATION_JS,
        "apply_default_css": False,
        "container": False,
        "padding": False,
        "server_functions": [
            ui_list_annotation_datasets,
            ui_open_annotation_dataset,
            ui_load_annotation_sample,
            ui_save_annotation_sample,
            ui_set_annotation_sample_reviewed,
        ],
    }
