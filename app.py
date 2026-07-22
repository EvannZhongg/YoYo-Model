import logging
import os
import json
import subprocess
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from annotation.annotator import annotate_image_for_dataset, run_detection_streaming
from annotation.prompts import EXAMPLE_PROMPTS, YOYO_DETECTION_PROMPT
from annotation.review import update_annotation_status
from annotation.video_frame_annotator import draw_visualization
from config import (
    BASE_DIR,
    DATASET_CONFIG,
    MODEL_CONFIG,
    SEMANTIC_STRING_CONFIG,
    STRING_SEGMENTATION_CONFIG,
    TRACKING_CONFIG,
)
from video_tracking.segment_review import load_segment_context, update_segment
from video_tracking.tracker import track_video
from video_dataset.string_review_queue import load_prediction_polylines


LOG_FILE = BASE_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


EXAMPLE_IMAGES = [
    BASE_DIR / "example1.jpg",
    BASE_DIR / "example2.png",
    BASE_DIR / "example3.png",
    None,
]


def _example_value(path: Path | None):
    if path is not None and path.exists():
        return str(path)
    return None


def _collect_dataset_images(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集图片目录不存在：{root}")

    pattern = "**/*" if DATASET_CONFIG.recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in DATASET_CONFIG.image_extensions
    )


def run_dataset_annotation_streaming(
    input_dir: str,
    output_dir: str,
    prompt: str,
    model: str,
    min_pixels_str: str,
    max_pixels_str: str,
):
    try:
        image_paths = _collect_dataset_images(input_dir)
    except Exception as exc:
        yield f"Error: {exc}"
        return

    if not image_paths:
        yield f"No images found in {input_dir}"
        return

    logs = [f"Found {len(image_paths)} image(s). Output: {output_dir}"]
    yield "\n".join(logs)

    success_count = 0
    failed_count = 0
    total_boxes = 0
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    for index, image_path in enumerate(image_paths, start=1):
        logs.append(f"[{index}/{len(image_paths)}] Annotating {image_path.name}")
        yield "\n".join(logs[-20:])

        try:
            result = annotate_image_for_dataset(
                image_path=image_path,
                input_dir=input_root,
                output_dir=output_root,
                prompt=prompt,
                model=model,
                min_pixels_str=min_pixels_str,
                max_pixels_str=max_pixels_str,
            )
            success_count += 1
            total_boxes += result["bbox_count"]
            logs.append(f"[{index}/{len(image_paths)}] Saved {result['bbox_count']} bbox(es): {result['label_path']}")
        except Exception as exc:
            failed_count += 1
            logger.exception("Failed to annotate %s", image_path)
            logs.append(f"[{index}/{len(image_paths)}] Failed: {image_path.name} - {exc}")

        yield "\n".join(logs[-20:])

    logs.append(
        f"Done. Success: {success_count}, Failed: {failed_count}, Total boxes: {total_boxes}. "
        f"Images/labels/visualizations are under {output_root}"
    )
    yield "\n".join(logs[-20:])


def _uploaded_video_path(video):
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("name")
    return getattr(video, "name", None)


def run_video_tracking(
    video,
    weights_path: str,
    output_dir: str,
    confidence: float,
    iou: float,
    imgsz: int,
    device: str,
    enable_pose: bool,
    pose_weights_path: str,
    enable_string_model: bool,
    string_weights_path: str,
    string_confidence: float,
    string_attachment_class: str,
    export_clips: bool,
    start_seconds: float,
    max_segment_seconds: float,
    activity_speed_diagonal_per_s: float,
):
    video_path = _uploaded_video_path(video)
    if not video_path:
        return None, None, None, None, None, None, [], "Error: No video provided."

    try:
        result = track_video(
            source_video_path=video_path,
            weights_path=weights_path,
            output_dir=output_dir,
            confidence=confidence,
            iou=iou,
            imgsz=int(imgsz),
            device=device.strip(),
            enable_pose=bool(enable_pose),
            pose_weights_path=pose_weights_path.strip() or None,
            auto_download_pose=TRACKING_CONFIG.auto_download_pose,
            enable_string_model=bool(enable_string_model),
            string_weights_path=string_weights_path.strip() or None,
            string_confidence=float(string_confidence),
            string_attachment_class=string_attachment_class,
            export_json=True,
            export_clips=bool(export_clips),
            activity_speed_diagonal_per_s=float(activity_speed_diagonal_per_s),
            start_seconds=float(start_seconds),
            max_segment_seconds=float(max_segment_seconds),
        )
    except Exception as exc:
        logger.exception("Video tracking failed")
        return None, None, None, None, None, None, [], f"Error: {exc}"

    status = (
        f"Done. Frames: {result['frame_count']}\n"
        f"Output: {result['output_video']}\n"
        f"Segments: {len(result['segments'])}\n"
        f"Approved clip-tokens: {result.get('trick_token_count', 0)}\n"
        f"Bad cases: {result['bad_case_counts']}\n"
        f"String model: {result.get('string_model', 'disabled')}\n"
        f"Run manifest: {result['run_manifest']}\n"
        f"Weights: {result['weights']}"
    )
    clips = [item["output_video"] for item in result["segments"] if item.get("output_video")]
    return (
        result["output_video"],
        result["metadata_jsonl"],
        result["segments_json"],
        result["run_manifest"],
        result.get("review_sheet") or None,
        result.get("trick_token_manifest") or None,
        clips,
        status,
    )


def _review_label_paths(dataset_dir: str, status: str, split: str, component: str = "all") -> list[Path]:
    root = Path(dataset_dir)
    labels_root = root / "annotations" / "labels"
    if not labels_root.exists():
        return []
    paths = sorted(labels_root.rglob("*.json"))
    results = []
    for path in paths:
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
            queue_rank = {}
    return sorted(results, key=lambda path: (queue_rank.get(str(path.resolve()), 10**9), str(path)))


def _review_visualization_path(label_path: Path, dataset_dir: str) -> Path:
    labels_root = Path(dataset_dir) / "annotations" / "labels"
    relative = label_path.relative_to(labels_root)
    return Path(dataset_dir) / "annotations" / "visualizations" / relative.with_name(f"{relative.stem}_vis.jpg")


def review_label_preview(label_path: str | None, dataset_dir: str):
    if not label_path:
        return None, "", ""
    path = Path(label_path)
    if not path.exists():
        return None, "Label not found", label_path
    data = json.loads(path.read_text(encoding="utf-8"))
    preview = _review_visualization_path(path, dataset_dir)
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
                data = dict(data)
                data["review_queue"] = selected
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
    summary = json.dumps(data, ensure_ascii=False, indent=2)
    return str(preview) if preview.exists() else None, summary, str(path)


def refresh_review_queue(dataset_dir: str, status: str, split: str, component: str):
    paths = _review_label_paths(dataset_dir, status, split, component)
    choices = [str(path) for path in paths]
    selected = choices[0] if choices else ""
    image, summary, selected = review_label_preview(selected, dataset_dir)
    return gr.update(choices=choices, value=selected), image, summary


def workbench_navigate(
    label_path: str | None,
    dataset_dir: str,
    status: str,
    split: str,
    component: str,
    direction: int,
):
    """Move through the active review queue while refreshing every editor field."""
    paths = [str(path) for path in _review_label_paths(dataset_dir, status, split, component)]
    if not paths:
        return (gr.update(choices=[], value=""),) + workbench_preview(None, dataset_dir)
    current = str(label_path or "")
    try:
        index = paths.index(current)
    except ValueError:
        index = 0 if int(direction) >= 0 else len(paths) - 1
    selected = paths[(index + (1 if int(direction) >= 0 else -1)) % len(paths)]
    return (gr.update(choices=paths, value=selected),) + workbench_preview(selected, dataset_dir)


def apply_review_status(
    status: str,
    label_path: str,
    notes: str,
    dataset_dir: str,
    split: str,
    component: str,
    string_attachment_class: str | None = None,
    string_visibility: str | None = None,
    yoyo_visibility: str | None = None,
    scene_label: str | None = None,
):
    if not label_path:
        return gr.update(), None, "No label selected"
    try:
        update_annotation_status(
            label_path,
            status,
            reviewer="gradio",
            notes=notes or None,
            component=component,
            string_attachment_class=string_attachment_class,
            string_visibility=string_visibility,
            yoyo_visibility=yoyo_visibility,
            scene_label=scene_label,
        )
    except Exception as exc:
        return gr.update(), None, f"Review update failed: {exc}"
    choices = [str(path) for path in _review_label_paths(dataset_dir, "auto_labeled_needs_review", split, component)]
    selected = choices[0] if choices else ""
    image, summary, selected = review_label_preview(selected, dataset_dir)
    return gr.update(choices=choices, value=selected), image, summary


def _workbench_stats(dataset_dir: str) -> str:
    """Return compact, refreshable counts for the visual workbench."""
    root = Path(dataset_dir)
    labels = sorted((root / "annotations" / "labels").rglob("*.json")) if (root / "annotations" / "labels").exists() else []
    counts = {
        "labels": len(labels),
        "bbox_pending": 0,
        "bbox_approved": 0,
        "string_pending": 0,
        "string_approved": 0,
        "rejected": 0,
        "trick": 0,
        "transition": 0,
        "non_trick": 0,
        "scene_unknown": 0,
    }
    for path in labels:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bbox_status = data.get("bbox_review_status", data.get("review_status", "auto_labeled_needs_review"))
        string_status = data.get("string_review_status", "auto_labeled_needs_review")
        counts["bbox_pending"] += bbox_status not in {"approved", "reviewed", "rejected"}
        counts["bbox_approved"] += bbox_status in {"approved", "reviewed"}
        counts["string_pending"] += string_status not in {"approved", "reviewed", "rejected"}
        counts["string_approved"] += string_status in {"approved", "reviewed"}
        counts["rejected"] += data.get("review_status") == "rejected"
        scene = str(data.get("scene_label", "unknown"))
        key = scene if scene in {"trick", "transition", "non_trick"} else "scene_unknown"
        counts[key] += 1
    frames_path = root / "frames.jsonl"
    frame_count = sum(1 for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()) if frames_path.exists() else 0
    return (
        f"Labels: {counts['labels']} | Frame records: {frame_count}\n"
        f"BBox pending: {counts['bbox_pending']} | BBox approved: {counts['bbox_approved']}\n"
        f"String pending: {counts['string_pending']} | String approved: {counts['string_approved']}\n"
        f"Scenes - trick: {counts['trick']} | transition: {counts['transition']} | "
        f"non-trick: {counts['non_trick']} | unknown: {counts['scene_unknown']}\n"
        f"Rejected frames: {counts['rejected']}"
    )


def _run_workbench_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        return f"Command failed to start: {exc}"
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return f"Exit code: {completed.returncode}\n{output}" if output else f"Exit code: {completed.returncode}"


def workbench_build(videos_dir: str, dataset_dir: str, sample_fps: float, max_frames_per_video: int) -> str:
    return _run_workbench_command([
        "-m", "video_dataset.build", "--videos-dir", videos_dir, "--output-dir", dataset_dir,
        "--sample-fps", str(sample_fps), "--max-frames-per-video", str(int(max_frames_per_video)),
        "--action-group", DATASET_CONFIG.current_action_group,
    ])


def workbench_audit(dataset_dir: str, strict: bool) -> str:
    args = ["-m", "video_dataset.audit", "--dataset-dir", dataset_dir]
    if strict:
        args.append("--strict")
    return _run_workbench_command(args)


def workbench_model_registry() -> str:
    return _run_workbench_command(["model_registry.py"])


def workbench_candidates(dataset_dir: str, weights: str, sample_fps: float, confidence: float, max_candidates: int) -> str:
    return _run_workbench_command([
        "-m", "video_dataset.select_candidates", "--dataset-dir", dataset_dir, "--weights", weights,
        "--sample-fps", str(sample_fps), "--confidence", str(confidence),
        "--max-candidates-per-video", str(int(max_candidates)),
    ])


def workbench_vlm(dataset_dir: str, split: str, limit: int, workers: int, candidates_only: bool) -> str:
    args = ["-m", "annotation.video_frame_annotator", "--dataset-dir", dataset_dir, "--split", split, "--workers", str(int(workers))]
    if int(limit) > 0:
        args.extend(["--limit", str(int(limit))])
    if candidates_only:
        args.append("--candidates-only")
    return _run_workbench_command(args)


def workbench_qa_export(dataset_dir: str, yolo_dir: str) -> str:
    qa_output = _run_workbench_command(["-m", "annotation.qa", "--dataset-dir", dataset_dir])
    export_output = _run_workbench_command([
        "-m", "yolo_training.prepare_dataset", "--annotations-dir", f"{dataset_dir}/annotations",
        "--output-dir", yolo_dir, "--clear",
    ])
    return f"QA\n{qa_output}\n\nYOLO export\n{export_output}"


def workbench_prepare_string(dataset_dir: str, output_dir: str) -> str:
    return _run_workbench_command([
        "-m", "string_segmentation.prepare_dataset",
        "--annotations-dir", f"{dataset_dir}/annotations",
        "--output-dir", output_dir,
        "--clear",
    ])


def workbench_train_string(dataset_dir: str, output_dir: str, epochs: int, device: str) -> str:
    args = [
        "-m", "string_segmentation.train",
        "--annotations-dir", f"{dataset_dir}/annotations",
        "--dataset-dir", output_dir,
        "--epochs", str(int(epochs)),
        "--auto-download",
        "--clear-dataset",
    ]
    if str(device).strip():
        args.extend(["--device", str(device).strip()])
    return _run_workbench_command(args)


def workbench_train_semantic(string_dataset_dir: str, output_dir: str, run_name: str, epochs: int, device: str) -> str:
    args = [
        "-m", "string_segmentation.train_semantic",
        "--dataset-dir", string_dataset_dir,
        "--project", output_dir,
        "--name", run_name.strip() or SEMANTIC_STRING_CONFIG.run_name,
        "--epochs", str(int(epochs)),
    ]
    if str(device).strip():
        args.extend(["--device", str(device).strip()])
    return _run_workbench_command(args)


def workbench_evaluate_semantic(weights: str, string_dataset_dir: str, device: str) -> str:
    args = [
        "-m", "string_segmentation.evaluate_semantic",
        "--weights", weights,
        "--dataset-dir", string_dataset_dir,
        "--split", "test",
    ]
    if str(device).strip():
        args.extend(["--device", str(device).strip()])
    return _run_workbench_command(args)


def workbench_prelabel_strings(dataset_dir: str, split: str, limit: int) -> str:
    args = ["-m", "annotation.string_prelabel", "--dataset-dir", dataset_dir, "--split", split]
    if int(limit) > 0:
        args.extend(["--limit", str(int(limit))])
    return _run_workbench_command(args)


def workbench_string_review_queue(
    dataset_dir: str,
    split: str,
    limit: int,
    with_model: bool,
    weights: str,
    device: str,
) -> tuple[str, str | None]:
    args = [
        "-m", "video_dataset.string_review_queue",
        "--dataset-dir", dataset_dir,
        "--split", split,
        "--limit", str(int(limit)),
    ]
    if with_model:
        args.extend(["--with-model", "--weights", weights])
        if str(device).strip():
            args.extend(["--device", str(device).strip()])
    output = _run_workbench_command(args)
    sheet = Path(dataset_dir) / "review_sheets" / "string_review_queue.jpg"
    return output, str(sheet) if sheet.exists() else None


def workbench_refresh(dataset_dir: str, status: str, split: str, component: str):
    queue, image, summary = refresh_review_queue(dataset_dir, status, split, component)
    return queue, image, summary, _workbench_stats(dataset_dir)


def workbench_apply(
    status: str,
    label_path: str,
    notes: str,
    dataset_dir: str,
    split: str,
    component: str,
    string_attachment_class: str,
    string_visibility: str,
    yoyo_visibility: str,
    scene_label: str,
):
    queue, image, summary = apply_review_status(
        status,
        label_path,
        notes,
        dataset_dir,
        split,
        component,
        string_attachment_class,
        string_visibility,
        yoyo_visibility,
        scene_label,
    )
    return queue, image, summary, _workbench_stats(dataset_dir)


def workbench_preview(label_path: str | None, dataset_dir: str):
    image, summary, selected = review_label_preview(label_path, dataset_dir)
    if not label_path or not Path(label_path).exists():
        return image, summary, "", "", "uncertain", "uncertain", "unknown", "unknown", []
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    bbox = json.dumps(data.get("yoyo_bbox_pixel"), ensure_ascii=False) if data.get("yoyo_bbox_pixel") else ""
    polylines = data.get("string_polylines_pixel")
    if not polylines and data.get("string_polyline_pixel"):
        polylines = [data["string_polyline_pixel"]]
    polyline = json.dumps(polylines, ensure_ascii=False) if polylines else ""
    return (
        image,
        summary,
        bbox,
        polyline,
        data.get("visibility", "uncertain"),
        data.get("string_visibility", "uncertain"),
        data.get("string_attachment_class", "unknown"),
        data.get("scene_label", "unknown"),
        data.get("bad_case", []),
    )


def _queue_prediction_preview(label_path: Path, dataset_dir: str | None) -> Path | None:
    root = Path(dataset_dir) if dataset_dir else None
    if root is None:
        root = next(
            (
                parent.parent.parent
                for parent in label_path.parents
                if parent.name == "labels" and parent.parent.name == "annotations"
            ),
            None,
        )
    if root is None:
        return None
    queue_path = root / "string_review_queue.json"
    if not queue_path.exists():
        return None
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        resolved = str(label_path.resolve())
        row = next(
            (
                item
                for item in queue.get("rows", [])
                if str(Path(str(item.get("label_path", ""))).resolve()) == resolved
            ),
            None,
        )
        preview = Path(str(((row or {}).get("model") or {}).get("prediction_preview", "")))
        return preview if preview.is_file() else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def workbench_detail_crop(label_path: str | None, dataset_dir: str | None = None):
    """Return raw, annotation, and optional model crops for visual review."""
    if not label_path or not Path(label_path).exists():
        return None
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    source = Path(str(data.get("source_image", "")))
    if not source.exists():
        return None
    image = Image.open(source).convert("RGB")
    anchors: list[tuple[float, float]] = []
    bbox = data.get("yoyo_bbox_pixel")
    if isinstance(bbox, list) and len(bbox) == 4:
        anchors.extend([(float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[3]))])
    polylines = data.get("string_polylines_pixel")
    if not polylines and data.get("string_polyline_pixel"):
        polylines = [data["string_polyline_pixel"]]
    for stroke in polylines or []:
        for point in stroke if isinstance(stroke, list) else []:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                anchors.append((float(point[0]), float(point[1])))
    for point in (data.get("hands_pixel") or {}).values():
        if isinstance(point, (list, tuple)) and len(point) == 2:
            anchors.append((float(point[0]), float(point[1])))
    if not anchors:
        anchors = [(image.width * 0.25, image.height * 0.2), (image.width * 0.75, image.height * 0.8)]
    min_x, max_x = min(point[0] for point in anchors), max(point[0] for point in anchors)
    min_y, max_y = min(point[1] for point in anchors), max(point[1] for point in anchors)
    center_x, center_y = (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
    crop_width = min(float(image.width), max(900.0, (max_x - min_x) * 1.8))
    crop_height = min(float(image.height), max(620.0, (max_y - min_y) * 1.8))
    left = max(0, min(image.width - crop_width, center_x - crop_width * 0.5))
    top = max(0, min(image.height - crop_height, center_y - crop_height * 0.5))
    crop_box = (int(round(left)), int(round(top)), int(round(left + crop_width)), int(round(top + crop_height)))

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    stroke_width = max(4, image.width // 800)
    if isinstance(bbox, list) and len(bbox) == 4:
        draw.rectangle(tuple(float(value) for value in bbox), outline=(40, 255, 70, 255), width=stroke_width)
    colors = [(0, 220, 255, 255), (255, 130, 20, 255), (255, 80, 210, 255)]
    for index, stroke in enumerate(polylines or []):
        points = [tuple(float(value) for value in point) for point in stroke if isinstance(point, (list, tuple)) and len(point) == 2]
        if len(points) >= 2:
            draw.line(points, fill=colors[index % len(colors)], width=stroke_width, joint="curve")
    for name, point in (data.get("hands_pixel") or {}).items():
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = float(point[0]), float(point[1])
        radius = max(10, image.width // 300)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 230, 30, 255), width=stroke_width)
        draw.text((x + radius + 2, y - radius), str(name), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for polygon in data.get("string_mask_polygons_pixel") or []:
        points = [tuple(float(value) for value in point) for point in polygon if isinstance(point, (list, tuple)) and len(point) == 2]
        if len(points) >= 3:
            draw.polygon(points, fill=(255, 120, 20, 70), outline=(255, 180, 30, 255))

    raw_crop = image.crop(crop_box)
    overlay_crop = overlay.crop(crop_box)
    prediction_path = _queue_prediction_preview(Path(label_path), dataset_dir)
    prediction_crop = None
    if prediction_path is not None:
        prediction = Image.open(prediction_path).convert("RGB")
        if prediction.size != image.size:
            prediction = prediction.resize(image.size)
        prediction_crop = prediction.crop(crop_box)
    header = 36
    columns = 3 if prediction_crop is not None else 2
    comparison = Image.new("RGB", (raw_crop.width * columns, raw_crop.height + header), (24, 24, 24))
    comparison.paste(raw_crop, (0, header))
    comparison.paste(overlay_crop, (raw_crop.width, header))
    if prediction_crop is not None:
        comparison.paste(prediction_crop, (raw_crop.width * 2, header))
    caption = ImageDraw.Draw(comparison)
    caption.text((12, 10), "RAW DETAIL", fill=(255, 255, 255))
    caption.text((raw_crop.width + 12, 10), "ANNOTATION OVERLAY", fill=(255, 255, 255))
    if prediction_crop is not None:
        caption.text((raw_crop.width * 2 + 12, 10), "SEMANTIC V3 REVIEW ONLY", fill=(255, 255, 255))
    return comparison


def _json_geometry(text: str):
    if not str(text or "").strip():
        return None
    return json.loads(text)


def _string_strokes(value) -> list[list[list[float]]]:
    if not isinstance(value, list) or not value:
        return []
    if all(isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point) for point in value):
        return [value]
    strokes = []
    for stroke in value:
        if isinstance(stroke, list) and all(
            isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point)
            for point in stroke
        ):
            strokes.append(stroke)
    return strokes


def workbench_use_semantic_prediction(
    label_path: str,
    dataset_dir: str,
    bbox_text: str,
    current_polyline_text: str,
):
    """Load model geometry into the unsaved editor for manual correction."""
    if not label_path or not Path(label_path).exists():
        return None, current_polyline_text, "No label selected."
    try:
        strokes = load_prediction_polylines(dataset_dir, label_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _editor_preview(label_path, bbox_text, current_polyline_text), current_polyline_text, f"Could not load semantic prediction: {exc}"
    if not strokes:
        return _editor_preview(label_path, bbox_text, current_polyline_text), current_polyline_text, "No editable semantic prediction is available for this frame."
    value = json.dumps(strokes, ensure_ascii=False)
    return (
        _editor_preview(label_path, bbox_text, value),
        value,
        f"Loaded {len(strokes)} semantic stroke(s) into the unsaved editor. Correct them before saving.",
    )


def _editor_preview(label_path: str, bbox_text: str, polyline_text: str):
    """Draw unsaved click edits over the source frame for immediate QA."""
    if not label_path or not Path(label_path).exists():
        return None
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    image = Image.open(data["source_image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = _json_geometry(bbox_text)
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
        draw.rectangle(tuple(float(value) for value in bbox), outline=(30, 255, 30), width=max(3, image.width // 900))
    elif isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
        x, y = bbox[0]
        radius = max(7, image.width // 350)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(30, 255, 30), width=4)

    strokes = _string_strokes(_json_geometry(polyline_text))
    colors = [(0, 220, 255), (255, 140, 20), (255, 80, 210), (80, 255, 120)]
    for stroke_index, stroke in enumerate(strokes, start=1):
        valid = [tuple(float(value) for value in point) for point in stroke]
        color = colors[(stroke_index - 1) % len(colors)]
        if len(valid) >= 2:
            draw.line(valid, fill=color, width=max(3, image.width // 900), joint="curve")
        radius = max(6, image.width // 450)
        for index, (x, y) in enumerate(valid, start=1):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 60, 80), width=2)
            draw.text((x + radius + 2, y - radius), f"{stroke_index}.{index}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def workbench_click_geometry(label_path: str, bbox_text: str, polyline_text: str, click_tool: str, evt: gr.SelectData):
    """Use source-image coordinates from a Gradio click to build geometry."""
    if not label_path:
        return None, bbox_text, polyline_text, "No label selected"
    if not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2:
        return _editor_preview(label_path, bbox_text, polyline_text), bbox_text, polyline_text, "Unsupported click coordinate"
    x, y = round(float(evt.index[0]), 2), round(float(evt.index[1]), 2)
    if click_tool == "BBox: two corners":
        current = _json_geometry(bbox_text)
        if isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
            x0, y0 = current[0]
            bbox_text = json.dumps([min(x0, x), min(y0, y), max(x0, x), max(y0, y)], ensure_ascii=False)
            hint = "BBox complete; save geometry or click again to restart."
        else:
            bbox_text = json.dumps([[x, y]], ensure_ascii=False)
            hint = "BBox first corner set; click the opposite corner."
    else:
        strokes = _string_strokes(_json_geometry(polyline_text))
        if not strokes:
            strokes = [[]]
        strokes[-1].append([x, y])
        polyline_text = json.dumps(strokes, ensure_ascii=False)
        hint = f"String stroke {len(strokes)}, point {len(strokes[-1])} added."
    return _editor_preview(label_path, bbox_text, polyline_text), bbox_text, polyline_text, hint


def workbench_undo_geometry(label_path: str, bbox_text: str, polyline_text: str, click_tool: str):
    if click_tool == "BBox: two corners":
        bbox = _json_geometry(bbox_text)
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_text = json.dumps([[bbox[0], bbox[1]]], ensure_ascii=False)
        else:
            bbox_text = ""
        hint = "Removed the last bbox corner."
    else:
        strokes = _string_strokes(_json_geometry(polyline_text))
        if strokes:
            if strokes[-1]:
                strokes[-1].pop()
            if not strokes[-1]:
                strokes.pop()
        polyline_text = json.dumps(strokes, ensure_ascii=False) if strokes else ""
        hint = "Removed the last string point."
    return _editor_preview(label_path, bbox_text, polyline_text), bbox_text, polyline_text, hint


def workbench_new_string_stroke(label_path: str, bbox_text: str, polyline_text: str):
    strokes = _string_strokes(_json_geometry(polyline_text))
    if strokes and len(strokes[-1]) < 2:
        return _editor_preview(label_path, bbox_text, polyline_text), polyline_text, "Current stroke needs at least two points."
    strokes.append([])
    value = json.dumps(strokes, ensure_ascii=False)
    return _editor_preview(label_path, bbox_text, value), value, f"Started string stroke {len(strokes)}."


def workbench_clear_geometry(label_path: str, bbox_text: str, polyline_text: str, click_tool: str):
    if click_tool == "BBox: two corners":
        bbox_text, hint = "", "BBox cleared. Set yoyo visibility to absent/out_of_frame/uncertain when appropriate."
    else:
        polyline_text, hint = "", "String cleared. Set string visibility to not_visible/uncertain when appropriate."
    return _editor_preview(label_path, bbox_text, polyline_text), bbox_text, polyline_text, hint


def workbench_save_geometry(
    label_path: str,
    bbox_text: str,
    polyline_text: str,
    dataset_dir: str,
    component: str,
    visibility: str,
    string_visibility: str,
    string_attachment_class: str,
    scene_label: str,
    bad_case: list[str] | None,
):
    if not label_path:
        return None, "No label selected", "", "", _workbench_stats(dataset_dir)
    path = Path(label_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = data.get("image_size", [0, 0])
    if scene_label not in {"trick", "transition", "non_trick", "unknown"}:
        raise ValueError(f"unsupported scene label: {scene_label}")

    def parse_value(text: str):
        if not text.strip():
            return None
        value = json.loads(text)
        return value

    def norm_point(point):
        return [round(float(point[0]) / float(width) * 999.0, 3), round(float(point[1]) / float(height) * 999.0, 3)]

    if component in {"bbox", "all"}:
        bbox = parse_value(bbox_text)
        if isinstance(bbox, list) and len(bbox) == 1:
            raise ValueError("bbox needs two corner clicks before it can be saved")
        if visibility in {"visible", "partially_visible"} and bbox is None:
            raise ValueError(f"visibility={visibility} requires a bbox")
        if visibility in {"absent", "out_of_frame"}:
            bbox = None
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4 or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                raise ValueError("bbox must be [x1, y1, x2, y2] in pixels")
            bbox = [float(value) for value in bbox]
            data["yoyo_bbox_pixel"] = [round(value, 2) for value in bbox]
            data["yoyo_bbox_2d"] = [round(max(0.0, min(999.0, value)), 3) for value in (bbox[0] / width * 999.0, bbox[1] / height * 999.0, bbox[2] / width * 999.0, bbox[3] / height * 999.0)]
            data["bbox"] = [{"label": "yoyo", "sub_label": "visible yoyo body", "bbox_2d": data["yoyo_bbox_2d"], "bbox_pixel": data["yoyo_bbox_pixel"]}]
        else:
            data["yoyo_bbox_pixel"] = None
            data["yoyo_bbox_2d"] = None
            data["bbox"] = []
        data["bbox_review_status"] = "auto_labeled_needs_review"
        data["visibility"] = visibility

    if component in {"string", "all"}:
        allowed_attachment_classes = {
            "hand_and_yoyo_attached",
            "yoyo_detached",
            "hand_detached",
            "unknown",
        }
        if string_attachment_class not in allowed_attachment_classes:
            raise ValueError(f"unsupported string attachment class: {string_attachment_class}")
        polylines = _string_strokes(parse_value(polyline_text))
        if string_visibility in {"visible", "partial"} and not polylines:
            raise ValueError(f"string_visibility={string_visibility} requires at least one string stroke")
        if string_visibility == "not_visible":
            polylines = []
        if polylines:
            if any(len(stroke) < 2 for stroke in polylines):
                raise ValueError("every string stroke must contain at least two [x, y] points")
            polylines = [[[float(point[0]), float(point[1])] for point in stroke] for stroke in polylines]
            data["string_polylines_pixel"] = [
                [[round(point[0], 2), round(point[1], 2)] for point in stroke] for stroke in polylines
            ]
            data["string_polylines_2d"] = [[norm_point(point) for point in stroke] for stroke in polylines]
            data["string_polyline_pixel"] = data["string_polylines_pixel"][0]
            data["string_polyline_2d"] = data["string_polylines_2d"][0]
        else:
            data["string_polylines_pixel"] = None
            data["string_polylines_2d"] = None
            data["string_polyline_pixel"] = None
            data["string_polyline_2d"] = None
        # A manual stroke edit invalidates any prior automatic color-mask proposal.
        data.pop("string_mask_polygons_pixel", None)
        data.pop("string_prelabel", None)
        data["string_review_status"] = "auto_labeled_needs_review"
        data["string_visibility"] = string_visibility
        data["string_attachment_class"] = string_attachment_class

    data["scene_label"] = scene_label
    data["bad_case"] = sorted(set(bad_case or []))
    data["bad_case"] = [value for value in data["bad_case"] if value not in {"non_trick_scene", "transition_scene"}]
    if scene_label == "non_trick":
        data["bad_case"].append("non_trick_scene")
    elif scene_label == "transition":
        data["bad_case"].append("transition_scene")
    data["bad_case"] = sorted(set(data["bad_case"]))
    if data.get("visibility") in {"absent", "out_of_frame"}:
        data["bad_case"] = sorted(set(data["bad_case"] + ["yoyo_not_visible"]))
    if data.get("string_visibility") == "not_visible":
        data["bad_case"] = sorted(set(data["bad_case"] + ["string_not_visible"]))

    data["review_status"] = "partially_reviewed"
    data["reviewed_at_utc"] = None
    data["reviewer"] = "geometry_editor"
    data["review_notes"] = "Geometry edited in Video Workbench; component returned to visual review queue."
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    vis_path = _review_visualization_path(path, dataset_dir)
    draw_visualization(Path(data["source_image"]), data, vis_path)
    image, summary, bbox, polyline, _, _, _, _, _ = workbench_preview(str(path), dataset_dir)
    return image, summary, bbox, polyline, _workbench_stats(dataset_dir)


def segment_review_preview(segment_id: str | int | None, segments_path: str):
    if not segment_id or not segments_path or not Path(segments_path).exists():
        return None, "", 0.0, 0.0, "", "", "auto_candidate_needs_review"
    try:
        _, _, segments = load_segment_context(segments_path)
        segment = next(item for item in segments if str(item.get("segment_id")) == str(segment_id))
    except Exception as exc:
        return None, f"Segment load failed: {exc}", 0.0, 0.0, "", "", "auto_candidate_needs_review"
    clip = segment.get("output_video") if segment.get("output_video") and Path(segment["output_video"]).exists() else None
    return (
        clip,
        json.dumps(segment, ensure_ascii=False, indent=2),
        float(segment.get("start_time_s", 0.0)),
        float(segment.get("end_time_s", 0.0)),
        str(segment.get("trick_label", "")),
        str(segment.get("review_notes", "")),
        str(segment.get("review_status", "auto_candidate_needs_review")),
    )


def load_segment_review_queue(segments_path: str):
    if not segments_path or not Path(segments_path).exists():
        return gr.update(choices=[], value=None), None, "No segments.json selected", 0.0, 0.0, "", "", "auto_candidate_needs_review"
    try:
        _, _, segments = load_segment_context(segments_path)
        choices = [str(item.get("segment_id")) for item in segments]
    except Exception as exc:
        return gr.update(choices=[], value=None), None, f"Segment load failed: {exc}", 0.0, 0.0, "", "", "auto_candidate_needs_review"
    selected = choices[0] if choices else None
    preview = segment_review_preview(selected, segments_path)
    return gr.update(choices=choices, value=selected), *preview


def save_segment_review(segment_id: str, segments_path: str, status: str, start_time_s: float, end_time_s: float, trick_label: str, notes: str):
    try:
        segment = update_segment(
            segments_path,
            int(segment_id),
            status,
            float(start_time_s),
            float(end_time_s),
            trick_label,
            notes,
            export_clip=True,
        )
    except Exception as exc:
        return None, f"Segment update failed: {exc}", start_time_s, end_time_s, trick_label, notes, status
    preview = segment_review_preview(str(segment["segment_id"]), segments_path)
    return preview


def create_demo():
    with gr.Blocks(title="YoYo Auto Annotation") as demo:
        gr.Markdown("# YoYo Auto Annotation")
        gr.Markdown("使用大模型先自动标注悠悠球图片，保存原图副本、坐标 JSON 和可视化图片，供后续训练检测/追踪模型使用。")

        with gr.Tabs():
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        example_selector = gr.Radio(
                            choices=["YoYo Prompt", "Example 2: Cars", "Example 3: People", "Upload your own image"],
                            value="YoYo Prompt",
                            label="Select Example",
                        )

                        image_input = gr.Image(
                            label="Input Image",
                            type="pil",
                            value=_example_value(EXAMPLE_IMAGES[0]),
                        )

                        user_prompt = gr.Textbox(
                            label="Prompt",
                            lines=10,
                            value=YOYO_DETECTION_PROMPT,
                            interactive=True,
                        )

                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=list(MODEL_CONFIG.available_models),
                            value=MODEL_CONFIG.default_model,
                            interactive=True,
                        )

                        min_pixels_input = gr.Textbox(
                            label="Min Image Tokens",
                            value=MODEL_CONFIG.min_image_tokens,
                            interactive=True,
                        )

                        max_pixels_input = gr.Textbox(
                            label="Max Image Tokens",
                            value=MODEL_CONFIG.max_image_tokens,
                            interactive=True,
                        )

                        run_btn = gr.Button("Run Object Detection", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_vis = gr.Image(label="Detection Result")
                        raw_output = gr.Textbox(
                            label="Raw Model Response",
                            lines=8,
                            interactive=False,
                            buttons=["copy"],
                        )
                        status_output = gr.Textbox(label="Status / Summary", lines=3, interactive=False)

                def on_example_select(selection):
                    idx = ["YoYo Prompt", "Example 2: Cars", "Example 3: People", "Upload your own image"].index(selection)
                    return _example_value(EXAMPLE_IMAGES[idx]), EXAMPLE_PROMPTS[idx]

                example_selector.change(
                    fn=on_example_select,
                    inputs=[example_selector],
                    outputs=[image_input, user_prompt],
                )

                run_btn.click(
                    fn=run_detection_streaming,
                    inputs=[image_input, user_prompt, model_dropdown, min_pixels_input, max_pixels_input],
                    outputs=[output_vis, raw_output, status_output],
                )

            with gr.Tab("Dataset Auto Label"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_input_dir = gr.Textbox(
                            label="Dataset Image Directory",
                            value=str(DATASET_CONFIG.image_input_dir),
                            interactive=True,
                        )
                        dataset_output_dir = gr.Textbox(
                            label="Annotation Output Directory",
                            value=str(DATASET_CONFIG.annotation_output_dir),
                            interactive=True,
                        )
                        dataset_prompt = gr.Textbox(
                            label="Dataset Prompt",
                            lines=10,
                            value=YOYO_DETECTION_PROMPT,
                            interactive=True,
                        )
                        dataset_model = gr.Dropdown(
                            label="Model",
                            choices=list(MODEL_CONFIG.available_models),
                            value=MODEL_CONFIG.default_model,
                            interactive=True,
                        )
                        dataset_min_pixels = gr.Textbox(
                            label="Min Image Tokens",
                            value=MODEL_CONFIG.min_image_tokens,
                            interactive=True,
                        )
                        dataset_max_pixels = gr.Textbox(
                            label="Max Image Tokens",
                            value=MODEL_CONFIG.max_image_tokens,
                            interactive=True,
                        )
                        annotate_btn = gr.Button("Annotate Dataset", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        dataset_status = gr.Textbox(
                            label="Batch Status",
                            lines=24,
                            interactive=False,
                            buttons=["copy"],
                        )

                annotate_btn.click(
                    fn=run_dataset_annotation_streaming,
                    inputs=[
                        dataset_input_dir,
                        dataset_output_dir,
                        dataset_prompt,
                        dataset_model,
                        dataset_min_pixels,
                        dataset_max_pixels,
                    ],
                    outputs=[dataset_status],
                )

            with gr.Tab("Video Workbench"):
                gr.Markdown(
                    f"当前数据组：**{DATASET_CONFIG.current_action_group}**。视频数据从抽帧、候选筛选、VLM 预标注到人工核验和 YOLO 导出的统一操作台；未来 2A/3A/4A/5A 暂不参与当前训练。所有预标注都必须经过组件级可视化审核。"
                )
                with gr.Row():
                    workbench_videos_dir = gr.Textbox(label="Videos Directory", value=str(BASE_DIR / "videos"))
                    workbench_dataset_dir = gr.Textbox(label="Video Dataset Directory", value=str(BASE_DIR / "datasets" / "video_v1"))
                    workbench_yolo_dir = gr.Textbox(label="YOLO Output Directory", value=str(BASE_DIR / "datasets" / "video_v1" / "yolo_v3"))
                    workbench_string_dir = gr.Textbox(label="String Segmentation Dataset", value=str(STRING_SEGMENTATION_CONFIG.dataset_dir))

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("数据准备")
                        wb_sample_fps = gr.Slider(label="Frame Sample FPS", minimum=0.05, maximum=2.0, value=1.0, step=0.05)
                        wb_max_frames = gr.Number(label="Max Frames / Video (0 = unlimited)", value=0, precision=0)
                        wb_build = gr.Button("Build / Refresh Frame Manifest", variant="primary")
                        wb_build_log = gr.Textbox(label="Frame Build Log", lines=7, interactive=False)
                        wb_audit_strict = gr.Checkbox(label="Strict audit (warnings fail)", value=False)
                        wb_audit = gr.Button("Audit Dataset Integrity")
                        wb_audit_log = gr.Textbox(label="Dataset Audit Log", lines=7, interactive=False)
                    with gr.Column():
                        gr.Markdown("候选筛选")
                        wb_weights = gr.Textbox(label="Bootstrap Weights", value=str(TRACKING_CONFIG.weights_path))
                        wb_candidate_conf = gr.Slider(label="Candidate Confidence", minimum=0.01, maximum=0.95, value=0.20, step=0.01)
                        wb_max_candidates = gr.Number(label="Max Candidates / Video (0 = unlimited)", value=5, precision=0)
                        wb_candidates = gr.Button("Select Candidate Frames")
                        wb_candidates_log = gr.Textbox(label="Candidate Log", lines=7, interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("VLM 预标注")
                        wb_split = gr.Dropdown(label="Split", choices=["all", "train", "val", "test"], value="all")
                        wb_limit = gr.Number(label="Annotation Limit (0 = all)", value=0, precision=0)
                        wb_workers = gr.Slider(label="Concurrent Requests", minimum=1, maximum=16, value=4, step=1)
                        wb_candidates_only = gr.Checkbox(label="Candidates Only", value=True)
                        wb_vlm = gr.Button("Run VLM Pre-annotation")
                        wb_vlm_log = gr.Textbox(label="VLM Log", lines=7, interactive=False)
                    with gr.Column():
                        gr.Markdown("QA 和导出")
                        wb_qa_export = gr.Button("Run QA + Export Reviewed YOLO Dataset", variant="primary")
                        wb_model_registry = gr.Button("Refresh Model Registry")
                        wb_qa_log = gr.Textbox(label="QA / Export Log", lines=11, interactive=False)
                        wb_stats = gr.Textbox(label="Dataset Review Statistics", lines=5, interactive=False)
                    with gr.Column():
                        gr.Markdown("绳子分割模型")
                        wb_string_epochs = gr.Number(label="Epochs", value=STRING_SEGMENTATION_CONFIG.epochs, precision=0)
                        wb_string_device = gr.Textbox(label="Device", value=STRING_SEGMENTATION_CONFIG.device, placeholder="0, cpu, or empty")
                        wb_string_prepare = gr.Button("Export Reviewed String Dataset")
                        wb_string_train = gr.Button("Train String Segmentation", variant="primary")
                        wb_semantic_project = gr.Textbox(
                            label="Semantic Model Project",
                            value=str(SEMANTIC_STRING_CONFIG.project),
                        )
                        wb_semantic_name = gr.Textbox(
                            label="Semantic Run Name",
                            value=SEMANTIC_STRING_CONFIG.run_name,
                        )
                        wb_semantic_weights = gr.Textbox(
                            label="Semantic Weights (for test evaluation)",
                            value=str(SEMANTIC_STRING_CONFIG.project / SEMANTIC_STRING_CONFIG.run_name / "weights" / "best.pt"),
                        )
                        wb_semantic_train = gr.Button("Train Semantic String Model")
                        wb_semantic_eval = gr.Button("Evaluate Semantic Model on Test")
                        wb_string_prelabel_split = gr.Dropdown(label="Color Prelabel Split", choices=["all", "train", "val", "test"], value="all")
                        wb_string_prelabel_limit = gr.Number(label="Color Prelabel Limit (0 = all)", value=0, precision=0)
                        wb_string_prelabel = gr.Button("Generate Color String Proposals")
                        wb_queue_split = gr.Dropdown(label="Review Queue Split", choices=["all", "train", "val", "test"], value="all")
                        wb_queue_limit = gr.Number(label="Review Queue Batch Size (0 = all)", value=16, precision=0)
                        wb_queue_with_model = gr.Checkbox(label="Use semantic v3 uncertainty", value=True)
                        wb_queue_device = gr.Textbox(label="Queue Device", value=TRACKING_CONFIG.device, placeholder="cuda, 0, or cpu")
                        wb_queue = gr.Button("Build String Review Queue")
                        wb_queue_log = gr.Textbox(label="Review Queue Log", lines=6, interactive=False)
                        wb_queue_sheet = gr.Image(label="Ranked String Review Batch", type="filepath", interactive=False)
                        wb_string_log = gr.Textbox(label="String Dataset / Training Log", lines=9, interactive=False)

                gr.Markdown("可视化审核队列。选择 String 后，沿画面中实际可见的绳段依次点击；遮挡或不可见间隔使用新的 stroke。BBox 使用两个对角点。黄色线/绿色框是尚未保存的编辑预览。")
                with gr.Row():
                    wb_review_status = gr.Dropdown(label="Queue Status", choices=["auto_labeled_needs_review", "partially_reviewed", "reviewed", "approved", "rejected"], value="auto_labeled_needs_review")
                    wb_review_split = gr.Dropdown(label="Queue Split", choices=["all", "train", "val", "test"], value="all")
                    wb_review_component = gr.Dropdown(label="Component", choices=["bbox", "string", "all"], value="bbox")
                    wb_review_refresh = gr.Button("Refresh Review Queue")
                wb_review_label = gr.Dropdown(label="Frame Annotation", choices=[])
                with gr.Row():
                    wb_review_previous = gr.Button("Previous Frame")
                    wb_review_next = gr.Button("Next Frame")
                with gr.Row():
                    wb_review_image = gr.Image(label="Visual Geometry Editor (click to add)", type="filepath", interactive=False)
                    wb_review_json = gr.Textbox(label="Annotation + QA", lines=20, interactive=False)
                wb_review_crop = gr.Image(label="Raw / Annotation / Semantic Detail", interactive=False)
                with gr.Row():
                    wb_click_tool = gr.Radio(
                        label="Click Tool",
                        choices=["String: add centerline point", "BBox: two corners"],
                        value="String: add centerline point",
                    )
                    wb_new_stroke = gr.Button("Start New String Stroke")
                    wb_undo_point = gr.Button("Undo Last Point")
                    wb_clear_geometry = gr.Button("Clear Selected Geometry")
                    wb_use_semantic = gr.Button("Load Semantic Prediction")
                wb_editor_hint = gr.Textbox(label="Geometry Editor Hint", interactive=False)
                with gr.Row():
                    wb_bbox_editor = gr.Textbox(label="BBox Pixel JSON [x1, y1, x2, y2]", lines=2)
                    wb_string_editor = gr.Textbox(label="String Strokes Pixel JSON [[[x, y], ...], ...]", lines=3)
                with gr.Row():
                    wb_yoyo_visibility = gr.Dropdown(
                        label="YoYo Visibility",
                        choices=["visible", "partially_visible", "occluded", "out_of_frame", "absent", "uncertain"],
                        value="uncertain",
                    )
                    wb_string_visibility = gr.Dropdown(
                        label="String Visibility",
                        choices=["visible", "partial", "not_visible", "uncertain"],
                        value="uncertain",
                    )
                    wb_string_attachment = gr.Dropdown(
                        label="String Attachment (current 1A)",
                        choices=[
                            ("Current 1A: hand and yoyo attached", "hand_and_yoyo_attached"),
                            ("Unknown / not visible / needs review", "unknown"),
                        ],
                        value=TRACKING_CONFIG.string_attachment_class,
                    )
                    wb_scene_label = gr.Dropdown(
                        label="Scene Label",
                        choices=[
                            ("Trick", "trick"),
                            ("Transition / setup", "transition"),
                            ("Non-trick / entrance / ceremony", "non_trick"),
                            ("Unknown / needs review", "unknown"),
                        ],
                        value="unknown",
                    )
                wb_bad_case = gr.CheckboxGroup(
                    label="Bad Case / Scene Flags",
                    choices=["yoyo_not_visible", "yoyo_edge_clipped", "motion_blur", "string_not_visible", "string_ambiguous", "hands_occluded", "multiple_yoyo", "non_trick_scene", "transition_scene"],
                )
                wb_review_notes = gr.Textbox(label="Review Notes", lines=2)
                with gr.Row():
                    wb_save_geometry = gr.Button("Save Geometry + Requeue")
                    wb_review_approve = gr.Button("Approve Component")
                    wb_review_marked = gr.Button("Mark Component Reviewed")
                    wb_review_reject = gr.Button("Reject Component")

                wb_build.click(workbench_build, [workbench_videos_dir, workbench_dataset_dir, wb_sample_fps, wb_max_frames], wb_build_log)
                wb_audit.click(workbench_audit, [workbench_dataset_dir, wb_audit_strict], wb_audit_log)
                wb_candidates.click(workbench_candidates, [workbench_dataset_dir, wb_weights, wb_sample_fps, wb_candidate_conf, wb_max_candidates], wb_candidates_log)
                wb_vlm.click(workbench_vlm, [workbench_dataset_dir, wb_split, wb_limit, wb_workers, wb_candidates_only], wb_vlm_log)
                wb_qa_export.click(workbench_qa_export, [workbench_dataset_dir, workbench_yolo_dir], wb_qa_log)
                wb_model_registry.click(workbench_model_registry, [], wb_qa_log)
                wb_string_prepare.click(workbench_prepare_string, [workbench_dataset_dir, workbench_string_dir], wb_string_log)
                wb_string_train.click(
                    workbench_train_string,
                    [workbench_dataset_dir, workbench_string_dir, wb_string_epochs, wb_string_device],
                    wb_string_log,
                )
                wb_semantic_train.click(
                    workbench_train_semantic,
                    [workbench_string_dir, wb_semantic_project, wb_semantic_name, wb_string_epochs, wb_string_device],
                    wb_string_log,
                )
                wb_semantic_eval.click(
                    workbench_evaluate_semantic,
                    [wb_semantic_weights, workbench_string_dir, wb_string_device],
                    wb_string_log,
                )
                wb_string_prelabel.click(
                    workbench_prelabel_strings,
                    [workbench_dataset_dir, wb_string_prelabel_split, wb_string_prelabel_limit],
                    wb_string_log,
                )
                wb_queue.click(
                    workbench_string_review_queue,
                    [workbench_dataset_dir, wb_queue_split, wb_queue_limit, wb_queue_with_model, wb_semantic_weights, wb_queue_device],
                    [wb_queue_log, wb_queue_sheet],
                )
                wb_review_refresh.click(workbench_refresh, [workbench_dataset_dir, wb_review_status, wb_review_split, wb_review_component], [wb_review_label, wb_review_image, wb_review_json, wb_stats])
                wb_review_label.change(
                    workbench_preview,
                    [wb_review_label, workbench_dataset_dir],
                    [wb_review_image, wb_review_json, wb_bbox_editor, wb_string_editor, wb_yoyo_visibility, wb_string_visibility, wb_string_attachment, wb_scene_label, wb_bad_case],
                )
                wb_review_label.change(workbench_detail_crop, [wb_review_label, workbench_dataset_dir], [wb_review_crop])
                for button, direction in ((wb_review_previous, -1), (wb_review_next, 1)):
                    button.click(
                        lambda label, dataset, status, split, component, direction=direction: workbench_navigate(
                            label, dataset, status, split, component, direction
                        ),
                        [wb_review_label, workbench_dataset_dir, wb_review_status, wb_review_split, wb_review_component],
                        [
                            wb_review_label, wb_review_image, wb_review_json, wb_bbox_editor, wb_string_editor,
                            wb_yoyo_visibility, wb_string_visibility, wb_string_attachment, wb_scene_label, wb_bad_case,
                        ],
                    )
                wb_review_image.select(
                    workbench_click_geometry,
                    [wb_review_label, wb_bbox_editor, wb_string_editor, wb_click_tool],
                    [wb_review_image, wb_bbox_editor, wb_string_editor, wb_editor_hint],
                )
                wb_undo_point.click(
                    workbench_undo_geometry,
                    [wb_review_label, wb_bbox_editor, wb_string_editor, wb_click_tool],
                    [wb_review_image, wb_bbox_editor, wb_string_editor, wb_editor_hint],
                )
                wb_new_stroke.click(
                    workbench_new_string_stroke,
                    [wb_review_label, wb_bbox_editor, wb_string_editor],
                    [wb_review_image, wb_string_editor, wb_editor_hint],
                )
                wb_clear_geometry.click(
                    workbench_clear_geometry,
                    [wb_review_label, wb_bbox_editor, wb_string_editor, wb_click_tool],
                    [wb_review_image, wb_bbox_editor, wb_string_editor, wb_editor_hint],
                )
                wb_use_semantic.click(
                    workbench_use_semantic_prediction,
                    [wb_review_label, workbench_dataset_dir, wb_bbox_editor, wb_string_editor],
                    [wb_review_image, wb_string_editor, wb_editor_hint],
                )
                wb_save_event = wb_save_geometry.click(
                    workbench_save_geometry,
                    [
                        wb_review_label, wb_bbox_editor, wb_string_editor, workbench_dataset_dir, wb_review_component,
                        wb_yoyo_visibility, wb_string_visibility, wb_string_attachment, wb_scene_label, wb_bad_case,
                    ],
                    [wb_review_image, wb_review_json, wb_bbox_editor, wb_string_editor, wb_stats],
                )
                wb_save_event.then(workbench_detail_crop, [wb_review_label, workbench_dataset_dir], [wb_review_crop])
                for button, status in ((wb_review_approve, "approved"), (wb_review_marked, "reviewed"), (wb_review_reject, "rejected")):
                    button.click(
                        lambda label, notes, dataset, split, component, attachment, string_visibility, yoyo_visibility, scene_label, status=status: workbench_apply(
                            status, label, notes, dataset, split, component, attachment, string_visibility, yoyo_visibility, scene_label
                        ),
                        [
                            wb_review_label, wb_review_notes, workbench_dataset_dir, wb_review_split,
                            wb_review_component, wb_string_attachment, wb_string_visibility,
                            wb_yoyo_visibility, wb_scene_label,
                        ],
                        [wb_review_label, wb_review_image, wb_review_json, wb_stats],
                    )

                gr.Markdown("招式片段审核")
                with gr.Row():
                    wb_segments_path = gr.Textbox(label="segments.json", value="")
                    wb_segments_load = gr.Button("Load Segment Queue")
                wb_segment_id = gr.Dropdown(label="Segment", choices=[])
                with gr.Row():
                    wb_segment_video = gr.Video(label="Candidate Trick Clip")
                    wb_segment_json = gr.Textbox(label="Segment Metadata", lines=14, interactive=False)
                with gr.Row():
                    wb_segment_start = gr.Number(label="Start (s)", minimum=0, value=0)
                    wb_segment_end = gr.Number(label="End (s)", minimum=0, value=1)
                    wb_segment_status = gr.Dropdown(label="Segment Status", choices=["auto_candidate_needs_review", "edited", "approved", "irrelevant", "rejected"], value="auto_candidate_needs_review")
                with gr.Row():
                    wb_segment_label = gr.Textbox(label="Trick Label", placeholder="e.g. mount / bind / throw")
                    wb_segment_notes = gr.Textbox(label="Segment Review Notes")
                wb_segment_save = gr.Button("Save Segment Review + Re-export Clip", variant="primary")
                wb_segments_load.click(
                    load_segment_review_queue,
                    [wb_segments_path],
                    [wb_segment_id, wb_segment_video, wb_segment_json, wb_segment_start, wb_segment_end, wb_segment_label, wb_segment_notes, wb_segment_status],
                )
                wb_segment_id.change(
                    segment_review_preview,
                    [wb_segment_id, wb_segments_path],
                    [wb_segment_video, wb_segment_json, wb_segment_start, wb_segment_end, wb_segment_label, wb_segment_notes, wb_segment_status],
                )
                wb_segment_save.click(
                    save_segment_review,
                    [wb_segment_id, wb_segments_path, wb_segment_status, wb_segment_start, wb_segment_end, wb_segment_label, wb_segment_notes],
                    [wb_segment_video, wb_segment_json, wb_segment_start, wb_segment_end, wb_segment_label, wb_segment_notes, wb_segment_status],
                )

            with gr.Tab("Annotation Review"):
                review_dataset_dir = gr.Textbox(
                    label="Video Dataset Directory",
                    value=str(BASE_DIR / "datasets" / "video_v1"),
                )
                with gr.Row():
                    review_status_filter = gr.Dropdown(
                        label="Review Status",
                        choices=["auto_labeled_needs_review", "partially_reviewed", "reviewed", "approved", "rejected"],
                        value="auto_labeled_needs_review",
                    )
                    review_split_filter = gr.Dropdown(
                        label="Split",
                        choices=["all", "train", "val", "test"],
                        value="all",
                    )
                    review_component = gr.Dropdown(
                        label="Review Component",
                        choices=["bbox", "string", "all"],
                        value="bbox",
                    )
                    review_refresh = gr.Button("Refresh Queue")
                review_label = gr.Dropdown(label="Annotation JSON", choices=[])
                review_image = gr.Image(label="Visual Verification", type="filepath")
                review_json = gr.Textbox(label="Annotation JSON", lines=18, interactive=False)
                review_notes = gr.Textbox(label="Review Notes", lines=3)
                with gr.Row():
                    review_approve = gr.Button("Approve")
                    review_marked = gr.Button("Mark Reviewed")
                    review_reject = gr.Button("Reject")

                review_refresh.click(
                    fn=refresh_review_queue,
                    inputs=[review_dataset_dir, review_status_filter, review_split_filter, review_component],
                    outputs=[review_label, review_image, review_json],
                )
                review_label.change(
                    fn=review_label_preview,
                    inputs=[review_label, review_dataset_dir],
                    outputs=[review_image, review_json, review_label],
                )
                for button, status in ((review_approve, "approved"), (review_marked, "reviewed"), (review_reject, "rejected")):
                    button.click(
                        fn=lambda label, notes, dataset, split, component, status=status: apply_review_status(status, label, notes, dataset, split, component),
                        inputs=[review_label, review_notes, review_dataset_dir, review_split_filter, review_component],
                        outputs=[review_label, review_image, review_json],
                    )

            with gr.Tab("Video Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Input Video")
                        tracking_weights = gr.Textbox(
                            label="YOLO Weights",
                            value=str(TRACKING_CONFIG.weights_path),
                            interactive=True,
                        )
                        tracking_output_dir = gr.Textbox(
                            label="Output Directory",
                            value=str(TRACKING_CONFIG.output_dir),
                            interactive=True,
                        )
                        tracking_conf = gr.Slider(
                            label="Confidence",
                            minimum=0.01,
                            maximum=0.99,
                            value=TRACKING_CONFIG.confidence,
                            step=0.01,
                        )
                        tracking_iou = gr.Slider(
                            label="IoU",
                            minimum=0.1,
                            maximum=0.95,
                            value=TRACKING_CONFIG.iou,
                            step=0.01,
                        )
                        tracking_imgsz = gr.Number(
                            label="Image Size",
                            value=TRACKING_CONFIG.imgsz,
                            precision=0,
                            interactive=True,
                        )
                        tracking_device = gr.Textbox(
                            label="Device",
                            value=TRACKING_CONFIG.device,
                            interactive=True,
                            placeholder="0, cpu, or empty for auto",
                        )
                        tracking_pose = gr.Checkbox(
                            label="Pose / hand landmarks",
                            value=TRACKING_CONFIG.enable_pose,
                        )
                        tracking_pose_weights = gr.Textbox(
                            label="Pose Weights",
                            value=str(TRACKING_CONFIG.pose_weights_path),
                            interactive=True,
                        )
                        tracking_string_model = gr.Checkbox(
                            label="String segmentation model",
                            value=TRACKING_CONFIG.enable_string_model,
                        )
                        tracking_string_weights = gr.Textbox(
                            label="String Segmentation Weights",
                            value=str(TRACKING_CONFIG.string_weights_path),
                            interactive=True,
                        )
                        tracking_string_conf = gr.Slider(
                            label="String Confidence",
                            minimum=0.01,
                            maximum=0.95,
                            value=TRACKING_CONFIG.string_confidence,
                            step=0.01,
                        )
                        tracking_string_attachment = gr.Dropdown(
                            label="String Attachment (current 1A)",
                            choices=[
                                ("Current 1A: hand and yoyo attached", "hand_and_yoyo_attached"),
                                ("Unknown / not visible / needs review", "unknown"),
                            ],
                            value=TRACKING_CONFIG.string_attachment_class,
                        )
                        tracking_export_clips = gr.Checkbox(
                            label="Export candidate clips",
                            value=TRACKING_CONFIG.export_clips,
                        )
                        tracking_start_seconds = gr.Number(
                            label="Start Time (seconds)",
                            value=0,
                            minimum=0,
                        )
                        tracking_max_segment = gr.Number(
                            label="Maximum Exported Valid Segment Seconds",
                            value=TRACKING_CONFIG.max_segment_seconds,
                            minimum=1,
                            maximum=180,
                            precision=1,
                        )
                        tracking_activity_speed = gr.Number(
                            label="Activity Speed (image diagonals / second)",
                            value=TRACKING_CONFIG.activity_speed_diagonal_per_s,
                            minimum=0.0,
                            maximum=2.0,
                            precision=3,
                        )
                        track_btn = gr.Button("Run Video Tracking", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        tracking_output_video = gr.Video(label="Tracked Video")
                        tracking_metadata = gr.File(label="Frame Metadata JSONL")
                        tracking_segments = gr.File(label="Segment Manifest")
                        tracking_run_manifest = gr.File(label="Run Manifest")
                        tracking_review_sheet = gr.Image(label="Tracking Visual Review", type="filepath")
                        tracking_token_manifest = gr.File(label="Valid Trick Clip-token Manifest")
                        tracking_clip_files = gr.File(label="Candidate Clips", file_count="multiple")
                        tracking_status = gr.Textbox(
                            label="Tracking Status",
                            lines=8,
                            interactive=False,
                            buttons=["copy"],
                        )

                track_btn.click(
                    fn=run_video_tracking,
                    inputs=[
                        video_input,
                        tracking_weights,
                        tracking_output_dir,
                        tracking_conf,
                        tracking_iou,
                        tracking_imgsz,
                        tracking_device,
                        tracking_pose,
                        tracking_pose_weights,
                        tracking_string_model,
                        tracking_string_weights,
                        tracking_string_conf,
                        tracking_string_attachment,
                        tracking_export_clips,
                        tracking_start_seconds,
                        tracking_max_segment,
                        tracking_activity_speed,
                    ],
                    outputs=[
                        tracking_output_video,
                        tracking_metadata,
                        tracking_segments,
                        tracking_run_manifest,
                        tracking_review_sheet,
                        tracking_token_manifest,
                        tracking_clip_files,
                        tracking_status,
                    ],
                )

    return demo


if __name__ == "__main__":
    os.makedirs(DATASET_CONFIG.temp_output_dir, exist_ok=True)
    os.makedirs(DATASET_CONFIG.annotation_output_dir, exist_ok=True)
    demo = create_demo()
    demo.launch(
        server_name=os.getenv("APP_HOST", "0.0.0.0"),
        server_port=int(os.getenv("APP_PORT", "7866")),
        share=False,
        theme=gr.themes.Soft(),
    )
