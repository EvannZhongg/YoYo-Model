import base64
import json
import logging
import mimetypes
import re
import shutil
import uuid
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import oss2
from openai import OpenAI
from oss2.credentials import EnvironmentVariableCredentialsProvider
from PIL import Image

from config import DATASET_CONFIG, MODEL_CONFIG, OSS_CONFIG, DatasetConfig, ModelConfig, OSSConfig


logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFA500",
    "#800080",
    "#00FFFF",
    "#FF00FF",
    "#FFFF00",
    "#008000",
    "#FFC0CB",
    "#A52A2A",
    "#808080",
    "#000080",
    "#FFD700",
    "#C0C0C0",
]

MAX_IMAGE_SIZE = 2048


def resize_image_if_needed(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width <= MAX_IMAGE_SIZE and height <= MAX_IMAGE_SIZE:
        return img

    if width >= height:
        new_width = MAX_IMAGE_SIZE
        new_height = int(height * (MAX_IMAGE_SIZE / width))
    else:
        new_height = MAX_IMAGE_SIZE
        new_width = int(width * (MAX_IMAGE_SIZE / height))

    logger.info("Resizing image from %sx%s to %sx%s", width, height, new_width, new_height)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image_to_base64(image_path_or_pil: str | Path | Image.Image) -> str:
    if isinstance(image_path_or_pil, (str, Path)):
        with open(image_path_or_pil, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    buffered = BytesIO()
    image_path_or_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_client(model_config: ModelConfig = MODEL_CONFIG) -> OpenAI:
    return OpenAI(base_url=model_config.base_url, api_key=model_config.api_key)


def upload_to_oss(
    data_or_path: str | Path | bytes,
    object_name: str,
    oss_config: OSSConfig = OSS_CONFIG,
) -> str:
    if not oss_config.endpoint or not oss_config.bucket_name or not oss_config.region:
        raise RuntimeError("OSS_ENDPOINT, OSS_BUCKET_NAME, and OSS_REGION must be configured.")

    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, oss_config.endpoint, oss_config.bucket_name, region=oss_config.region)

    if isinstance(data_or_path, (str, Path)):
        response = bucket.put_object_from_file(object_name, str(data_or_path))
    else:
        response = bucket.put_object(object_name, data_or_path)

    if response.status == HTTPStatus.OK:
        return bucket.sign_url(
            "GET",
            object_name,
            oss_config.signed_url_expires_seconds,
            slash_safe=True,
        )

    raise RuntimeError(f"OSS upload failed with status {response.status}")


def image_url_for_model(
    local_img_path: str | Path,
    object_name: str,
    model_config: ModelConfig = MODEL_CONFIG,
) -> str:
    transport = model_config.image_transport.lower()
    local_img_path = Path(local_img_path)

    if transport == "base64":
        mime_type = mimetypes.guess_type(local_img_path.name)[0] or "image/png"
        encoded = encode_image_to_base64(local_img_path)
        return f"data:{mime_type};base64,{encoded}"

    if transport == "oss":
        return upload_to_oss(local_img_path, object_name)

    raise ValueError("model.image_transport must be either 'oss' or 'base64'.")


def _json_candidates(text: str) -> Iterable[str]:
    yield text.strip()

    for match in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        yield match.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        yield text[start : end + 1]


def _normalize_bbox_object(obj: dict, index: int) -> dict | None:
    coords = obj.get("bbox_2d")
    if not isinstance(coords, list) or len(coords) != 4:
        return None

    try:
        bbox_2d = [float(value) for value in coords]
    except (TypeError, ValueError):
        return None

    return {
        "bbox_2d": bbox_2d,
        "label": str(obj.get("label") or f"object_{index + 1}"),
        "sub_label": str(obj.get("sub_label") or ""),
    }


def parse_bboxes_from_response(text: str) -> list[dict]:
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items = parsed if isinstance(parsed, list) else [parsed]
        results = [
            normalized
            for i, item in enumerate(items)
            if isinstance(item, dict)
            if (normalized := _normalize_bbox_object(item, i)) is not None
        ]
        if results:
            return results
        if isinstance(parsed, list) and not parsed:
            return []

    results = []
    object_pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
    for i, match in enumerate(re.findall(object_pattern, text, re.DOTALL)):
        cleaned = re.sub(r"#.*$", "", match, flags=re.MULTILINE).strip()
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        normalized = _normalize_bbox_object(obj, i)
        if normalized:
            results.append(normalized)

    if results:
        return results

    bbox_pattern = r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    for i, (x1, y1, x2, y2) in enumerate(re.findall(bbox_pattern, text)):
        results.append(
            {
                "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                "label": f"object_{i + 1}",
                "sub_label": "",
            }
        )

    return results


def denormalize_bbox(bbox_coords: list[float], img_width: int, img_height: int) -> list[int]:
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_coords
    x1 = int(x1_norm / 999.0 * img_width)
    y1 = int(y1_norm / 999.0 * img_height)
    x2 = int(x2_norm / 999.0 * img_width)
    y2 = int(y2_norm / 999.0 * img_height)
    return [x1, y1, x2, y2]


def add_pixel_bboxes(bboxes: list[dict], image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    normalized = []
    for bbox in bboxes:
        item = dict(bbox)
        item["bbox_pixel"] = denormalize_bbox(item["bbox_2d"], width, height)
        normalized.append(item)
    return normalized


def visualize_bboxes(
    image: str | Path | Image.Image,
    bboxes: list[dict],
    save_path: str | Path,
    header_text: str | None = None,
) -> str:
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    img_width, img_height = img.size
    img_array = np.array(img)

    if header_text:
        fig, ax = plt.subplots(1, figsize=(12, 10))
    else:
        fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.imshow(img_array)

    if header_text:
        fig.text(
            0.02,
            0.98,
            header_text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"),
            wrap=True,
        )
        plt.subplots_adjust(top=0.88)

    for i, bbox in enumerate(bboxes):
        coords = bbox.get("bbox_2d", [0, 0, 0, 0])
        if len(coords) != 4:
            continue

        x1, y1, x2, y2 = denormalize_bbox(coords, img_width, img_height)
        color = COLORS[i % len(COLORS)]
        label = bbox.get("label", "unknown")
        sub_label = bbox.get("sub_label", "")
        display_text = label if not sub_label else f"{label}\n({sub_label})"

        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.25)
        ax.add_patch(rect)

        box_width = x2 - x1
        font_size = max(6, min(9, box_width / 36))
        ax.text(
            x1,
            y1,
            display_text,
            fontsize=font_size,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor=color, alpha=0.6, linewidth=0),
            color="white",
        )

    ax.set_axis_off()
    if not header_text:
        plt.tight_layout(pad=0.5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    return str(save_path)


def eval_pixel_string(s: str) -> int:
    try:
        return int(eval(s.strip().replace(" ", ""), {"__builtins__": {}}))
    except Exception as exc:
        raise ValueError(f"Invalid pixel expression: {s}") from exc


def _image_token_limits(min_pixels_str: str, max_pixels_str: str) -> tuple[int, int]:
    min_tokens = eval_pixel_string(min_pixels_str)
    max_tokens = eval_pixel_string(max_pixels_str)
    return min_tokens * 32 * 32, max_tokens * 32 * 32


def _prepare_local_image(image: str | Path | Image.Image, output_dir: Path, model: str, run_id: uuid.UUID) -> tuple[Image.Image, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        img = Image.open(image_path)
        ext = image_path.suffix.lstrip(".").lower() or "png"
    else:
        img = image
        ext = "png"

    local_img_path = output_dir / f"{model}_img_{run_id}.{ext}"
    img.save(local_img_path)
    return img, local_img_path


def _messages_for_image(file_url: str, prompt: str, min_pixels: int, max_pixels: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": file_url},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _save_result_json(result_data: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    return output_path


def _call_model_once(
    messages: list[dict],
    model: str,
    model_config: ModelConfig = MODEL_CONFIG,
) -> str:
    response = create_client(model_config).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=model_config.max_response_tokens,
        extra_body={"enable_thinking": model_config.enable_thinking},
        stream=False,
    )
    return response.choices[0].message.content or ""


def run_detection_streaming(
    image: str | Path | Image.Image,
    user_prompt: str,
    model: str,
    min_pixels_str: str,
    max_pixels_str: str,
):
    logger.info("=" * 60)
    logger.info("Starting detection request")
    logger.info("Model: %s", model)
    logger.info("User prompt: %s", user_prompt)

    try:
        min_pixels, max_pixels = _image_token_limits(min_pixels_str, max_pixels_str)
    except ValueError as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        yield None, "", error_msg
        return

    if image is None:
        error_msg = "Error: No image provided."
        logger.error(error_msg)
        yield None, "", error_msg
        return

    run_id = uuid.uuid4()
    img, local_img_path = _prepare_local_image(image, DATASET_CONFIG.temp_output_dir, model, run_id)
    logger.info("Local image saved: %s", local_img_path)
    logger.info("Image size: %sx%s", img.size[0], img.size[1])

    object_name_img = f"{OSS_CONFIG.object_prefix}/{model}_img_{run_id}{local_img_path.suffix}"
    try:
        file_url = image_url_for_model(local_img_path, object_name_img)
        logger.info("Image prepared via %s: %s...", MODEL_CONFIG.image_transport, file_url[:100])
    except Exception as e:
        error_msg = f"Image Prepare Error: {str(e)}"
        logger.error(error_msg)
        yield None, "", error_msg
        return

    messages = _messages_for_image(file_url, user_prompt, min_pixels, max_pixels)
    full_response = ""

    try:
        logger.info("Sending API request to %s...", MODEL_CONFIG.base_url)
        stream = create_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MODEL_CONFIG.max_response_tokens,
            extra_body={"enable_thinking": MODEL_CONFIG.enable_thinking},
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                yield None, full_response, "Streaming response..."
    except Exception as e:
        error_msg = f"API Error: {str(e)}"
        logger.error(error_msg)
        yield None, f"[ERROR] {error_msg}\n\n---\n\nPartial response before error:\n{full_response}", error_msg
        return

    bboxes = parse_bboxes_from_response(full_response)
    logger.info("Parsed %s bounding box(es)", len(bboxes))

    if not bboxes:
        warning_msg = "No bounding boxes detected in the response. See raw output for details."
        logger.warning(warning_msg)
        yield None, full_response, warning_msg
        return

    result_data = {
        "model": model,
        "image_size": [img.size[0], img.size[1]],
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "user_prompt": user_prompt,
        "bbox": add_pixel_bboxes(bboxes, img.size),
    }

    local_json_path = DATASET_CONFIG.temp_output_dir / f"{model}_result_{run_id}_result.json"
    _save_result_json(result_data, local_json_path)
    logger.info("JSON saved: %s", local_json_path)

    local_vis_path = DATASET_CONFIG.temp_output_dir / f"{model}_vis_{run_id}.png"
    vis_path = visualize_bboxes(img, bboxes, local_vis_path)
    logger.info("Visualization saved: %s", vis_path)

    summary = f"Detected {len(bboxes)} object(s). Image size: {img.size[0]}x{img.size[1]}. JSON: {local_json_path}"
    logger.info(summary)
    logger.info("=" * 60)
    yield vis_path, full_response, summary


def _iter_image_paths(input_dir: Path, dataset_config: DatasetConfig = DATASET_CONFIG) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset image directory does not exist: {input_dir}")
    pattern = "**/*" if dataset_config.recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in dataset_config.image_extensions
    )


def _relative_output_paths(image_path: Path, input_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    rel_path = image_path.relative_to(input_dir)
    saved_image_path = output_dir / "images" / rel_path
    label_path = output_dir / "labels" / rel_path.with_suffix(".json")
    vis_path = output_dir / "visualizations" / rel_path.with_name(f"{rel_path.stem}_vis.png")
    return saved_image_path, label_path, vis_path


def annotation_output_paths(
    image_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    saved_image_path, label_path, vis_path = _relative_output_paths(
        Path(image_path),
        Path(input_dir),
        Path(output_dir),
    )
    return {
        "image": saved_image_path,
        "label": label_path,
        "visualization": vis_path,
    }


def has_complete_annotation(
    image_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_config: DatasetConfig = DATASET_CONFIG,
) -> bool:
    paths = annotation_output_paths(image_path, input_dir, output_dir)
    # Existing labels may come from the richer video-frame annotator, whose
    # visualization extension differs from this module. Never overwrite a
    # label implicitly; callers can still opt in with --force.
    if paths["label"].exists():
        return True
    required_paths = [paths["label"], paths["visualization"]]
    if dataset_config.keep_source_images:
        required_paths.append(paths["image"])
    return all(path.exists() for path in required_paths)


def annotate_image_for_dataset(
    image_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    prompt: str,
    model: str,
    min_pixels_str: str,
    max_pixels_str: str,
    dataset_config: DatasetConfig = DATASET_CONFIG,
) -> dict:
    image_path = Path(image_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    min_pixels, max_pixels = _image_token_limits(min_pixels_str, max_pixels_str)

    run_id = uuid.uuid4()
    img, local_img_path = _prepare_local_image(image_path, dataset_config.temp_output_dir, model, run_id)
    object_name_img = f"{OSS_CONFIG.object_prefix}/{model}_dataset_{run_id}{local_img_path.suffix}"
    file_url = image_url_for_model(local_img_path, object_name_img)
    messages = _messages_for_image(file_url, prompt, min_pixels, max_pixels)
    raw_response = _call_model_once(messages, model)
    bboxes = parse_bboxes_from_response(raw_response)
    bboxes_with_pixels = add_pixel_bboxes(bboxes, img.size)

    saved_image_path, label_path, vis_path = _relative_output_paths(image_path, input_dir, output_dir)
    saved_image_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_config.keep_source_images:
        shutil.copy2(image_path, saved_image_path)

    visualize_bboxes(img, bboxes, vis_path)

    result_data = {
        "model": model,
        "source_image": str(image_path),
        "saved_image": str(saved_image_path) if dataset_config.keep_source_images else "",
        "visualization": str(vis_path),
        "image_size": [img.size[0], img.size[1]],
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "user_prompt": prompt,
        "raw_response": raw_response,
        "bbox": bboxes_with_pixels,
    }
    _save_result_json(result_data, label_path)

    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "saved_image_path": str(saved_image_path) if dataset_config.keep_source_images else "",
        "visualization_path": str(vis_path),
        "bbox_count": len(bboxes),
    }


def annotate_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    prompt: str,
    model: str,
    min_pixels_str: str,
    max_pixels_str: str,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_paths = _iter_image_paths(input_dir)

    if progress_callback:
        progress_callback(f"Found {len(image_paths)} image(s) in {input_dir}")

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        if progress_callback:
            progress_callback(f"[{index}/{len(image_paths)}] Annotating {image_path.name}")
        result = annotate_image_for_dataset(
            image_path=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
            prompt=prompt,
            model=model,
            min_pixels_str=min_pixels_str,
            max_pixels_str=max_pixels_str,
        )
        results.append(result)
        if progress_callback:
            progress_callback(f"[{index}/{len(image_paths)}] Saved {result['bbox_count']} bbox(es): {result['label_path']}")

    return results
