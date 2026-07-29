import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.yaml"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def _load_yaml_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_yaml_config()


def _get_config(path: str, default: Any = None) -> Any:
    value: Any = _CONFIG
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _env_or_config(env_name: str, config_path: str, default: Any = None) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    return _get_config(config_path, default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _as_optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return _as_path(value)


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = _env_or_config("BASE_URL", "model.base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key_env: str = _get_config("model.api_key_env", "API_KEY")
    api_key: str | None = os.getenv(api_key_env)
    default_model: str = _env_or_config("DEFAULT_MODEL", "model.default_model", "qwen3.6-35b-a3b")
    available_models: tuple[str, ...] = _as_tuple(
        _env_or_config("AVAILABLE_MODELS", "model.available_models", None),
        ("qwen3.6-35b-a3b", "qwen3.5-35b-a3b"),
    )
    min_image_tokens: str = str(_env_or_config("MIN_IMAGE_TOKENS", "model.min_image_tokens", "1024"))
    max_image_tokens: str = str(_env_or_config("MAX_IMAGE_TOKENS", "model.max_image_tokens", "9800"))
    max_response_tokens: int = int(_env_or_config("MAX_RESPONSE_TOKENS", "model.max_response_tokens", 20000))
    enable_thinking: bool = _as_bool(_env_or_config("ENABLE_THINKING", "model.enable_thinking", True), True)
    image_transport: str = str(_env_or_config("IMAGE_TRANSPORT", "model.image_transport", "oss")).lower()


@dataclass(frozen=True)
class OSSConfig:
    endpoint: str | None = _env_or_config("OSS_ENDPOINT", "oss.endpoint", None)
    region: str | None = _env_or_config("OSS_REGION", "oss.region", None)
    bucket_name: str | None = _env_or_config("OSS_BUCKET_NAME", "oss.bucket_name", None)
    object_prefix: str = _env_or_config("OSS_OBJECT_PREFIX", "oss.object_prefix", "studio-temp/qwen-det")
    signed_url_expires_seconds: int = int(
        _env_or_config("OSS_SIGNED_URL_EXPIRES_SECONDS", "oss.signed_url_expires_seconds", 3600)
    )


@dataclass(frozen=True)
class DatasetConfig:
    image_input_dir: Path = _as_path(
        _env_or_config("DATASET_IMAGE_DIR", "dataset.image_input_dir", "dataset/Positive_Sample/1A")
    )
    current_action_group: str = str(
        _env_or_config("DATASET_CURRENT_ACTION_GROUP", "dataset.current_action_group", "1A")
    )
    annotation_output_dir: Path = _as_path(
        _env_or_config("ANNOTATION_OUTPUT_DIR", "dataset.annotation_output_dir", "annotations")
    )
    temp_output_dir: Path = _as_path(_env_or_config("TEMP_OUTPUT_DIR", "dataset.temp_output_dir", "tmp"))
    image_extensions: tuple[str, ...] = _as_tuple(
        _get_config("dataset.image_extensions", None),
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    )
    recursive: bool = _as_bool(_env_or_config("DATASET_RECURSIVE", "dataset.recursive", True), True)
    keep_source_images: bool = _as_bool(
        _env_or_config("KEEP_SOURCE_IMAGES", "dataset.keep_source_images", True),
        True,
    )


@dataclass(frozen=True)
class YOLOConfig:
    model_name: str = _env_or_config("YOLO_MODEL_NAME", "yolo.model_name", "yolo11n.pt")
    models_dir: Path = _as_path(_env_or_config("YOLO_MODELS_DIR", "yolo.models_dir", "models"))
    model_url_template: str = _env_or_config(
        "YOLO_MODEL_URL_TEMPLATE",
        "yolo.model_url_template",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}",
    )


@dataclass(frozen=True)
class TrackingConfig:
    weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_WEIGHTS_PATH",
            "tracking.weights_path",
            "runs/candidates/yoyo_unified_1ae945ed3856_detection_best_incremental8-lr5e5-v1/weights/best.pt",
        )
    )
    output_dir: Path = _as_path(_env_or_config("TRACKING_OUTPUT_DIR", "tracking.output_dir", "tracked_videos"))
    confidence: float = float(_env_or_config("TRACKING_CONFIDENCE", "tracking.confidence", 0.15))
    iou: float = float(_env_or_config("TRACKING_IOU", "tracking.iou", 0.7))
    imgsz: int = int(_env_or_config("TRACKING_IMGSZ", "tracking.imgsz", 1280))
    device: str = str(_env_or_config("TRACKING_DEVICE", "tracking.device", ""))
    trace_length: int = int(_env_or_config("TRACKING_TRACE_LENGTH", "tracking.trace_length", 30))
    line_thickness: int = int(_env_or_config("TRACKING_LINE_THICKNESS", "tracking.line_thickness", 2))
    text_scale: float = float(_env_or_config("TRACKING_TEXT_SCALE", "tracking.text_scale", 0.6))
    visualization_max_width: int = int(
        _env_or_config("TRACKING_VISUALIZATION_MAX_WIDTH", "tracking.visualization_max_width", 1920)
    )
    pose_weights_path: Path = _as_path(_env_or_config("TRACKING_POSE_WEIGHTS_PATH", "tracking.pose_weights_path", "models/yolo11n-pose.pt"))
    enable_pose: bool = _as_bool(_env_or_config("TRACKING_ENABLE_POSE", "tracking.enable_pose", True), True)
    auto_download_pose: bool = _as_bool(_env_or_config("TRACKING_AUTO_DOWNLOAD_POSE", "tracking.auto_download_pose", False), False)
    string_weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_STRING_WEIGHTS_PATH",
            "tracking.string_weights_path",
            "runs/candidates/yoyo_unified_1ae945ed3856_semantic_string_lraspp_incremental8-lr1e5-v1/weights/best.pt",
        )
    )
    enable_string_model: bool = _as_bool(_env_or_config("TRACKING_ENABLE_STRING_MODEL", "tracking.enable_string_model", True), True)
    string_confidence: float = float(_env_or_config("TRACKING_STRING_CONFIDENCE", "tracking.string_confidence", 0.20))
    string_inference_scale: float = float(
        _env_or_config("TRACKING_STRING_INFERENCE_SCALE", "tracking.string_inference_scale", 1.0)
    )
    string_inference_fps: float = float(
        _env_or_config("TRACKING_STRING_INFERENCE_FPS", "tracking.string_inference_fps", 10.0)
    )
    string_max_propagation_frames: int = int(
        _env_or_config("TRACKING_STRING_MAX_PROPAGATION_FRAMES", "tracking.string_max_propagation_frames", 12)
    )
    string_flow_fb_max_error: float = float(
        _env_or_config("TRACKING_STRING_FLOW_FB_MAX_ERROR", "tracking.string_flow_fb_max_error", 4.0)
    )
    string_fusion_distance_px: float = float(
        _env_or_config("TRACKING_STRING_FUSION_DISTANCE_PX", "tracking.string_fusion_distance_px", 48.0)
    )
    string_attachment_class: str = str(
        _env_or_config("TRACKING_STRING_ATTACHMENT_CLASS", "tracking.string_attachment_class", "hand_and_yoyo_attached")
    )
    orientation_weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_ORIENTATION_WEIGHTS_PATH",
            "tracking.orientation_weights_path",
            "runs/v2v3/yoyo_v2v3_c48ce78a1181_orientation_roi_0c0225b2e6ed/weights/best.pt",
        )
    )
    enable_orientation_model: bool = _as_bool(
        _env_or_config("TRACKING_ENABLE_ORIENTATION_MODEL", "tracking.enable_orientation_model", True),
        True,
    )
    orientation_imgsz: int = int(_env_or_config("TRACKING_ORIENTATION_IMGSZ", "tracking.orientation_imgsz", 320))
    orientation_inference_fps: float = float(
        _env_or_config("TRACKING_ORIENTATION_INFERENCE_FPS", "tracking.orientation_inference_fps", 5.0)
    )
    export_json: bool = _as_bool(_env_or_config("TRACKING_EXPORT_JSON", "tracking.export_json", True), True)


@dataclass(frozen=True)
class SemanticStringConfig:
    dataset_dir: Path = _as_path(
        _env_or_config("SEMANTIC_STRING_DATASET_DIR", "semantic_string.dataset_dir", "datasets/yoyo_dataset/string_segmentation")
    )
    project: Path = _as_path(_env_or_config("SEMANTIC_STRING_PROJECT", "semantic_string.project", "runs/semantic"))
    run_name: str = str(_env_or_config("SEMANTIC_STRING_RUN_NAME", "semantic_string.run_name", "yoyo_string_semantic_candidate"))
    epochs: int = int(_env_or_config("SEMANTIC_STRING_EPOCHS", "semantic_string.epochs", 40))
    input_width: int = int(_env_or_config("SEMANTIC_STRING_INPUT_WIDTH", "semantic_string.input_width", 960))
    input_height: int = int(_env_or_config("SEMANTIC_STRING_INPUT_HEIGHT", "semantic_string.input_height", 544))
    batch: int = int(_env_or_config("SEMANTIC_STRING_BATCH", "semantic_string.batch", 2))
    workers: int = int(_env_or_config("SEMANTIC_STRING_WORKERS", "semantic_string.workers", 0))
    learning_rate: float = float(
        _env_or_config("SEMANTIC_STRING_LEARNING_RATE", "semantic_string.learning_rate", 0.001)
    )
    base_channels: int = int(
        _env_or_config("SEMANTIC_STRING_BASE_CHANNELS", "semantic_string.base_channels", 16)
    )
    min_mask_width_px: int = int(
        _env_or_config("SEMANTIC_STRING_MIN_MASK_WIDTH_PX", "semantic_string.min_mask_width_px", 2)
    )
    seed: int = int(_env_or_config("SEMANTIC_STRING_SEED", "semantic_string.seed", 42))
    device: str = str(_env_or_config("SEMANTIC_STRING_DEVICE", "semantic_string.device", "cuda"))


MODEL_CONFIG = ModelConfig()
OSS_CONFIG = OSSConfig()
DATASET_CONFIG = DatasetConfig()
YOLO_CONFIG = YOLOConfig()
TRACKING_CONFIG = TrackingConfig()
SEMANTIC_STRING_CONFIG = SemanticStringConfig()
