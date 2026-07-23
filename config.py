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
    dataset_dir: Path = _as_path(_env_or_config("YOLO_DATASET_DIR", "yolo.dataset_dir", "yolo_dataset"))
    data_yaml: Path = _as_path(_env_or_config("YOLO_DATA_YAML", "yolo.data_yaml", "yolo_dataset/data.yaml"))
    class_names: tuple[str, ...] = _as_tuple(_get_config("yolo.class_names", None), ("yoyo",))
    train_split: float = float(_env_or_config("YOLO_TRAIN_SPLIT", "yolo.train_split", 0.8))
    seed: int = int(_env_or_config("YOLO_SEED", "yolo.seed", 42))
    epochs: int = int(_env_or_config("YOLO_EPOCHS", "yolo.epochs", 100))
    imgsz: int = int(_env_or_config("YOLO_IMGSZ", "yolo.imgsz", 640))
    batch: str = str(_env_or_config("YOLO_BATCH", "yolo.batch", 8))
    workers: int = int(_env_or_config("YOLO_WORKERS", "yolo.workers", 4))
    device: str = str(_env_or_config("YOLO_DEVICE", "yolo.device", ""))
    project: Path = _as_path(_env_or_config("YOLO_PROJECT", "yolo.project", "runs/yolo"))
    run_name: str = _env_or_config("YOLO_RUN_NAME", "yolo.run_name", "yoyo_video_v1")
    exist_ok: bool = _as_bool(_env_or_config("YOLO_EXIST_OK", "yolo.exist_ok", True), True)

    @property
    def weights_path(self) -> Path:
        return self.models_dir / self.model_name


@dataclass(frozen=True)
class TrackingConfig:
    weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_WEIGHTS_PATH",
            "tracking.weights_path",
            "runs/yolo/yoyo_video_v8_fresh_holdout/weights/best.pt",
        )
    )
    output_dir: Path = _as_path(_env_or_config("TRACKING_OUTPUT_DIR", "tracking.output_dir", "tracked_videos"))
    confidence: float = float(_env_or_config("TRACKING_CONFIDENCE", "tracking.confidence", 0.25))
    iou: float = float(_env_or_config("TRACKING_IOU", "tracking.iou", 0.7))
    imgsz: int = int(_env_or_config("TRACKING_IMGSZ", "tracking.imgsz", 640))
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
            "runs/semantic/yoyo_string_semantic_v17_reviewed_expansion_hn005/weights/best.pt",
        )
    )
    enable_string_model: bool = _as_bool(_env_or_config("TRACKING_ENABLE_STRING_MODEL", "tracking.enable_string_model", True), True)
    string_confidence: float = float(_env_or_config("TRACKING_STRING_CONFIDENCE", "tracking.string_confidence", 0.20))
    string_inference_scale: float = float(
        _env_or_config("TRACKING_STRING_INFERENCE_SCALE", "tracking.string_inference_scale", 2.0)
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
    export_json: bool = _as_bool(_env_or_config("TRACKING_EXPORT_JSON", "tracking.export_json", True), True)
    export_clips: bool = _as_bool(_env_or_config("TRACKING_EXPORT_CLIPS", "tracking.export_clips", True), True)
    activity_speed_diagonal_per_s: float = float(
        _env_or_config("TRACKING_ACTIVITY_SPEED_DIAGONAL_PER_S", "tracking.activity_speed_diagonal_per_s", 0.08)
    )
    padding_seconds: float = float(_env_or_config("TRACKING_PADDING_SECONDS", "tracking.padding_seconds", 0.4))
    min_segment_seconds: float = float(_env_or_config("TRACKING_MIN_SEGMENT_SECONDS", "tracking.min_segment_seconds", 0.5))
    max_gap_seconds: float = float(_env_or_config("TRACKING_MAX_GAP_SECONDS", "tracking.max_gap_seconds", 0.4))
    max_segment_seconds: float = float(
        _env_or_config("TRACKING_MAX_SEGMENT_SECONDS", "tracking.max_segment_seconds", 180.0)
    )


@dataclass(frozen=True)
class StringSegmentationConfig:
    model_name: str = _env_or_config("STRING_MODEL_NAME", "string_segmentation.model_name", "yolo11n-seg.pt")
    models_dir: Path = _as_path(_env_or_config("STRING_MODELS_DIR", "string_segmentation.models_dir", "models"))
    dataset_dir: Path = _as_path(
        _env_or_config("STRING_DATASET_DIR", "string_segmentation.dataset_dir", "datasets/video_v1/string_seg")
    )
    annotations_dir: Path = _as_path(
        _env_or_config("STRING_ANNOTATIONS_DIR", "string_segmentation.annotations_dir", "datasets/video_v1/annotations")
    )
    project: Path = _as_path(_env_or_config("STRING_PROJECT", "string_segmentation.project", "runs/yolo"))
    run_name: str = str(_env_or_config("STRING_RUN_NAME", "string_segmentation.run_name", "yoyo_string_v1"))
    epochs: int = int(_env_or_config("STRING_EPOCHS", "string_segmentation.epochs", 100))
    imgsz: int = int(_env_or_config("STRING_IMGSZ", "string_segmentation.imgsz", 960))
    batch: str = str(_env_or_config("STRING_BATCH", "string_segmentation.batch", 4))
    workers: int = int(_env_or_config("STRING_WORKERS", "string_segmentation.workers", 0))
    device: str = str(_env_or_config("STRING_DEVICE", "string_segmentation.device", ""))
    line_width_px: int = int(_env_or_config("STRING_LINE_WIDTH_PX", "string_segmentation.line_width_px", 8))
    mask_ratio: int = int(_env_or_config("STRING_MASK_RATIO", "string_segmentation.mask_ratio", 1))
    translate: float = float(_env_or_config("STRING_TRANSLATE", "string_segmentation.translate", 0.03))
    scale: float = float(_env_or_config("STRING_SCALE", "string_segmentation.scale", 0.15))
    mosaic: float = float(_env_or_config("STRING_MOSAIC", "string_segmentation.mosaic", 0.0))

    @property
    def weights_path(self) -> Path:
        return self.models_dir / self.model_name


@dataclass(frozen=True)
class SemanticStringConfig:
    dataset_dir: Path = _as_path(
        _env_or_config("SEMANTIC_STRING_DATASET_DIR", "semantic_string.dataset_dir", "datasets/video_v1/string_seg")
    )
    project: Path = _as_path(_env_or_config("SEMANTIC_STRING_PROJECT", "semantic_string.project", "runs/semantic"))
    run_name: str = str(_env_or_config("SEMANTIC_STRING_RUN_NAME", "semantic_string.run_name", "yoyo_string_semantic_v1"))
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
STRING_SEGMENTATION_CONFIG = StringSegmentationConfig()
SEMANTIC_STRING_CONFIG = SemanticStringConfig()
