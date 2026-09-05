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
class DatasetConfig:
    image_input_dir: Path = _as_path(
        _env_or_config("DATASET_IMAGE_DIR", "dataset.image_input_dir", "dataset/Positive_Sample/1A")
    )
    annotation_output_dir: Path = _as_path(
        _env_or_config("ANNOTATION_OUTPUT_DIR", "dataset.annotation_output_dir", "annotations")
    )
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
            "runs/experiments/det_replay_soup_a25/weights/best.pt",
        )
    )
    output_dir: Path = _as_path(_env_or_config("TRACKING_OUTPUT_DIR", "tracking.output_dir", "tracked_videos"))
    confidence: float = float(_env_or_config("TRACKING_CONFIDENCE", "tracking.confidence", 0.15))
    iou: float = float(_env_or_config("TRACKING_IOU", "tracking.iou", 0.7))
    imgsz: int = int(_env_or_config("TRACKING_IMGSZ", "tracking.imgsz", 1024))
    device: str = str(_env_or_config("TRACKING_DEVICE", "tracking.device", ""))
    trace_length: int = int(_env_or_config("TRACKING_TRACE_LENGTH", "tracking.trace_length", 30))
    line_thickness: int = int(_env_or_config("TRACKING_LINE_THICKNESS", "tracking.line_thickness", 2))
    text_scale: float = float(_env_or_config("TRACKING_TEXT_SCALE", "tracking.text_scale", 0.6))
    visualization_max_width: int = int(
        _env_or_config("TRACKING_VISUALIZATION_MAX_WIDTH", "tracking.visualization_max_width", 1920)
    )
    pose_weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_POSE_WEIGHTS_PATH",
            "tracking.pose_weights_path",
            "models/rtmpose/rtmpose-m-wholebody-256x192.onnx",
        )
    )
    pose_detector_path: Path = _as_path(
        _env_or_config(
            "TRACKING_POSE_DETECTOR_PATH",
            "tracking.pose_detector_path",
            "models/rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
        )
    )
    enable_pose: bool = _as_bool(_env_or_config("TRACKING_ENABLE_POSE", "tracking.enable_pose", False), False)
    string_weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_STRING_WEIGHTS_PATH",
            "tracking.string_weights_path",
            "runs/experiments/semantic_ablation_nomorph_foundation_r1/weights/best.pt",
        )
    )
    enable_string_model: bool = _as_bool(_env_or_config("TRACKING_ENABLE_STRING_MODEL", "tracking.enable_string_model", True), True)
    string_confidence: float = float(_env_or_config("TRACKING_STRING_CONFIDENCE", "tracking.string_confidence", 0.40))
    string_low_threshold: float | None = (
        None
        if _env_or_config("TRACKING_STRING_LOW_THRESHOLD", "tracking.string_low_threshold", None) in (None, "", "null", "None")
        else float(_env_or_config("TRACKING_STRING_LOW_THRESHOLD", "tracking.string_low_threshold", None))
    )
    string_inference_scale: float = float(
        _env_or_config("TRACKING_STRING_INFERENCE_SCALE", "tracking.string_inference_scale", 1.125)
    )
    string_cuda_graph: bool = _as_bool(
        _env_or_config("TRACKING_STRING_CUDA_GRAPH", "tracking.string_cuda_graph", True),
        True,
    )
    string_inference_fps: float = float(
        _env_or_config("TRACKING_STRING_INFERENCE_FPS", "tracking.string_inference_fps", 0.0)
    )
    string_color_probability_augment: bool = _as_bool(
        _env_or_config(
            "TRACKING_STRING_COLOR_PROBABILITY_AUGMENT",
            "tracking.string_color_probability_augment",
            True,
        ),
        True,
    )
    string_bright_line_augment: bool = _as_bool(
        _env_or_config(
            "TRACKING_STRING_BRIGHT_LINE_AUGMENT",
            "tracking.string_bright_line_augment",
            True,
        ),
        True,
    )
    string_bright_line_min_mean: float = float(
        _env_or_config(
            "TRACKING_STRING_BRIGHT_LINE_MIN_MEAN",
            "tracking.string_bright_line_min_mean",
            0.70,
        )
    )
    string_color_semantic_prefilter: bool = _as_bool(
        _env_or_config(
            "TRACKING_STRING_COLOR_SEMANTIC_PREFILTER",
            "tracking.string_color_semantic_prefilter",
            True,
        ),
        True,
    )
    string_color_probability_min_mean: float = float(
        _env_or_config(
            "TRACKING_STRING_COLOR_PROBABILITY_MIN_MEAN",
            "tracking.string_color_probability_min_mean",
            0.70,
        )
    )
    string_color_probability_min_fraction: float = float(
        _env_or_config(
            "TRACKING_STRING_COLOR_PROBABILITY_MIN_FRACTION",
            "tracking.string_color_probability_min_fraction",
            0.50,
        )
    )
    string_max_components: int = int(
        _env_or_config("TRACKING_STRING_MAX_COMPONENTS", "tracking.string_max_components", 32)
    )
    string_max_propagation_frames: int = int(
        _env_or_config("TRACKING_STRING_MAX_PROPAGATION_FRAMES", "tracking.string_max_propagation_frames", 12)
    )
    string_flow_fb_max_error: float = float(
        _env_or_config("TRACKING_STRING_FLOW_FB_MAX_ERROR", "tracking.string_flow_fb_max_error", 4.0)
    )
    yoyo_division: str = str(
        _env_or_config("TRACKING_YOYO_DIVISION", "tracking.yoyo_division", "1A")
    )
    orientation_weights_path: Path = _as_path(
        _env_or_config(
            "TRACKING_ORIENTATION_WEIGHTS_PATH",
            "tracking.orientation_weights_path",
            "runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/weights/best.pt",
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
    orientation_adaptive_inference: bool = _as_bool(
        _env_or_config("TRACKING_ORIENTATION_ADAPTIVE_INFERENCE", "tracking.orientation_adaptive_inference", True),
        True,
    )
    orientation_burst_inference_fps: float = float(
        _env_or_config("TRACKING_ORIENTATION_BURST_INFERENCE_FPS", "tracking.orientation_burst_inference_fps", 25.0)
    )
    orientation_adaptive_min_confidence: float = float(
        _env_or_config("TRACKING_ORIENTATION_ADAPTIVE_MIN_CONFIDENCE", "tracking.orientation_adaptive_min_confidence", 0.4)
    )
    orientation_adaptive_stable_observations: int = int(
        _env_or_config("TRACKING_ORIENTATION_ADAPTIVE_STABLE_OBSERVATIONS", "tracking.orientation_adaptive_stable_observations", 4)
    )
    orientation_temporal_filter: bool = _as_bool(
        _env_or_config("TRACKING_ORIENTATION_TEMPORAL_FILTER", "tracking.orientation_temporal_filter", True),
        True,
    )
    orientation_ema_alpha: float = float(
        _env_or_config("TRACKING_ORIENTATION_EMA_ALPHA", "tracking.orientation_ema_alpha", 0.5)
    )
    orientation_switch_margin: float = float(
        _env_or_config("TRACKING_ORIENTATION_SWITCH_MARGIN", "tracking.orientation_switch_margin", 0.05)
    )
    orientation_switch_confirmations: int = int(
        _env_or_config("TRACKING_ORIENTATION_SWITCH_CONFIRMATIONS", "tracking.orientation_switch_confirmations", 4)
    )
    orientation_strong_switch_confidence: float = float(
        _env_or_config("TRACKING_ORIENTATION_STRONG_SWITCH_CONFIDENCE", "tracking.orientation_strong_switch_confidence", 0.9)
    )
    orientation_strong_switch_margin: float = float(
        _env_or_config("TRACKING_ORIENTATION_STRONG_SWITCH_MARGIN", "tracking.orientation_strong_switch_margin", 0.1)
    )
    orientation_switch_confirmation_seconds: float = float(
        _env_or_config(
            "TRACKING_ORIENTATION_SWITCH_CONFIRMATION_SECONDS",
            "tracking.orientation_switch_confirmation_seconds",
            0.0,
        )
    )
    orientation_ema_time_constant_seconds: float = float(
        _env_or_config(
            "TRACKING_ORIENTATION_EMA_TIME_CONSTANT_SECONDS",
            "tracking.orientation_ema_time_constant_seconds",
            0.0,
        )
    )


@dataclass(frozen=True)
class SemanticStringConfig:
    dataset_dir: Path = _as_path(
        _env_or_config("SEMANTIC_STRING_DATASET_DIR", "semantic_string.dataset_dir", "datasets/1Ayoyo_dataset/string_segmentation")
    )
    project: Path = _as_path(_env_or_config("SEMANTIC_STRING_PROJECT", "semantic_string.project", "runs/v2v3"))
    run_name: str = str(_env_or_config("SEMANTIC_STRING_RUN_NAME", "semantic_string.run_name", "yoyo_v2v3_semantic_string"))
    epochs: int = int(_env_or_config("SEMANTIC_STRING_EPOCHS", "semantic_string.epochs", 12))
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
        _env_or_config("SEMANTIC_STRING_MIN_MASK_WIDTH_PX", "semantic_string.min_mask_width_px", 1)
    )
    motion_blur_probability: float = float(
        _env_or_config("SEMANTIC_STRING_MOTION_BLUR_PROBABILITY", "semantic_string.motion_blur_probability", 0.0)
    )
    motion_blur_min_sharpness: float = float(
        _env_or_config("SEMANTIC_STRING_MOTION_BLUR_MIN_SHARPNESS", "semantic_string.motion_blur_min_sharpness", 37.0)
    )
    architecture: str = str(
        _env_or_config("SEMANTIC_STRING_ARCHITECTURE", "semantic_string.architecture", "mobilenet_v3_fpn")
    )
    pretrained_backbone: bool = _as_bool(
        _env_or_config("SEMANTIC_STRING_PRETRAINED_BACKBONE", "semantic_string.pretrained_backbone", True),
        True,
    )
    freeze_backbone_epochs: int = int(
        _env_or_config("SEMANTIC_STRING_FREEZE_BACKBONE_EPOCHS", "semantic_string.freeze_backbone_epochs", 3)
    )
    backbone_lr_multiplier: float = float(
        _env_or_config("SEMANTIC_STRING_BACKBONE_LR_MULTIPLIER", "semantic_string.backbone_lr_multiplier", 0.05)
    )
    hard_negative_weight: float = float(
        _env_or_config("SEMANTIC_STRING_HARD_NEGATIVE_WEIGHT", "semantic_string.hard_negative_weight", 0.2)
    )
    negative_sample_weight: float = float(
        _env_or_config("SEMANTIC_STRING_NEGATIVE_SAMPLE_WEIGHT", "semantic_string.negative_sample_weight", 4.0)
    )
    early_stopping_patience: int = int(
        _env_or_config("SEMANTIC_STRING_EARLY_STOPPING_PATIENCE", "semantic_string.early_stopping_patience", 0)
    )
    early_stopping_min_epochs: int = int(
        _env_or_config("SEMANTIC_STRING_EARLY_STOPPING_MIN_EPOCHS", "semantic_string.early_stopping_min_epochs", 10)
    )
    seed: int = int(_env_or_config("SEMANTIC_STRING_SEED", "semantic_string.seed", 20260830))
    device: str = str(_env_or_config("SEMANTIC_STRING_DEVICE", "semantic_string.device", "cuda"))


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration owned by the yoyo detection path."""

    weights_path: Path = _as_path(_env_or_config("DETECTION_WEIGHTS_PATH", "detection.weights_path", "runs/experiments/det_replay_soup_a25/weights/best.pt"))
    confidence: float = float(_env_or_config("DETECTION_CONFIDENCE", "detection.confidence", 0.15))
    iou: float = float(_env_or_config("DETECTION_IOU", "detection.iou", 0.7))
    imgsz: int = int(_env_or_config("DETECTION_IMGSZ", "detection.imgsz", 1024))
    device: str = str(_env_or_config("DETECTION_DEVICE", "detection.device", ""))


@dataclass(frozen=True)
class StringTrackingConfig:
    """Configuration owned by the string recognition/tracking path."""

    weights_path: Path = _as_path(_env_or_config("STRING_TRACKING_WEIGHTS_PATH", "string_tracking.weights_path", "runs/experiments/semantic_ablation_nomorph_foundation_r1/weights/best.pt"))
    confidence: float = float(_env_or_config("STRING_TRACKING_CONFIDENCE", "string_tracking.confidence", 0.40))
    imgsz: int = int(_env_or_config("STRING_TRACKING_IMGSZ", "string_tracking.imgsz", 544))
    device: str = str(_env_or_config("STRING_TRACKING_DEVICE", "string_tracking.device", ""))


@dataclass(frozen=True)
class OrientationConfig:
    """Configuration owned by the yoyo orientation path."""

    weights_path: Path = _as_path(_env_or_config("ORIENTATION_WEIGHTS_PATH", "orientation.weights_path", "runs/experiments/yoyo_unified_5673a7faf873_orientation_roi_afbae9c0cd2a_yolo11n-cls_current5673-foundation-e30-b32/weights/best.pt"))
    imgsz: int = int(_env_or_config("ORIENTATION_IMGSZ", "orientation.imgsz", 320))
    device: str = str(_env_or_config("ORIENTATION_DEVICE", "orientation.device", ""))


DATASET_CONFIG = DatasetConfig()
YOLO_CONFIG = YOLOConfig()
TRACKING_CONFIG = TrackingConfig()
SEMANTIC_STRING_CONFIG = SemanticStringConfig()
DETECTION_CONFIG = DetectionConfig()
STRING_TRACKING_CONFIG = StringTrackingConfig()
ORIENTATION_CONFIG = OrientationConfig()
