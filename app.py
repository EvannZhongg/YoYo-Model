import logging
import os
from pathlib import Path

# The system TEMP directory can receive sandbox-specific ACLs on Windows,
# making Gradio's copied media intermittently unreadable by the ASGI worker.
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(__file__).resolve().parent / "tmp" / "gradio"))
Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

import gradio as gr

from config import BASE_DIR, TRACKING_CONFIG
from video_tracking.frame_review import append_tracking_frame_review, load_tracking_frame_selection
from video_tracking.tracker import track_video
from workbench.commands import workbench_evaluate_v2v3, workbench_train_v2v3
from workbench.consecutive_annotation import consecutive_annotation_component_kwargs
from workbench.dataset_annotation import dataset_annotation_component_kwargs
from workbench.score_annotation import score_annotation_component_kwargs
from workbench.tracking import tracking_review_gallery as _tracking_review_gallery


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


def _uploaded_video_path(video):
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("name")
    return getattr(video, "name", None)


def select_tracking_review_frame(metadata_path: str | Path | None, evt: gr.SelectData):
    if not metadata_path:
        return {}, {}, "Select a completed tracking run first."
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        selection = load_tracking_frame_selection(metadata_path, index)
    except Exception as exc:
        return {}, {}, f"Frame selection failed: {exc}"
    binding = selection["binding"]
    return (
        selection["frame_record"],
        binding,
        f"Selected frame {binding['frame_index']} ({binding['view']}); digest {binding['frame_record_sha256']}.",
    )


def save_tracking_frame_review(
    metadata_path: str | Path | None,
    binding: dict | None,
    decision: str,
    reviewer: str,
    notes: str,
):
    if not metadata_path:
        return None, "Frame review failed: select a completed tracking run first."
    try:
        output_path, event = append_tracking_frame_review(
            metadata_path,
            binding or {},
            decision,
            reviewer,
            notes,
        )
    except Exception as exc:
        return None, f"Frame review failed: {exc}"
    return (
        str(output_path),
        f"Saved {event['decision']} review for frame {event['frame_index']} by {event['reviewer']}.",
    )


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
    string_inference_scale: float,
    string_inference_fps: float,
    yoyo_division: str,
    enable_orientation_model: bool,
    orientation_weights_path: str,
    orientation_inference_fps: float,
    visualization_max_width: int,
):
    empty = (None, None, None, None, [], None, {}, {}, None, "")
    video_path = _uploaded_video_path(video)
    if not video_path:
        return (*empty, "Error: No video provided.")

    try:
        selected_string_weights = Path(string_weights_path.strip()).resolve() if string_weights_path.strip() else None
        use_default_string_ensemble = bool(
            selected_string_weights is not None
            and selected_string_weights == TRACKING_CONFIG.string_weights_path.resolve()
        )
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
            string_ensemble_weights_path=(
                TRACKING_CONFIG.string_ensemble_weights_path
                if use_default_string_ensemble else None
            ),
            string_ensemble_alpha=(
                TRACKING_CONFIG.string_ensemble_alpha
                if use_default_string_ensemble else 0.0
            ),
            string_ensemble_candidate_threshold=(
                TRACKING_CONFIG.string_ensemble_candidate_threshold
            ),
            string_confidence=float(string_confidence),
            string_inference_scale=float(string_inference_scale),
            string_inference_fps=float(string_inference_fps),
            yoyo_division=yoyo_division,
            enable_orientation_model=bool(enable_orientation_model),
            orientation_weights_path=orientation_weights_path.strip() or None,
            orientation_imgsz=TRACKING_CONFIG.orientation_imgsz,
            orientation_inference_fps=float(orientation_inference_fps),
            export_json=True,
            visualization_max_width=max(0, int(visualization_max_width)),
        )
    except Exception as exc:
        logger.exception("Video tracking failed")
        return (*empty, f"Error: {exc}")

    gr.set_static_paths(paths=[result["run_dir"]])
    review_gallery = _tracking_review_gallery(result.get("run_dir"))
    status = (
        f"Done. Frames: {result['frame_count']}\n"
        f"Output: {result['output_video']}\n"
        f"Bad cases: {result['bad_case_counts']}\n"
        f"String geometry: {result.get('string_geometry_counts', {})}\n"
        f"String model: {result.get('string_model', 'disabled')}\n"
        f"Semantic inference frames: {result.get('string_inference_frame_count', 0)}\n"
        f"Orientation model: {result.get('orientation_model', 'disabled')}\n"
        f"Orientation inference frames: {result.get('orientation_inference_frame_count', 0)}\n"
        f"Orientation summary: {result.get('orientation_summary', {})}\n"
        f"Tracking rate: {result.get('tracking_loop_fps', 0):.2f} frames/s\n"
        f"Preview resolution: {result.get('output_width', 0)}x{result.get('output_height', 0)}\n"
        f"Review images: {len(review_gallery)}\n"
        f"Run manifest: {result['run_manifest']}\n"
        f"Weights: {result['weights']}"
    )
    return (
        result["output_video"],
        result["metadata_jsonl"],
        result["run_manifest"],
        result.get("review_sheet") or None,
        review_gallery,
        result["metadata_jsonl"],
        {},
        {},
        None,
        "",
        status,
    )


def create_demo():
    gr.set_static_paths(paths=[TRACKING_CONFIG.output_dir])
    unified_dataset = BASE_DIR / "datasets" / "1Ayoyo_dataset"
    with gr.Blocks(title="YoYo Model") as demo:
        gr.Markdown("# YoYo Model")
        with gr.Tabs():
            with gr.Tab("数据标注"):
                gr.HTML(**dataset_annotation_component_kwargs())

            with gr.Tab("连续帧标注"):
                gr.HTML(**consecutive_annotation_component_kwargs())

            with gr.Tab("Unified Training"):
                training_dataset = gr.Textbox(label="Unified Dataset", value=str(unified_dataset))
                with gr.Row():
                    training_task = gr.Dropdown(
                        label="Training Task",
                        choices=["detection", "semantic_string", "orientation_roi", "orientation", "string_segmentation", "all"],
                        value="detection",
                    )
                    training_epochs = gr.Number(label="Epochs", value=80, precision=0)
                    training_device = gr.Textbox(label="Device", value="0")
                    training_project = gr.Textbox(label="Model Runs", value=str(BASE_DIR / "runs" / "v2v3"))
                    training_start = gr.Button("Train", variant="primary")
                with gr.Row():
                    evaluation_run = gr.Textbox(label="Run Directory")
                    evaluation_start = gr.Button("Evaluate Run")
                training_log = gr.Textbox(label="Training / Evaluation Log", lines=14, interactive=False, buttons=["copy"])
                training_start.click(
                    workbench_train_v2v3,
                    [training_dataset, training_project, training_task, training_epochs, training_device],
                    [training_log],
                )
                evaluation_start.click(workbench_evaluate_v2v3, [evaluation_run, training_device], [training_log])

            with gr.Tab("Score Annotation"):
                gr.HTML(**score_annotation_component_kwargs())

            with gr.Tab("Video Tracking"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.FileExplorer(
                            label="Input Video",
                            root_dir=BASE_DIR / "videos",
                            glob="*.mp4",
                            file_count="single",
                            height=280,
                        )
                        tracking_weights = gr.Textbox(label="YOLO Weights", value=str(TRACKING_CONFIG.weights_path))
                        tracking_output_dir = gr.Textbox(label="Output Directory", value=str(TRACKING_CONFIG.output_dir))
                        tracking_preview_width = gr.Number(
                            label="Tracked Preview Maximum Width (0 = source)",
                            value=TRACKING_CONFIG.visualization_max_width,
                            minimum=0,
                            precision=0,
                        )
                        with gr.Row():
                            tracking_conf = gr.Slider(label="Confidence", minimum=0.01, maximum=0.99, value=TRACKING_CONFIG.confidence, step=0.01)
                            tracking_iou = gr.Slider(label="IoU", minimum=0.1, maximum=0.95, value=TRACKING_CONFIG.iou, step=0.01)
                            tracking_imgsz = gr.Number(label="Image Size", value=TRACKING_CONFIG.imgsz, precision=0)
                            tracking_device = gr.Textbox(label="Device", value=TRACKING_CONFIG.device)
                        tracking_pose = gr.Checkbox(label="Pose / hand landmarks", value=TRACKING_CONFIG.enable_pose)
                        tracking_pose_weights = gr.Textbox(label="Pose Weights", value=str(TRACKING_CONFIG.pose_weights_path))
                        tracking_string_model = gr.Checkbox(label="String segmentation model", value=TRACKING_CONFIG.enable_string_model)
                        tracking_string_weights = gr.Textbox(label="String Segmentation Weights", value=str(TRACKING_CONFIG.string_weights_path))
                        with gr.Row():
                            tracking_string_conf = gr.Slider(
                                label="String Confidence", minimum=0.01, maximum=0.95,
                                value=TRACKING_CONFIG.string_confidence, step=0.01,
                            )
                            tracking_string_scale = gr.Slider(
                                label="Semantic Inference Scale", minimum=0.5, maximum=2.0,
                                value=TRACKING_CONFIG.string_inference_scale, step=0.25,
                            )
                            tracking_string_fps = gr.Number(
                                label="Semantic Model FPS (0 = every frame)",
                                value=TRACKING_CONFIG.string_inference_fps,
                                minimum=0,
                                precision=1,
                            )
                        tracking_yoyo_division = gr.Dropdown(
                            label="YoYo Division",
                            choices=["1A", "2A", "3A", "4A", "5A"],
                            value=TRACKING_CONFIG.yoyo_division,
                        )
                        tracking_orientation_model = gr.Checkbox(
                            label="Coarse trick orientation model",
                            value=TRACKING_CONFIG.enable_orientation_model,
                        )
                        tracking_orientation_weights = gr.Textbox(
                            label="Orientation Weights",
                            value=str(TRACKING_CONFIG.orientation_weights_path),
                        )
                        tracking_orientation_fps = gr.Number(
                            label="Orientation Model FPS (0 = every frame)",
                            value=TRACKING_CONFIG.orientation_inference_fps,
                            minimum=0,
                            precision=1,
                        )
                        track_btn = gr.Button("Run Full Video Tracking", variant="primary")

                    with gr.Column(scale=1):
                        tracking_metadata_source = gr.State(value=None)
                        tracking_output_video = gr.Video(label="Tracked Video")
                        tracking_metadata = gr.File(label="Frame Metadata JSONL")
                        tracking_run_manifest = gr.File(label="Run Manifest")
                        tracking_review_sheet = gr.Image(label="Tracking Visual Review", type="filepath")
                        tracking_review_gallery = gr.Gallery(
                            label="Tracking Review Frames (Raw / Overlay)",
                            columns=2,
                            height=560,
                            allow_preview=True,
                            object_fit="contain",
                            type="filepath",
                            buttons=["fullscreen", "download"],
                        )
                        tracking_selected_frame = gr.JSON(label="Selected Tracking Frame JSON")
                        tracking_review_binding = gr.JSON(label="Digest-bound Review Identity")
                        with gr.Row():
                            tracking_review_decision = gr.Radio(
                                choices=["correct", "incorrect", "unresolved"],
                                value="unresolved",
                                label="Frame Decision",
                            )
                            tracking_review_reviewer = gr.Textbox(label="Reviewer", value="workbench-reviewer")
                        tracking_review_notes = gr.Textbox(label="Frame Review Notes", lines=2)
                        tracking_review_save = gr.Button("Save Frame Review", variant="primary")
                        tracking_review_log = gr.File(label="Tracking Frame Review Log")
                        tracking_review_status = gr.Textbox(label="Frame Review Status", interactive=False)
                        tracking_status = gr.Textbox(label="Tracking Status", lines=10, interactive=False, buttons=["copy"])

                track_btn.click(
                    run_video_tracking,
                    [
                        video_input, tracking_weights, tracking_output_dir, tracking_conf, tracking_iou,
                        tracking_imgsz, tracking_device, tracking_pose, tracking_pose_weights,
                        tracking_string_model, tracking_string_weights, tracking_string_conf,
                        tracking_string_scale, tracking_string_fps, tracking_yoyo_division,
                        tracking_orientation_model, tracking_orientation_weights,
                        tracking_orientation_fps, tracking_preview_width,
                    ],
                    [
                        tracking_output_video, tracking_metadata, tracking_run_manifest,
                        tracking_review_sheet, tracking_review_gallery, tracking_metadata_source,
                        tracking_selected_frame, tracking_review_binding, tracking_review_log,
                        tracking_review_status, tracking_status,
                    ],
                )
                tracking_review_gallery.select(
                    select_tracking_review_frame,
                    [tracking_metadata_source],
                    [tracking_selected_frame, tracking_review_binding, tracking_review_status],
                )
                tracking_review_save.click(
                    save_tracking_frame_review,
                    [
                        tracking_metadata_source, tracking_review_binding, tracking_review_decision,
                        tracking_review_reviewer, tracking_review_notes,
                    ],
                    [tracking_review_log, tracking_review_status],
                )

    return demo


if __name__ == "__main__":
    create_demo().launch(
        server_name=os.getenv("APP_HOST", "0.0.0.0"),
        server_port=int(os.getenv("APP_PORT", "7866")),
        share=False,
        theme=gr.themes.Soft(),
    )
