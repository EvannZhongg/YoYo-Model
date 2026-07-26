import logging
import os
from pathlib import Path

import gradio as gr

from annotation.annotator import annotate_image_for_dataset, run_detection_streaming
from annotation.prompts import EXAMPLE_PROMPTS, YOYO_DETECTION_PROMPT
from common.files import collect_files
from config import BASE_DIR, DATASET_CONFIG, MODEL_CONFIG, TRACKING_CONFIG
from video_tracking.frame_review import append_tracking_frame_review, load_tracking_frame_selection
from video_tracking.tracker import track_video
from workbench.commands import workbench_evaluate_v2v3, workbench_train_v2v3
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


EXAMPLE_IMAGES = [
    BASE_DIR / "example1.jpg",
    BASE_DIR / "example2.png",
    BASE_DIR / "example3.png",
    None,
]


def _example_value(path: Path | None):
    return str(path) if path is not None and path.exists() else None


def run_dataset_annotation_streaming(
    input_dir: str,
    output_dir: str,
    prompt: str,
    model: str,
    min_pixels_str: str,
    max_pixels_str: str,
):
    try:
        image_paths = collect_files(
            Path(input_dir),
            DATASET_CONFIG.image_extensions,
            recursive=DATASET_CONFIG.recursive,
        )
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
    string_attachment_class: str,
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
            string_inference_scale=float(string_inference_scale),
            string_inference_fps=float(string_inference_fps),
            string_attachment_class=string_attachment_class,
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
    unified_dataset = BASE_DIR / "datasets" / "yoyo_dataset"
    with gr.Blocks(title="YoYo Model") as demo:
        gr.Markdown("# YoYo Model")
        with gr.Tabs():
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        example_selector = gr.Radio(
                            choices=["YoYo Prompt", "Example 2: Cars", "Example 3: People", "Upload your own image"],
                            value="YoYo Prompt",
                            label="Select Example",
                        )
                        image_input = gr.Image(label="Input Image", type="pil", value=_example_value(EXAMPLE_IMAGES[0]))
                        user_prompt = gr.Textbox(label="Prompt", lines=10, value=YOYO_DETECTION_PROMPT)
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=list(MODEL_CONFIG.available_models),
                            value=MODEL_CONFIG.default_model,
                        )
                        min_pixels_input = gr.Textbox(label="Min Image Tokens", value=MODEL_CONFIG.min_image_tokens)
                        max_pixels_input = gr.Textbox(label="Max Image Tokens", value=MODEL_CONFIG.max_image_tokens)
                        run_btn = gr.Button("Run Object Detection", variant="primary")
                    with gr.Column(scale=1):
                        output_vis = gr.Image(label="Detection Result")
                        raw_output = gr.Textbox(label="Raw Model Response", lines=8, interactive=False, buttons=["copy"])
                        status_output = gr.Textbox(label="Status / Summary", lines=3, interactive=False)

                def on_example_select(selection):
                    index = ["YoYo Prompt", "Example 2: Cars", "Example 3: People", "Upload your own image"].index(selection)
                    return _example_value(EXAMPLE_IMAGES[index]), EXAMPLE_PROMPTS[index]

                example_selector.change(on_example_select, [example_selector], [image_input, user_prompt])
                run_btn.click(
                    run_detection_streaming,
                    [image_input, user_prompt, model_dropdown, min_pixels_input, max_pixels_input],
                    [output_vis, raw_output, status_output],
                )

            with gr.Tab("Dataset Auto Label"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_input_dir = gr.Textbox(label="Dataset Image Directory", value=str(DATASET_CONFIG.image_input_dir))
                        dataset_output_dir = gr.Textbox(label="Annotation Output Directory", value=str(DATASET_CONFIG.annotation_output_dir))
                        dataset_prompt = gr.Textbox(label="Dataset Prompt", lines=10, value=YOYO_DETECTION_PROMPT)
                        dataset_model = gr.Dropdown(
                            label="Model",
                            choices=list(MODEL_CONFIG.available_models),
                            value=MODEL_CONFIG.default_model,
                        )
                        dataset_min_pixels = gr.Textbox(label="Min Image Tokens", value=MODEL_CONFIG.min_image_tokens)
                        dataset_max_pixels = gr.Textbox(label="Max Image Tokens", value=MODEL_CONFIG.max_image_tokens)
                        annotate_btn = gr.Button("Annotate Dataset", variant="primary")
                    dataset_status = gr.Textbox(label="Batch Status", lines=24, interactive=False, buttons=["copy"])
                annotate_btn.click(
                    run_dataset_annotation_streaming,
                    [dataset_input_dir, dataset_output_dir, dataset_prompt, dataset_model, dataset_min_pixels, dataset_max_pixels],
                    [dataset_status],
                )

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
                        video_input = gr.Video(label="Input Video")
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
                        tracking_string_attachment = gr.Dropdown(
                            label="String Attachment (current 1A)",
                            choices=[
                                ("Current 1A: hand and yoyo attached", "hand_and_yoyo_attached"),
                                ("Unknown / not visible / needs review", "unknown"),
                            ],
                            value=TRACKING_CONFIG.string_attachment_class,
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
                        tracking_string_scale, tracking_string_fps, tracking_string_attachment,
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
    os.makedirs(DATASET_CONFIG.temp_output_dir, exist_ok=True)
    os.makedirs(DATASET_CONFIG.annotation_output_dir, exist_ok=True)
    create_demo().launch(
        server_name=os.getenv("APP_HOST", "0.0.0.0"),
        server_port=int(os.getenv("APP_PORT", "7866")),
        share=False,
        theme=gr.themes.Soft(),
    )
