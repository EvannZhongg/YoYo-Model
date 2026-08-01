#!/usr/bin/env python3
"""Run a standalone multi-scenario acceptance suite for the annotation skill."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import annotation_pipeline as pipeline


WIDTH, HEIGHT = 640, 360
HAND_LEFT = [88, 74]
HAND_RIGHT = [552, 74]
YOYO_BOX = [294, 268, 346, 320]


def synthetic_provenance(source_group: str, frame_index: int) -> dict[str, Any]:
    return {
        "source_video": str(Path("source-videos") / f"{source_group}.mp4"),
        "source_video_sha256": hashlib.sha256(source_group.encode("utf-8")).hexdigest(),
        "source_group": source_group,
        "sequence_id": f"seq-001-anchor-{frame_index:08d}",
        "role": "anchor",
        "anchor_frame_index": frame_index,
        "frame_index": frame_index,
        "timestamp_s": frame_index / 10.0,
    }


def endpoint_anchor(point: list[float]) -> str:
    distances = {
        "left_hand": ((point[0] - HAND_LEFT[0]) ** 2 + (point[1] - HAND_LEFT[1]) ** 2) ** 0.5,
        "right_hand": ((point[0] - HAND_RIGHT[0]) ** 2 + (point[1] - HAND_RIGHT[1]) ** 2) ** 0.5,
        "yoyo": pipeline.point_box_distance(point, YOYO_BOX),
    }
    anchor, distance = min(distances.items(), key=lambda item: item[1])
    return anchor if distance <= 36 else "unknown"


def observed_path(strokes: list[list[list[float]]], topology: str = "open") -> dict[str, Any]:
    paths = []
    for index, stroke in enumerate(strokes):
        paths.append(
            {
                "path_id": f"visible-{index + 1}",
                "start_anchor": endpoint_anchor(stroke[0]),
                "end_anchor": endpoint_anchor(stroke[-1]),
                "points_pixel": stroke,
                "edges": [
                    {"from": edge, "to": edge + 1, "evidence": "observed", "confidence": 0.98}
                    for edge in range(len(stroke) - 1)
                ],
            }
        )
    return {
        "topology": topology,
        "reconstruction_status": "complete" if len(strokes) == 1 else "partial",
        "paths": paths,
        "unresolved_gaps": [] if len(strokes) == 1 else ["visible strokes are separated by an intentional occlusion"],
    }


def positive_candidate(
    strokes: list[list[list[float]]],
    features: list[str],
    visibility: str = "visible",
    topology: str = "open",
    notes: str = "Synthetic acceptance truth.",
) -> dict[str, Any]:
    return {
        "visibility": "visible",
        "yoyo_bbox_pixel": YOYO_BOX,
        "string_visibility": visibility,
        "string_polylines_pixel": strokes,
        "string_mask_polygons_pixel": None,
        "hands_pixel": {"left": HAND_LEFT, "right": HAND_RIGHT},
        "yoyo_division": "1A",
        "scene_label": "trick",
        "trick_orientation": "normal",
        "string_path": observed_path(strokes, topology),
        "bad_case": [feature for feature in features if feature in {"motion_blur", "low_contrast", "occluded", "edge_clipped"}],
        "notes": notes,
    }


def negative_candidate(features: list[str], yoyo_visible: bool) -> dict[str, Any]:
    return {
        "visibility": "visible" if yoyo_visible else "absent",
        "yoyo_bbox_pixel": YOYO_BOX if yoyo_visible else None,
        "string_visibility": "not_visible",
        "string_polylines_pixel": None,
        "string_mask_polygons_pixel": None,
        "hands_pixel": {"left": None, "right": None},
        "yoyo_division": "1A",
        "scene_label": "non_trick",
        "trick_orientation": "not_applicable",
        "string_path": {
            "topology": "uncertain",
            "reconstruction_status": "not_applicable",
            "paths": [],
            "unresolved_gaps": [],
        },
        "bad_case": ["string_not_visible"],
        "notes": "Reviewed synthetic hard negative.",
    }


def shifted_candidate(candidate: dict[str, Any], dy: float) -> dict[str, Any]:
    shifted = copy.deepcopy(candidate)
    for stroke in shifted.get("string_polylines_pixel") or []:
        for point in stroke:
            point[1] += dy
    for path_item in (shifted.get("string_path") or {}).get("paths") or []:
        for point in path_item.get("points_pixel") or []:
            point[1] += dy
    shifted["notes"] = f"Deliberately shifted {dy}px acceptance candidate; refinement required."
    return shifted


def draw_scene(
    case_id: str,
    strokes: list[list[list[float]]],
    style: str,
    draw_yoyo: bool = True,
    draw_hands: bool = True,
) -> Image.Image:
    background = "#777b7e" if style in {"low_contrast", "ambiguous"} else "#d5d8dc"
    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)
    if draw_yoyo:
        draw.ellipse(YOYO_BOX, fill="#c0392b", outline="#202020", width=3)
    if draw_hands:
        draw.ellipse((HAND_LEFT[0] - 10, HAND_LEFT[1] - 10, HAND_LEFT[0] + 10, HAND_LEFT[1] + 10), fill="#f0b27a")
        draw.ellipse((HAND_RIGHT[0] - 10, HAND_RIGHT[1] - 10, HAND_RIGHT[0] + 10, HAND_RIGHT[1] + 10), fill="#f0b27a")
    color = "#7b7f81" if style == "ambiguous" else "#858a8d" if style == "low_contrast" else "#17202a"
    width = 2 if style in {"thin", "low_contrast"} else 4
    for stroke in strokes:
        draw.line([tuple(point) for point in stroke], fill=color, width=width, joint="curve")
    if style == "occlusion":
        draw.rectangle((270, 164, 370, 204), fill="#566573")
    if style in {"blur", "ambiguous"}:
        image = image.filter(ImageFilter.GaussianBlur(radius=4.0 if style == "ambiguous" else 1.7))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), case_id, fill="#111111", font=ImageFont.load_default())
    return image


def cases() -> list[dict[str, Any]]:
    return [
        {"id": "01-thin-straight", "style": "thin", "features": ["straight", "thin_string"], "strokes": [[[88, 74], [190, 135], [270, 220], [320, 270]]]},
        {"id": "02-deep-curve", "style": "normal", "features": ["curved", "long_route"], "strokes": [[[88, 74], [150, 42], [250, 92], [205, 190], [320, 270]]]},
        {"id": "03-open-v", "style": "normal", "features": ["v_shape", "open_formation", "thin_string"], "strokes": [[[88, 74], [320, 286], [552, 74]]]},
        {"id": "04-occluded-multistroke", "style": "occlusion", "features": ["occluded", "multi_segment", "curved"], "visibility": "partial", "strokes": [[[88, 74], [180, 128], [265, 172]], [[375, 196], [420, 230], [320, 270]]]},
        {"id": "05-loop", "style": "normal", "features": ["loop", "curved", "self_near"], "topology": "loop", "strokes": [[[88, 74], [240, 110], [360, 110], [420, 190], [340, 250], [260, 190], [320, 270]]]},
        {"id": "06-hard-negative", "negative": True, "yoyo_visible": True, "features": ["no_string", "background_edge", "hard_negative"], "strokes": []},
        {"id": "07-crossing", "style": "normal", "features": ["crossing", "multi_segment", "complex_topology"], "topology": "multiple", "strokes": [[[88, 74], [420, 240], [320, 270]], [[552, 74], [220, 240], [320, 270]]]},
        {"id": "08-edge-clipped", "style": "normal", "features": ["edge_clipped", "partial", "straight"], "visibility": "partial", "strokes": [[[0, 120], [125, 142], [245, 205], [320, 270]]]},
        {"id": "09-low-contrast", "style": "low_contrast", "features": ["low_contrast", "curved", "thin_string"], "strokes": [[[88, 74], [180, 98], [245, 170], [320, 270]]]},
        {"id": "10-non-trick-negative", "negative": True, "yoyo_visible": False, "features": ["no_string", "non_trick", "background_edge"], "strokes": []},
        {"id": "11-motion-blur", "style": "blur", "features": ["motion_blur", "curved", "partial"], "visibility": "partial", "strokes": [[[88, 74], [170, 115], [250, 205], [320, 270]]]},
        {"id": "12-repeated-refinement", "style": "normal", "features": ["repeated_refinement", "thin_string", "curved"], "requires_refinement": True, "strokes": [[[88, 74], [150, 155], [250, 205], [320, 270]]]},
        {"id": "13-ambiguous-unresolved", "style": "ambiguous", "features": ["ambiguous", "motion_blur", "low_contrast"], "unresolved": True, "strokes": [[[88, 74], [260, 180], [320, 270]], [[88, 74], [380, 170], [320, 270]]]},
        {"id": "14-temporal-previous", "style": "normal", "features": ["temporal_pair", "curved", "thin_string"], "source_group": "temporal-sequence", "strokes": [[[88, 74], [175, 120], [250, 195], [320, 270]]]},
        {"id": "15-temporal-current", "style": "normal", "features": ["temporal_pair", "curved", "translated"], "source_group": "temporal-sequence", "temporal_target": True, "strokes": [[[94, 78], [181, 124], [256, 199], [326, 274]]]},
    ]


def approve(label_path: Path, case_id: str) -> None:
    for reviewer, role, note in (
        (f"{case_id}-geometry", "geometry-critic", "Every consecutive edge follows the rendered ground-truth stroke."),
        (f"{case_id}-semantic", "semantic-critic", "Visibility, topology, continuity, and trick orientation match the scene."),
    ):
        pipeline.command_review(
            SimpleNamespace(
                label=str(label_path),
                decision="approve",
                reviewer=reviewer,
                role=role,
                model="acceptance-agent",
                notes=note,
            )
        )


def make_sheet(paths: list[Path], output: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((420, 250), Image.Resampling.LANCZOS)
        thumbs.append(image)
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 420, rows * 250), "#202326")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 420, (index // columns) * 250))
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(cases()) < 10:
        raise AssertionError("acceptance suite must contain at least ten scenarios")
    output = Path(args.output).resolve()
    if output.exists() and any(output.rglob("*")):
        parser.error(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    labels: dict[str, Path] = {}
    raw_paths: list[Path] = []
    overlay_paths: list[Path] = []
    for index, case in enumerate(cases(), start=1):
        case_id = case["id"]
        source_group = case.get("source_group", case_id)
        image_path = output / "images" / source_group / f"frame_{index:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = draw_scene(
            case_id,
            case["strokes"],
            case.get("style", "normal"),
            draw_yoyo=not case.get("negative") or bool(case.get("yoyo_visible")),
            draw_hands=not case.get("negative"),
        )
        image.save(image_path)
        raw_paths.append(image_path)
        label_path = output / "labels" / source_group / f"frame_{index:04d}.json"
        pipeline.write_json(
            label_path,
            pipeline.initial_label(
                image_path,
                synthetic_provenance(source_group, index),
                2,
                hashlib.sha256(b"synthetic-acceptance-manifest").hexdigest(),
            ),
        )
        labels[case_id] = label_path

        if case.get("unresolved"):
            candidate = {
                "visibility": "uncertain",
                "string_visibility": "uncertain",
                "string_path": {"topology": "uncertain", "reconstruction_status": "uncertain", "paths": [], "unresolved_gaps": ["two blurred routes are equally plausible"]},
                "bad_case": ["ambiguous_string", "motion_blur"],
                "notes": "Repeated direct inspection cannot choose between two blurred routes.",
            }
        elif case.get("negative"):
            candidate = negative_candidate(case["features"], case["yoyo_visible"])
        else:
            candidate = positive_candidate(
                case["strokes"],
                case["features"],
                case.get("visibility", "visible"),
                case.get("topology", "open"),
            )
        initial_candidate = shifted_candidate(candidate, 18.0) if case.get("requires_refinement") else candidate
        pipeline.apply_candidate(label_path, initial_candidate, "acceptance-agent", "model-annotator", "acceptance-agent", "direct synthetic scene annotation")
        pipeline.command_render(SimpleNamespace(label=str(label_path), output=str(output / "review"), max_side=1200))
        if case.get("unresolved"):
            pipeline.command_review(
                SimpleNamespace(
                    label=str(label_path), decision="unresolved", reviewer=f"{case_id}-geometry", role="geometry-critic",
                    model="acceptance-agent", notes="No defensible centerline remains after repeated inspection.",
                )
            )
            outcome = "handled_unresolved"
        elif case.get("temporal_target"):
            previous_path = labels["14-temporal-previous"]
            pipeline.command_propagate(
                SimpleNamespace(
                    previous_label=str(previous_path), target_label=str(label_path), actor="acceptance-temporal-seed",
                    model="deterministic-optical-flow", message="adjacent-frame seed",
                    max_error=2.5,
                )
            )
            seeded = pipeline.read_json(label_path)
            if seeded["string_review_status"] != "auto_labeled_needs_review":
                raise AssertionError("temporal propagation must remain pending")
            pipeline.apply_candidate(label_path, candidate, "acceptance-agent", "model-annotator", "acceptance-agent", "current-frame correction after temporal seed")
            approve(label_path, case_id)
            outcome = "accepted_after_temporal_refinement"
        elif case.get("requires_refinement"):
            pipeline.command_review(
                SimpleNamespace(
                    label=str(label_path), decision="request_changes", reviewer=f"{case_id}-geometry-initial",
                    role="geometry-critic", model="acceptance-agent",
                    notes="The centerline is displaced onto background pixels; move every point toward the rendered string.",
                )
            )
            intermediate = shifted_candidate(candidate, 5.0)
            pipeline.apply_candidate(label_path, intermediate, "acceptance-agent", "model-annotator", "acceptance-agent", "first point-level correction")
            pipeline.command_review(
                SimpleNamespace(
                    label=str(label_path), decision="approve", reviewer=f"{case_id}-geometry-intermediate",
                    role="geometry-critic", model="acceptance-agent", notes="Intermediate geometry is structurally valid.",
                )
            )
            if len(pipeline.current_approvals(pipeline.read_json(label_path))) != 1:
                raise AssertionError("intermediate revision should have one current approval")
            pipeline.apply_candidate(label_path, candidate, "acceptance-agent", "model-annotator", "acceptance-agent", "final pixel-level correction")
            if pipeline.current_approvals(pipeline.read_json(label_path)):
                raise AssertionError("editing geometry must invalidate approvals from the previous digest")
            approve(label_path, case_id)
            outcome = "accepted_after_repeated_refinement"
        else:
            approve(label_path, case_id)
            outcome = "accepted"
        pipeline.command_render(SimpleNamespace(label=str(label_path), output=str(output / "review"), max_side=1200))
        overlay_paths.append(output / "review" / f"frame_{index:04d}_overlay.jpg")
        final = pipeline.read_json(label_path)
        case_results.append(
            {
                "id": case_id,
                "scenario": case["features"],
                "outcome": outcome,
                "revision": final["quality"]["revision"],
                "status": final["string_review_status"],
                "current_approvals": len(pipeline.current_approvals(final)),
                "trick_orientation": final["trick_orientation"],
            }
        )

    previous = pipeline.read_json(labels["14-temporal-previous"])
    if previous["string_review_status"] != "approved":
        raise AssertionError("temporal source must be approved")

    audit = pipeline.audit_collection(
        output / "labels",
        check_image=True,
        require_approved=False,
    )
    if not audit["ok"]:
        raise AssertionError(json.dumps(audit["collection_errors"], ensure_ascii=False))
    export = output / "export"
    pipeline.command_export(SimpleNamespace(labels=str(output / "labels"), output=str(export), min_approvals=None))
    export_manifest = pipeline.read_json(export / "manifest.json")
    if export_manifest["exported_count"] != len(cases()) - 1 or export_manifest["excluded_count"] != 1:
        raise AssertionError("unresolved acceptance case was not excluded exactly once")
    for item in export_manifest["exported"]:
        if not Path(item["image"]).is_file() or not Path(item["visualization"]).is_file():
            raise AssertionError("portable export must include every source frame and terminal overlay")
        if pipeline.sha256_file(Path(item["visualization"])) != item["visualization_sha256"]:
            raise AssertionError("exported visualization digest mismatch")

    make_sheet(raw_paths, output / "raw_contact_sheet.jpg")
    make_sheet(overlay_paths, output / "overlay_contact_sheet.jpg")
    summary = {
        "schema_version": "agent_yoyo_string_acceptance_v1",
        "case_count": len(case_results),
        "accepted_count": sum(item["outcome"].startswith("accepted") for item in case_results),
        "handled_unresolved_count": sum(item["outcome"] == "handled_unresolved" for item in case_results),
        "pending_or_failed_count": sum(item["status"] not in {"approved", "unresolved"} for item in case_results),
        "audit_ok": audit["ok"],
        "exported_count": export_manifest["exported_count"],
        "excluded_count": export_manifest["excluded_count"],
        "cases": case_results,
    }
    pipeline.write_json(output / "acceptance_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
