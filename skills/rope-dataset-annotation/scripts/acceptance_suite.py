#!/usr/bin/env python3
"""Prepare and summarize reproducible multi-scenario rope acceptance suites."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from rope_pipeline import (
    apply_candidate,
    audit_collection,
    clean_id,
    content_digest,
    initial_label,
    read_json,
    render_layer,
    write_json,
)


def point_box_distance(point: list[float], box: list[float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def nearest_anchor(point: list[float], label: dict[str, Any], threshold: float) -> str:
    candidates: list[tuple[float, str]] = []
    for name, hand in (label.get("hands_pixel") or {}).items():
        if hand:
            candidates.append((math.hypot(point[0] - hand[0], point[1] - hand[1]), f"{name}_hand"))
    box = label.get("yoyo_bbox_pixel")
    if box:
        candidates.append((point_box_distance(point, box), "yoyo"))
    if not candidates:
        return "unknown"
    distance, name = min(candidates)
    return name if distance <= threshold else "unknown"


def temporal_path_seed(candidate: dict[str, Any]) -> dict[str, Any]:
    width, height = [int(item) for item in candidate["image_size"]]
    threshold = math.hypot(width, height) * 0.06
    paths = []
    unresolved = []
    for index, stroke in enumerate(candidate.get("string_polylines_pixel") or []):
        if not isinstance(stroke, list) or len(stroke) < 2:
            continue
        start_anchor = nearest_anchor(stroke[0], candidate, threshold)
        end_anchor = nearest_anchor(stroke[-1], candidate, threshold)
        paths.append(
            {
                "path_id": f"legacy-stroke-{index + 1}",
                "start_anchor": start_anchor,
                "end_anchor": end_anchor,
                "points_pixel": stroke,
                "edges": [
                    {"from": edge, "to": edge + 1, "evidence": "temporal", "confidence": 0.55}
                    for edge in range(len(stroke) - 1)
                ],
            }
        )
        if start_anchor == "unknown" or end_anchor == "unknown":
            unresolved.append(f"legacy stroke {index + 1} has an unanchored endpoint")
    return {
        "topology": "uncertain" if len(paths) != 1 else "open",
        "reconstruction_status": "partial" if paths else "not_applicable",
        "paths": paths,
        "unresolved_gaps": unresolved,
    }


def legacy_candidate(legacy: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "visibility",
        "yoyo_bbox_pixel",
        "string_visibility",
        "string_polylines_pixel",
        "string_mask_polygons_pixel",
        "hands_pixel",
        "scene_label",
        "bad_case",
        "notes",
    )
    candidate = {key: legacy.get(key) for key in fields}
    candidate["image_size"] = legacy["image_size"]
    candidate["string_attachment_class"] = "unknown"
    candidate["string_path"] = temporal_path_seed({**legacy, **candidate})
    candidate["notes"] = (str(candidate.get("notes") or "") + " Imported only as a low-confidence acceptance seed.").strip()
    return candidate


def save_thumb(source: Path, label: dict[str, Any], output: Path, overlay: bool, width: int = 560) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    scale = width / image.width
    resized = image.resize((width, round(image.height * scale)), Image.Resampling.LANCZOS)
    rendered = render_layer(resized, label, scale, False) if overlay else resized
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output, quality=92)


def contact_sheet(rows: list[dict[str, Any]], image_key: str, output: Path, columns: int = 2) -> None:
    images = []
    font = ImageFont.load_default()
    for row in rows:
        with Image.open(row[image_key]) as opened:
            image = opened.convert("RGB")
        header = 42
        tile = Image.new("RGB", (image.width, image.height + header), "#111111")
        tile.paste(image, (0, header))
        draw = ImageDraw.Draw(tile)
        text = f"{row['id']} | {row['scenario']}"
        draw.text((8, 8), text[:95], fill="white", font=font)
        images.append(tile)
    if not images:
        return
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    rows_count = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows_count), "#222222")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def command_prepare(args: argparse.Namespace) -> int:
    manifest_path = Path(args.cases).resolve()
    output = Path(args.output).resolve()
    manifest = read_json(manifest_path)
    rows = []
    for case in manifest.get("cases") or []:
        case_id = clean_id(str(case["id"]))
        legacy_path = Path(case["legacy_label"])
        if not legacy_path.is_absolute():
            legacy_path = (manifest_path.parent / legacy_path).resolve()
        legacy = read_json(legacy_path)
        source = Path(legacy["source_image"])
        split = str(legacy.get("split") or "train")
        group = clean_id(str(legacy.get("source_group") or source.parent.name))
        target = output / "labels" / split / group / f"{case_id}.json"
        draft = initial_label(source, split, group, args.min_approvals)
        draft["frame_index"] = legacy.get("frame_index")
        draft["timestamp_s"] = legacy.get("timestamp_s")
        write_json(target, draft)
        candidate = legacy_candidate(legacy)
        applied = apply_candidate(
            target,
            candidate,
            actor="acceptance-legacy-seed",
            role="prelabel-import",
            model=str(legacy.get("model") or "legacy"),
            message="low-confidence seed for independent model acceptance",
        )
        raw_thumb = output / "review" / f"{case_id}_raw.jpg"
        overlay_thumb = output / "review" / f"{case_id}_overlay.jpg"
        save_thumb(source, applied, raw_thumb, overlay=False)
        save_thumb(source, applied, overlay_thumb, overlay=True)
        rows.append(
            {
                "id": case_id,
                "scenario": str(case["scenario"]),
                "legacy_label": str(legacy_path),
                "label": str(target),
                "raw": str(raw_thumb),
                "overlay": str(overlay_thumb),
            }
        )
    contact_sheet(rows, "raw", output / "review" / "raw_contact_sheet.jpg")
    contact_sheet(rows, "overlay", output / "review" / "overlay_contact_sheet.jpg")
    suite = {
        "schema_version": "rope_acceptance_suite_v1",
        "cases_manifest": str(manifest_path),
        "output": str(output),
        "case_count": len(rows),
        "cases": rows,
    }
    write_json(output / "suite.json", suite)
    print(json.dumps({"case_count": len(rows), "suite": str(output / 'suite.json'), "review": str(output / 'review')}, ensure_ascii=False, indent=2))
    return 0


def command_report(args: argparse.Namespace) -> int:
    root = Path(args.output).resolve()
    suite = read_json(root / "suite.json")
    audit = audit_collection(root / "labels", check_image=True, require_approved=False)
    records = {item["label"]: item for item in audit["records"]}
    results = []
    for case in suite["cases"]:
        label = read_json(Path(case["label"]))
        record = records.get(case["label"], {"errors": ["missing audit record"], "warnings": []})
        status = str(label.get("string_review_status"))
        digest = content_digest(label)
        current_decisions = {
            str(review.get("decision"))
            for review in (label.get("quality") or {}).get("reviews") or []
            if review.get("content_sha256") == digest
        }
        handled = bool(current_decisions & {"unresolved", "reject"}) or label.get("string_visibility") == "uncertain"
        outcome = "accepted" if status in {"approved", "reviewed"} and not record["errors"] else "handled_unresolved" if handled else "pending_or_failed"
        results.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "outcome": outcome,
                "string_visibility": label.get("string_visibility"),
                "review_status": status,
                "errors": record["errors"],
                "warnings": record["warnings"],
            }
        )
    summary = {
        "accepted": sum(item["outcome"] == "accepted" for item in results),
        "handled_unresolved": sum(item["outcome"] == "handled_unresolved" for item in results),
        "pending_or_failed": sum(item["outcome"] == "pending_or_failed" for item in results),
    }
    report = {"schema_version": "rope_acceptance_report_v1", "summary": summary, "results": results, "audit": audit}
    write_json(root / "acceptance_report.json", report)
    print(json.dumps({"summary": summary, "report": str(root / 'acceptance_report.json')}, ensure_ascii=False, indent=2))
    return 0 if summary["pending_or_failed"] == 0 else 1


def command_confirm(args: argparse.Namespace) -> int:
    path = Path(args.label).resolve()
    label = read_json(path)
    candidate = json.loads(json.dumps(label))
    promoted = 0
    for path_item in (candidate.get("string_path") or {}).get("paths") or []:
        for edge in path_item.get("edges") or []:
            if edge.get("evidence") == "temporal":
                edge["evidence"] = "observed"
                edge["confidence"] = max(float(edge.get("confidence", 0.0)), args.confidence)
                promoted += 1
    if args.visibility:
        candidate["string_visibility"] = args.visibility
    if args.clear_mask:
        candidate["string_mask_polygons_pixel"] = None
    candidate["notes"] = (str(candidate.get("notes") or "") + " " + args.notes).strip()
    result = apply_candidate(
        path,
        candidate,
        actor=args.actor,
        role="model-acceptance-refiner",
        model=args.model,
        message=args.notes,
    )
    print(json.dumps({"label": str(path), "promoted_edges": promoted, "revision": result["quality"]["revision"]}, ensure_ascii=False, indent=2))
    return 0


def command_defer(args: argparse.Namespace) -> int:
    path = Path(args.label).resolve()
    label = read_json(path)
    candidate = json.loads(json.dumps(label))
    candidate["string_visibility"] = "uncertain"
    candidate["string_polylines_pixel"] = None
    candidate["string_polyline_pixel"] = None
    candidate["string_polylines_2d"] = None
    candidate["string_polyline_2d"] = None
    candidate["string_mask_polygons_pixel"] = None
    candidate["bad_case"] = sorted(set((candidate.get("bad_case") or []) + ["model_review_unresolved"]))
    for path_item in (candidate.get("string_path") or {}).get("paths") or []:
        for edge in path_item.get("edges") or []:
            edge["evidence"] = "inferred"
            edge["confidence"] = min(float(edge.get("confidence", 0.0)), 0.35)
    candidate["notes"] = (str(candidate.get("notes") or "") + " " + args.notes).strip()
    result = apply_candidate(
        path,
        candidate,
        actor=args.actor,
        role="model-acceptance-unresolved",
        model=args.model,
        message=args.notes,
    )
    print(json.dumps({"label": str(path), "visibility": "uncertain", "revision": result["quality"]["revision"]}, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cases", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--min-approvals", type=int, default=2)
    prepare.set_defaults(func=command_prepare)
    report = commands.add_parser("report")
    report.add_argument("--output", required=True)
    report.set_defaults(func=command_report)
    confirm = commands.add_parser("confirm")
    confirm.add_argument("--label", required=True)
    confirm.add_argument("--actor", default="model-acceptance-refiner")
    confirm.add_argument("--model", default="codex-model-review")
    confirm.add_argument("--confidence", type=float, default=0.9)
    confirm.add_argument("--visibility", choices=("visible", "partial", "not_visible", "uncertain"), default="")
    confirm.add_argument("--clear-mask", action="store_true")
    confirm.add_argument("--notes", required=True)
    confirm.set_defaults(func=command_confirm)
    defer = commands.add_parser("defer")
    defer.add_argument("--label", required=True)
    defer.add_argument("--actor", default="model-acceptance-unresolved")
    defer.add_argument("--model", default="codex-model-review")
    defer.add_argument("--notes", required=True)
    defer.set_defaults(func=command_defer)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
