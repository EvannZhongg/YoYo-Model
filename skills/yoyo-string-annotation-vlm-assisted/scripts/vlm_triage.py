"""Run bounded weak-VLM triage and build deterministic visual-agent handoffs."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

import annotation_pipeline as labels


PROMPT_VERSION = "weak_yoyo_triage_v2"
RESULT_SCHEMA = "weak_vlm_triage_result_v1"
MANIFEST_SCHEMA = "weak_vlm_triage_manifest_v1"
HANDOFF_SCHEMA = "yoyo_visual_agent_handoff_v1"
SKILL_ROOT = Path(__file__).resolve().parents[1]

DOMAIN_STATUS = {"in_domain", "out_of_domain", "invalid_source", "uncertain"}
SCENE_LABELS = {"trick", "non_trick", "unknown"}
PRESENCE = {"present", "absent", "uncertain"}
STRING_EVIDENCE = {"obvious", "possible", "none_obvious", "uncertain"}
USABILITY = {"usable", "severely_degraded", "uncertain"}
PRIORITY_SUGGESTIONS = {"quick_verify", "clear_candidate", "standard", "hard_case", "uncertain"}
TRIAGE_BAD_CASES = {"motion_blur", "low_contrast", "edge_clipped", "severe_occlusion"}
PROHIBITED_KEY_PARTS = {
    "bbox",
    "box",
    "coordinate",
    "geometry",
    "hand",
    "mask",
    "orientation",
    "point",
    "polyline",
    "polygon",
    "pose",
    "review_status",
    "string_visibility",
    "wrist",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scalar_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, parsed)), 4)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("VLM response does not contain a JSON object")


def prohibited_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            current = f"{prefix}.{key}" if prefix else str(key)
            if any(token in normalized for token in PROHIBITED_KEY_PARTS):
                found.append(current)
            found.extend(prohibited_keys(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(prohibited_keys(item, f"{prefix}[{index}]"))
    return found


def normalize_assessment(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    blocked = prohibited_keys(raw)
    if blocked:
        warnings.append("discarded prohibited visual-authority fields: " + ", ".join(sorted(set(blocked))))

    domain = str(raw.get("domain_status", "uncertain")).strip().lower()
    scene = str(raw.get("scene_label", "unknown")).strip().lower()
    yoyo = str(raw.get("obvious_yoyo_presence", "uncertain")).strip().lower()
    string_evidence = str(raw.get("coarse_string_evidence", "uncertain")).strip().lower()
    usability = str(raw.get("frame_usability", "uncertain")).strip().lower()
    priority = str(raw.get("priority_suggestion", "uncertain")).strip().lower()
    bad_cases = sorted(
        {
            str(item).strip().lower()
            for item in (raw.get("obvious_bad_cases") or [])
            if str(item).strip().lower() in TRIAGE_BAD_CASES
        }
    )
    confidence_raw = raw.get("confidence") if isinstance(raw.get("confidence"), dict) else {}
    notes = re.sub(r"\s+", " ", str(raw.get("notes", "")).strip())[:500]
    assessment = {
        "domain_status": domain if domain in DOMAIN_STATUS else "uncertain",
        "scene_label": scene if scene in SCENE_LABELS else "unknown",
        "scene_is_obvious": bool(raw.get("scene_is_obvious", False)),
        "obvious_yoyo_presence": yoyo if yoyo in PRESENCE else "uncertain",
        "coarse_string_evidence": string_evidence if string_evidence in STRING_EVIDENCE else "uncertain",
        "frame_usability": usability if usability in USABILITY else "uncertain",
        "priority_suggestion": priority if priority in PRIORITY_SUGGESTIONS else "uncertain",
        "obvious_bad_cases": bad_cases,
        "notes": notes,
        "confidence": {
            "domain": scalar_confidence(confidence_raw.get("domain")),
            "scene": scalar_confidence(confidence_raw.get("scene")),
            "yoyo_presence": scalar_confidence(confidence_raw.get("yoyo_presence")),
            "bad_cases": scalar_confidence(confidence_raw.get("bad_cases")),
            "priority": scalar_confidence(confidence_raw.get("priority")),
            "overall": scalar_confidence(confidence_raw.get("overall")),
        },
    }
    return assessment, warnings


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str | None
    model: str
    min_pixels: int
    max_pixels: int
    max_response_tokens: int
    enable_thinking: bool
    timeout_seconds: float
    retries: int
    promotion_confidence: float
    quick_verify_confidence: float
    notes_confidence: float
    safe_bad_cases: tuple[str, ...]


def resolve_env_file(env_file: Path | None) -> Path:
    return env_file.resolve() if env_file else SKILL_ROOT / ".env"


def load_settings(config_path: Path, env_file: Path | None, model_override: str = "") -> Settings:
    selected_env = resolve_env_file(env_file)
    if selected_env.exists():
        load_dotenv(selected_env, override=False)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model = config.get("model") or {}
    triage = config.get("triage") or {}
    api_key_env = str(model.get("api_key_env", "API_KEY"))
    return Settings(
        base_url=str(os.getenv("BASE_URL") or model.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv(api_key_env),
        model=str(model_override or os.getenv("DEFAULT_MODEL") or model.get("default_model") or "qwen3.6-35b-a3b"),
        min_pixels=int(model.get("min_image_tokens", 1024)) * 32 * 32,
        max_pixels=int(model.get("max_image_tokens", 9800)) * 32 * 32,
        max_response_tokens=int(model.get("max_response_tokens", 3000)),
        enable_thinking=bool(model.get("enable_thinking", False)),
        timeout_seconds=float(model.get("request_timeout_seconds", 180)),
        retries=max(0, int(model.get("retries", 2))),
        promotion_confidence=scalar_confidence(triage.get("promotion_confidence", 0.9)),
        quick_verify_confidence=scalar_confidence(triage.get("quick_verify_confidence", 0.95)),
        notes_confidence=scalar_confidence(triage.get("notes_confidence", 0.7)),
        safe_bad_cases=tuple(str(item) for item in triage.get("safe_bad_cases", ["motion_blur", "low_contrast", "edge_clipped"])),
    )


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def triage_prompt() -> str:
    return """You are a low-cost visual triage worker for yoyo video frames.
Perform only coarse, obvious judgments. Do not trace thin strings and do not output any coordinates, bounding boxes, points, masks, polylines, polygons, trick orientation, final string visibility, review decision, or approval.

Return exactly one JSON object with these fields:
{
  "domain_status": "in_domain|out_of_domain|invalid_source|uncertain",
  "scene_label": "trick|non_trick|unknown",
  "scene_is_obvious": true,
  "obvious_yoyo_presence": "present|absent|uncertain",
  "coarse_string_evidence": "obvious|possible|none_obvious|uncertain",
  "frame_usability": "usable|severely_degraded|uncertain",
  "priority_suggestion": "quick_verify|clear_candidate|standard|hard_case|uncertain",
  "obvious_bad_cases": ["motion_blur|low_contrast|edge_clipped|severe_occlusion"],
  "notes": "one short factual sentence about only obvious frame-level evidence",
  "confidence": {
    "domain": 0.0,
    "scene": 0.0,
    "yoyo_presence": 0.0,
    "bad_cases": 0.0,
    "priority": 0.0,
    "overall": 0.0
  }
}

Use uncertain when evidence is not obvious. none_obvious means only that no string is obvious at this resolution; it never means a reviewed negative label."""


def call_vlm(image_path: Path, settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    if not settings.api_key:
        raise RuntimeError(
            f"API key is missing; set the configured api_key_env, add it to {SKILL_ROOT / '.env'}, "
            "or pass --env-file"
        )
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=settings.timeout_seconds)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels,
                    "image_url": {"url": image_data_url(image_path)},
                },
                {"type": "text", "text": triage_prompt()},
            ],
        }
    ]
    last_error: Exception | None = None
    for attempt in range(settings.retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                max_tokens=settings.max_response_tokens,
                extra_body={"enable_thinking": settings.enable_thinking},
                stream=False,
            )
            content = response.choices[0].message.content or ""
            raw = extract_json_object(content)
            return raw, {
                "request_id": getattr(response, "id", None),
                "model": settings.model,
                "base_url": settings.base_url,
                "attempt": attempt + 1,
            }
        except Exception as exc:
            last_error = exc
            if attempt < settings.retries:
                time.sleep(min(8.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"VLM request failed after {settings.retries + 1} attempts: {last_error}")


def compute_promotions(assessment: dict[str, Any], settings: Settings) -> dict[str, Any]:
    confidence = assessment["confidence"]
    promoted: dict[str, Any] = {}
    if (
        assessment["domain_status"] == "in_domain"
        and confidence["domain"] >= settings.promotion_confidence
        and assessment["scene_is_obvious"]
        and assessment["scene_label"] in {"trick", "non_trick"}
        and confidence["scene"] >= settings.promotion_confidence
    ):
        promoted["scene_label"] = assessment["scene_label"]
    safe_bad_cases = sorted(set(assessment["obvious_bad_cases"]) & set(settings.safe_bad_cases))
    if safe_bad_cases and confidence["bad_cases"] >= settings.promotion_confidence:
        promoted["bad_case"] = safe_bad_cases
    if assessment["notes"] and confidence["overall"] >= settings.notes_confidence:
        promoted["notes"] = f"Weak-VLM API-resolution observation, not string truth: {assessment['notes']}"
    return promoted


def compute_handoff(assessment: dict[str, Any], promotions: dict[str, Any], settings: Settings) -> dict[str, Any]:
    confidence = assessment["confidence"]
    quick_verify = (
        assessment["domain_status"] in {"out_of_domain", "invalid_source"}
        and confidence["domain"] >= settings.quick_verify_confidence
    )
    if quick_verify:
        queue = "quick_verify"
        rank = 0
        tasks = [
            "Inspect the raw frame once and confirm whether the source is invalid or outside the yoyo domain.",
            "Record reject only when the raw pixels confirm the triage result; otherwise return the record to full annotation.",
        ]
    else:
        suggested = assessment["priority_suggestion"] if confidence["priority"] >= 0.75 else "uncertain"
        hard = (
            suggested == "hard_case"
            or assessment["frame_usability"] == "severely_degraded"
            or "severe_occlusion" in assessment["obvious_bad_cases"]
            or confidence["overall"] < 0.65
        )
        clear_positive = (
            assessment["domain_status"] == "in_domain"
            and assessment["frame_usability"] == "usable"
            and assessment["obvious_yoyo_presence"] == "present"
            and confidence["overall"] >= 0.8
        )
        queue = "hard_case" if hard else "clear_candidate" if clear_positive and suggested in {"clear_candidate", "uncertain"} else "standard"
        rank = 3 if hard else 1 if clear_positive else 2
        tasks = [
            "Inspect original pixels and annotate every defensible visible string centerline segment.",
            "Set final string visibility, yoyo bbox, yoyo/unknown anchors, ordered path, hidden gaps, and trick orientation.",
            "Render and refine geometry, then obtain independent geometry and semantic approvals.",
        ]
    skipped = []
    if "scene_label" in promotions:
        skipped.append("coarse scene classification")
    if promotions.get("bad_case"):
        skipped.append("obvious bad-case tagging for: " + ", ".join(promotions["bad_case"]))
    if "notes" in promotions:
        skipped.append("initial factual note formatting")
    return {
        "queue": queue,
        "priority_rank": rank,
        "skip_decisions": skipped,
        "required_visual_agent_tasks": tasks,
        "override_rule": "Keep promoted coarse fields unless original pixels clearly contradict them.",
    }


def resolve_image(label: dict[str, Any], label_path: Path) -> Path:
    return labels.resolve_source_image(label, label_path)


def already_applied(label: dict[str, Any], message: str) -> bool:
    history = (label.get("quality") or {}).get("history") or []
    return any(isinstance(item, dict) and item.get("message") == message for item in history)


def apply_safe_promotions(
    label_path: Path,
    label: dict[str, Any],
    promotions: dict[str, Any],
    model: str,
    result_digest: str,
) -> tuple[dict[str, Any], str]:
    if not promotions:
        return label, "no_safe_fields"
    message = f"trusted weak-VLM triage {result_digest[:16]}"
    if already_applied(label, message):
        return label, "already_applied"
    quality = label.get("quality") or {}
    if label.get("string_review_status") in labels.ACCEPTED_REVIEW or quality.get("reviews"):
        return label, "skipped_reviewed_label"
    if label.get("string_polylines_pixel") or label.get("string_mask_polygons_pixel"):
        return label, "skipped_existing_geometry"
    candidate = copy.deepcopy(label)
    if "scene_label" in promotions:
        candidate["scene_label"] = promotions["scene_label"]
    if promotions.get("bad_case"):
        candidate["bad_case"] = sorted(set(candidate.get("bad_case") or []) | set(promotions["bad_case"]))
    if promotions.get("notes"):
        existing = str(candidate.get("notes") or "").strip()
        candidate["notes"] = " ".join(item for item in (existing, promotions["notes"]) if item)[:2000]
    updated = labels.apply_candidate(
        label_path,
        candidate,
        actor="weak-vlm-triage",
        role="weak-vlm-triager",
        model=model,
        message=message,
    )
    return updated, "applied"


def load_replay_response(replay_dir: Path, label_path: Path) -> dict[str, Any]:
    path = replay_dir / f"{label_path.stem}.json"
    return read_json(path)


def run_record(
    label_path: Path,
    output: Path,
    settings: Settings,
    replay_dir: Path | None,
    apply_fields: bool,
    force: bool,
) -> dict[str, Any]:
    label = read_json(label_path)
    image_path = resolve_image(label, label_path)
    if not image_path.exists():
        raise FileNotFoundError(f"source image does not exist: {image_path}")
    result_path = output / "results" / str(label.get("source_group") or "ungrouped") / f"{label_path.stem}.json"
    if result_path.exists() and not force:
        existing = read_json(result_path)
        if (
            existing.get("prompt_version") == PROMPT_VERSION
            and existing.get("model") == settings.model
            and existing.get("image_sha256") == label.get("image_sha256")
        ):
            return existing
    if replay_dir:
        raw = load_replay_response(replay_dir, label_path)
        request_meta = {"request_id": None, "model": settings.model, "base_url": "replay", "attempt": 0}
    else:
        raw, request_meta = call_vlm(image_path, settings)
    assessment, warnings = normalize_assessment(raw)
    promotions = compute_promotions(assessment, settings)
    handoff = compute_handoff(assessment, promotions, settings)
    result = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": utc_now(),
        "prompt_version": PROMPT_VERSION,
        "model": settings.model,
        "request": request_meta,
        "label": str(label_path),
        "source_image": str(image_path),
        "image_sha256": label.get("image_sha256"),
        "source_group": label.get("source_group"),
        "sequence_id": label.get("sequence_id"),
        "frame_index": label.get("frame_index"),
        "assessment": assessment,
        "warnings": warnings,
        "promotions": promotions,
        "handoff": handoff,
    }
    result_digest = canonical_digest(result)
    result["result_sha256"] = result_digest
    apply_status = "disabled"
    if apply_fields:
        updated, apply_status = apply_safe_promotions(label_path, label, promotions, settings.model, result_digest)
        result["label_revision_after_triage"] = int((updated.get("quality") or {}).get("revision", 0))
    result["apply_status"] = apply_status
    write_json(result_path, result)
    return result


def command_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve() if args.config else SKILL_ROOT / "config.yaml"
    env_file = Path(args.env_file).resolve() if args.env_file else None
    settings = load_settings(config_path, env_file, args.model)
    labels_root = Path(args.labels).resolve()
    output = Path(args.output).resolve() if args.output else labels_root.parent / "triage"
    replay_dir = Path(args.response_replay).resolve() if args.response_replay else None
    paths = labels.label_files(labels_root)
    if args.max_records is not None:
        paths = paths[: max(0, args.max_records)]
    records = []
    failures = []
    for label_path in paths:
        try:
            records.append(
                run_record(
                    label_path,
                    output,
                    settings,
                    replay_dir,
                    args.apply_safe_fields,
                    args.force,
                )
            )
        except Exception as exc:
            failures.append({"label": str(label_path), "error": str(exc)})
    handoff_records = [
        {
            "label": item["label"],
            "source_image": item["source_image"],
            "source_group": item.get("source_group"),
            "sequence_id": item.get("sequence_id"),
            "frame_index": item.get("frame_index"),
            "assessment": item["assessment"],
            "promotions": item["promotions"],
            **item["handoff"],
        }
        for item in records
    ]
    handoff_records.sort(key=lambda item: (item["priority_rank"], str(item.get("source_group")), int(item.get("frame_index") or 0)))
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "created_at_utc": utc_now(),
        "record_count": len(handoff_records),
        "records": handoff_records,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "prompt_version": PROMPT_VERSION,
        "model": settings.model,
        "labels": str(labels_root),
        "output": str(output),
        "record_count": len(records),
        "failure_count": len(failures),
        "queue_counts": {
            queue: sum(1 for item in handoff_records if item["queue"] == queue)
            for queue in sorted({item["queue"] for item in handoff_records})
        },
        "failures": failures,
    }
    write_json(output / "agent_handoff.json", handoff)
    write_json(output / "triage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Triage label frames and write a sorted visual-agent handoff.")
    run.add_argument("--labels", required=True)
    run.add_argument("--output", default="")
    run.add_argument("--config", default="")
    run.add_argument("--env-file", default="")
    run.add_argument("--model", default="")
    run.add_argument("--response-replay", default="", help="Read per-label VLM JSON responses from this directory instead of calling the API.")
    run.add_argument("--max-records", type=int)
    run.add_argument("--apply-safe-fields", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=command_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
