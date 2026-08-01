"""Run sample, init, weak-VLM triage, and handoff validation for five videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs=5, required=True, help="Exactly five source video paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--hash-cache", default="")
    parser.add_argument("--response-replay", default="")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing sampled and initialized project under OUTPUT/project.")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("run this validation with a project virtual-environment interpreter")
    scripts = Path(__file__).resolve().parent
    skill_root = scripts.parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    list_path = output / "five_videos_utf8.txt"
    videos = [Path(item).resolve() for item in args.videos]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing videos: " + ", ".join(missing))
    list_path.write_text("\n".join(str(path) for path in videos) + "\n", encoding="utf-8")

    project = output / "project"
    hash_cache = Path(args.hash_cache).resolve() if args.hash_cache else output / "source_video_sha256_cache.json"
    if args.resume:
        if not (project / "sampling_manifest.json").is_file() or not any((project / "labels").rglob("*.json")):
            raise RuntimeError("--resume requires an initialized OUTPUT/project")
    else:
        run(
            [
                sys.executable,
                str(scripts / "sample_video_frames.py"),
                "--videos-list",
                str(list_path),
                "--output",
                str(project),
                "--total-anchors",
                "5",
                "--oversample-factor",
                "2",
                "--neighbor-offsets=-1,1",
                "--separate-context",
                "--hash-cache",
                str(hash_cache),
            ]
        )
        run(
            [
                sys.executable,
                str(scripts / "annotation_pipeline.py"),
                "init",
                "--images",
                str(project / "images"),
                "--output",
                str(project),
                "--min-approvals",
                "2",
            ]
        )
    triage_command = [
        sys.executable,
        str(scripts / "vlm_triage.py"),
        "run",
        "--labels",
        str(project / "labels"),
        "--output",
        str(project / "triage"),
        "--config",
        str(Path(args.config).resolve() if args.config else skill_root / "config.yaml"),
    ]
    if args.env_file:
        triage_command.extend(["--env-file", str(Path(args.env_file).resolve())])
    if args.response_replay:
        triage_command.extend(["--response-replay", str(Path(args.response_replay).resolve())])
    run(triage_command)

    sampling = read_json(project / "sampling_manifest.json")
    triage = read_json(project / "triage" / "triage_manifest.json")
    handoff = read_json(project / "triage" / "agent_handoff.json")
    label_paths = sorted((project / "labels").rglob("*.json"))
    checks = {
        "source_count_is_five": sampling.get("source_count") == 5,
        "anchor_label_count_is_five": len(label_paths) == 5,
        "triage_record_count_is_five": triage.get("record_count") == 5,
        "triage_has_no_failures": triage.get("failure_count") == 0,
        "handoff_record_count_is_five": handoff.get("record_count") == 5,
        "all_handoffs_have_visual_tasks": all(
            bool(item.get("required_visual_agent_tasks")) for item in handoff.get("records") or []
        ),
    }
    authority_checks = []
    for path in label_paths:
        label = read_json(path)
        authority_checks.append(
            label.get("string_polylines_pixel") is None
            and label.get("string_mask_polygons_pixel") is None
            and label.get("string_visibility") == "uncertain"
            and label.get("yoyo_bbox_pixel") is None
            and label.get("string_review_status") != "approved"
        )
    checks["weak_vlm_did_not_claim_visual_authority"] = all(authority_checks)
    report = {
        "schema_version": "five_video_scene_validation_v1",
        "created_at_utc": utc_now(),
        "python": sys.executable,
        "project": str(project),
        "videos": [str(path) for path in videos],
        "checks": checks,
        "ok": all(checks.values()),
    }
    report_path = output / "five_video_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
