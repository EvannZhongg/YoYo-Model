"""Export reviewed consecutive labels through a running Workbench read API."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _component_call(url: str, component_id: int, function: str, data: Any) -> Any:
    endpoint = url.rstrip("/") + "/gradio_api/component_server/"
    payload = json.dumps(
        {
            "data": data,
            "component_id": int(component_id),
            "fn_name": function,
            "session_hash": "tracking-ground-truth-export",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Workbench component call failed: {function}") from exc


def export_snapshot(dataset_dir: str, output: str | Path, workbench_url: str, component_id: int) -> dict[str, Any]:
    opened = _component_call(
        workbench_url,
        component_id,
        "ui_open_annotation_dataset",
        {"dataset_path": dataset_dir},
    )
    frames = []
    for sample in opened.get("samples") or []:
        loaded = _component_call(
            workbench_url,
            component_id,
            "ui_load_annotation_sample",
            {"dataset_path": dataset_dir, "sample_key": sample["key"]},
        )
        frames.append(
            {
                "sample_key": loaded["key"],
                "label_path": loaded.get("label_path"),
                "reviewed": bool(loaded.get("reviewed")),
                "reviewed_at_utc": loaded.get("reviewed_at_utc"),
                "reviewer": loaded.get("reviewer"),
                "annotation": loaded["annotation"],
            }
        )
    result = {
        "schema_version": "yoyo_consecutive_ground_truth_snapshot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(opened.get("dataset_path") or dataset_dir),
        "source": "gradio_html_component_read_api",
        "frame_count": len(frames),
        "reviewed_count": sum(bool(item["reviewed"]) for item in frames),
        "frames": frames,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: result[key] for key in ("schema_version", "dataset_dir", "frame_count", "reviewed_count")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export consecutive labels via a running Workbench.")
    parser.add_argument("dataset_dir")
    parser.add_argument("output")
    parser.add_argument("--workbench-url", default="http://127.0.0.1:7866")
    parser.add_argument("--component-id", type=int, default=34)
    args = parser.parse_args()
    print(json.dumps(export_snapshot(args.dataset_dir, args.output, args.workbench_url, args.component_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

