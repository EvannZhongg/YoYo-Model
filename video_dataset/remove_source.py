"""Remove one polluted video and its directly derived dataset artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _same_source(value: Any, source: dict[str, Any], source_id: str) -> bool:
    text = str(value or "")
    return text in {
        source_id,
        str(source.get("path", "")),
        str(source.get("filename", "")),
        str(source.get("sha256", "")),
    }


def remove_source(
    dataset_dir: Path,
    video: Path,
    archive_root: Path | None = None,
    delete_video: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    dataset_dir = _resolve(dataset_dir)
    video = _resolve(video)
    sources_path = dataset_dir / "sources.json"
    if not sources_path.exists():
        raise FileNotFoundError(f"Source manifest not found: {sources_path}")
    manifest = json.loads(sources_path.read_text(encoding="utf-8"))
    sources = list(manifest.get("sources") or [])
    matches = [
        item for item in sources
        if _resolve(str(item.get("path", ""))) == video
        or str(item.get("filename", "")) == video.name
        or str(item.get("video_id", "")) == video.stem
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source match for {video}; found {len(matches)}")
    source = matches[0]
    source_id = str(source.get("video_id") or source.get("source_group") or "")
    split = str(source.get("split") or "")
    if not source_id or split not in {"train", "val", "test"}:
        raise ValueError(f"Source has invalid video_id/split: {source}")

    rows_path = dataset_dir / "frames.jsonl"
    rows = _read_jsonl(rows_path)
    kept_rows = [
        row for row in rows
        if not (
            str(row.get("video_id")) == source_id
            or str(row.get("source_group")) == source_id
            or _same_source(row.get("source_video"), source, source_id)
            or _same_source(row.get("source_video_sha256"), source, source_id)
        )
    ]
    auto_path = dataset_dir / "auto_annotations.jsonl"
    auto_rows = _read_jsonl(auto_path)
    kept_auto = [
        row for row in auto_rows
        if source_id not in str(row.get("label_path", ""))
        and source_id not in str(row.get("frame_path", ""))
        and not _same_source(row.get("source_video"), source, source_id)
    ]

    derived_dirs = [
        dataset_dir / "frames" / split / source_id,
        dataset_dir / "candidate_frames" / split / source_id,
        dataset_dir / "annotations" / "labels" / split / source_id,
        dataset_dir / "annotations" / "visualizations" / split / source_id,
        dataset_dir / "yolo" / "images" / split / source_id,
        dataset_dir / "yolo" / "labels" / split / source_id,
        dataset_dir / "string_seg" / "images" / split / source_id,
        dataset_dir / "string_seg" / "labels" / split / source_id,
    ]
    workspace_root = dataset_dir.parent.parent
    targets = [path for path in derived_dirs if path.exists()]
    if delete_video:
        targets.append(video)
    for target in targets:
        if not _under(target, workspace_root):
            raise ValueError(f"Refusing target outside workspace: {target}")

    archive_sources: list[Path] = []
    archive_jsonl: list[Path] = []
    if archive_root and archive_root.exists():
        for path in archive_root.rglob("sources.json"):
            try:
                archive = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if any(
                _same_source(item.get("path"), source, source_id)
                or _same_source(item.get("filename"), source, source_id)
                or _same_source(item.get("sha256"), source, source_id)
                for item in archive.get("sources", [])
            ):
                archive_sources.append(path)
        for path in archive_root.rglob("*.jsonl"):
            rows_archive = _read_jsonl(path)
            if any(
                _same_source(row.get("source_video"), source, source_id)
                or _same_source(row.get("source_video_sha256"), source, source_id)
                or str(row.get("video_id")) == source_id
                for row in rows_archive
            ):
                archive_jsonl.append(path)

    result = {
        "source": source,
        "source_id": source_id,
        "split": split,
        "frame_records_removed": len(rows) - len(kept_rows),
        "auto_annotation_records_removed": len(auto_rows) - len(kept_auto),
        "derived_targets": [str(path) for path in targets],
        "archive_sources": [str(path) for path in archive_sources],
        "archive_jsonl": [str(path) for path in archive_jsonl],
        "applied": apply,
    }
    if not apply:
        return result

    manifest["sources"] = [item for item in sources if item is not source]
    sources_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(rows_path, kept_rows)
    if auto_path.exists():
        _write_jsonl(auto_path, kept_auto)
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
    for path in archive_sources:
        archive = json.loads(path.read_text(encoding="utf-8"))
        archive["sources"] = [
            item for item in archive.get("sources", [])
            if not (
                _same_source(item.get("path"), source, source_id)
                or _same_source(item.get("filename"), source, source_id)
                or _same_source(item.get("sha256"), source, source_id)
            )
        ]
        path.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in archive_jsonl:
        rows_archive = _read_jsonl(path)
        rows_archive = [
            row for row in rows_archive
            if not (
                _same_source(row.get("source_video"), source, source_id)
                or _same_source(row.get("source_video_sha256"), source, source_id)
                or str(row.get("video_id")) == source_id
            )
        ]
        _write_jsonl(path, rows_archive)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove one polluted video and its derived artifacts.")
    parser.add_argument("video", help="Video path or filename present in sources.json")
    parser.add_argument("--dataset-dir", default="datasets/video_v1")
    parser.add_argument("--archive-root", default="archive")
    parser.add_argument("--delete-video", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Actually remove files; otherwise print a dry-run.")
    args = parser.parse_args()
    result = remove_source(
        Path(args.dataset_dir),
        Path(args.video),
        Path(args.archive_root) if args.archive_root else None,
        args.delete_video,
        args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
