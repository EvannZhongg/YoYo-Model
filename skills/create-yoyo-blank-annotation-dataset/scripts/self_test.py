#!/usr/bin/env python3
"""Deterministic end-to-end test for create_blank_dataset.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

import create_blank_dataset as generator

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
from workbench import dataset_annotation as workbench_annotation


def write_video(path: Path, frame_count: int = 40) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    if not writer.isOpened():
        raise RuntimeError("test video writer could not open")
    for index in range(frame_count):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 17) % 255
        frame[:, :, 1] = np.arange(96, dtype=np.uint8)[None, :]
        cv2.circle(frame, (10 + index * 2 % 80, 32), 6, (30, 240, 180), -1)
        writer.write(frame)
    writer.release()


def write_reference(dataset: Path, video: Path, frame_index: int) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("test reference frame could not be read")
    image_path = dataset / "canonical" / "images" / "reference" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(image_path, quality=96, subsampling=0)
    label_path = dataset / "canonical" / "labels" / "reference" / "frame.json"
    label_path.parent.mkdir(parents=True)
    label_path.write_text(json.dumps({
        "schema_version": generator.ANNOTATION_SCHEMA_VERSION,
        "source_video_sha256": generator.sha256_file(video),
        "frame_index": frame_index,
        "image_sha256": generator.sha256_file(image_path),
    }), encoding="utf-8")


def keys(dataset: Path) -> set[tuple[str, int]]:
    return {
        (str(value["source_video_sha256"]), int(value["frame_index"]))
        for path in (dataset / "canonical" / "labels").rglob("*.json")
        for value in [json.loads(path.read_text(encoding="utf-8"))]
    }


def assert_initial_defaults(dataset: Path) -> None:
    for path in (dataset / "canonical" / "labels").rglob("*.json"):
        label = json.loads(path.read_text(encoding="utf-8"))
        assert label["visibility"] == "visible"
        assert label["trick_orientation"] == "normal"
        assert label["string_visibility"] == "partial"
        assert label["yoyo_bbox_pixel"] is None
        assert label["yoyo_bbox_2d"] is None
        assert label["bbox"] == []
        assert label["string_polylines_pixel"] is None
        assert label["string_polylines_2d"] is None
        assert label["string_mask_polygons_pixel"] is None
        assert label["hands_pixel"] == {"left": None, "right": None}
        assert label["hands_2d"] == {"left": None, "right": None}
        assert label["string_path"]["paths"] == []
        assert label["string_path"]["unresolved_gaps"] == []


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="yoyo-blank-skill-test-") as raw:
        root = Path(raw)
        datasets = root / "datasets"
        videos = root / "videos"
        videos.mkdir()
        video = videos / "unicode-测试.avi"
        write_video(video)
        reference = datasets / "1Ayoyo_dataset"
        write_reference(reference, video, frame_index=20)
        cache = root / "hash_cache.json"
        common = [
            "--videos", str(videos), "--datasets-root", str(datasets),
            "--frames-per-video", "4",
            "--oversample-factor", "8", "--perceptual-hamming-threshold", "0",
            "--hash-cache", str(cache),
        ]
        assert generator.main([*common, "--dataset-name", "batch-one"]) == 0
        first = datasets / "batch-one"
        create_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        create_sampling = json.loads((first / "sampling_manifest.json").read_text(encoding="utf-8"))
        source_metadata = create_sampling["sources"][0]
        assert source_metadata["candidate_attempted"] <= 8
        assert source_metadata["candidate_attempted"] < source_metadata["candidate_budget"]
        assert create_manifest["source_frame_cache"]["video_seek_count"] <= 8
        cached_record = create_manifest["records"][0]
        cached_stats = {"hit_count": 0, "miss_count": 0, "invalid_count": 0, "video_seek_count": 0}
        cached_capture = cv2.VideoCapture(str(video))
        cached_candidate = generator.decode_frame_candidate(
            cached_capture,
            str(cached_record["source_video_sha256"]),
            int(cached_record["frame_index"]),
            tuple(source_metadata["image_size"]),
            generator.argparse.Namespace(
                jpeg_quality=96,
                frame_cache_root=cache.parent / "source_frame_jpeg_cache",
            ),
            cached_stats,
        )
        cached_capture.release()
        assert cached_candidate is not None
        assert cached_stats == {"hit_count": 1, "miss_count": 0, "invalid_count": 0, "video_seek_count": 0}
        reference_key = (generator.sha256_file(video), 20)
        assert len(keys(first)) == 4
        assert reference_key not in keys(first)
        assert_initial_defaults(first)
        workbench_annotation.DATASETS_DIR = datasets
        workbench_annotation.REVIEW_MAP_PATH = root / "workbench_state" / "dataset_review_status.json"
        opened = workbench_annotation.open_annotation_dataset(str(first))
        protected_key = opened["samples"][0]["key"]
        workbench_annotation.save_annotation_sample(str(first), protected_key, {
            "yoyo_visibility": "uncertain",
            "trick_orientation": "normal",
            "yoyo_bbox_pixel": None,
            "string_visibility": "not_visible",
            "string_polylines_pixel": [],
            "string_review_status": "reviewed",
            "bbox_review_status": "reviewed",
            "reviewer": "human-reviewer",
            "notes": "manual edit that incremental append must preserve",
        })
        workbench_annotation.set_annotation_sample_reviewed(
            str(first), protected_key, "human-reviewer", confirmed=True
        )
        edited_only_key = opened["samples"][1]["key"]
        workbench_annotation.save_annotation_sample(str(first), edited_only_key, {
            "yoyo_visibility": "uncertain",
            "trick_orientation": "horizontal",
            "yoyo_bbox_pixel": None,
            "string_visibility": "uncertain",
            "string_polylines_pixel": [],
            "string_review_status": "needs_review",
            "bbox_review_status": "needs_review",
            "reviewer": "human-editor",
            "notes": "manual edit without human verification must also be preserved",
        })
        protected_label = first / "canonical" / "labels" / Path(protected_key)
        edited_only_label = first / "canonical" / "labels" / Path(edited_only_key)
        protected_label_bytes = protected_label.read_bytes()
        edited_only_label_bytes = edited_only_label.read_bytes()
        review_map_bytes = workbench_annotation.REVIEW_MAP_PATH.read_bytes()
        old_keys = keys(first)
        assert generator.main([*common, "--dataset-name", "batch-one"]) == 1
        assert generator.main([*common, "--dataset-name", "batch-one", "--append"]) == 0
        assert protected_label.read_bytes() == protected_label_bytes
        assert edited_only_label.read_bytes() == edited_only_label_bytes
        assert workbench_annotation.REVIEW_MAP_PATH.read_bytes() == review_map_bytes
        assert len(keys(first)) == 8
        assert old_keys < keys(first)
        first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        assert first_manifest["sample_count"] == 8
        assert len(first_manifest["generation_runs"]) == 2
        assert (first / first_manifest["generation_runs"][1]["sampling_manifest"]).is_file()
        reopened = workbench_annotation.open_annotation_dataset(str(first))
        protected_summary = next(sample for sample in reopened["samples"] if sample["key"] == protected_key)
        assert protected_summary["reviewed"] is True

        protected_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        original_copy = generator.copy_file_exclusive
        copy_calls = 0

        def fail_during_publish(source: Path, destination: Path) -> None:
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 3:
                raise OSError("injected append publication failure")
            original_copy(source, destination)

        generator.copy_file_exclusive = fail_during_publish
        try:
            assert generator.main([*common, "--dataset-name", "batch-one", "--append"]) == 1
        finally:
            generator.copy_file_exclusive = original_copy
        after_failure = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        assert after_failure == protected_files
        assert workbench_annotation.REVIEW_MAP_PATH.read_bytes() == review_map_bytes

        assert generator.main([*common, "--dataset-name", "batch-two"]) == 0
        second = datasets / "batch-two"
        assert len(keys(second)) == 4
        assert reference_key not in keys(second)
        assert keys(first).isdisjoint(keys(second))
        assert_initial_defaults(second)
        for dataset, expected_count in ((first, 8), (second, 4)):
            manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["schema_version"] == generator.DATASET_SCHEMA_VERSION
            assert manifest["sample_count"] == expected_count
            labels = list((dataset / "canonical" / "labels").rglob("*.json"))
            assert len(labels) == expected_count
        discovered = workbench_annotation.list_annotation_datasets()
        assert {item["name"] for item in discovered} >= {"batch-one", "batch-two"}
        assert reopened["sample_count"] == 8
        assert reopened["error_count"] == 0
    print(json.dumps({"ok": True, "test": "create-yoyo-blank-annotation-dataset"}))


if __name__ == "__main__":
    run()
