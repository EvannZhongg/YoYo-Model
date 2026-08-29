#!/usr/bin/env python3
"""End-to-end smoke test for the standalone agent annotation tools."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT = Path(__file__).with_name("annotation_pipeline.py")
SAMPLER = Path(__file__).with_name("sample_video_frames.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_sampling_manifest(images_root: Path, output: Path) -> Path:
    records = []
    sources = {}
    for image_path in sorted(path for path in images_root.rglob("*") if path.is_file()):
        source_group = image_path.parent.name
        frame_match = re.search(r"(\d+)$", image_path.stem)
        frame_index = int(frame_match.group(1)) if frame_match else 0
        source_video = str((output.parent / "source-videos" / f"{source_group}.mp4").resolve())
        source_video_sha256 = hashlib.sha256(source_group.encode("utf-8")).hexdigest()
        sources[source_group] = {
            "source_video": source_video,
            "source_video_sha256": source_video_sha256,
            "source_group": source_group,
            "fps": 10.0,
            "frame_count": max(frame_index + 1, 2),
            "image_size": list(Image.open(image_path).size),
            "candidate_count": 1,
            "anchor_count": 1,
            "written_count": 1,
        }
        records.append(
            {
                "source_video": source_video,
                "source_video_sha256": source_video_sha256,
                "source_group": source_group,
                "sequence_id": f"seq-001-anchor-{frame_index:08d}",
                "role": "anchor",
                "anchor_frame_index": frame_index,
                "frame_index": frame_index,
                "timestamp_s": frame_index / 10.0,
                "image_size": list(Image.open(image_path).size),
                "output_image": str(image_path.resolve()),
                "output_image_sha256": file_sha256(image_path),
            }
        )
    payload = {
        "schema_version": "agent_video_sampling_v1",
        "source_count": len(sources),
        "record_count": len(records),
        "failure_count": 0,
        "sources": list(sources.values()),
        "records": records,
        "failures": [],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def run(*args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_sampler(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SAMPLER), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"sampler failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="agent-yoyo-string-skill-test-") as temp:
        root = Path(temp)
        images = root / "images" / "video-a"
        images.mkdir(parents=True)
        first = images / "frame_0001.png"
        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((300, 245, 340, 285), fill="#DD2222", outline="black", width=2)
        draw.line([(140, 60), (210, 105), (260, 180), (320, 245)], fill="black", width=3)
        image.save(first)

        project = root / "project"
        sampling_manifest_path = write_sampling_manifest(images, root / "sampling_manifest.json")
        run(
            "init",
            "--images",
            str(images),
            "--output",
            str(project),
            "--sampling-manifest",
            str(sampling_manifest_path),
        )
        label = project / "labels" / "video-a" / "frame_0001.json"
        initialized = json.loads(label.read_text(encoding="utf-8"))
        assert "hands_pixel" not in initialized
        assert "hands_2d" not in initialized
        candidate = root / "candidate.json"
        candidate.write_text(
            json.dumps(
                {
                    "visibility": "visible",
                    "yoyo_bbox_pixel": [300, 245, 340, 285],
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[140, 60], [210, 105], [260, 180], [320, 245]]],
                    "yoyo_division": "1A",
                    "scene_label": "trick",
                    "trick_orientation": "normal",
                    "string_path": {
                        "topology": "open",
                        "reconstruction_status": "complete",
                        "paths": [
                            {
                                "path_id": "visible-route-to-yoyo",
                                "start_anchor": "unknown",
                                "end_anchor": "yoyo",
                                "points_pixel": [[140, 60], [210, 105], [260, 180], [320, 245]],
                                "edges": [
                                    {"from": 0, "to": 1, "evidence": "observed", "confidence": 0.98},
                                    {"from": 1, "to": 2, "evidence": "observed", "confidence": 0.98},
                                    {"from": 2, "to": 3, "evidence": "observed", "confidence": 0.96},
                                ],
                            }
                        ],
                        "unresolved_gaps": [],
                    },
                    "bad_case": [],
                }
            ),
            encoding="utf-8",
        )
        run(
            "apply",
            "--label",
            str(label),
            "--candidate",
            str(candidate),
            "--actor",
            "model-annotator",
            "--role",
            "model-annotator",
            "--model",
            "self-test-model",
        )
        unknown_orientation = root / "unknown-orientation.json"
        unknown_orientation_payload = json.loads(candidate.read_text(encoding="utf-8"))
        unknown_orientation_payload["trick_orientation"] = "unknown"
        unknown_orientation.write_text(json.dumps(unknown_orientation_payload), encoding="utf-8")
        run("apply", "--label", str(label), "--candidate", str(unknown_orientation), "--actor", "orientation-gate-test")
        orientation_review = run(
            "review",
            "--label",
            str(label),
            "--decision",
            "approve",
            "--reviewer",
            "orientation-reviewer",
            "--role",
            "semantic-critic",
            expected=2,
        )
        assert "requires trick_orientation=normal or horizontal" in orientation_review.stderr
        run("apply", "--label", str(label), "--candidate", str(candidate), "--actor", "orientation-gate-fix")
        scaled_candidate = root / "scaled-candidate.json"
        scaled_candidate.write_text(
            json.dumps(
                {
                    "coordinate_frame": {"type": "rendered_image", "render_size": [320, 180]},
                    "visibility": "visible",
                    "yoyo_bbox_pixel": [150, 122.5, 170, 142.5],
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[70, 30], [105, 52.5], [130, 90], [160, 122.5]]],
                    "yoyo_division": "1A",
                    "scene_label": "trick",
                    "trick_orientation": "normal",
                    "string_path": {
                        "topology": "open",
                        "reconstruction_status": "complete",
                        "paths": [
                            {
                                "path_id": "scaled-visible-route-to-yoyo",
                                "start_anchor": "unknown",
                                "end_anchor": "yoyo",
                                "points_pixel": [[70, 30], [105, 52.5], [130, 90], [160, 122.5]],
                                "edges": [
                                    {"from": 0, "to": 1, "evidence": "observed", "confidence": 0.98},
                                    {"from": 1, "to": 2, "evidence": "observed", "confidence": 0.98},
                                    {"from": 2, "to": 3, "evidence": "observed", "confidence": 0.96},
                                ],
                            }
                        ],
                        "unresolved_gaps": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        run("apply", "--label", str(label), "--candidate", str(scaled_candidate), "--actor", "scale-gate-test")
        scaled_saved = json.loads(label.read_text(encoding="utf-8"))
        assert scaled_saved["string_polylines_pixel"][0][0] == [140.0, 60.0]
        assert scaled_saved["yoyo_bbox_pixel"] == [300.0, 245.0, 340.0, 285.0]
        patch = root / "point-patch.json"
        patch.write_text(
            json.dumps(
                {
                    "coordinate_frame": "original_pixel",
                    "stroke_ops": [{"op": "move_point", "stroke": 0, "point": 1, "value": [212, 106]}],
                }
            ),
            encoding="utf-8",
        )
        run(
            "apply-patch",
            "--label",
            str(label),
            "--patch",
            str(patch),
            "--actor",
            "patch-test",
            "--model",
            "self-test-model",
        )
        patched_saved = json.loads(label.read_text(encoding="utf-8"))
        assert patched_saved["string_polylines_pixel"][0][1] == [212.0, 106.0]
        run("apply", "--label", str(label), "--candidate", str(candidate), "--actor", "patch-reset")
        review_dir = root / "review"
        run("render", "--label", str(label), "--output", str(review_dir))
        assert (review_dir / "frame_0001_grid.jpg").exists()
        assert (review_dir / "frame_0001_overlay.jpg").exists()
        assert (review_dir / "frame_0001_detail.jpg").exists()
        assert (review_dir / "frame_0001_render.json").exists()

        pre_audit = run("audit", "--labels", str(project / "labels"), "--require-approved", "--strict", expected=1)
        assert '"ok": false' in pre_audit.stdout.lower()
        run(
            "review",
            "--label",
            str(label),
            "--decision",
            "approve",
            "--reviewer",
            "geometry-model-pass",
            "--role",
            "geometry-critic",
            "--model",
            "self-test-model",
        )
        run(
            "review",
            "--label",
            str(label),
            "--decision",
            "approve",
            "--reviewer",
            "semantic-model-pass",
            "--role",
            "semantic-critic",
            "--model",
            "self-test-model",
        )
        run("audit", "--labels", str(project / "labels"), "--require-approved", "--strict")
        run("render", "--label", str(label), "--output", str(review_dir))
        final_render = json.loads((review_dir / "frame_0001_render.json").read_text(encoding="utf-8"))
        assert final_render["string_review_status"] == "approved"
        export = root / "export"
        run("export", "--labels", str(project / "labels"), "--output", str(export))
        manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["exported_count"] == 1
        assert manifest["excluded_count"] == 0
        exported_item = manifest["exported"][0]
        assert Path(exported_item["image"]).is_file()
        assert Path(exported_item["visualization"]).is_file()
        with Image.open(exported_item["image"]) as exported_image, Image.open(exported_item["visualization"]) as visualization:
            assert visualization.size == exported_image.size
        assert exported_item["visualization_sha256"]
        assert exported_item["trick_orientation"] == "normal"
        assert exported_item["source_group"] == "video-a"
        assert exported_item["video_id"] == "video-a"
        assert exported_item["frame_index"] == 1
        assert exported_item["timestamp_s"] == 0.1
        assert exported_item["source_video_sha256"]
        assert manifest["source_count"] == 1
        assert manifest["trick_orientation_counts"] == {"normal": 1}
        run("audit", "--labels", str(export / "labels"), "--require-approved", "--strict")
        portable_path = next((export / "labels" / "video-a").rglob("*.json"))
        portable_label = json.loads(portable_path.read_text(encoding="utf-8"))
        assert "split" not in portable_label
        assert not Path(portable_label["source_image"]).is_absolute()
        assert not Path(portable_label["visualization"]).is_absolute()
        assert portable_label["source_video"] == exported_item["source_video"]
        assert portable_label["source_video_sha256"] == exported_item["source_video_sha256"]
        run("refresh-visualizations", "--export", str(export))
        refreshed_manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
        assert refreshed_manifest["visualizations_refreshed_at_utc"]

        bad_candidate = root / "bad_candidate.json"
        bad_candidate.write_text(
            json.dumps({"string_visibility": "not_visible", "string_polylines_pixel": [[[0, 0], [10, 10]]]}),
            encoding="utf-8",
        )
        # Normalization must clear stale visible geometry for a reviewed negative.
        run(
            "apply",
            "--label",
            str(label),
            "--candidate",
            str(bad_candidate),
            "--actor",
            "negative-test",
        )
        saved = json.loads(label.read_text(encoding="utf-8"))
        assert saved["string_polylines_pixel"] is None
        assert saved["string_review_status"] == "auto_labeled_needs_review"
        assert not [
            review
            for review in saved["quality"]["reviews"]
            if review["content_sha256"]
            == __import__("hashlib").sha256(
                json.dumps(
                    {key: saved.get(key) for key in (
                        "image_sha256",
                        "image_size",
                        "source_group",
                        "visibility",
                        "yoyo_bbox_pixel",
                        "string_visibility",
                        "string_polylines_pixel",
                        "string_mask_polygons_pixel",
                        "yoyo_division",
                        "scene_label",
                        "trick_orientation",
                        "string_path",
                        "bad_case",
                    )},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ]

        missing_path = root / "missing_path.json"
        missing_path.write_text(
            json.dumps(
                {
                    "visibility": "visible",
                    "yoyo_bbox_pixel": [300, 245, 340, 285],
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[140, 60], [320, 245]]],
                    "string_path": {"topology": "uncertain", "reconstruction_status": "uncertain", "paths": []},
                }
            ),
            encoding="utf-8",
        )
        run("apply", "--label", str(label), "--candidate", str(missing_path), "--actor", "path-gate-test")
        rejected_review = run(
            "review",
            "--label",
            str(label),
            "--decision",
            "approve",
            "--reviewer",
            "path-gate-reviewer",
            "--role",
            "geometry-critic",
            expected=2,
        )
        assert "requires an ordered string_path" in rejected_review.stderr

        invalid_anchor = root / "invalid-anchor.json"
        invalid_anchor_payload = json.loads(candidate.read_text(encoding="utf-8"))
        invalid_anchor_payload["string_path"]["paths"][0]["start_anchor"] = "left_hand"
        invalid_anchor.write_text(json.dumps(invalid_anchor_payload), encoding="utf-8")
        run("apply", "--label", str(label), "--candidate", str(invalid_anchor), "--actor", "anchor-gate-test")
        invalid_anchor_review = run(
            "review",
            "--label",
            str(label),
            "--decision",
            "approve",
            "--reviewer",
            "anchor-gate-reviewer",
            "--role",
            "semantic-critic",
            expected=2,
        )
        assert "start_anchor=left_hand is unsupported" in invalid_anchor_review.stderr
        run(
            "review",
            "--label",
            str(label),
            "--decision",
            "unresolved",
            "--reviewer",
            "unresolved-test",
            "--role",
            "geometry-critic",
        )
        unresolved = json.loads(label.read_text(encoding="utf-8"))
        assert unresolved["string_review_status"] == "unresolved"

        # Regression: a thin V-shaped mask must follow both connected arms.
        # Joining the two far endpoints would invent a nonexistent top edge.
        v_images = root / "v-images" / "video-v"
        v_images.mkdir(parents=True)
        v_image = v_images / "frame_0001.png"
        v_frame = Image.new("RGB", (640, 360), "#202020")
        v_draw = ImageDraw.Draw(v_frame)
        v_draw.line([(100, 80), (320, 285), (540, 80)], fill="#DDF45A", width=8, joint="curve")
        v_frame.save(v_image)
        v_project = root / "v-project"
        v_manifest = write_sampling_manifest(v_images, root / "v-sampling-manifest.json")
        run(
            "init",
            "--images",
            str(v_images),
            "--output",
            str(v_project),
            "--sampling-manifest",
            str(v_manifest),
        )
        v_label = v_project / "labels" / "video-v" / "frame_0001.json"
        v_polygon = [[96, 75], [320, 279], [544, 75], [549, 84], [320, 291], [91, 84]]
        v_candidate = root / "v-candidate.json"
        v_candidate.write_text(
            json.dumps(
                {
                    "visibility": "uncertain",
                    "string_visibility": "visible",
                    "string_mask_polygons_pixel": [v_polygon],
                    "string_path": {
                        "topology": "uncertain",
                        "reconstruction_status": "uncertain",
                        "paths": [],
                        "unresolved_gaps": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        run("apply", "--label", str(v_label), "--candidate", str(v_candidate), "--actor", "mask-seed")
        derive = run("derive-centerlines", "--label", str(v_label), "--actor", "mask-skeleton-test")
        assert '"mask_support_fraction": 1.0' in derive.stdout
        derived_v = json.loads(v_label.read_text(encoding="utf-8"))
        derived_points = [point for stroke in derived_v["string_polylines_pixel"] for point in stroke]
        assert min(((point[0] - 320) ** 2 + (point[1] - 285) ** 2) ** 0.5 for point in derived_points) < 18
        run("audit", "--labels", str(v_project / "labels"), "--strict")

        shortcut_candidate = root / "v-shortcut.json"
        shortcut_candidate.write_text(
            json.dumps(
                {
                    "visibility": "uncertain",
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[100, 80], [540, 80]]],
                    "string_mask_polygons_pixel": [v_polygon],
                    "string_path": {
                        "topology": "open",
                        "reconstruction_status": "partial",
                        "paths": [
                            {
                                "path_id": "invalid-shortcut",
                        "start_anchor": "unknown",
                        "end_anchor": "unknown",
                                "points_pixel": [[100, 80], [540, 80]],
                                "edges": [{"from": 0, "to": 1, "evidence": "observed", "confidence": 0.9}],
                            }
                        ],
                        "unresolved_gaps": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        run("apply", "--label", str(v_label), "--candidate", str(shortcut_candidate), "--actor", "shortcut-test")
        shortcut_review = run(
            "review",
            "--label",
            str(v_label),
            "--decision",
            "approve",
            "--reviewer",
            "shortcut-critic",
            "--role",
            "geometry-critic",
            expected=2,
        )
        assert "support from mask geometry" in shortcut_review.stderr

        # Verify that a consecutive frame is seeded, not approved, and that the
        # reconstructed path carries temporal evidence after optical flow.
        sequence = root / "sequence"
        sequence.mkdir()
        for frame_number, shift in ((1, 0), (2, 5)):
            frame = Image.new("RGB", (640, 360), "white")
            frame_draw = ImageDraw.Draw(frame)
            frame_draw.ellipse((300 + shift, 245, 340 + shift, 285), fill="#DD2222", outline="black", width=2)
            frame_draw.line(
                [(140 + shift, 60), (210 + shift, 105), (260 + shift, 180), (320 + shift, 245)],
                fill="black",
                width=3,
            )
            frame.save(sequence / f"frame_{frame_number:04d}.png")
        temporal_project = root / "temporal-project"
        temporal_manifest = write_sampling_manifest(sequence, root / "temporal-sampling-manifest.json")
        run(
            "init",
            "--images",
            str(sequence),
            "--output",
            str(temporal_project),
            "--sampling-manifest",
            str(temporal_manifest),
        )
        previous_label = temporal_project / "labels" / "sequence" / "frame_0001.json"
        target_label = temporal_project / "labels" / "sequence" / "frame_0002.json"
        run(
            "apply",
            "--label",
            str(previous_label),
            "--candidate",
            str(candidate),
            "--actor",
            "model-annotator",
        )
        run(
            "propagate",
            "--previous-label",
            str(previous_label),
            "--target-label",
            str(target_label),
            "--actor",
            "temporal-self-test",
            "--model",
            "optical-flow-test",
        )
        propagated = json.loads(target_label.read_text(encoding="utf-8"))
        assert propagated["string_review_status"] == "auto_labeled_needs_review"
        assert propagated["temporal_seed"]["requires_current_frame_model_review"] is True
        assert propagated["temporal_seed"]["tracked_fraction"] > 0.5
        assert propagated["string_polylines_pixel"]
        evidences = {
            edge["evidence"]
            for path_item in propagated["string_path"]["paths"]
            for edge in path_item["edges"]
        }
        assert "temporal" in evidences

        video_root = root / "videos"
        video_root.mkdir()
        video_path = video_root / "synthetic.avi"
        if cv2 is None:
            raise RuntimeError("self-test video sampling path requires OpenCV in the active Python environment")
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 180))
        if not writer.isOpened():
            raise RuntimeError("could not create synthetic sampler video")
        for index in range(48):
            frame = np.full((180, 320, 3), 245, dtype=np.uint8)
            x = 20 + index * 5
            cv2.line(frame, (20, 20 + index % 30), (min(300, x), 130), (20, 20, 20), 2)
            cv2.circle(frame, (min(300, x), 130), 9, (210, 30, 30), -1)
            writer.write(frame)
        writer.release()
        sampling_output = root / "sampling"
        run_sampler(
            "--videos",
            str(video_root),
            "--output",
            str(sampling_output),
            "--total-anchors",
            "4",
            "--oversample-factor",
            "3",
            "--neighbor-offsets=-1,1",
        )
        sampling_manifest = json.loads((sampling_output / "sampling_manifest.json").read_text(encoding="utf-8"))
        assert sampling_manifest["recognition_model_used"] is False
        assert sampling_manifest["parameters"]["decode_strategy"] == "random_seek_requested_frames_only"
        assert sampling_manifest["parameters"]["total_anchors"] == 4
        assert sampling_manifest["parameters"]["separate_context"] is False
        assert sampling_manifest["source_hash_cache"]["computed_count"] == 1
        assert sampling_manifest["source_count"] == 1
        assert all("split" not in item for item in sampling_manifest["records"])
        anchors = [item for item in sampling_manifest["records"] if item["role"] == "anchor"]
        assert len(anchors) == 4
        assert len({item["sequence_id"] for item in anchors}) == 4
        assert max(item["frame_index"] for item in anchors) - min(item["frame_index"] for item in anchors) > 20
        assert (sampling_output / "anchor_contact_sheet.jpg").is_file()
        assert (sampling_output / "anchor_contact_sheet_large.jpg").is_file()
        assert (sampling_output / "anchor_center_crop_sheet.jpg").is_file()
        assert (sampling_output / "source_hash_cache.json").is_file()
        sampled_project = root / "sampled-project"
        run("init", "--images", str(sampling_output / "images"), "--output", str(sampled_project))
        sampled_labels = sorted((sampled_project / "labels").rglob("*.json"))
        assert len(sampled_labels) == sampling_manifest["record_count"]
        sampled_label = json.loads(sampled_labels[0].read_text(encoding="utf-8"))
        sampled_record = next(
            item
            for item in sampling_manifest["records"]
            if item["output_image_sha256"] == sampled_label["image_sha256"]
        )
        for field in ("source_video", "source_video_sha256", "source_group", "frame_index", "timestamp_s"):
            assert sampled_label[field] == sampled_record[field]
        tampered_image = Path(sampling_manifest["records"][0]["output_image"])
        with tampered_image.open("ab") as handle:
            handle.write(b"tampered-after-sampling")
        tampered_init = run(
            "init",
            "--images",
            str(sampling_output / "images"),
            "--output",
            str(root / "tampered-project"),
            expected=2,
        )
        assert "sampling record image hash does not match" in tampered_init.stderr
        retained_audit = root / "final_label_audit.json"
        retained_audit.write_text(json.dumps({"ok": True}), encoding="utf-8")
        cleanup_preview = run("cleanup", "--project", str(root))
        cleanup_report = json.loads(cleanup_preview.stdout)
        assert cleanup_report["dry_run"] is True
        assert any(item.endswith("candidate.json") for item in cleanup_report["candidates"])
        assert any("review" in item for item in cleanup_report["candidates"])
        assert str(retained_audit) not in cleanup_report["candidates"]
        assert str(Path(exported_item["visualization"])) not in cleanup_report["candidates"]
        run("cleanup", "--project", str(root), "--confirm")
        assert not review_dir.exists()
        assert not candidate.exists()
        assert retained_audit.is_file()
        assert Path(exported_item["image"]).is_file()
        assert Path(exported_item["visualization"]).is_file()

    print("agent yoyo string annotation self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
