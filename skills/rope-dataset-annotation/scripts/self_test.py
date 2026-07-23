#!/usr/bin/env python3
"""End-to-end smoke test for rope_pipeline.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT = Path(__file__).with_name("rope_pipeline.py")


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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rope-skill-test-") as temp:
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
        run("init", "--images", str(images), "--output", str(project), "--split", "train")
        label = project / "labels" / "train" / "video-a" / "frame_0001.json"
        candidate = root / "candidate.json"
        candidate.write_text(
            json.dumps(
                {
                    "visibility": "visible",
                    "yoyo_bbox_pixel": [300, 245, 340, 285],
                    "string_visibility": "visible",
                    "string_polylines_pixel": [[[140, 60], [210, 105], [260, 180], [320, 245]]],
                    "hands_pixel": {"left": [140, 60], "right": None},
                    "string_attachment_class": "hand_and_yoyo_attached",
                    "scene_label": "trick",
                    "string_path": {
                        "topology": "open",
                        "reconstruction_status": "complete",
                        "paths": [
                            {
                                "path_id": "hand-to-yoyo",
                                "start_anchor": "left_hand",
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
                    "notes": "Synthetic rope is fully visible.",
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
            "--notes",
            "Centerline points follow the visible rope pixels.",
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
            "--notes",
            "Visibility, anchors, and full path agree with the image.",
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
                        "visibility",
                        "yoyo_bbox_pixel",
                        "string_visibility",
                        "string_polylines_pixel",
                        "string_mask_polygons_pixel",
                        "hands_pixel",
                        "string_attachment_class",
                        "scene_label",
                        "string_path",
                        "bad_case",
                        "notes",
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
            "--notes",
            "This must be rejected because the ordered path is absent.",
            expected=2,
        )
        assert "requires an ordered string_path" in rejected_review.stderr

        invalid_anchor = root / "invalid-anchor.json"
        invalid_anchor_payload = json.loads(candidate.read_text(encoding="utf-8"))
        invalid_anchor_payload["string_path"]["paths"][0]["start_anchor"] = "hand"
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
            "--notes",
            "Generic hand anchors must be rejected.",
            expected=2,
        )
        assert "start_anchor=hand is unsupported" in invalid_anchor_review.stderr

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
        run("init", "--images", str(v_images), "--output", str(v_project), "--split", "train")
        v_label = v_project / "labels" / "train" / "video-v" / "frame_0001.json"
        v_polygon = [[96, 75], [320, 279], [544, 75], [549, 84], [320, 291], [91, 84]]
        v_candidate = root / "v-candidate.json"
        v_candidate.write_text(
            json.dumps(
                {
                    "visibility": "visible",
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
                    "visibility": "visible",
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
            "--notes",
            "The nonexistent top edge must fail mask support.",
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
        run("init", "--images", str(sequence), "--output", str(temporal_project), "--split", "train")
        previous_label = temporal_project / "labels" / "train" / "sequence" / "frame_0001.json"
        target_label = temporal_project / "labels" / "train" / "sequence" / "frame_0002.json"
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

    print("rope pipeline self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
