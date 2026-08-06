"""Self-tests for weak-VLM triage boundaries and Unicode media I/O."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import annotation_pipeline as annotation
import unicode_media
import vlm_triage


def settings() -> vlm_triage.Settings:
    return vlm_triage.Settings(
        base_url="https://example.invalid/v1",
        api_key=None,
        model="replay-model",
        min_pixels=32 * 32,
        max_pixels=1024 * 32 * 32,
        max_response_tokens=1000,
        enable_thinking=False,
        timeout_seconds=10,
        retries=0,
        promotion_confidence=0.9,
        quick_verify_confidence=0.95,
        notes_confidence=0.7,
        safe_bad_cases=("motion_blur", "low_contrast", "edge_clipped"),
    )


def draft_label(image_path: Path) -> dict:
    return {
        "schema_version": annotation.SCHEMA_VERSION,
        "source_image": str(image_path),
        "image_sha256": annotation.sha256_file(image_path),
        "image_size": [32, 24],
        "source_video": "source.mp4",
        "source_video_sha256": "1" * 64,
        "source_group": "source-group",
        "video_id": "source-group",
        "frame_index": 10,
        "timestamp_s": 1.0,
        "sequence_id": "seq-001",
        "sampling_role": "anchor",
        "anchor_frame_index": 10,
        "sampling_manifest_sha256": "2" * 64,
        "visibility": "uncertain",
        "yoyo_bbox_pixel": None,
        "yoyo_bbox_2d": None,
        "bbox": [],
        "string_visibility": "uncertain",
        "string_polylines_pixel": None,
        "string_polylines_2d": None,
        "string_polyline_pixel": None,
        "string_polyline_2d": None,
        "string_mask_polygons_pixel": None,
        "yoyo_division": "1A",
        "scene_label": "unknown",
        "trick_orientation": "unknown",
        "string_path": {
            "topology": "uncertain",
            "reconstruction_status": "uncertain",
            "paths": [],
            "unresolved_gaps": [],
        },
        "bad_case": [],
        "notes": "",
        "string_review_status": "auto_labeled_needs_review",
        "review_status": "partially_reviewed",
        "bbox_review_status": "auto_labeled_needs_review",
        "reviewed_at_utc": None,
        "reviewer": None,
        "quality": {"revision": 0, "min_model_approvals": 2, "history": [], "reviews": []},
    }


class UnicodeMediaTests(unittest.TestCase):
    def test_unicode_image_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "中文目录" / "帧图像.png"
            image = np.zeros((18, 24, 3), dtype=np.uint8)
            image[:, :, 1] = 173
            unicode_media.imwrite(target, image)
            decoded = unicode_media.imread(target)
            self.assertEqual(decoded.shape, image.shape)
            self.assertTrue(np.array_equal(decoded, image))


class TriageBoundaryTests(unittest.TestCase):
    def test_default_env_file_is_skill_local(self) -> None:
        self.assertEqual(vlm_triage.resolve_env_file(None), vlm_triage.SKILL_ROOT / ".env")
        explicit = Path("explicit.env")
        self.assertEqual(vlm_triage.resolve_env_file(explicit), explicit.resolve())

    def test_prohibited_geometry_is_discarded(self) -> None:
        raw = {
            "domain_status": "in_domain",
            "scene_label": "trick",
            "scene_is_obvious": True,
            "obvious_yoyo_presence": "present",
            "coarse_string_evidence": "obvious",
            "frame_usability": "usable",
            "obvious_bad_cases": ["motion_blur"],
            "notes": "Obvious yoyo performance frame.",
            "string_polylines_pixel": [[[1, 2], [3, 4]]],
            "yoyo_bbox": [1, 2, 3, 4],
            "hands_pixel": {"left": [1, 2], "right": None},
            "confidence": {"domain": 0.99, "scene": 0.99, "yoyo_presence": 0.9, "bad_cases": 0.95, "overall": 0.95},
        }
        assessment, warnings = vlm_triage.normalize_assessment(raw)
        self.assertNotIn("string_polylines_pixel", assessment)
        self.assertNotIn("yoyo_bbox", assessment)
        self.assertNotIn("hands_pixel", assessment)
        self.assertTrue(any("prohibited" in item for item in warnings))

    def test_only_safe_fields_are_promoted(self) -> None:
        assessment, _ = vlm_triage.normalize_assessment(
            {
                "domain_status": "in_domain",
                "scene_label": "trick",
                "scene_is_obvious": True,
                "obvious_yoyo_presence": "present",
                "coarse_string_evidence": "obvious",
                "frame_usability": "usable",
                "priority_suggestion": "clear_candidate",
                "obvious_bad_cases": ["motion_blur", "severe_occlusion"],
                "notes": "Obvious motion blur across the frame.",
                "confidence": {"domain": 0.99, "scene": 0.98, "bad_cases": 0.96, "priority": 0.95, "overall": 0.92},
            }
        )
        promoted = vlm_triage.compute_promotions(assessment, settings())
        self.assertEqual(promoted["scene_label"], "trick")
        self.assertEqual(promoted["bad_case"], ["motion_blur"])
        self.assertTrue(promoted["notes"].startswith("Weak-VLM API-resolution observation, not string truth:"))
        self.assertNotIn("string_visibility", promoted)
        self.assertNotIn("trick_orientation", promoted)

    def test_apply_preserves_visual_authority_fields(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            image_path = root_path / "中文图像" / "帧.jpg"
            image_path.parent.mkdir(parents=True)
            unicode_media.imwrite(image_path, np.zeros((24, 32, 3), dtype=np.uint8))
            label_path = root_path / "labels" / "frame.json"
            label_path.parent.mkdir(parents=True)
            label_path.write_text(json.dumps(draft_label(image_path)), encoding="utf-8")
            label = vlm_triage.read_json(label_path)
            updated, status = vlm_triage.apply_safe_promotions(
                label_path,
                label,
                {"scene_label": "trick", "bad_case": ["low_contrast"], "notes": "Weak-VLM triage: low contrast."},
                "replay-model",
                "a" * 64,
            )
            self.assertEqual(status, "applied")
            self.assertEqual(updated["scene_label"], "trick")
            self.assertEqual(updated["string_visibility"], "uncertain")
            self.assertEqual(updated["trick_orientation"], "unknown")
            self.assertIsNone(updated["string_polylines_pixel"])
            self.assertEqual(updated["quality"]["history"][-1]["role"], "weak-vlm-triager")

    def test_quick_verify_never_auto_rejects(self) -> None:
        assessment, _ = vlm_triage.normalize_assessment(
            {
                "domain_status": "out_of_domain",
                "scene_label": "unknown",
                "scene_is_obvious": False,
                "obvious_yoyo_presence": "absent",
                "coarse_string_evidence": "none_obvious",
                "frame_usability": "usable",
                "obvious_bad_cases": [],
                "notes": "No obvious yoyo activity.",
                "confidence": {"domain": 0.99, "overall": 0.95},
            }
        )
        handoff = vlm_triage.compute_handoff(assessment, {}, settings())
        self.assertEqual(handoff["queue"], "quick_verify")
        self.assertTrue(any("confirm" in item.lower() for item in handoff["required_visual_agent_tasks"]))

    def test_clear_priority_is_gated_by_obvious_yoyo(self) -> None:
        assessment, _ = vlm_triage.normalize_assessment(
            {
                "domain_status": "in_domain",
                "scene_label": "non_trick",
                "scene_is_obvious": True,
                "obvious_yoyo_presence": "absent",
                "coarse_string_evidence": "none_obvious",
                "frame_usability": "usable",
                "priority_suggestion": "clear_candidate",
                "obvious_bad_cases": [],
                "notes": "No obvious yoyo at API resolution.",
                "confidence": {"domain": 0.99, "scene": 0.98, "priority": 0.95, "overall": 0.9},
            }
        )
        handoff = vlm_triage.compute_handoff(assessment, {}, settings())
        self.assertEqual(handoff["queue"], "standard")


if __name__ == "__main__":
    unittest.main(verbosity=2)
