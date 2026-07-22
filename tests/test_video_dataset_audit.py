import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from video_dataset.action_group import set_action_group
from video_dataset.audit import audit


class VideoDatasetAuditTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        source = root / "source.mp4"
        source.write_bytes(b"test-video")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = dataset / "frames" / "train" / "video-a" / "frame_00000000.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"frame")
        (dataset / "sources.json").write_text(
            json.dumps({"sources": [{"video_id": "video-a", "source_group": "group-a", "split": "train", "path": str(source), "sha256": digest}]}),
            encoding="utf-8",
        )
        record = {"video_id": "video-a", "source_group": "group-a", "split": "train", "frame_index": 0, "frame_path": str(frame), "source_video_sha256": digest}
        (dataset / "frames.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        label = dataset / "annotations" / "labels" / "train" / "video-a" / "frame_00000000.json"
        label.parent.mkdir(parents=True)
        label.write_text(
            json.dumps({
                **record,
                "source_image": str(frame),
                "bbox_review_status": "approved",
                "string_review_status": "reviewed",
                "visibility": "absent",
                "bbox": [],
                "string_visibility": "not_visible",
            }),
            encoding="utf-8",
        )
        visualization = dataset / "annotations" / "visualizations" / "train" / "video-a" / "frame_00000000_vis.jpg"
        visualization.parent.mkdir(parents=True)
        visualization.write_bytes(b"preview")
        return dataset

    def test_valid_dataset_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit(self._dataset(Path(tmp)))
        self.assertTrue(report["ok"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["labels"]["bbox_accepted_by_split"]["train"], 1)

    def test_detects_split_and_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            label = next((dataset / "annotations" / "labels").rglob("*.json"))
            data = json.loads(label.read_text(encoding="utf-8"))
            data["split"] = "test"
            data["source_video_sha256"] = "stale"
            label.write_text(json.dumps(data), encoding="utf-8")
            report = audit(dataset)
        kinds = {item["kind"] for item in report["errors"]}
        self.assertIn("label_split_mismatch", kinds)
        self.assertIn("label_source_sha256_mismatch", kinds)
        self.assertFalse(report["ok"])

    def test_detects_accepted_annotation_that_fails_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            label = next((dataset / "annotations" / "labels").rglob("*.json"))
            data = json.loads(label.read_text(encoding="utf-8"))
            data["visibility"] = "visible"
            label.write_text(json.dumps(data), encoding="utf-8")
            report = audit(dataset)
        kinds = {item["kind"] for item in report["errors"]}
        self.assertIn("accepted_bbox_failed_review_gate", kinds)

    def test_action_group_migration_is_explicit_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            dry_run = set_action_group(dataset, "1A")
            self.assertFalse(dry_run["apply"])
            self.assertEqual(dry_run["changes"]["labels"], 1)

            set_action_group(dataset, "1A", apply=True)
            report = audit(dataset)
            self.assertEqual(report["current_action_group"], "1A")
            self.assertTrue(report["ok"])

            label = next((dataset / "annotations" / "labels").rglob("*.json"))
            data = json.loads(label.read_text(encoding="utf-8"))
            data["action_group"] = "4A"
            label.write_text(json.dumps(data), encoding="utf-8")
            kinds = {item["kind"] for item in audit(dataset)["errors"]}
            self.assertIn("label_action_group_mismatch", kinds)


if __name__ == "__main__":
    unittest.main()
