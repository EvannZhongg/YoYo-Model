import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_tracking.frame_review import (
    append_tracking_frame_review,
    frame_record_digest,
    load_tracking_frame_selection,
    tracking_review_gallery_items,
)


class TrackingFrameReviewTests(unittest.TestCase):
    def _make_run(self, root: Path) -> tuple[Path, dict]:
        run_dir = root / "tracking_run"
        raw_dir = run_dir / "review_raw_frames"
        overlay_dir = run_dir / "review_frames"
        raw_dir.mkdir(parents=True)
        overlay_dir.mkdir(parents=True)
        raw = raw_dir / "frame_00000020.jpg"
        overlay = overlay_dir / "frame_00000020.jpg"
        raw.write_bytes(b"raw")
        overlay.write_bytes(b"overlay")
        record = {
            "frame_index": 20,
            "timestamp_s": 0.4,
            "yoyo": {"bbox": [10, 20, 30, 40], "confidence": 0.91},
            "string": {"method": "semantic", "confidence": 0.7},
            "bad_case": ["string_needs_review"],
        }
        frames_path = run_dir / "frames.jsonl"
        frames_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "1.2",
            "run_id": "run-123",
            "source_video_sha256": "source-digest",
            "outputs": {"frames_jsonl": str(frames_path.resolve())},
        }
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        index = [{
            "frame": record,
            "source_frame_index": 20,
            "raw_image": str(raw.resolve()),
            "overlay_image": str(overlay.resolve()),
        }]
        (run_dir / "tracking_review_index.json").write_text(json.dumps(index), encoding="utf-8")
        return frames_path, record

    def test_selection_exposes_authoritative_record_and_digest_binding(self):
        with TemporaryDirectory() as directory:
            frames_path, record = self._make_run(Path(directory))
            first = load_tracking_frame_selection(frames_path, 0)
            second = load_tracking_frame_selection(frames_path, 1)

        self.assertEqual(first["frame_record"], record)
        self.assertEqual(first["binding"]["run_id"], "run-123")
        self.assertEqual(first["binding"]["frame_index"], 20)
        self.assertEqual(first["binding"]["view"], "raw")
        self.assertEqual(second["binding"]["view"], "overlay")
        self.assertEqual(first["binding"]["frame_record_sha256"], frame_record_digest(record))
        self.assertEqual(len(first["binding"]["frame_record_sha256"]), 64)

    def test_reviews_append_history_without_mutating_tracking_truth(self):
        with TemporaryDirectory() as directory:
            frames_path, record = self._make_run(Path(directory))
            run_dir = frames_path.parent
            protected = [
                frames_path,
                run_dir / "run.json",
                run_dir / "tracking_review_index.json",
            ]
            before = {path: path.read_bytes() for path in protected}
            selection = load_tracking_frame_selection(frames_path, 1)

            output_path, first = append_tracking_frame_review(
                frames_path,
                selection["binding"],
                "incorrect",
                "reviewer-a",
                "String follows the sleeve edge.",
            )
            _, second = append_tracking_frame_review(
                frames_path,
                selection["binding"],
                "unresolved",
                "reviewer-b",
                "Raw pixels remain ambiguous.",
            )

            events = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            after = {path: path.read_bytes() for path in protected}

        self.assertEqual(before, after)
        self.assertEqual([event["decision"] for event in events], ["incorrect", "unresolved"])
        self.assertNotEqual(first["review_id"], second["review_id"])
        self.assertEqual(events[0]["frame_record"], record)
        self.assertEqual(events[0]["frame_record_sha256"], frame_record_digest(record))
        self.assertEqual(output_path.name, "tracking_frame_reviews.jsonl")

    def test_rejects_stale_or_tampered_binding(self):
        with TemporaryDirectory() as directory:
            frames_path, _ = self._make_run(Path(directory))
            selection = load_tracking_frame_selection(frames_path, 0)
            tampered = dict(selection["binding"])
            tampered["frame_record_sha256"] = hashlib.sha256(b"other").hexdigest()

            with self.assertRaisesRegex(ValueError, "stale"):
                append_tracking_frame_review(frames_path, tampered, "correct", "reviewer")
            with self.assertRaisesRegex(ValueError, "Reviewer is required"):
                append_tracking_frame_review(frames_path, selection["binding"], "correct", "")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                append_tracking_frame_review(frames_path, selection["binding"], "approved", "reviewer")

        self.assertFalse((frames_path.parent / "tracking_frame_reviews.jsonl").exists())

    def test_rejects_invalid_run_and_frame_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path, _ = self._make_run(root)
            outside = root / "outside.jpg"
            outside.write_bytes(b"outside")
            index_path = frames_path.parent / "tracking_review_index.json"
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            entries[0]["raw_image"] = str(outside.resolve())
            index_path.write_text(json.dumps(entries), encoding="utf-8")

            items = tracking_review_gallery_items(frames_path.parent)
            wrong_name = root / "other.jsonl"
            wrong_name.write_text(frames_path.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frames.jsonl"):
                load_tracking_frame_selection(wrong_name, 0)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["view"], "overlay")

    def test_rejects_review_index_record_mismatch(self):
        with TemporaryDirectory() as directory:
            frames_path, _ = self._make_run(Path(directory))
            index_path = frames_path.parent / "tracking_review_index.json"
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            entries[0]["frame"]["timestamp_s"] = 99.0
            index_path.write_text(json.dumps(entries), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_tracking_frame_selection(frames_path, 0)


if __name__ == "__main__":
    unittest.main()
