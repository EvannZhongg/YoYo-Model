import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import atomic_write_text, collect_files, sha256_file
from cli.models.model_registry import _metric_summary


class CommonFileTests(unittest.TestCase):
    def test_collect_files_filters_extensions_and_respects_recursion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "first.JPG").write_bytes(b"first")
            (root / "ignored.txt").write_bytes(b"ignored")
            (nested / "second.png").write_bytes(b"second")

            shallow = collect_files(root, {".jpg", ".png"}, recursive=False)
            recursive = collect_files(root, {".jpg", ".png"})

        self.assertEqual([path.name for path in shallow], ["first.JPG"])
        self.assertEqual([path.name for path in recursive], ["first.JPG", "second.png"])

    def test_collect_files_rejects_missing_root(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                collect_files(missing, {".jpg"})

    def test_sha256_file_streams_the_expected_digest(self):
        payload = b"yoyo-model" * 1024
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(payload)
            digest = sha256_file(path, chunk_size=17)

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_atomic_write_text_replaces_file_and_creates_parent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.txt"
            atomic_write_text(path, "first\n")
            atomic_write_text(path, "second\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")
            if sys.platform == "win32":
                result = subprocess.run(
                    ["icacls", str(path)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                self.assertIn("(I)", result.stdout, f"atomic replacement lost inherited ACL:\n{result.stdout}")


class ModelRegistryMetricTests(unittest.TestCase):
    def test_summarizes_ultralytics_detection_and_segmentation_metrics(self):
        summary = _metric_summary(
            {
                "split": "test",
                "metrics": {
                    "metrics/mAP50(B)": 0.8,
                    "metrics/mAP50-95(B)": 0.5,
                    "metrics/mAP50(M)": 0.7,
                    "metrics/mAP50-95(M)": 0.4,
                },
            }
        )
        self.assertEqual(summary["map50"], 0.8)
        self.assertEqual(summary["mask_map50_95"], 0.4)
        self.assertEqual(summary["split"], "test")

    def test_summarizes_ultralytics_classification_metrics(self):
        summary = _metric_summary(
            {
                "metrics": {
                    "metrics/accuracy_top1": 0.75,
                    "metrics/accuracy_top5": 1.0,
                    "macro_recall": 0.8,
                    "per_class_recall": {"horizontal": 0.8, "normal": 0.6},
                }
            }
        )
        self.assertEqual(summary["top1_accuracy"], 0.75)
        self.assertEqual(summary["top5_accuracy"], 1.0)
        self.assertEqual(summary["macro_recall"], 0.8)
        self.assertEqual(summary["per_class_recall"]["horizontal"], 0.8)


if __name__ == "__main__":
    unittest.main()
