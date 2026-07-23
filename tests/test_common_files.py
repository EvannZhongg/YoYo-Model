import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from common.files import collect_files, sha256_file


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


if __name__ == "__main__":
    unittest.main()
