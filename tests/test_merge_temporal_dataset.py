import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from helpers import make_consecutive_dataset
import importlib.util

_SCRIPT = Path(__file__).parents[1] / "skills" / "merge-yoyo-temporal-dataset" / "scripts" / "merge_temporal_dataset.py"
_SPEC = importlib.util.spec_from_file_location("merge_temporal_dataset", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
merge = _MODULE.merge


class MergeTemporalDatasetTests(unittest.TestCase):
    def test_merges_confirmed_non_contiguous_selection_and_carries_frame_reviews(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            source, keys = make_consecutive_dataset(datasets)
            source = source.rename(datasets / "batch")
            manifest = json.loads((source / "manifest.json").read_text()) if (source / "manifest.json").exists() else {}
            records = []
            for i, key in enumerate(keys):
                records.append({"image": f"canonical/images/video-a/frame-{10+i:03d}.jpg", "label": f"canonical/labels/{key}", "image_sha256": ""})
            # Build the minimal skill manifests expected by the merge validator.
            import hashlib
            for record in records:
                record["image_sha256"] = hashlib.sha256((source / record["image"]).read_bytes()).hexdigest()
                record.update({"source_video_sha256": "video-hash", "source_group": "video-a", "frame_index": 10 + len(records) - len(records), "group_id": "video-a--run-10-13"})
            for i, record in enumerate(records):
                record["frame_index"] = 10 + i
            (source / "manifest.json").write_text(json.dumps({"schema_version": "yoyo_consecutive_annotation_dataset_v1", "dataset_id": source.name, "records": records}), encoding="utf-8")
            (source / "sampling_manifest.json").write_text(json.dumps({"sampling_method": "evenly_spaced_non_overlapping_consecutive_groups"}), encoding="utf-8")
            groups = json.loads((source / "consecutive_groups.json").read_text())
            groups["dataset_id"] = source.name
            (source / "consecutive_groups.json").write_text(json.dumps(groups), encoding="utf-8")
            (source / "temporal_review.json").write_text(json.dumps({"schema_version": "yoyo_temporal_review_v1", "dataset_id": source.name, "groups": {"video-a--run-10-13": {"status": "confirmed", "selected_sample_keys": [keys[0], keys[2], keys[3]], "selected_frame_indices": [10, 12, 13]}}}), encoding="utf-8")
            review_map = root / "review.json"
            review_map.write_text(json.dumps({"schema_version": "yoyo_dataset_review_v3", "datasets": {"batch": {"samples": {keys[0]: {"confirmed": True, "reviewer": "r"}}}}}), encoding="utf-8")
            target = datasets / "1Ayoyo_temporal"
            result = merge(source, target, review_map)
            self.assertTrue(result["ok"])
            self.assertFalse((target / "temporal_review.json").exists())
            merged = json.loads((target / "consecutive_groups.json").read_text())
            self.assertEqual([f["frame_index"] for f in merged["groups"][0]["frames"]], [10, 12, 13])
            carried = json.loads(review_map.read_text())["datasets"]["1Ayoyo_temporal"]["samples"]
            self.assertIn(keys[0], carried)


if __name__ == "__main__":
    unittest.main()
