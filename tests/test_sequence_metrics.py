import json
import tempfile
import unittest
from pathlib import Path

from video_tracking.sequence_metrics import centerline_pair_metrics, evaluate_sequence


class SequenceMetricsTests(unittest.TestCase):
    def test_centerline_tolerance_is_source_pixel_based(self):
        target = [[[10.0, 20.0], [30.0, 20.0]]]
        prediction = [[[10.0, 23.0], [30.0, 23.0]]]
        metrics = centerline_pair_metrics(target, prediction, (2.0, 4.0), spacing_px=1.0)
        self.assertEqual(metrics["tolerances"]["2"]["f1"], 0.0)
        self.assertEqual(metrics["tolerances"]["4"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["chamfer_mean_px"], 3.0, places=4)

    def test_centerline_recall_penalizes_a_missing_segment(self):
        target = [
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 10.0], [10.0, 10.0]],
        ]
        prediction = [[[0.0, 0.0], [10.0, 0.0]]]
        metrics = centerline_pair_metrics(target, prediction, (2.0,), spacing_px=1.0)

        self.assertEqual(metrics["target_samples"], 22)
        self.assertEqual(metrics["prediction_samples"], 11)
        self.assertEqual(metrics["tolerances"]["2"]["precision"], 1.0)
        self.assertEqual(metrics["tolerances"]["2"]["recall"], 0.5)
        self.assertEqual(metrics["tolerances"]["2"]["f1"], 0.666667)
        self.assertAlmostEqual(metrics["chamfer_mean_px"], 3.3333, places=4)

    def _dataset(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "consecutive"
        label_root = dataset / "canonical" / "labels" / "video-a"
        label_root.mkdir(parents=True)
        frames = []
        for index in range(4):
            key = f"video-a/frame-{index:03d}.json"
            label = {
                "source_group": "video-a",
                "frame_index": index,
                "timestamp_s": index / 30.0,
                "active_yoyo": {
                    "visibility": "visible",
                    "bbox_review_status": "reviewed",
                    "bbox_pixel": [10 + index, 10, 20 + index, 20],
                    "trick_orientation": "normal" if index < 2 else "horizontal",
                },
                "string_visibility": "partial",
                "string_review_status": "reviewed",
                "string_polylines_pixel": [[[5 + index, 30], [25 + index, 30]]],
            }
            (label_root / f"frame-{index:03d}.json").write_text(json.dumps(label), encoding="utf-8")
            frames.append({
                "sample_key": key,
                "frame_index": index,
                "timestamp_s": index / 30.0,
            })
        (dataset / "consecutive_groups.json").write_text(json.dumps({
            "schema_version": "yoyo_consecutive_groups_v1",
            "groups": [{
                "group_id": "video-a--run-0-3",
                "source_group": "video-a",
                "frames": frames,
            }],
        }), encoding="utf-8")
        predictions = root / "frames.jsonl"
        records = []
        for index in range(4):
            records.append({
                "frame_index": index,
                "yoyo": {"bbox": [10 + index, 10, 20 + index, 20], "center": [15 + index, 15], "track_id": 7},
                "string": {
                    "points": [[5 + index, 30], [25 + index, 30]],
                    "method": "lucas_kanade_optical_flow" if index else "semantic_segmentation",
                    "propagation_age_frames": 1 if index == 1 else 0,
                },
                "bad_case": [],
                "trick_orientation": {
                    "label": "normal" if index < 2 else "horizontal",
                    "confidence": 0.9,
                },
            })
        predictions.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
        return dataset, predictions

    def test_perfect_sequence_has_geometry_and_tracking_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            result = evaluate_sequence(dataset, predictions)
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["yoyo"]["presence"]["f1"], 1.0)
        self.assertEqual(result["yoyo"]["localization"]["mean_iou"], 1.0)
        self.assertEqual(result["string"]["centerline"]["tolerances"]["2"]["f1"], 1.0)
        self.assertEqual(result["string"]["centerline"]["chamfer_mean_px"], 0.0)
        self.assertEqual(result["string"]["propagated_frames"], 1)
        self.assertEqual(result["yoyo"]["temporal"]["matched_motion_pairs"], 3)
        self.assertEqual(result["yoyo"]["temporal"]["track_id_switch_count"], 0)
        self.assertEqual(result["string"]["temporal"]["matched_motion_pairs"], 3)
        self.assertEqual(result["string"]["temporal"]["mean_centroid_motion_error_px"], 0.0)
        self.assertEqual(result["orientation"]["accuracy"], 1.0)
        self.assertEqual(result["orientation"]["temporal"]["target_switch_count"], 1)
        self.assertEqual(result["orientation"]["temporal"]["predicted_switch_count"], 1)

    def test_temporal_metrics_count_switch_and_recovery_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            records[1]["yoyo"] = None
            records[1]["string"] = None
            records[2]["yoyo"]["track_id"] = 9
            predictions.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            result = evaluate_sequence(dataset, predictions)
        self.assertEqual(result["yoyo"]["temporal"]["missing_episode_count"], 1)
        self.assertEqual(result["yoyo"]["temporal"]["mean_recovery_latency_frames"], 1.0)
        self.assertEqual(result["yoyo"]["temporal"]["track_id_switch_count"], 2)
        self.assertEqual(result["string"]["temporal"]["mean_recovery_latency_frames"], 1.0)

    def test_unknown_string_negative_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            label = dataset / "canonical" / "labels" / "video-a" / "frame-003.json"
            data = json.loads(label.read_text(encoding="utf-8"))
            data["string_visibility"] = "not_visible"
            data["string_review_status"] = "unresolved"
            data["string_polylines_pixel"] = []
            label.write_text(json.dumps(data), encoding="utf-8")
            result = evaluate_sequence(dataset, predictions)
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["excluded_unknown"]["string"], 1)
        self.assertEqual(result["string"]["presence"]["known_frames"], 3)

    def test_uncertain_yoyo_is_excluded_but_reviewed_absent_is_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            uncertain_path = dataset / "canonical" / "labels" / "video-a" / "frame-002.json"
            uncertain = json.loads(uncertain_path.read_text(encoding="utf-8"))
            uncertain["active_yoyo"]["visibility"] = "uncertain"
            uncertain["active_yoyo"]["bbox_pixel"] = None
            uncertain["active_yoyo"]["bbox_review_status"] = "needs_review"
            uncertain_path.write_text(json.dumps(uncertain), encoding="utf-8")
            absent_path = dataset / "canonical" / "labels" / "video-a" / "frame-003.json"
            absent = json.loads(absent_path.read_text(encoding="utf-8"))
            absent["active_yoyo"]["visibility"] = "not_visible"
            absent["active_yoyo"]["not_visible_reason"] = "absent"
            absent["active_yoyo"]["bbox_pixel"] = None
            absent_path.write_text(json.dumps(absent), encoding="utf-8")
            result = evaluate_sequence(dataset, predictions)
        self.assertEqual(result["excluded_unknown"]["yoyo"], 1)
        self.assertEqual(result["yoyo"]["presence"]["known_frames"], 3)
        self.assertEqual(result["yoyo"]["presence"]["fp"], 1)

    def test_nested_active_yoyo_review_status_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            label = dataset / "canonical" / "labels" / "video-a" / "frame-000.json"
            data = json.loads(label.read_text(encoding="utf-8"))
            data["active_yoyo"] = {
                "visibility": "visible",
                "bbox_pixel": [10, 10, 20, 20],
                "bbox_review_status": "reviewed",
            }
            label.write_text(json.dumps(data), encoding="utf-8")
            result = evaluate_sequence(dataset, predictions)
        self.assertEqual(result["yoyo"]["presence"]["known_frames"], 4)
        self.assertEqual(result["yoyo"]["localization"]["matched_frames"], 4)

    def test_frame_range_can_hold_out_a_contiguous_subsequence(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, predictions = self._dataset(Path(directory))
            result = evaluate_sequence(dataset, predictions, start_frame=1, end_frame=2)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["frame_range"], {"start": 1, "end": 2})


if __name__ == "__main__":
    unittest.main()
