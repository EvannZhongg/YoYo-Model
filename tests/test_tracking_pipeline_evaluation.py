import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from video_tracking.evaluate_pipeline import (
    _annotation_path,
    _observation_mask,
    _truth_bbox,
    backfill_detector_truth,
    detector_metrics,
)


class TrackingPipelineEvaluationTests(unittest.TestCase):
    def test_observation_mask_renders_all_polygons(self):
        mask = _observation_mask(
            {
                "polygons": [
                    [[2, 2], [8, 2], [8, 8], [2, 8]],
                    [[12, 12], [18, 12], [18, 18], [12, 18]],
                ]
            },
            24,
            24,
        )
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[5, 5]), 1)
        self.assertEqual(int(mask[15, 15]), 1)
        self.assertEqual(int(mask[10, 10]), 0)

    def test_truth_bbox_supports_nested_annotation_schema(self):
        self.assertEqual(
            _truth_bbox({"bbox": [{"bbox_pixel": [10, 20, 30, 40]}]}),
            [10.0, 20.0, 30.0, 40.0],
        )

    def test_detector_metrics_use_only_reviewed_bbox_truth(self):
        metrics = detector_metrics(
            [
                {
                    "bbox_truth_accepted": True,
                    "truth_bbox": [0, 0, 10, 10],
                    "predicted_bbox": [0, 0, 10, 10],
                },
                {
                    "bbox_truth_accepted": True,
                    "truth_bbox": [20, 20, 30, 30],
                    "predicted_bbox": None,
                },
                {
                    "bbox_truth_accepted": True,
                    "truth_bbox": None,
                    "predicted_bbox": None,
                },
                {
                    "bbox_truth_accepted": False,
                    "truth_bbox": None,
                    "predicted_bbox": [5, 5, 8, 8],
                },
            ]
        )
        self.assertEqual(metrics["accepted_images"], 3)
        self.assertEqual(metrics["presence"]["tp"], 1)
        self.assertEqual(metrics["presence"]["fn"], 1)
        self.assertEqual(metrics["presence"]["tn"], 1)
        self.assertEqual(metrics["iou50_matches"], 1)
        self.assertEqual(metrics["iou50_recall"], 0.5)

    def test_annotation_path_finds_canonical_split_for_derived_holdout(self):
        with TemporaryDirectory() as temp_dir:
            annotations_dir = Path(temp_dir)
            canonical = annotations_dir / "labels" / "train" / "source-a" / "frame_0001.json"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("{}", encoding="utf-8")
            derived_image = Path("dataset/images/test/source-a/frame_0001.jpg")

            self.assertEqual(_annotation_path(annotations_dir, "test", derived_image), canonical)

    def test_backfill_repairs_detector_truth_without_model_inference(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation = root / "annotations" / "labels" / "train" / "source-a" / "frame_0001.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(
                json.dumps({"review_status": "reviewed", "bbox": [{"bbox_pixel": [0, 0, 10, 10]}]}),
                encoding="utf-8",
            )
            metrics_path = root / "test_tracking_pipeline_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "split": "test",
                        "string": {
                            "images": [
                                {
                                    "image_path": "dataset/images/test/source-a/frame_0001.jpg",
                                    "predicted_bbox": [0, 0, 10, 10],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = backfill_detector_truth(metrics_path, root / "annotations")

            self.assertEqual(result["detector_on_accepted_subset"]["iou50_matches"], 1)
            self.assertEqual(result["detector_truth_resolution"]["canonical_split_fallbacks"], 1)
            self.assertFalse(result["detector_truth_resolution"]["model_inference_rerun"])


if __name__ == "__main__":
    unittest.main()
