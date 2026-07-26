import unittest

from model_registry import _metric_summary


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
