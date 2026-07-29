import unittest

from training_v2.evaluate import _artifact_suffix, _check_dataset_manifest, _detection_recall_from_confusion


class DetectionRecallTests(unittest.TestCase):
    def test_background_predictions_count_as_false_negatives(self) -> None:
        # Rows are predicted classes plus background; columns are true classes plus background.
        recall, matrix = _detection_recall_from_confusion(
            [[10.0, 2.0], [5.0, 0.0]],
            ["yoyo"],
        )

        self.assertEqual(matrix, [[10.0, 2.0], [5.0, 0.0]])
        self.assertAlmostEqual(recall["yoyo"], 10.0 / 15.0)


class DatasetComparisonTests(unittest.TestCase):
    def test_native_manifest_has_no_artifact_suffix(self) -> None:
        matches, warning = _check_dataset_manifest("abc", "abc", False)

        self.assertTrue(matches)
        self.assertEqual(warning, "")
        self.assertEqual(_artifact_suffix(matches, "abc"), "")

    def test_external_manifest_requires_explicit_opt_in(self) -> None:
        with self.assertRaises(RuntimeError):
            _check_dataset_manifest("old", "new", False)

        matches, warning = _check_dataset_manifest("old", "1234567890abcdef", True)
        self.assertFalse(matches)
        self.assertIn("cross-model comparison", warning)
        self.assertEqual(_artifact_suffix(matches, "1234567890abcdef"), "_external_1234567890ab")


if __name__ == "__main__":
    unittest.main()
