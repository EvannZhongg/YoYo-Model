import unittest

from training_v2.evaluate import _detection_recall_from_confusion


class DetectionRecallTests(unittest.TestCase):
    def test_background_predictions_count_as_false_negatives(self) -> None:
        # Rows are predicted classes plus background; columns are true classes plus background.
        recall, matrix = _detection_recall_from_confusion(
            [[10.0, 2.0], [5.0, 0.0]],
            ["yoyo"],
        )

        self.assertEqual(matrix, [[10.0, 2.0], [5.0, 0.0]])
        self.assertAlmostEqual(recall["yoyo"], 10.0 / 15.0)


if __name__ == "__main__":
    unittest.main()
