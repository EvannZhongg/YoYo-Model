import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from cli.tracking.evaluate_orientation import _read_image, _replay
from video_tracking.orientation import (
    OrientationTemporalFilter,
    carry_orientation,
    orientation_observation_is_unstable,
    orientation_crop_box,
    predict_orientation,
)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class _Vector(_Scalar):
    def tolist(self):
        return list(self.value)


class OrientationTrackingTests(unittest.TestCase):
    @staticmethod
    def _prediction(horizontal: float, normal: float, not_applicable: float) -> dict:
        probabilities = {
            "horizontal": horizontal,
            "normal": normal,
            "not_applicable": not_applicable,
        }
        label = max(probabilities, key=probabilities.get)
        return {
            "label": label,
            "confidence": probabilities[label],
            "probabilities": probabilities,
            "inference_status": "ran",
            "age_frames": 0,
        }

    def test_crop_matches_yoyo_only_training_policy(self):
        crop = orientation_crop_box(
            1000,
            600,
            {"bbox": [450, 250, 550, 350]},
        )
        self.assertEqual(crop, (350, 150, 650, 450))

    def test_evaluation_image_reader_supports_unicode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "方向样本.jpg"
            expected = np.full((8, 10, 3), 127, dtype=np.uint8)
            encoded_ok, encoded = cv2.imencode(".jpg", expected)
            self.assertTrue(encoded_ok)
            encoded.tofile(path)
            actual = _read_image(path)
        self.assertEqual(actual.shape, expected.shape)

    def test_adaptive_replay_returns_to_stable_cadence(self):
        records = [
            {
                "group_id": "group-a",
                "frame_index": index,
                "timestamp_s": index / 50.0,
                "target": "normal",
                "predicted": "normal",
                "confidence": 0.9,
                "probabilities": {
                    "horizontal": 0.05,
                    "normal": 0.9,
                    "not_applicable": 0.05,
                },
            }
            for index in range(10)
        ]
        replay = _replay(
            records,
            inference_fps=5.0,
            filter_kwargs={},
            adaptive_kwargs={
                "burst_inference_fps": 25.0,
                "min_confidence": 0.5,
                "stable_observations": 4,
            },
        )

        self.assertEqual(replay["inference_count"], 5)
        self.assertEqual(replay["burst_inference_count"], 4)
        self.assertEqual(set(replay["predictions"].values()), {"normal"})

    def test_crop_uses_deterministic_center_negative_without_yoyo(self):
        self.assertEqual(orientation_crop_box(640, 360, None), (270, 130, 370, 230))

    def test_prediction_is_json_ready_and_carry_is_explicit(self):
        class Probs:
            data = _Vector([0.2, 0.7, 0.1])
            top1 = 1
            top1conf = _Scalar(0.7)

        class Result:
            probs = Probs()

        class Model:
            names = {0: "horizontal", 1: "normal", 2: "not_applicable"}

            def predict(self, **kwargs):
                self.kwargs = kwargs
                return [Result()]

        model = Model()
        prediction = predict_orientation(model, np.zeros((360, 640, 3), dtype=np.uint8), None, 320, "cpu")
        self.assertEqual(prediction["label"], "normal")
        self.assertEqual(prediction["crop_box_pixel"], [270, 130, 370, 230])
        self.assertEqual(
            prediction["crop_policy"],
            "yoyo_bbox_square_3p0_min_12pct; no_yoyo_center_square_28pct",
        )
        self.assertEqual(model.kwargs["imgsz"], 320)
        carried = carry_orientation(prediction, 2)
        self.assertEqual(carried["inference_status"], "carried")
        self.assertEqual(carried["age_frames"], 2)
        self.assertEqual(prediction["inference_status"], "ran")

    def test_four_way_presentation_prediction_maps_to_coarse_label(self):
        class Probs:
            data = _Vector([0.1, 0.2, 0.6, 0.1])
            top1 = 2
            top1conf = _Scalar(0.6)

        class Result:
            probs = Probs()

        class Model:
            names = {0: "frontal", 1: "edge_horizontal", 2: "edge_vertical", 3: "unknown"}

            def predict(self, **kwargs):
                return [Result()]

        prediction = predict_orientation(Model(), np.zeros((360, 640, 3), dtype=np.uint8), None, 320, "cpu")
        self.assertEqual(prediction["presentation_label"], "edge_vertical")
        self.assertEqual(prediction["label"], "normal")
        self.assertAlmostEqual(prediction["probabilities"]["normal"], 0.7)

    def test_temporal_filter_rejects_an_isolated_flip(self):
        temporal_filter = OrientationTemporalFilter()
        first = temporal_filter.update(self._prediction(0.05, 0.9, 0.05))
        spike = temporal_filter.update(self._prediction(0.8, 0.15, 0.05))
        recovered = temporal_filter.update(self._prediction(0.05, 0.9, 0.05))

        self.assertEqual(first["label"], "normal")
        self.assertEqual(spike["label"], "normal")
        self.assertEqual(spike["raw_label"], "horizontal")
        self.assertEqual(recovered["label"], "normal")
        self.assertTrue(orientation_observation_is_unstable(spike, 0.5))
        self.assertFalse(orientation_observation_is_unstable(recovered, 0.5))

    def test_temporal_filter_switches_after_sustained_evidence(self):
        temporal_filter = OrientationTemporalFilter(strong_switch_confidence=1.0)
        temporal_filter.update(self._prediction(0.05, 0.9, 0.05))
        outputs = [
            temporal_filter.update(self._prediction(0.8, 0.15, 0.05))
            for _ in range(4)
        ]

        self.assertEqual([item["label"] for item in outputs[:3]], ["normal"] * 3)
        self.assertEqual(outputs[3]["label"], "horizontal")
        self.assertEqual(outputs[3]["temporal_filter"]["status"], "confirmed_switched")

if __name__ == "__main__":
    unittest.main()
