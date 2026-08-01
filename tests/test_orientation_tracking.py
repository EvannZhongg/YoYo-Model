import unittest

import numpy as np

from video_tracking.orientation import carry_orientation, orientation_crop_box, predict_orientation


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
    def test_crop_matches_training_union_policy(self):
        crop = orientation_crop_box(
            1000,
            600,
            {"bbox": [450, 250, 550, 350]},
            [{"x": 300, "y": 200}, {"x": 700, "y": 200}],
            {"points": [[300, 200], [500, 500]]},
        )
        self.assertEqual(crop, (200, 0, 800, 600))

    def test_crop_uses_full_frame_without_tracking_points(self):
        self.assertEqual(orientation_crop_box(640, 360, None, [], None), (0, 0, 640, 360))

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
        prediction = predict_orientation(model, np.zeros((360, 640, 3), dtype=np.uint8), None, [], None, 320, "cpu")
        self.assertEqual(prediction["label"], "normal")
        self.assertEqual(prediction["crop_box_pixel"], [0, 0, 640, 360])
        self.assertEqual(model.kwargs["imgsz"], 320)
        carried = carry_orientation(prediction, 2)
        self.assertEqual(carried["inference_status"], "carried")
        self.assertEqual(carried["age_frames"], 2)
        self.assertEqual(prediction["inference_status"], "ran")

if __name__ == "__main__":
    unittest.main()
