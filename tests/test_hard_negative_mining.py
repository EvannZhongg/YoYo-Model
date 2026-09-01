import unittest

from cli.training.mine_hard_negatives import _extract_candidates


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    xyxy = _Tensor([[1.2, 2.3, 10.4, 12.5], [4.0, 5.0, 8.0, 9.0]])
    conf = _Tensor([0.81, 0.72])
    cls = _Tensor([0, 1])


class _Result:
    boxes = _Boxes()


class HardNegativeMiningTests(unittest.TestCase):
    def test_extracts_only_yoyo_candidates_above_threshold(self):
        candidates = _extract_candidates(_Result(), {0: "yoyo", 1: "person"}, 0.75)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["class_name"], "yoyo")
        self.assertEqual(candidates[0]["bbox"], [1.2, 2.3, 10.4, 12.5])


if __name__ == "__main__":
    unittest.main()
