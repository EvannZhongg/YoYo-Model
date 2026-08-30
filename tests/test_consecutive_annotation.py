import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from helpers import make_consecutive_dataset
from workbench.consecutive_annotation import (
    CONSECUTIVE_FILENAME,
    list_consecutive_annotation_datasets,
    open_consecutive_annotation_dataset,
    save_consecutive_annotation_sample,
    select_consecutive_group_range,
)


class ConsecutiveAnnotationWorkbenchTests(unittest.TestCase):
    @staticmethod
    def _edit(orientation: str, x1: int) -> dict:
        return {
            "yoyo_visibility": "visible",
            "trick_orientation": orientation,
            "presentation_orientation": "frontal" if orientation == "normal" else "edge_horizontal",
            "yoyo_bbox_pixel": [x1, 10, x1 + 20, 30],
            "string_visibility": "partial",
            "string_polylines_pixel": [[[5, 5], [25, 25]]],
            "string_review_status": "reviewed",
            "bbox_review_status": "reviewed",
            "reviewer": "tester",
        }

    def test_lists_only_datasets_with_valid_consecutive_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = make_consecutive_dataset(root)
            make_consecutive_dataset(root / "experiments")
            ordinary = root / "ordinary"
            (ordinary / "canonical" / "images").mkdir(parents=True)
            (ordinary / "canonical" / "labels").mkdir(parents=True)
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                self.assertEqual(
                    list_consecutive_annotation_datasets(),
                    [{"name": dataset.name, "path": str(dataset.resolve())}],
                )

    def test_range_confirmation_removes_only_metadata_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, keys = make_consecutive_dataset(root)
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                opened = select_consecutive_group_range(
                    str(dataset), "video-a--run-10-13", 11, 12
                )
            metadata = json.loads((dataset / CONSECUTIVE_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual([frame["frame_index"] for frame in metadata["groups"][0]["frames"]], [11, 12])
            self.assertEqual([sample["frame_index"] for sample in opened["samples"]], [11, 12])
            self.assertTrue((dataset / "canonical" / "labels" / keys[0]).is_file())
            self.assertTrue((dataset / "canonical" / "labels" / keys[-1]).is_file())

    def test_sync_save_overwrites_only_later_frames_each_time(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, keys = make_consecutive_dataset(root)
            review = root / "review.json"
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review),
            ):
                first = save_consecutive_annotation_sample(
                    str(dataset), keys[0], self._edit("normal", 20), True
                )
                second = save_consecutive_annotation_sample(
                    str(dataset), keys[1], self._edit("horizontal", 40), True
                )
                opened = open_consecutive_annotation_dataset(str(dataset))

            self.assertEqual(first["propagated_count"], 3)
            self.assertEqual(second["propagated_count"], 2)
            labels = [
                json.loads((dataset / "canonical" / "labels" / key).read_text(encoding="utf-8"))
                for key in keys
            ]
            self.assertEqual(labels[0]["active_yoyo"]["bbox_pixel"], [20.0, 10.0, 40.0, 30.0])
            self.assertEqual([label["active_yoyo"]["trick_orientation"] for label in labels], [
                "normal", "horizontal", "horizontal", "horizontal",
            ])
            self.assertEqual([label["active_yoyo"]["presentation_orientation"] for label in labels], [
                "frontal", "edge_horizontal", "edge_horizontal", "edge_horizontal",
            ])
            self.assertEqual(opened["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
