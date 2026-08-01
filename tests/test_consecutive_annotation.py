import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from workbench.consecutive_annotation import (
    CONSECUTIVE_FILENAME,
    CONSECUTIVE_SCHEMA_VERSION,
    consecutive_annotation_component_kwargs,
    list_consecutive_annotation_datasets,
    open_consecutive_annotation_dataset,
    save_consecutive_annotation_sample,
    select_consecutive_group_range,
)
from workbench.dataset_annotation import ANNOTATION_SCHEMA_VERSION


class ConsecutiveAnnotationWorkbenchTests(unittest.TestCase):
    def _dataset(self, root: Path, frame_count: int = 4) -> tuple[Path, list[str]]:
        dataset = root / "sequence-set"
        group = "video-a"
        keys = []
        frames = []
        for offset in range(frame_count):
            frame_index = 10 + offset
            stem = f"frame-{frame_index:03d}"
            image = dataset / "canonical" / "images" / group / f"{stem}.jpg"
            label = dataset / "canonical" / "labels" / group / f"{stem}.json"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 100), (240 - offset, 240, 240)).save(image)
            key = f"{group}/{stem}.json"
            keys.append(key)
            label.write_text(json.dumps({
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "source_image": f"../../images/{group}/{stem}.jpg",
                "image_size": [200, 100],
                "source_group": group,
                "frame_index": frame_index,
                "visibility": "visible",
                "trick_orientation": "normal",
                "yoyo_bbox_pixel": [10, 10, 30, 30],
                "string_visibility": "partial",
                "string_polylines_pixel": [[[5, 5], [25, 25]]],
                "string_review_status": "unresolved",
                "string_path": {"topology": "single_path", "paths": []},
            }), encoding="utf-8")
            frames.append({
                "sample_key": key,
                "image": f"canonical/images/{group}/{stem}.jpg",
                "frame_index": frame_index,
                "timestamp_s": offset / 30,
            })
        (dataset / CONSECUTIVE_FILENAME).write_text(json.dumps({
            "schema_version": CONSECUTIVE_SCHEMA_VERSION,
            "dataset_id": dataset.name,
            "groups": [{
                "group_id": "video-a--run-10-13",
                "source_group": group,
                "source_video": "video-a.mp4",
                "original_start_frame": 10,
                "original_end_frame": 13,
                "selected_start_frame": 10,
                "selected_end_frame": 13,
                "start_sample_key": keys[0],
                "frames": frames,
            }],
        }), encoding="utf-8")
        return dataset, keys

    @staticmethod
    def _edit(orientation: str, x1: int) -> dict:
        return {
            "yoyo_visibility": "visible",
            "trick_orientation": orientation,
            "yoyo_bbox_pixel": [x1, 10, x1 + 20, 30],
            "string_visibility": "partial",
            "string_polylines_pixel": [[[5, 5], [25, 25]]],
            "string_review_status": "reviewed",
            "bbox_review_status": "reviewed",
            "reviewer": "tester",
            "notes": orientation,
        }

    def test_lists_only_datasets_with_valid_consecutive_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self._dataset(root)
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
            dataset, keys = self._dataset(root)
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
            dataset, keys = self._dataset(root)
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
            self.assertEqual(labels[0]["yoyo_bbox_pixel"], [20.0, 10.0, 40.0, 30.0])
            self.assertEqual([label["trick_orientation"] for label in labels], [
                "normal", "horizontal", "horizontal", "horizontal",
            ])
            self.assertEqual(opened["sample_count"], 4)

    def test_component_has_separate_current_and_sync_save_actions(self):
        component = consecutive_annotation_component_kwargs()
        self.assertIn("连续帧标注", component["value"])
        self.assertIn('id="yca-confirm-range"', component["value"])
        self.assertIn('id="yca-save"', component["value"])
        self.assertIn('id="yca-sync-save"', component["value"])
        self.assertIn("propagate_remaining:false", component["js_on_load"])
        self.assertIn("propagate_remaining:true", component["js_on_load"])
        self.assertIn("ui_select_consecutive_group_range", component["js_on_load"])
        self.assertIn("function deleteSelectedPoint", component["js_on_load"])
        self.assertIn('event.key!=="Delete"', component["js_on_load"])
        self.assertIn("line.splice(selected.point,1)", component["js_on_load"])


if __name__ == "__main__":
    unittest.main()
