import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from helpers import make_consecutive_dataset
from workbench.temporal_annotation import (
    temporal_annotation_component_kwargs,
    list_temporal_annotation_datasets,
    open_temporal_annotation_dataset,
    set_temporal_group_selection,
    set_temporal_group_reviewed,
)


def mark_temporal(dataset: Path) -> None:
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "yoyo_consecutive_annotation_dataset_v1",
                "dataset_id": dataset.name,
                "deduplication": {"within_group_repeated_frames_allowed": True},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "sampling_manifest.json").write_text(
        json.dumps(
            {"sampling_method": "evenly_spaced_non_overlapping_consecutive_groups"}
        ),
        encoding="utf-8",
    )


class TemporalAnnotationWorkbenchTests(unittest.TestCase):
    def test_temporal_component_registers_model_preannotation_endpoint(self):
        from workbench.preannotation import ui_preannotate_dataset

        component = temporal_annotation_component_kwargs()
        self.assertIn(ui_preannotate_dataset, component["server_functions"])
        self.assertNotIn("yta-save-selection", component["value"] + component["js_on_load"])
        self.assertIn("确认本组可用", component["value"])
        self.assertIn("temporalSampleCache.clear()", component["js_on_load"])

    def test_only_skill_temporal_datasets_are_listed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            temporal, _ = make_consecutive_dataset(root)
            mark_temporal(temporal)
            ordinary, _ = make_consecutive_dataset(root / "ordinary")
            ordinary.rename(root / "ordinary-sequence")
            aggregate, _ = make_consecutive_dataset(root / "aggregate")
            aggregate.rename(root / "1Ayoyo_temporal")
            group_metadata = json.loads(
                (root / "1Ayoyo_temporal" / "consecutive_groups.json").read_text(encoding="utf-8")
            )
            group_metadata["dataset_id"] = "1Ayoyo_temporal"
            (root / "1Ayoyo_temporal" / "consecutive_groups.json").write_text(
                json.dumps(group_metadata), encoding="utf-8"
            )
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                listed = list_temporal_annotation_datasets()
            self.assertEqual(
                {item["name"] for item in listed},
                {"sequence-set", "1Ayoyo_temporal"},
            )

    def test_group_progress_and_group_review_round_trip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = make_consecutive_dataset(root)
            mark_temporal(dataset)
            review = root / "review.json"
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review),
            ):
                opened = open_temporal_annotation_dataset(str(dataset))
                self.assertEqual(opened["dataset_type"], "temporal")
                self.assertEqual(opened["groups"][0]["review_progress"], 0.0)
                keys = [sample["key"] for sample in opened["samples"]]
                selected = set_temporal_group_selection(
                    str(dataset), "video-a--run-10-13", [keys[0], keys[2], keys[3]]
                )
                self.assertEqual(selected["groups"][0]["selected_frame_indices"], [10, 12, 13])
                reviewed = set_temporal_group_reviewed(
                    str(dataset), "video-a--run-10-13", "tester", True
                )
            self.assertEqual(reviewed["groups"][0]["review_progress"], 0.0)
            self.assertEqual(reviewed["temporal_reviewed_group_count"], 1)

    def test_aggregate_temporal_dataset_accepts_non_contiguous_groups(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, keys = make_consecutive_dataset(root)
            dataset.rename(root / "1Ayoyo_temporal")
            metadata_path = root / "1Ayoyo_temporal" / "consecutive_groups.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_id"] = "1Ayoyo_temporal"
            metadata["groups"][0]["frames"] = [metadata["groups"][0]["frames"][i] for i in (0, 2, 3)]
            metadata["groups"][0]["selected_end_frame"] = 13
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("workbench.dataset_annotation.DATASETS_DIR", root):
                opened = open_temporal_annotation_dataset(str(root / "1Ayoyo_temporal"))
            self.assertEqual(opened["groups"][0]["frame_count"], 3)

    def test_group_confirmation_selects_reviewed_frames_automatically(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, keys = make_consecutive_dataset(root)
            mark_temporal(dataset)
            review = root / "review.json"
            with (
                patch("workbench.dataset_annotation.DATASETS_DIR", root),
                patch("workbench.dataset_annotation.REVIEW_MAP_PATH", review),
            ):
                for key in keys:
                    from workbench.dataset_annotation import set_annotation_sample_reviewed

                    set_annotation_sample_reviewed(str(dataset), key, "tester", True)
                result = set_temporal_group_reviewed(str(dataset), "video-a--run-10-13", "tester", True)
            self.assertEqual(result["groups"][0]["selected_frame_count"], len(keys))
            self.assertEqual(result["groups"][0]["selected_sample_keys"], keys)
            self.assertEqual(result["groups"][0]["group_review_status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
