import unittest

from video_dataset.split_policy import derived_split, parse_source_groups, remove_cross_split_duplicate_hashes


class DerivedSplitPolicyTests(unittest.TestCase):
    def test_holdout_source_is_routed_to_test_before_old_test_exclusion(self):
        split, reason = derived_split("train", "fresh-source", {"fresh-source"}, True)
        self.assertEqual(split, "test")
        self.assertIsNone(reason)

    def test_old_test_is_excluded_and_other_splits_are_preserved(self):
        self.assertEqual(
            derived_split("test", "old-test", {"fresh-source"}, True),
            (None, "original_test_excluded_for_fresh_holdout"),
        )
        self.assertEqual(derived_split("val", "validation", {"fresh-source"}, True), ("val", None))

    def test_source_group_parser_is_deduplicated(self):
        self.assertEqual(parse_source_groups("a, b,a, "), {"a", "b"})

    def test_duplicate_content_is_kept_in_highest_priority_split(self):
        kept, dropped = remove_cross_split_duplicate_hashes(
            [
                {"id": "train-copy", "split": "train", "image_sha256": "same"},
                {"id": "test-copy", "split": "test", "image_sha256": "same"},
                {"id": "unique", "split": "train", "image_sha256": "unique"},
            ]
        )
        self.assertEqual({item["id"] for item in kept}, {"test-copy", "unique"})
        self.assertEqual(dropped[0]["duplicate_owner_split"], "test")


if __name__ == "__main__":
    unittest.main()
