import unittest

from video_tracking.tracker import _segments_from_records


def records(active_indexes: set[int], count: int = 20, fps: float = 10.0) -> list[dict]:
    return [
        {
            "frame_index": index,
            "timestamp_s": index / fps,
            "active": index in active_indexes,
        }
        for index in range(count)
    ]


class SegmentGenerationTests(unittest.TestCase):
    def test_padding_is_trimmed_so_neighboring_candidates_do_not_overlap(self):
        segments = _segments_from_records(
            records(set(range(2, 5)) | set(range(8, 11))),
            fps=10.0,
            padding_seconds=0.4,
            min_segment_seconds=0.2,
            max_gap_seconds=0.1,
            max_segment_seconds=180.0,
        )

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0]["end_frame"], segments[1]["start_frame"])
        self.assertLessEqual(segments[0]["start_frame"], segments[0]["active_start_frame"])
        self.assertGreaterEqual(segments[0]["end_frame"], segments[0]["active_end_frame"])
        self.assertTrue(segments[0]["padding_trimmed_for_neighbor"])
        self.assertTrue(segments[1]["padding_trimmed_for_neighbor"])

    def test_duration_limited_chunks_remain_non_overlapping(self):
        segments = _segments_from_records(
            records(set(range(20)), count=20),
            fps=10.0,
            padding_seconds=0.2,
            min_segment_seconds=0.2,
            max_gap_seconds=0.1,
            max_segment_seconds=1.0,
        )

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(item["duration_s"] <= 1.0 for item in segments))
        self.assertTrue(
            all(left["end_frame"] < right["start_frame"] for left, right in zip(segments, segments[1:]))
        )


if __name__ == "__main__":
    unittest.main()
