import json
import tempfile
import unittest
from pathlib import Path

from audit_attachment_dataset import audit
from annotation.qa import qa_annotation


class AttachmentAuditTests(unittest.TestCase):
    def test_qa_promotes_unreliable_string_proposal_to_high_priority(self):
        result = qa_annotation(
            {
                "visibility": "visible",
                "string_visibility": "visible",
                "string_polyline_pixel": [[10, 10], [20, 20]],
                "string_prelabel": {"status": "no_mask"},
            },
            None,
        )
        self.assertEqual(result["priority"], "high")
        self.assertIn("string_color_proposal_no_mask", result["warnings"])

    def test_counts_reviewed_classes_and_missing_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir()
            (root / "train" / "a.json").write_text(
                json.dumps(
                    {
                        "source_group": "group-a",
                        "split": "train",
                        "string_review_status": "approved",
                        "string_attachment_class": "hand_and_yoyo_attached",
                    }
                ),
                encoding="utf-8",
            )
            (root / "val").mkdir()
            (root / "val" / "b.json").write_text(
                json.dumps(
                    {
                        "source_group": "group-b",
                        "split": "val",
                        "string_review_status": "auto_labeled_needs_review",
                        "string_attachment_class": "yoyo_detached",
                    }
                ),
                encoding="utf-8",
            )

            report = audit(root)

        self.assertEqual(report["totals"]["annotations"], 2)
        self.assertEqual(report["totals"]["string_reviewed"], 1)
        self.assertEqual(report["reviewed_attachment_classes"]["hand_and_yoyo_attached"], 1)
        self.assertIn("yoyo_detached", report["missing_reviewed_classes"])
        self.assertEqual(report["reviewed_source_groups"]["hand_and_yoyo_attached"], ["group-a"])


if __name__ == "__main__":
    unittest.main()
