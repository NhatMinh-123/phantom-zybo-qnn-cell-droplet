from __future__ import annotations

import unittest

from scripts.build_teacher_refined_dataset import Box, refine_labels


class TeacherRefinedDatasetTests(unittest.TestCase):
    def test_blends_a_matched_box_and_keeps_unmatched_ground_truth(self) -> None:
        ground_truth = [
            Box(0, (0.10, 0.10, 0.20, 0.20)),
            Box(1, (0.40, 0.40, 0.70, 0.70)),
        ]
        teacher = [
            Box(0, (0.11, 0.10, 0.21, 0.20), 0.90),
        ]

        refined, metrics = refine_labels(
            ground_truth,
            teacher,
            match_iou=0.5,
            blend=0.5,
            add_confidence=(0.75, 0.85),
            duplicate_iou=0.3,
        )

        self.assertEqual(len(refined), 2)
        self.assertEqual(metrics["matched"], 1)
        self.assertEqual(metrics["added"], 0)
        self.assertAlmostEqual(refined[0].xyxy[0], 0.105)
        self.assertEqual(refined[1], ground_truth[1])

    def test_adds_only_high_confidence_nonduplicate_teacher_boxes(self) -> None:
        ground_truth = [Box(0, (0.10, 0.10, 0.20, 0.20))]
        teacher = [
            Box(0, (0.50, 0.10, 0.60, 0.20), 0.80),
            Box(0, (0.70, 0.10, 0.80, 0.20), 0.70),
            Box(0, (0.11, 0.10, 0.21, 0.20), 0.95),
        ]

        refined, metrics = refine_labels(
            ground_truth,
            teacher,
            match_iou=0.5,
            blend=0.0,
            add_confidence=(0.75, 0.85),
            duplicate_iou=0.3,
        )

        self.assertEqual(len(refined), 2)
        self.assertEqual(metrics["matched"], 1)
        self.assertEqual(metrics["added"], 1)

    def test_does_not_cross_match_classes(self) -> None:
        ground_truth = [Box(0, (0.10, 0.10, 0.20, 0.20))]
        teacher = [Box(1, (0.10, 0.10, 0.20, 0.20), 0.90)]

        refined, metrics = refine_labels(
            ground_truth,
            teacher,
            match_iou=0.5,
            blend=0.5,
            add_confidence=(0.75, 0.85),
            duplicate_iou=0.3,
        )

        self.assertEqual(metrics["matched"], 0)
        self.assertEqual(metrics["added"], 1)
        self.assertEqual({item.class_id for item in refined}, {0, 1})


if __name__ == "__main__":
    unittest.main()
