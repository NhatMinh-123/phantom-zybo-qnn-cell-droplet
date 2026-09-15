from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.feature_ml_motion_candidates import (
    extract_motion_fused_candidates,
)
from scripts.run_feature_ml_small_roi_motion_test import (
    install_simple_geometry,
)


class MotionFusedCandidateTests(unittest.TestCase):
    def candidate_kwargs(self) -> dict[str, object]:
        return {
            "video_slug": "test",
            "video_name": "test.mp4",
            "frame_index": 3,
            "timestamp_sec": 0.03,
            "sequence_id": 1,
            "droplet_radius_px": 60.0,
            "core_radius_px": 18,
            "patch_size": 16,
            "blackhat_kernel": 7,
            "blackhat_sigma": 30.0,
            "temporal_sigma": 1.5,
            "min_area": 2,
            "max_area": 100,
            "max_side": 18,
            "max_candidates": 8,
            "dark_sigma": 1.0,
        }

    def test_motion_branch_recovers_dark_candidate(self) -> None:
        previous = np.full((48, 48), 120, dtype=np.uint8)
        current = previous.copy()
        current[22:25, 25:28] = 100

        without_reference, _ = extract_motion_fused_candidates(
            current,
            None,
            **self.candidate_kwargs(),
        )
        with_reference, diagnostics = extract_motion_fused_candidates(
            current,
            previous,
            **self.candidate_kwargs(),
        )

        self.assertEqual(without_reference, [])
        self.assertEqual(len(with_reference), 1)
        self.assertEqual(
            with_reference[0].row["proposal_source"],
            "motion_dark",
        )
        self.assertGreaterEqual(
            int(with_reference[0].row["pixel_area"]),
            6,
        )
        self.assertGreater(diagnostics["motion_dark_pixels"], 0)

    def test_simple_geometry_replaces_two_circles_with_one_roi(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original_circle, original_rectangle = install_simple_geometry(48)
        try:
            cv2.circle(image, (50, 50), 30, (0, 220, 255), 2)
            cv2.circle(image, (50, 50), 18, (255, 120, 0), 1)
        finally:
            cv2.circle = original_circle
            cv2.rectangle = original_rectangle

        self.assertTrue(np.any(image[26, 26] == (40, 220, 40)))
        self.assertTrue(np.all(image[50, 68] == 0))


if __name__ == "__main__":
    unittest.main()
