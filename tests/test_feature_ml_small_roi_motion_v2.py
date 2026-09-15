from __future__ import annotations

import unittest

import numpy as np

from scripts.run_feature_ml_small_roi_motion_v2 import (
    MaxHistoryMotionExtractor,
)


class MaxHistoryMotionExtractorTests(unittest.TestCase):
    def test_four_frame_background_recovers_current_dark_particle(self) -> None:
        extractor = MaxHistoryMotionExtractor(lag=4, dark_sigma=1.0)
        uniform = np.full((48, 48), 120, dtype=np.uint8)
        common = {
            "video_slug": "test",
            "video_name": "test.mp4",
            "timestamp_sec": 0.0,
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
        }
        previous = None
        for frame_index in range(4):
            candidates, _ = extractor(
                uniform,
                previous,
                frame_index=frame_index,
                **common,
            )
            self.assertEqual(candidates, [])
            previous = uniform

        current = uniform.copy()
        current[22:25, 25:28] = 100
        candidates, _ = extractor(
            current,
            previous,
            frame_index=4,
            **common,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].row["proposal_source"], "motion_dark")


if __name__ == "__main__":
    unittest.main()
