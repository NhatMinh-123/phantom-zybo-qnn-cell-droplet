from __future__ import annotations

import unittest

import numpy as np

from scripts.extract_droplet_microplastic_features import (
    CandidateFeatureTracker,
    entropy,
    extract_candidates,
    masked_stats,
)


class DropletMicroplasticFeatureTests(unittest.TestCase):
    def test_constant_image_has_zero_entropy_and_contrast(self) -> None:
        image = np.full((16, 16), 120, dtype=np.uint8)
        mask = np.full_like(image, 255)
        stats = masked_stats(image, mask)
        self.assertAlmostEqual(entropy(image.reshape(-1)), 0.0)
        self.assertAlmostEqual(stats["std"], 0.0)
        self.assertAlmostEqual(stats["contrast"], 0.0)

    def test_dark_particle_generates_finite_candidate_features(self) -> None:
        current = np.full((128, 128), 180, dtype=np.uint8)
        previous = current.copy()
        current[62:66, 62:66] = 90
        candidates, thresholds = extract_candidates(
            current,
            previous,
            video_slug="video",
            video_name="video.mp4",
            frame_index=10,
            timestamp_sec=0.1,
            sequence_id=2,
            droplet_radius_px=60.0,
            core_radius_px=46,
            patch_size=32,
            blackhat_kernel=7,
            blackhat_sigma=3.2,
            temporal_sigma=2.5,
            min_area=2,
            max_area=100,
            max_side=18,
            max_candidates=8,
        )
        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(candidate.patch.shape, (32, 32))
        self.assertGreater(float(candidate.row["local_contrast_gray"]), 0.0)
        self.assertLess(float(candidate.row["radial_distance_norm"]), 0.1)
        self.assertEqual(candidate.row["label_status"], "unlabeled")
        self.assertGreaterEqual(thresholds["blackhat_threshold"], 3.0)

    def test_tracker_links_repeated_candidate_without_creating_label(self) -> None:
        current = np.full((128, 128), 180, dtype=np.uint8)
        current[62:66, 62:66] = 90
        candidates, _ = extract_candidates(
            current,
            np.full_like(current, 180),
            video_slug="video",
            video_name="video.mp4",
            frame_index=1,
            timestamp_sec=0.01,
            sequence_id=1,
            droplet_radius_px=60.0,
            core_radius_px=46,
            patch_size=32,
            blackhat_kernel=7,
            blackhat_sigma=3.2,
            temporal_sigma=2.5,
            min_area=2,
            max_area=100,
            max_side=18,
            max_candidates=8,
        )
        tracker = CandidateFeatureTracker(max_distance=8.0, max_missed=1)
        for frame_index in range(1, 4):
            copied = [
                type(item)(row=dict(item.row), patch=item.patch.copy())
                for item in candidates
            ]
            finished, assignments = tracker.update(
                copied,
                frame_index=frame_index,
                sequence_id=1,
            )
            self.assertFalse(finished)
            self.assertEqual(assignments[0], 1)
        tracks = tracker.reset()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].observations), 3)
        self.assertEqual(
            tracks[0].observations[0]["ground_truth_class"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
