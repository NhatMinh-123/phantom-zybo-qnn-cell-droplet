from __future__ import annotations

import unittest

import numpy as np
import torch

from qnn.patch_classifier import (
    PatchClassifierConfig,
    TinyQuantPatchClassifier,
    count_patch_parameters,
)
from qnn.patch_preprocess import preprocess_patch
from scripts.run_microplastic_one_droplet_hybrid import (
    DropletObservation,
    DropletSequenceTracker,
    ParticleCandidate,
    Rect,
    TemporalParticleTracker,
    find_droplet,
    find_particle_candidates,
    robust_threshold,
)


def candidate(x: float = 30.0, y: float = 28.0) -> ParticleCandidate:
    return ParticleCandidate(
        x=x,
        y=y,
        width=3,
        height=3,
        area=7,
        contrast=0.9,
        temporal=0.8,
        score=0.9,
        bbox=(int(x) - 1, int(y) - 1, 3, 3),
    )


class OneDropletPipelineTests(unittest.TestCase):
    def test_bright_candidate_mode_ignores_dark_spot(self) -> None:
        image = np.full((64, 64), 100, dtype=np.uint8)
        image[30:33, 30:33] = 220
        image[44:47, 44:47] = 20
        mask = np.full_like(image, 255)

        candidates, _, _, _, _ = find_particle_candidates(
            image,
            None,
            core_mask=mask,
            blackhat_kernel=9,
            sigma=3.0,
            min_area=1,
            max_area=50,
            max_side=10,
            max_candidates=8,
            polarity="bright",
        )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].x, 31.0, delta=1.0)
        self.assertAlmostEqual(candidates[0].y, 31.0, delta=1.0)
    def test_both_candidate_mode_keeps_bright_and_dark_spots(self) -> None:
        image = np.full((64, 64), 100, dtype=np.uint8)
        image[20:23, 20:23] = 220
        image[44:47, 44:47] = 20
        mask = np.full_like(image, 255)

        candidates, _, _, _, _ = find_particle_candidates(
            image,
            None,
            core_mask=mask,
            blackhat_kernel=9,
            sigma=3.0,
            min_area=1,
            max_area=50,
            max_side=10,
            max_candidates=8,
            polarity="both",
        )

        self.assertEqual(len(candidates), 2)

    def test_roi_must_fit_inside_source_frame(self) -> None:
        Rect(621, 40, 180, 260).validate(1280, 800)
        with self.assertRaises(ValueError):
            Rect(1200, 40, 180, 260).validate(1280, 800)

    def test_robust_threshold_ignores_values_outside_core_mask(self) -> None:
        image = np.full((8, 8), 10, dtype=np.uint8)
        image[0, 0] = 255
        mask = np.zeros_like(image)
        mask[2:6, 2:6] = 255
        threshold = robust_threshold(image, mask, sigma=4.0, floor=3.0)
        self.assertAlmostEqual(threshold, 14.0)

    def test_droplet_center_uses_measured_contour_position(self) -> None:
        background = np.zeros((120, 120), dtype=np.uint8)
        gray = background.copy()
        yy, xx = np.ogrid[:120, :120]
        gray[(xx - 43) ** 2 + (yy - 70) ** 2 <= 20**2] = 255

        observation, _, _ = find_droplet(
            gray,
            background,
            threshold=10,
            min_area=300,
            max_area=3000,
            channel_center_x=70,
            expected_radius=20,
        )

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertAlmostEqual(observation.center_x, 43.0, delta=1.0)
        self.assertNotAlmostEqual(observation.center_x, 70.0, delta=1.0)

    def test_droplet_tracker_smooths_measured_x(self) -> None:
        tracker = DropletSequenceTracker(new_droplet_jump=55.0, max_missing=3)
        first, _ = tracker.update(
            DropletObservation(43.0, 70.0, 20.0, 1200.0, 1.0, (23, 50, 41, 41)),
            frame_index=1,
        )
        second, _ = tracker.update(
            DropletObservation(47.0, 72.0, 20.0, 1200.0, 1.0, (27, 52, 41, 41)),
            frame_index=2,
        )

        assert first is not None and second is not None
        self.assertAlmostEqual(first.center_x, 43.0)
        self.assertAlmostEqual(second.center_x, 46.0)
    def test_track_records_best_candidate_frame_and_position(self) -> None:
        tracker = TemporalParticleTracker(
            max_distance=5.0,
            min_hits=2,
            max_missed=1,
            confidence_threshold=0.5,
            patch_size=16,
        )
        tracker.update([candidate(30.0, 28.0)], np.zeros((64, 64), np.uint8), 7)

        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[0]
        self.assertEqual(track.best_frame, 7)
        self.assertAlmostEqual(track.best_x, 30.0)
        self.assertAlmostEqual(track.best_y, 28.0)
    def test_temporal_gate_rejects_single_frame_response(self) -> None:
        tracker = TemporalParticleTracker(
            max_distance=5.0,
            min_hits=3,
            max_missed=1,
            confidence_threshold=0.5,
            patch_size=16,
        )
        visible, _ = tracker.update([candidate()], np.zeros((64, 64), np.uint8), 1)
        self.assertEqual(len(visible), 1)
        self.assertFalse(visible[0].confirmed)

    def test_temporal_gate_confirms_persistent_particle(self) -> None:
        tracker = TemporalParticleTracker(
            max_distance=5.0,
            min_hits=3,
            max_missed=1,
            confidence_threshold=0.5,
            patch_size=16,
        )
        gray = np.zeros((64, 64), np.uint8)
        for frame_index, offset in enumerate((0.0, 0.5, 1.0), start=1):
            visible, _ = tracker.update(
                [candidate(30.0 + offset, 28.0)],
                gray,
                frame_index,
            )
        self.assertEqual(len(tracker.tracks), 1)
        self.assertTrue(visible[0].confirmed)
        self.assertEqual(visible[0].hits, 3)


class PatchQnnTests(unittest.TestCase):
    def test_bright_residual_has_opposite_polarity_to_dark_residual(self) -> None:
        image = np.full((32, 32), 100, dtype=np.uint8)
        image[16, 16] = 220
        image[8, 8] = 20

        bright = preprocess_patch(image, "bright_residual")
        dark = preprocess_patch(image, "dark_residual")
        contrast = preprocess_patch(image, "contrast_residual")

        self.assertGreater(int(bright[16, 16]), int(bright[8, 8]))
        self.assertGreater(int(dark[8, 8]), int(dark[16, 16]))
        self.assertGreater(int(contrast[16, 16]), 0)
        self.assertGreater(int(contrast[8, 8]), 0)
    def test_patch_qnn_shape_and_parameter_budget(self) -> None:
        model = TinyQuantPatchClassifier(PatchClassifierConfig())
        output = model(torch.rand(4, 1, 32, 32))
        total, trainable = count_patch_parameters(model)
        self.assertEqual(tuple(output.shape), (4, 1))
        self.assertEqual(total, 2758)
        self.assertEqual(trainable, 2758)

    def test_patch_qnn_rejects_wrong_spatial_shape(self) -> None:
        model = TinyQuantPatchClassifier(PatchClassifierConfig())
        with self.assertRaises(ValueError):
            model(torch.rand(1, 1, 24, 24))


if __name__ == "__main__":
    unittest.main()
