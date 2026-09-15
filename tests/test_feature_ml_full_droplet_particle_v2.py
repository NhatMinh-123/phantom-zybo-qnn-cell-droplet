from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.extract_droplet_microplastic_features import CandidateData
from scripts.run_feature_ml_full_droplet_particle_v2 import (
    align_translation,
    physically_plausible,
)


class RegisteredParticleProposalTests(unittest.TestCase):
    def test_phase_registration_recovers_two_pixel_translation(self) -> None:
        historical = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(historical, (64, 64), 28, 180, 2)
        transform = np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        current = cv2.warpAffine(
            historical,
            transform,
            (128, 128),
            borderMode=cv2.BORDER_REFLECT_101,
        )

        aligned, shift, response = align_translation(
            historical,
            current,
        )

        self.assertAlmostEqual(shift[0], 2.0, delta=0.4)
        self.assertAlmostEqual(shift[1], -1.0, delta=0.4)
        self.assertGreater(response, 0.08)
        self.assertLess(
            np.mean(cv2.absdiff(aligned, current)),
            1.0,
        )

    def test_filter_keeps_dark_blob_and_rejects_edge_jitter(self) -> None:
        def candidate(**row) -> CandidateData:
            return CandidateData(
                row=row,
                patch=np.zeros((8, 8), dtype=np.uint8),
            )

        dark_blob = candidate(
            radial_distance_norm=0.7,
            local_contrast_gray=2.2,
            blackhat_max=3.0,
            temporal_max=6.0,
        )
        edge_jitter = candidate(
            radial_distance_norm=0.97,
            local_contrast_gray=8.0,
            blackhat_max=8.0,
            temporal_max=20.0,
        )
        weak_texture = candidate(
            radial_distance_norm=0.6,
            local_contrast_gray=0.3,
            blackhat_max=2.0,
            temporal_max=7.0,
        )

        self.assertTrue(physically_plausible(dark_blob))
        self.assertFalse(physically_plausible(edge_jitter))
        self.assertFalse(physically_plausible(weak_texture))


if __name__ == "__main__":
    unittest.main()
