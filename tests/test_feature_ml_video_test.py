from __future__ import annotations

import unittest

import numpy as np

from scripts.extract_droplet_microplastic_features import (
    CandidateData,
    CandidateTrack,
)
from scripts.run_feature_ml_video_test import (
    candidate_global_box,
    suppress_nearby_candidates,
    track_probability,
    track_record,
)


class FeatureMlVideoTestRunnerTests(unittest.TestCase):
    def test_candidate_box_maps_from_patch_to_source_frame(self) -> None:
        row = {
            "bbox_x": 60,
            "bbox_y": 62,
            "bbox_width": 5,
            "bbox_height": 7,
        }

        box = candidate_global_box(
            row,
            droplet_x=700.0,
            droplet_y=200.0,
            processing_size=128,
        )

        self.assertEqual(box, (696, 198, 5, 7))

    def test_track_confirmation_requires_probability_and_hits(self) -> None:
        track = CandidateTrack(
            track_id=4,
            sequence_id=2,
            first_frame=10,
            last_frame=11,
            x=64.0,
            y=64.0,
            observations=[
                {"ml_probability": 0.80},
                {"ml_probability": 0.60},
            ],
        )

        self.assertAlmostEqual(track_probability(track), 0.70)
        accepted = track_record(
            track,
            threshold=0.50,
            minimum_hits=2,
        )
        rejected = track_record(
            track,
            threshold=0.75,
            minimum_hits=2,
        )

        self.assertEqual(accepted["predicted_particle"], 1)
        self.assertEqual(rejected["predicted_particle"], 0)

    def test_spatial_suppression_keeps_best_nearby_candidate(self) -> None:
        def candidate(x: float, y: float) -> CandidateData:
            return CandidateData(
                row={"center_x_px": x, "center_y_px": y},
                patch=np.zeros((32, 32), dtype=np.uint8),
            )

        candidates = [
            candidate(60.0, 60.0),
            candidate(63.0, 62.0),
            candidate(80.0, 80.0),
        ]
        probabilities = np.array([0.40, 0.90, 0.70])

        kept, kept_probabilities = suppress_nearby_candidates(
            candidates,
            probabilities,
            minimum_distance=8.0,
        )

        self.assertEqual(len(kept), 2)
        self.assertEqual(
            [item.row["center_x_px"] for item in kept],
            [63.0, 80.0],
        )
        np.testing.assert_allclose(kept_probabilities, [0.90, 0.70])

        top_one, top_probability = suppress_nearby_candidates(
            candidates,
            probabilities,
            minimum_distance=0.0,
            maximum_candidates=1,
        )
        self.assertEqual(len(top_one), 1)
        self.assertEqual(top_one[0].row["center_x_px"], 63.0)
        np.testing.assert_allclose(top_probability, [0.90])


if __name__ == "__main__":
    unittest.main()
