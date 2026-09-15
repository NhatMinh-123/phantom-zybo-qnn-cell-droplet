from __future__ import annotations

import unittest

from scripts.bright_round_feature_gate import (
    BrightRoundGateConfig,
    evaluate_bright_round_candidate,
)


class BrightRoundFeatureGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BrightRoundGateConfig()
        self.strong_round_particle = {
            "bbox_width": 8,
            "bbox_height": 7,
            "blackhat_threshold": 4.0,
            "blackhat_max": 32.0,
            "blackhat_mean": 12.0,
            "pixel_area": 44.0,
            "circularity": 0.72,
            "solidity": 0.88,
            "extent": 0.78,
            "radial_distance_norm": 0.25,
        }

    def test_strong_compact_round_response_passes(self) -> None:
        result = evaluate_bright_round_candidate(
            self.strong_round_particle,
            self.config,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())

    def test_dim_response_is_rejected(self) -> None:
        row = dict(self.strong_round_particle)
        row.update(blackhat_max=10.0, blackhat_mean=4.0)

        result = evaluate_bright_round_candidate(row, self.config)

        self.assertFalse(result.passed)
        self.assertIn("dim_peak", result.reasons)
        self.assertIn("dim_mean", result.reasons)

    def test_elongated_diffuse_rim_is_rejected(self) -> None:
        row = dict(self.strong_round_particle)
        row.update(
            bbox_width=18,
            bbox_height=3,
            circularity=0.18,
            solidity=0.48,
            extent=0.30,
        )

        result = evaluate_bright_round_candidate(row, self.config)

        self.assertFalse(result.passed)
        self.assertIn("elongated", result.reasons)
        self.assertIn("not_circular", result.reasons)
        self.assertIn("diffuse", result.reasons)


if __name__ == "__main__":
    unittest.main()