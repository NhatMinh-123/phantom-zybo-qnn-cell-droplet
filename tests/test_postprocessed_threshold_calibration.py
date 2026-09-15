from __future__ import annotations

import unittest

from scripts.calibrate_qnn_postprocessed_thresholds import (
    combined_metrics,
    threshold_values,
)


class PostprocessedThresholdCalibrationTests(unittest.TestCase):
    def test_threshold_values_include_both_endpoints(self) -> None:
        self.assertEqual(
            threshold_values(0.80, 0.82, 0.01),
            [0.80, 0.81, 0.82],
        )

    def test_combined_metrics_use_micro_counts_and_macro_f1(self) -> None:
        rows = (
            {
                "true_positive": 8,
                "false_positive": 2,
                "false_negative": 2,
                "f1": 0.8,
            },
            {
                "true_positive": 6,
                "false_positive": 1,
                "false_negative": 4,
                "f1": 12 / 17,
            },
        )
        result = combined_metrics(rows)

        self.assertEqual(result["true_positive"], 14)
        self.assertEqual(result["false_positive"], 3)
        self.assertEqual(result["false_negative"], 6)
        self.assertAlmostEqual(float(result["f1"]), 28 / 37)
        self.assertAlmostEqual(float(result["macro_f1"]), (0.8 + 12 / 17) / 2)


if __name__ == "__main__":
    unittest.main()
