from __future__ import annotations

import unittest

import numpy as np

from scripts.build_quality_augmented_dataset import (
    VARIANTS,
    apply_quality_variant,
    quality_metrics,
    stable_rng,
)


class QualityAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:96, 0:96]
        base = 110.0 + x * 0.7 + y * 0.3
        self.image = np.stack(
            (base * 0.85, base, base * 1.12),
            axis=-1,
        ).clip(0, 255).astype(np.uint8)

    def test_variants_are_deterministic_and_preserve_image_contract(self) -> None:
        for variant in VARIANTS:
            first, first_parameters = apply_quality_variant(
                self.image,
                variant,
                stable_rng(42, "sample", variant),
            )
            second, second_parameters = apply_quality_variant(
                self.image,
                variant,
                stable_rng(42, "sample", variant),
            )
            self.assertEqual(first.shape, self.image.shape)
            self.assertEqual(first.dtype, np.uint8)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first_parameters, second_parameters)
            if variant != "original":
                self.assertFalse(np.array_equal(first, self.image))

    def test_quality_metrics_are_finite(self) -> None:
        metrics = quality_metrics(self.image)
        self.assertEqual(
            set(metrics),
            {
                "brightness_mean",
                "contrast_std",
                "sharpness_laplacian_var",
                "clipped_dark_fraction",
                "clipped_bright_fraction",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
