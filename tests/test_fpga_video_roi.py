from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.detection import Detection
from scripts.run_video_finn_uart import (
    map_roi_detections,
    prepare_roi_input,
    scaled_roi,
)


class FpgaVideoRoiTests(unittest.TestCase):
    def test_roi_scales_from_reference_frame(self) -> None:
        config = {
            "reference_width": 1280,
            "reference_height": 800,
            "x": 611,
            "y": 419,
            "width": 154,
            "height": 115,
        }
        self.assertEqual(scaled_roi(1280, 800, config), (611, 419, 765, 534))
        self.assertEqual(scaled_roi(640, 400, config), (306, 210, 382, 267))

    def test_median_letterbox_matches_training_geometry(self) -> None:
        roi = np.full((115, 154, 3), (40, 80, 120), dtype=np.uint8)
        canvas, transform = prepare_roi_input(roi, 192, 192, 192, 144)
        self.assertEqual(canvas.shape, (192, 192, 3))
        self.assertEqual(transform.offset_x, 0)
        self.assertEqual(transform.offset_y, 24)
        np.testing.assert_array_equal(canvas[0, 0], (40, 80, 120))
        np.testing.assert_array_equal(canvas[24, 0], (40, 80, 120))

    def test_inverse_mapping_removes_letterbox_padding(self) -> None:
        _, transform = prepare_roi_input(
            np.full((115, 154, 3), 128, dtype=np.uint8),
            192,
            192,
            192,
            144,
        )
        detections = map_roi_detections(
            [Detection(1, 0.95, (0.0, 0.125, 1.0, 0.875))],
            ["cell", "droplet"],
            (611, 419, 765, 534),
            transform,
        )
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(
            detections[0].box,
            (611.0, 419.0, 765.0, 534.0),
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
