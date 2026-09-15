import unittest

import cv2
import numpy as np

from scripts.dual_roi_rtl_classical import (
    RtlDownstreamGate,
    RtlGateConfig,
    cell_gate_candidates,
    droplet_ring_gate_score,
)


def circle_frame(center_x: int = 48) -> np.ndarray:
    image = np.full((96, 96), 190, dtype=np.uint8)
    cv2.circle(image, (center_x, 48), 25, 75, 3)
    return image


def cell_frame() -> np.ndarray:
    image = np.full((96, 96), 170, dtype=np.uint8)
    cv2.circle(image, (48, 38), 3, 70, 1)
    image[38, 48] = 245
    cv2.circle(image, (49, 64), 3, 80, 1)
    image[64, 49] = 240
    return image


class RtlClassicalReferenceTests(unittest.TestCase):
    def test_dark_ring_scores_above_threshold_at_gate(self) -> None:
        score, center_y, radius = droplet_ring_gate_score(circle_frame())
        self.assertGreaterEqual(score, RtlGateConfig().droplet_score_threshold)
        self.assertLessEqual(abs(center_y - 48), 2)
        self.assertLessEqual(abs(radius - 25), 3)

    def test_two_radial_cell_peaks_are_retained(self) -> None:
        candidates = cell_gate_candidates(cell_frame())
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0].response, 40)
        self.assertGreaterEqual(abs(candidates[0].y - candidates[1].y), 6)

    def test_droplet_temporal_peak_emits_once(self) -> None:
        gate = RtlDownstreamGate()
        events = []
        for frame_index, center_x in enumerate((39, 44, 48, 52, 58, 65)):
            frame_events, _ = gate.process_frame(circle_frame(center_x), frame_index)
            events.extend(item for item in frame_events if item.class_name == "droplet")
        self.assertEqual(len(events), 1)

    def test_cell_requires_two_hits_and_emits_once(self) -> None:
        gate = RtlDownstreamGate()
        first, _ = gate.process_frame(cell_frame(), 0)
        second, _ = gate.process_frame(cell_frame(), 1)
        third, _ = gate.process_frame(cell_frame(), 2)
        self.assertFalse(any(item.class_name == "cell" for item in first))
        self.assertEqual(sum(item.class_name == "cell" for item in second), 2)
        self.assertFalse(any(item.class_name == "cell" for item in third))


if __name__ == "__main__":
    unittest.main()
