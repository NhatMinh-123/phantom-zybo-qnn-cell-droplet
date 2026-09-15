from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_cell_droplet_realtime import (  # noqa: E402
    ClassAwareTracker,
    InputTransform,
    ObjectDetection,
    map_box_to_frame,
    prepare_model_input,
)


def detection(center_x: float) -> ObjectDetection:
    box = np.array([center_x - 10.0, 120.0, center_x + 10.0, 150.0])
    return ObjectDetection(0, 0.9, box, box.copy())


class InputTransformTests(unittest.TestCase):
    def test_median_letterbox_and_inverse_mapping(self) -> None:
        roi = np.full((115, 154, 3), 180, dtype=np.uint8)
        model_input, transform = prepare_model_input(roi, 384, 384, 384, 288)
        self.assertEqual(model_input.shape, (384, 384, 3))
        self.assertEqual(transform.offset_x, 0)
        self.assertEqual(transform.offset_y, 48)
        frame_box = map_box_to_frame(
            np.array([0.0, 48.0, 384.0, 336.0]),
            (611, 419, 765, 534),
            transform,
        )
        np.testing.assert_allclose(frame_box, [611, 419, 765, 534])


class TrackerTests(unittest.TestCase):
    def make_tracker(self) -> ClassAwareTracker:
        return ClassAwareTracker(
            model_width=384,
            max_misses=3,
            max_center_distance=0.5,
            count_direction="left_to_right",
            count_hysteresis=0.04,
            minimum_hits=3,
        )

    def test_counts_left_to_right_once_after_hysteresis(self) -> None:
        tracker = self.make_tracker()
        events = []
        for frame_index, center_x in enumerate((100.0, 170.0, 220.0, 260.0, 300.0)):
            item = detection(center_x)
            tracker.update([item], frame_index, count_line_x=230.0)
            events.append(item.crossed)
        self.assertEqual(events.count(True), 1)

    def test_does_not_count_reverse_motion(self) -> None:
        tracker = self.make_tracker()
        events = []
        for frame_index, center_x in enumerate((300.0, 270.0, 220.0, 170.0, 100.0)):
            item = detection(center_x)
            tracker.update([item], frame_index, count_line_x=230.0)
            events.append(item.crossed)
        self.assertNotIn(True, events)


if __name__ == "__main__":
    unittest.main()
