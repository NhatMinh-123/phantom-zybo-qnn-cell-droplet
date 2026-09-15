from __future__ import annotations

import unittest

from scripts.run_video_finn_uart import FrameDetection
from scripts.run_video_finn_uart_sparse_realtime import (
    HardwareResult,
    estimate_detection_velocity,
)


def result(
    frame_index: int,
    boxes: tuple[tuple[float, float, float, float], ...],
) -> HardwareResult:
    return HardwareResult(
        frame_index=frame_index,
        completed_at=0.0,
        detections=tuple(
            FrameDetection(
                class_id=1,
                class_name="droplet",
                confidence=0.9,
                box=box,
            )
            for box in boxes
        ),
        stream_cycles=1,
        stream_fps=1.0,
        round_trip_seconds=0.01,
        sparse_bytes=8,
    )


class SparseRealtimeMotionTests(unittest.TestCase):
    def test_estimates_median_motion_per_source_frame(self) -> None:
        previous = result(
            10,
            (
                (10.0, 20.0, 30.0, 40.0),
                (60.0, 22.0, 80.0, 42.0),
                (110.0, 21.0, 130.0, 41.0),
            ),
        )
        current = result(
            12,
            (
                (14.0, 22.0, 34.0, 42.0),
                (64.0, 24.0, 84.0, 44.0),
                (114.0, 23.0, 134.0, 43.0),
            ),
        )

        velocity_x, velocity_y, matches = estimate_detection_velocity(
            previous,
            current,
        )

        self.assertEqual(matches, 3)
        self.assertAlmostEqual(velocity_x, 2.0)
        self.assertAlmostEqual(velocity_y, 1.0)

    def test_returns_zero_without_a_valid_pair(self) -> None:
        velocity_x, velocity_y, matches = estimate_detection_velocity(
            None,
            result(1, ((10.0, 20.0, 30.0, 40.0),)),
        )

        self.assertEqual((velocity_x, velocity_y, matches), (0.0, 0.0, 0))

    def test_rejects_implausibly_large_motion(self) -> None:
        previous = result(1, ((0.0, 0.0, 10.0, 10.0),))
        current = result(2, ((200.0, 0.0, 210.0, 10.0),))

        velocity_x, velocity_y, matches = estimate_detection_velocity(
            previous,
            current,
        )

        self.assertEqual((velocity_x, velocity_y, matches), (0.0, 0.0, 0))


if __name__ == "__main__":
    unittest.main()
