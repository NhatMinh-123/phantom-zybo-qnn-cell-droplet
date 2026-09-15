#!/usr/bin/env python3
"""Benchmark one fixed 256x256 camera ROI at several YOLO input sizes."""

from __future__ import annotations

import benchmark_square_roi as runner
from benchmark_roi_candidates import Candidate


FULL_ROI = Candidate("full", 0, 0, 640, 640)
runner.MODES = tuple(
    runner.Mode(f"full_{size}", FULL_ROI, size)
    for size in (640, 512, 416, 320)
)


if __name__ == "__main__":
    runner.main()
