#!/usr/bin/env python3
"""Run the 15 um fixed ROI through the Arty S7 with training-exact resize."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_video_finn_uart import InputTransform
from scripts import run_video_finn_uart_sparse_realtime as runtime


def training_exact_roi(
    roi,
    canvas_width: int,
    canvas_height: int,
    content_width: int,
    content_height: int,
):
    """Leave ROI pixels untouched so prepare_image performs the sole resize.

    The 15 um QNN was trained by converting the ROI to grayscale and resizing
    it once with PIL bilinear interpolation. The legacy video driver first
    applied an OpenCV cubic resize, which is intentionally bypassed here.
    """

    if (content_width, content_height) != (canvas_width, canvas_height):
        raise ValueError(
            "15 um runtime requires full-canvas content without letterboxing"
        )
    return roi, InputTransform(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_width=canvas_width,
        content_height=canvas_height,
        offset_x=0,
        offset_y=0,
    )


def main() -> None:
    runtime.prepare_roi_input = training_exact_roi
    runtime.main()


if __name__ == "__main__":
    main()
