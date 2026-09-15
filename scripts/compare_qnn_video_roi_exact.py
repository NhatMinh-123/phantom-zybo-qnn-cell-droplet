"""Dual-QNN comparison for the exact compact ROI requested by the user.

This wrapper preserves the original vertical orientation and corrects the
on-video preprocessing label so it reports the real source ROI dimensions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

import compare_qnn_video_roi as runner


ORIGINAL_MAKE_PANEL = runner.make_panel


def make_panel_with_exact_geometry(
    roi,
    detections,
    runtime,
    *,
    frame_index,
    timestamp,
    inference_ms,
    decode_ms,
    display_size,
):
    panel = ORIGINAL_MAKE_PANEL(
        roi,
        detections,
        runtime,
        frame_index=frame_index,
        timestamp=timestamp,
        inference_ms=inference_ms,
        decode_ms=decode_ms,
        display_size=display_size,
    )
    cv2.rectangle(panel, (0, 62), (panel.shape[1] - 1, 95), (0, 0, 0), -1)
    text = (
        f"decode={decode_ms:5.2f} ms | ROI {roi.shape[1]}x{roi.shape[0]} "
        f"-> QNN {runtime.input_width}x{runtime.input_height}"
    )
    cv2.putText(
        panel,
        text,
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError("--output is required") from error


def annotate_summary(output: Path) -> None:
    summary_path = output / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["roi"]["reference"] = (
        "Compact vertical microchannel region matched to the user-provided image."
    )
    payload["runtime"]["interpretation"] = (
        "Each model inference is independently faster than the encoded 30 FPS "
        "source. The side-by-side export runs both models and writes three video "
        "streams, so its offline export FPS is not the single-model realtime FPS."
    )
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    runner.make_panel = make_panel_with_exact_geometry
    output = output_directory()
    runner.main()
    annotate_summary(output)


if __name__ == "__main__":
    main()
