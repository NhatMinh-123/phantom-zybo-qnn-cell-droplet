"""Run the dual-QNN video comparison with a vertical-to-horizontal ROI rotation.

Both deployed QNNs were trained on horizontal microchannels. The new source
video has a vertical channel, so this wrapper rotates the cropped ROI clockwise
for inference and maps every decoded box back to the original vertical view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

import compare_qnn_video_roi as runner
from qnn.detection import Detection


ORIGINAL_PREPROCESS = runner.preprocess_roi
ORIGINAL_INFER = runner.infer


def preprocess_rotated(roi, *, width, height, device):
    horizontal_roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    return ORIGINAL_PREPROCESS(
        horizontal_roi,
        width=width,
        height=height,
        device=device,
    )


def map_clockwise_box_back(item: Detection) -> Detection:
    x1, y1, x2, y2 = item.box
    return Detection(
        class_id=item.class_id,
        confidence=item.confidence,
        box=(y1, 1.0 - x2, y2, 1.0 - x1),
    )


def infer_and_map(runtime, tensor, device):
    detections, inference_ms, decode_ms = ORIGINAL_INFER(runtime, tensor, device)
    return (
        [map_clockwise_box_back(item) for item in detections],
        inference_ms,
        decode_ms,
    )


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError("--output is required") from error


def annotate_outputs(output: Path) -> None:
    original_path = output / "selected_roi_first_frame.jpg"
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if original is not None:
        cv2.imwrite(
            str(output / "selected_roi_model_input_orientation.jpg"),
            cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE),
        )

    summary_path = output / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["roi"]["inference_rotation"] = "90_degrees_clockwise"
    payload["runtime"]["orientation_note"] = (
        "The vertical source ROI is rotated clockwise to match the horizontal "
        "training distribution. Decoded boxes are mapped back to the vertical "
        "source view before drawing."
    )
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    runner.preprocess_roi = preprocess_rotated
    runner.infer = infer_and_map
    output = output_directory()
    runner.main()
    annotate_outputs(output)


if __name__ == "__main__":
    main()
