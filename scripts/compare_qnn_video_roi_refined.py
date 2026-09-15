"""Dual-QNN video comparison with geometry-aware droplet refinement.

The QNN remains responsible for object presence and confidence. A lightweight
Hough-circle check selects the physically plausible droplet prediction, tightens
its box to the visible outer ring, and rejects cell boxes outside that droplet.
This is intended for the compact one-droplet observation ROI used by the
``droplet 100fps.mp4`` experiment.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

import compare_qnn_video_roi as runner
import compare_qnn_video_roi_exact as exact
from qnn.detection import Detection


ORIGINAL_PREPROCESS = runner.preprocess_roi
ORIGINAL_INFER = runner.infer
CURRENT_CIRCLE: tuple[float, float, float] | None = None
CURRENT_SHAPE = (0, 0)
CIRCLE_CALLS = 0
CIRCLE_FOUND = 0


def find_primary_circle(roi) -> tuple[float, float, float] | None:
    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=55,
        param1=45,
        param2=32,
        minRadius=25,
        maxRadius=75,
    )
    if circles is None:
        return None
    height, width = grayscale.shape
    candidates = []
    for center_x, center_y, radius in circles[0]:
        if not 0.18 * width <= center_x <= 0.82 * width:
            continue
        if center_y - radius < -8 or center_y + radius > height + 8:
            continue
        candidates.append((float(center_x), float(center_y), float(radius)))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item[2], -abs(item[0] - width / 2)),
    )


def preprocess_with_geometry(roi, *, width, height, device):
    global CURRENT_CIRCLE, CURRENT_SHAPE, CIRCLE_CALLS, CIRCLE_FOUND
    CURRENT_SHAPE = roi.shape[:2]
    CURRENT_CIRCLE = find_primary_circle(roi)
    CIRCLE_CALLS += 1
    if CURRENT_CIRCLE is not None:
        CIRCLE_FOUND += 1
    return ORIGINAL_PREPROCESS(roi, width=width, height=height, device=device)


def detection_center_pixels(
    item: Detection,
    width: int,
    height: int,
) -> tuple[float, float]:
    x1, y1, x2, y2 = item.box
    return ((x1 + x2) * width / 2, (y1 + y2) * height / 2)


def refine_detections(detections: list[Detection]) -> list[Detection]:
    if CURRENT_CIRCLE is None:
        return []
    height, width = CURRENT_SHAPE
    center_x, center_y, radius = CURRENT_CIRCLE
    droplets = [item for item in detections if item.class_id == 1]
    if not droplets:
        return []

    ranked = []
    for item in droplets:
        predicted_x, predicted_y = detection_center_pixels(item, width, height)
        distance = math.hypot(predicted_x - center_x, predicted_y - center_y)
        ranked.append((distance, -item.confidence, item))
    distance, _, selected = min(ranked, key=lambda value: (value[0], value[1]))
    if distance > max(55.0, radius * 1.25):
        return []

    margin_radius = radius * 1.04
    droplet_box = (
        max(0.0, (center_x - margin_radius) / width),
        max(0.0, (center_y - margin_radius) / height),
        min(1.0, (center_x + margin_radius) / width),
        min(1.0, (center_y + margin_radius) / height),
    )
    refined = [
        Detection(
            class_id=selected.class_id,
            confidence=selected.confidence,
            box=droplet_box,
        )
    ]
    cell_radius = radius * 0.92
    for item in detections:
        if item.class_id != 0:
            continue
        cell_x, cell_y = detection_center_pixels(item, width, height)
        if math.hypot(cell_x - center_x, cell_y - center_y) <= cell_radius:
            refined.append(item)
    return sorted(refined, key=lambda item: item.confidence, reverse=True)


def infer_and_refine(runtime, tensor, device):
    detections, inference_ms, decode_ms = ORIGINAL_INFER(runtime, tensor, device)
    return refine_detections(detections), inference_ms, decode_ms


def make_refined_panel(
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
    panel = exact.make_panel_with_exact_geometry(
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
    status = "circle matched" if CURRENT_CIRCLE is not None else "no complete droplet"
    text = (
        f"QNN {runtime.input_width}x{runtime.input_height} + geometry refine "
        f"| {status}"
    )
    cv2.putText(
        panel,
        text,
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 230, 160) if CURRENT_CIRCLE is not None else (150, 150, 150),
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
    processed = int(payload["source"]["processed_frames"])
    warmup_probe = 1 if CIRCLE_CALLS > processed else 0
    found = max(0, CIRCLE_FOUND - warmup_probe)
    payload["roi"]["reference"] = (
        "Compact vertical microchannel region matched to the user-provided image."
    )
    payload["geometry_refinement"] = {
        "purpose": (
            "Select the QNN droplet prediction supported by a visible circular "
            "outer ring, tighten its box, and keep only cells inside that ring."
        ),
        "does_not_create_model_detections": True,
        "circle_supported_frames": found,
        "processed_frames": processed,
        "circle_supported_fraction": found / processed if processed else 0.0,
        "hough": {
            "dp": 1.2,
            "minimum_distance": 55,
            "canny_threshold": 45,
            "accumulator_threshold": 32,
            "minimum_radius": 25,
            "maximum_radius": 75,
        },
    }
    payload["runtime"]["interpretation"] = (
        "Each QNN inference is independently faster than the encoded 30 FPS "
        "source. The side-by-side export runs both models and writes three video "
        "streams, so its export FPS is not the single-model realtime FPS."
    )
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    runner.preprocess_roi = preprocess_with_geometry
    runner.infer = infer_and_refine
    runner.make_panel = make_refined_panel
    output = output_directory()
    runner.main()
    annotate_summary(output)


if __name__ == "__main__":
    main()
