#!/usr/bin/env python3
"""Run the 15-micron QNN on one fixed ROI and export an auditable video."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.compare_qnn_video_roi import (
    CLASS_COLORS,
    count_classes,
    infer,
    load_runtime,
    preprocess_roi,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-x", type=int, default=560)
    parser.add_argument("--roi-y", type=int, default=342)
    parser.add_argument("--roi-width", type=int, default=256)
    parser.add_argument("--roi-height", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def draw_detection(
    frame: np.ndarray,
    detection,
    *,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    class_names: tuple[str, ...],
) -> None:
    x1, y1, x2, y2 = detection.box
    px1 = roi_x + int(round(x1 * (roi_width - 1)))
    py1 = roi_y + int(round(y1 * (roi_height - 1)))
    px2 = roi_x + int(round(x2 * (roi_width - 1)))
    py2 = roi_y + int(round(y2 * (roi_height - 1)))
    color = CLASS_COLORS.get(detection.class_id, (255, 255, 255))
    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2, cv2.LINE_AA)
    label = f"{class_names[detection.class_id]} {detection.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(0, py1 - th - 8)
    cv2.rectangle(frame, (px1, top), (px1 + tw + 8, py1), (10, 10, 10), -1)
    cv2.putText(
        frame,
        label,
        (px1 + 4, max(th + 2, py1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_header(
    frame: np.ndarray,
    *,
    frame_index: int,
    counts: list[int],
    infer_ms: float,
    total_ms: float,
    roi_width: int,
    roi_height: int,
    input_width: int,
    input_height: int,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
    model_fps = 1000.0 / infer_ms if infer_ms else 0.0
    pipeline_fps = 1000.0 / total_ms if total_ms else 0.0
    lines = (
        f"PC QNN pre-FPGA | frame {frame_index} | cell={counts[0]} droplet={counts[1]}",
        f"ROI {roi_width}x{roi_height} -> QNN {input_width}x{input_height} | "
        f"infer {infer_ms:.2f} ms ({model_fps:.1f} FPS) | pipeline {pipeline_fps:.1f} FPS",
    )
    for line, y, color in (
        (lines[0], 27, (245, 245, 245)),
        (lines[1], 55, (80, 220, 255)),
    ):
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runtime = load_runtime("15micro W4A6", args.checkpoint, args.postprocess, device)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = (args.roi_x, args.roi_y, args.roi_width, args.roi_height)
    if (
        min(roi) < 0
        or args.roi_x + args.roi_width > width
        or args.roi_y + args.roi_height > height
    ):
        raise ValueError(f"ROI {roi} is outside source {width}x{height}")

    args.output.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "pc_qnn_roi256_pre_fpga.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*args.codec),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    limit = source_frames if args.max_frames <= 0 else min(source_frames, args.max_frames)
    sample_indices = set(
        int(round(value))
        for value in np.linspace(0, max(limit - 1, 0), max(args.sample_count, 1))
    )
    rows: list[dict[str, float | int]] = []
    infer_times: list[float] = []
    decode_times: list[float] = []
    total_times: list[float] = []
    totals = [0] * len(runtime.class_names)
    start_run = time.perf_counter()
    frame_index = 0
    try:
        while frame_index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_start = time.perf_counter()
            x, y, w, h = roi
            roi_bgr = frame[y : y + h, x : x + w]
            tensor = preprocess_roi(
                roi_bgr,
                width=runtime.input_width,
                height=runtime.input_height,
                device=device,
            )
            detections, infer_ms, decode_ms = infer(runtime, tensor, device)
            counts = count_classes(detections, len(runtime.class_names))
            for class_id, value in enumerate(counts):
                totals[class_id] += value
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 210, 80), 2)
            for detection in detections:
                draw_detection(
                    frame,
                    detection,
                    roi_x=x,
                    roi_y=y,
                    roi_width=w,
                    roi_height=h,
                    class_names=runtime.class_names,
                )
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            draw_header(
                frame,
                frame_index=frame_index,
                counts=counts,
                infer_ms=infer_ms,
                total_ms=total_ms,
                roi_width=w,
                roi_height=h,
                input_width=runtime.input_width,
                input_height=runtime.input_height,
            )
            writer.write(frame)
            if frame_index in sample_indices:
                cv2.imwrite(str(samples_dir / f"frame_{frame_index:06d}.jpg"), frame)
            rows.append(
                {
                    "frame": frame_index,
                    "timestamp_s": frame_index / source_fps,
                    "cell": counts[0],
                    "droplet": counts[1],
                    "inference_ms": infer_ms,
                    "decode_ms": decode_ms,
                    "pipeline_ms_excluding_write": total_ms,
                }
            )
            infer_times.append(infer_ms)
            decode_times.append(decode_ms)
            total_times.append(total_ms)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_run
    with (args.output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    report = {
        "truth_boundary": "PC GPU QNN pre-FPGA; this is not FPGA execution",
        "source": str(args.source.resolve()),
        "output_video": str(video_path.resolve()),
        "source_metadata": {
            "width": width,
            "height": height,
            "fps": source_fps,
            "frames": source_frames,
        },
        "processed_frames": frame_index,
        "wall_clock_s": elapsed,
        "export_throughput_fps_including_video_write": frame_index / max(elapsed, 1e-9),
        "roi": {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]},
        "qnn_input": {"width": runtime.input_width, "height": runtime.input_height},
        "quantization": {"weights_bits": 4, "activations_bits": 6, "output_bits": 8},
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_epoch": runtime.checkpoint_epoch,
        "postprocess": str(runtime.postprocess_path),
        "device": str(device),
        "detections_total": dict(zip(runtime.class_names, totals)),
        "latency_ms": {
            "inference_mean": statistics.fmean(infer_times),
            "inference_median": statistics.median(infer_times),
            "inference_p95": percentile(infer_times, 95),
            "decode_mean": statistics.fmean(decode_times),
            "pipeline_excluding_write_mean": statistics.fmean(total_times),
            "pipeline_excluding_write_p95": percentile(total_times, 95),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
