#!/usr/bin/env python3
"""Run one compact video ROI through the Arty S7-25 FINN UART bitstream."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.detection import Detection
from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_output_axis,
)
from scripts.send_frame_finn_uart import (
    RESPONSE_HEADER_SIZE,
    STATUS_NAMES,
    checksum16,
    checkpoint_output_tensor,
    make_request,
    parse_response_header,
    read_exact,
    resolve_checkpoint,
)


DEFAULT_VIDEO = ROOT / "data" / "raw" / "3.4.mp4"
DEFAULT_MANIFEST = (
    ROOT
    / "exports"
    / "qnn_cell_droplet_v2"
    / "tiny_detector_192x192_w4a6_fpga.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "fpga_video_single_roi_w4a6_square192"
DEFAULT_ROI_CONFIG = (
    ROOT / "models" / "cell_droplet_yolo11n" / "roi384_baseline_config.json"
)


@dataclass(frozen=True)
class FrameDetection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class InputTransform:
    canvas_width: int
    canvas_height: int
    content_width: int
    content_height: int
    offset_x: int
    offset_y: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="QAT checkpoint; defaults to model.checkpoint from the manifest",
    )
    parser.add_argument(
        "--skip-checkpoint-compare",
        action="store_true",
        help="Decode FPGA output without exact checkpoint comparison",
    )
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--start-sec", type=float, default=3.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--search-radius", type=int, default=12)
    parser.add_argument("--search-step", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--display-seconds", type=float, default=1.0)
    return parser.parse_args()


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read source frame {frame_index + 1}")
    return frame


def sharpness(
    frame: np.ndarray, roi_geometry: tuple[int, int, int, int]
) -> float:
    x1, y1, x2, y2 = roi_geometry
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_frames(
    capture: cv2.VideoCapture,
    first: int,
    last: int,
    count: int,
    search_radius: int,
    search_step: int,
    roi_geometry: tuple[int, int, int, int],
) -> list[int]:
    if count <= 0:
        raise ValueError("samples must be positive")
    targets = np.linspace(first, last, count).round().astype(int)
    selected: list[int] = []
    used: set[int] = set()
    for target in targets:
        candidates = sorted(
            {
                max(first, min(last, int(target) + offset))
                for offset in range(-search_radius, search_radius + 1, search_step)
            }
            | {int(target)}
        )
        ranked = sorted(
            (
                (
                    sharpness(read_frame(capture, candidate), roi_geometry),
                    candidate,
                )
                for candidate in candidates
                if candidate not in used
            ),
            reverse=True,
        )
        if not ranked:
            raise RuntimeError(f"No frame candidate near {target + 1}")
        selected.append(ranked[0][1])
        used.add(ranked[0][1])
    return selected


def scaled_roi(
    frame_width: int,
    frame_height: int,
    roi_config: dict[str, Any],
) -> tuple[int, int, int, int]:
    reference_width = int(roi_config["reference_width"])
    reference_height = int(roi_config["reference_height"])
    scale_x = frame_width / reference_width
    scale_y = frame_height / reference_height
    x1 = int(round(float(roi_config["x"]) * scale_x))
    y1 = int(round(float(roi_config["y"]) * scale_y))
    x2 = int(round((float(roi_config["x"]) + float(roi_config["width"])) * scale_x))
    y2 = int(round((float(roi_config["y"]) + float(roi_config["height"])) * scale_y))
    if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
        raise ValueError(f"ROI {(x1, y1, x2, y2)} is outside the video frame")
    return x1, y1, x2, y2


def prepare_roi_input(
    roi: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    content_width: int,
    content_height: int,
) -> tuple[np.ndarray, InputTransform]:
    resized = cv2.resize(
        roi,
        (content_width, content_height),
        interpolation=cv2.INTER_CUBIC,
    )
    median_color = np.median(roi.reshape(-1, roi.shape[2]), axis=0).astype(np.uint8)
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = median_color
    offset_x = (canvas_width - content_width) // 2
    offset_y = (canvas_height - content_height) // 2
    canvas[
        offset_y : offset_y + content_height,
        offset_x : offset_x + content_width,
    ] = resized
    return canvas, InputTransform(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_width=content_width,
        content_height=content_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def transact(
    port: Any,
    payload: bytes,
    manifest: dict[str, Any],
    frame_id: int,
    timeout: float,
) -> tuple[np.ndarray, int, float]:
    request = make_request(payload, frame_id)
    started = time.perf_counter()
    written = port.write(request)
    port.flush()
    if written != len(request):
        raise IOError(f"Only wrote {written} of {len(request)} request bytes")

    header = read_exact(port, RESPONSE_HEADER_SIZE, timeout)
    response_frame_id, status, payload_length, accelerator_cycles = (
        parse_response_header(header)
    )
    if response_frame_id != frame_id:
        raise ValueError(
            f"Frame ID mismatch: sent {frame_id}, got {response_frame_id}"
        )
    if status != 0:
        raise RuntimeError(
            f"FPGA rejected tile: {STATUS_NAMES.get(status, f'UNKNOWN_{status}')}"
        )
    expected_length = int(manifest["fpga_core"]["output_stream"]["bytes_per_frame"])
    if payload_length != expected_length:
        raise ValueError(
            f"Expected {expected_length} output bytes, got {payload_length}"
        )
    response = read_exact(port, payload_length, timeout)
    response_checksum = int.from_bytes(read_exact(port, 2, timeout), "little")
    if response_checksum != checksum16(response):
        raise ValueError("FPGA output checksum mismatch")
    return (
        unpack_output_axis(response, manifest),
        accelerator_cycles,
        time.perf_counter() - started,
    )


def map_roi_detections(
    decoded: list[Detection],
    class_names: list[str],
    roi_geometry: tuple[int, int, int, int],
    transform: InputTransform,
) -> list[FrameDetection]:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    scale_x = (roi_x2 - roi_x1) / transform.content_width
    scale_y = (roi_y2 - roi_y1) / transform.content_height
    output: list[FrameDetection] = []
    for item in decoded:
        canvas_box = np.asarray(
            (
                item.box[0] * transform.canvas_width,
                item.box[1] * transform.canvas_height,
                item.box[2] * transform.canvas_width,
                item.box[3] * transform.canvas_height,
            ),
            dtype=np.float32,
        )
        content_box = canvas_box.copy()
        content_box[[0, 2]] = np.clip(
            content_box[[0, 2]] - transform.offset_x,
            0,
            transform.content_width,
        )
        content_box[[1, 3]] = np.clip(
            content_box[[1, 3]] - transform.offset_y,
            0,
            transform.content_height,
        )
        if content_box[2] <= content_box[0] or content_box[3] <= content_box[1]:
            continue
        global_box = (
            roi_x1 + float(content_box[0]) * scale_x,
            roi_y1 + float(content_box[1]) * scale_y,
            roi_x1 + float(content_box[2]) * scale_x,
            roi_y1 + float(content_box[3]) * scale_y,
        )
        output.append(
            FrameDetection(
                class_id=item.class_id,
                class_name=class_names[item.class_id],
                confidence=item.confidence,
                box=global_box,
            )
        )
    return output


def draw_result(
    frame: np.ndarray,
    detections: list[FrameDetection],
    frame_index: int,
    source_time: float,
    sample_index: int,
    sample_count: int,
    roi_geometry: tuple[int, int, int, int],
    accelerator_fps: float,
    hardware_exact: bool | None,
) -> np.ndarray:
    output = frame.copy()
    colors = {"cell": (40, 40, 235), "droplet": (235, 130, 25)}
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    cv2.rectangle(
        output,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (40, 210, 40),
        2,
    )
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item.box)
        color = colors.get(item.class_name, (40, 200, 40))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if item.class_name != "droplet":
            continue
        label = f"{item.class_name} {item.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )
        label_y = max(label_height + baseline + 2, y1 - 3)
        cv2.rectangle(
            output,
            (x1, label_y - label_height - baseline - 2),
            (x1 + label_width + 4, label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    counts = {
        name: sum(item.class_name == name for item in detections)
        for name in ("cell", "droplet")
    }
    banner = (
        f"Arty S7-25 FPGA | sample {sample_index}/{sample_count} | "
        f"source frame {frame_index + 1} ({source_time:.2f}s) | "
        f"single ROI {roi_x2 - roi_x1}x{roi_y2 - roi_y1}"
    )
    equivalence = (
        "not compared"
        if hardware_exact is None
        else "EXACT"
        if hardware_exact
        else "MISMATCH"
    )
    result_banner = (
        f"accelerator={accelerator_fps:.2f} FPS | "
        f"FPGA=checkpoint {equivalence} | "
        f"cell={counts['cell']} droplet={counts['droplet']}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (15, 15, 15), -1)
    cv2.putText(
        output,
        banner,
        (12, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        result_banner,
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (70, 230, 70) if hardware_exact is not False else (40, 40, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Offline UART transport; inference tensor produced by FPGA",
        (12, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 220),
        1,
        cv2.LINE_AA,
    )
    zoom_scale = 3
    roi_view = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    zoomed = cv2.resize(
        roi_view,
        (
            (roi_x2 - roi_x1) * zoom_scale,
            (roi_y2 - roi_y1) * zoom_scale,
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    for item in detections:
        x1, y1, x2, y2 = (
            int(round((item.box[0] - roi_x1) * zoom_scale)),
            int(round((item.box[1] - roi_y1) * zoom_scale)),
            int(round((item.box[2] - roi_x1) * zoom_scale)),
            int(round((item.box[3] - roi_y1) * zoom_scale)),
        )
        color = colors.get(item.class_name, (40, 200, 40))
        cv2.rectangle(zoomed, (x1, y1), (x2, y2), color, 2)
        if item.class_name == "droplet":
            cv2.putText(
                zoomed,
                f"{item.confidence:.2f}",
                (x1 + 2, max(14, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    inset_x = output.shape[1] - zoomed.shape[1] - 12
    inset_y = 68
    output[
        inset_y : inset_y + zoomed.shape[0],
        inset_x : inset_x + zoomed.shape[1],
    ] = zoomed
    cv2.rectangle(
        output,
        (inset_x - 2, inset_y - 2),
        (inset_x + zoomed.shape[1] + 1, inset_y + zoomed.shape[0] + 1),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        "ROI zoom x3",
        (inset_x, inset_y + zoomed.shape[0] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    thumbnails = [cv2.resize(image, (640, 400)) for image in images]
    rows = []
    for index in range(0, len(thumbnails), 2):
        pair = thumbnails[index : index + 2]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 245))
        rows.append(np.hstack(pair))
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    checkpoint_path = None
    if not args.skip_checkpoint_compare:
        checkpoint_path = resolve_checkpoint(manifest, args.checkpoint)
    runtime_config = json.loads(args.roi_config.read_text(encoding="utf-8"))
    roi_config = runtime_config["roi"]
    preprocess_config = runtime_config["preprocess"]
    video_path = args.video.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_geometry = scaled_roi(width, height, roi_config)
    resize = manifest["preprocessing"]["resize"]
    canvas_width = int(resize["width"])
    canvas_height = int(resize["height"])
    source_canvas_width = int(preprocess_config["content_width"])
    source_canvas_height = int(preprocess_config["content_height"])
    source_image_size = int(runtime_config["image_size"])
    content_width = int(round(canvas_width * source_canvas_width / source_image_size))
    content_height = int(round(canvas_height * source_canvas_height / source_image_size))
    if content_width > canvas_width or content_height > canvas_height:
        raise ValueError("Letterbox content is larger than the FPGA input canvas")

    first = max(0, int(round(args.start_sec * fps)))
    end_sec = args.end_sec
    if end_sec is None:
        end_sec = max(args.start_sec, total_frames / fps - 3.0)
    last = min(total_frames - 1, int(round(end_sec * fps)))
    selected = select_frames(
        capture,
        first,
        last,
        args.samples,
        args.search_radius,
        args.search_step,
        roi_geometry,
    )

    import serial

    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    annotated_frames: list[np.ndarray] = []
    detection_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    tile_transaction = 0
    started_all = time.perf_counter()
    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
        write_timeout=10.0,
    ) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        for sample_index, source_index in enumerate(selected, start=1):
            frame = read_frame(capture, source_index)
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            model_input, transform = prepare_roi_input(
                roi,
                canvas_width,
                canvas_height,
                content_width,
                content_height,
            )
            codes = prepare_image(model_input, manifest, array_is_bgr=True)
            payload = pack_input_axis(codes, manifest)
            tile_transaction += 1
            tensor, accelerator_cycles, elapsed = transact(
                port,
                payload,
                manifest,
                tile_transaction,
                args.timeout,
            )
            accelerator_fps = (
                float(manifest["fpga_core"]["clock_hz"]) / accelerator_cycles
            )
            hardware_exact: bool | None = None
            mismatch_count = 0
            maximum_absolute_error = 0
            if checkpoint_path is not None:
                golden = checkpoint_output_tensor(
                    codes,
                    manifest,
                    checkpoint_path,
                )
                delta = tensor.astype(np.int64) - golden.astype(np.int64)
                mismatch_count = int(np.count_nonzero(delta))
                maximum_absolute_error = int(np.max(np.abs(delta)))
                hardware_exact = mismatch_count == 0
                np.save(
                    tensors_dir / f"sample_{sample_index:02d}_checkpoint.npy",
                    golden,
                )
            np.save(
                tensors_dir / f"sample_{sample_index:02d}_fpga.npy",
                tensor,
            )
            decoded = decode_output_tensor(tensor, manifest)[0]
            frame_detections = map_roi_detections(
                decoded,
                class_names,
                roi_geometry,
                transform,
            )
            print(
                f"sample={sample_index}/{len(selected)} "
                f"roi=1/1 "
                f"fpga={accelerator_cycles / manifest['fpga_core']['clock_hz'] * 1000:.3f}ms "
                f"fps={accelerator_fps:.3f} "
                f"uart={elapsed:.3f}s detections={len(decoded)} "
                f"exact={hardware_exact}",
                flush=True,
            )
            source_time = source_index / fps
            annotated = draw_result(
                frame,
                frame_detections,
                source_index,
                source_time,
                sample_index,
                len(selected),
                roi_geometry,
                accelerator_fps,
                hardware_exact,
            )
            annotated_path = frames_dir / f"sample_{sample_index:02d}.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            annotated_frames.append(annotated)
            for item in frame_detections:
                detection_rows.append(
                    {
                        "sample": sample_index,
                        "source_frame": source_index + 1,
                        "source_time_sec": f"{source_time:.6f}",
                        "class_id": item.class_id,
                        "class": item.class_name,
                        "confidence": f"{item.confidence:.6f}",
                        "x1": f"{item.box[0]:.3f}",
                        "y1": f"{item.box[1]:.3f}",
                        "x2": f"{item.box[2]:.3f}",
                        "y2": f"{item.box[3]:.3f}",
                    }
                )
            sample_rows.append(
                {
                    "sample": sample_index,
                    "source_frame": source_index + 1,
                    "source_time_sec": f"{source_time:.6f}",
                    "cells": sum(
                        item.class_name == "cell" for item in frame_detections
                    ),
                    "droplets": sum(
                        item.class_name == "droplet" for item in frame_detections
                    ),
                    "mean_accelerator_ms": f"{accelerator_cycles / manifest['fpga_core']['clock_hz'] * 1000:.6f}",
                    "accelerator_fps": f"{accelerator_fps:.6f}",
                    "uart_seconds": f"{elapsed:.6f}",
                    "hardware_exact": hardware_exact,
                    "mismatch_count": mismatch_count,
                    "maximum_absolute_error": maximum_absolute_error,
                    "image": annotated_path.relative_to(output_dir).as_posix(),
                }
            )
            comparison_rows.append(
                {
                    "sample": sample_index,
                    "source_frame": source_index + 1,
                    "hardware_exact": hardware_exact,
                    "mismatch_count": mismatch_count,
                    "maximum_absolute_error": maximum_absolute_error,
                    "accelerator_cycles": accelerator_cycles,
                    "accelerator_fps": f"{accelerator_fps:.6f}",
                }
            )
    capture.release()

    output_video = output_dir / "fpga_video_result.mp4"
    display_fps = 15.0
    repeat_count = max(1, int(round(args.display_seconds * display_fps)))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        display_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")
    for frame in annotated_frames:
        for _ in range(repeat_count):
            writer.write(frame)
    writer.release()
    make_contact_sheet(annotated_frames, output_dir / "contact_sheet.jpg")

    for name, rows in (
        ("detections.csv", detection_rows),
        ("samples.csv", sample_rows),
        ("hardware_comparison.csv", comparison_rows),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer_csv.writeheader()
                writer_csv.writerows(rows)

    compared_rows = [
        row for row in comparison_rows if row["hardware_exact"] is not None
    ]
    frames_exact = sum(bool(row["hardware_exact"]) for row in compared_rows)
    all_exact = bool(compared_rows) and frames_exact == len(compared_rows)
    accelerator_fps_values = [
        float(row["accelerator_fps"]) for row in sample_rows
    ]
    report = {
        "video": str(video_path),
        "video_frames": total_frames,
        "video_fps": fps,
        "video_size": [width, height],
        "selected_source_frames": [index + 1 for index in selected],
        "samples": len(selected),
        "roi": {
            "x": roi_geometry[0],
            "y": roi_geometry[1],
            "width": roi_geometry[2] - roi_geometry[0],
            "height": roi_geometry[3] - roi_geometry[1],
        },
        "fpga_input": [canvas_width, canvas_height],
        "letterbox_content": [content_width, content_height],
        "tiles_per_sample": 1,
        "fpga_transactions": tile_transaction,
        "port": args.port,
        "baud": args.baud,
        "elapsed_seconds": time.perf_counter() - started_all,
        "mean_accelerator_ms": float(
            np.mean([float(row["mean_accelerator_ms"]) for row in sample_rows])
        ),
        "accelerator_fps": {
            "minimum": float(np.min(accelerator_fps_values)),
            "mean": float(np.mean(accelerator_fps_values)),
            "maximum": float(np.max(accelerator_fps_values)),
            "target": 50.0,
            "target_met": float(np.min(accelerator_fps_values)) >= 50.0,
        },
        "hardware_vs_checkpoint": {
            "enabled": checkpoint_path is not None,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "frames_compared": len(compared_rows),
            "frames_exact": frames_exact,
            "all_exact": all_exact if compared_rows else None,
            "mismatch_count_total": sum(
                int(row["mismatch_count"]) for row in compared_rows
            ),
            "maximum_absolute_error": max(
                (
                    int(row["maximum_absolute_error"])
                    for row in compared_rows
                ),
                default=0,
            ),
        },
        "total_detections": len(detection_rows),
        "output_video": output_video.name,
        "note": (
            "All tensors were produced by the Arty S7-25 from one compact "
            "median-letterboxed ROI. The 115200-baud UART transport is "
            "offline validation, not realtime video I/O."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if compared_rows and not all_exact:
        raise RuntimeError(
            f"FPGA video comparison failed: {frames_exact}/{len(compared_rows)} exact"
        )
    if not report["accelerator_fps"]["target_met"]:
        raise RuntimeError(
            "FPGA video accelerator did not meet the 50 FPS target"
        )
    print(f"FPGA_VIDEO_PASS: {output_video}")
    print(f"Report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
