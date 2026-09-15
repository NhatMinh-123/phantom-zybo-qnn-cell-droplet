#!/usr/bin/env python3
"""Batch the 15 um YOLO teacher over one fixed ROI and export an annotated video."""

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
from ultralytics import YOLO

from run_cell_droplet_realtime import (
    ClassAwareTracker,
    ObjectDetection,
    add_roi_inset,
    box_center,
    draw_box,
    make_contact_sheet,
    map_box_to_frame,
    prepare_model_input,
    scaled_roi,
)


RUNTIME_LABEL = "PC GPU reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--inset-size", type=int, default=260)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() != "cpu":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config = json.loads(args.config.resolve().read_text(encoding="ascii"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "samples"
    samples_dir.mkdir(exist_ok=True)

    image_size = int(config["image_size"])
    preprocess = config.get("preprocess", {})
    content_width = int(preprocess.get("content_width", image_size))
    content_height = int(preprocess.get("content_height", image_size))
    nms_iou = float(config.get("nms_iou", 0.5))
    roi_config = config["roi"]
    tracking = config.get("tracking", {})

    model = YOLO(str(args.model.resolve()))
    names = {int(index): str(name) for index, name in model.names.items()}
    class_ids = {name: class_id for class_id, name in names.items()}
    thresholds = {
        class_id: float(config["confidence"].get(class_name, 0.5))
        for class_id, class_name in names.items()
    }
    minimum_confidence = min(thresholds.values())

    capture = cv2.VideoCapture(str(args.source.resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    available = max(0, source_frames - args.start_frame)
    planned = min(available, args.max_frames) if args.max_frames else available

    roi_width = int(roi_config.get("width", roi_config.get("size", 256)))
    roi_height = int(roi_config.get("height", roi_config.get("size", 256)))
    roi_geometry = scaled_roi(
        frame_width,
        frame_height,
        int(roi_config["reference_width"]),
        int(roi_config["reference_height"]),
        int(roi_config["x"]),
        int(roi_config["y"]),
        roi_width,
        roi_height,
    )
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry
    count_fraction = float(roi_config.get("count_line_fraction", 0.65))
    count_line_model_x = count_fraction * content_width + (image_size - content_width) / 2
    count_line_frame_x = roi_x1 + count_fraction * (roi_x2 - roi_x1)

    tracker = ClassAwareTracker(
        image_size,
        int(tracking.get("max_misses", 6)),
        float(tracking.get("max_center_distance", 0.22)),
        str(tracking.get("direction", "left_to_right")),
        float(tracking.get("count_hysteresis", 0.04)),
        int(tracking.get("minimum_hits", 3)),
    )
    cumulative = {class_id: 0 for class_id in names}

    video_path = output / "pc_gpu_teacher_roi256.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")

    detection_path = output / "detections.csv"
    frame_path = output / "frame_summary.csv"
    batch_times: list[float] = []
    draw_times: list[float] = []
    sheet_frames: list[np.ndarray] = []
    processed = 0
    wall_started = time.perf_counter()
    sample_interval = max(1, planned // 12) if planned else 1

    warmup = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for _ in range(int(config.get("warmup_frames", 5))):
        model.predict(
            warmup,
            imgsz=image_size,
            conf=minimum_confidence,
            iou=nms_iou,
            device=args.device,
            verbose=False,
        )

    detection_fields = [
        "frame", "time_s", "track_id", "class_id", "class", "confidence",
        "frame_x1", "frame_y1", "frame_x2", "frame_y2", "crossed",
    ]
    frame_fields = [
        "frame", "time_s", "detected_cell", "detected_droplet",
        "counted_cell", "counted_droplet", "batch_inference_ms_per_frame",
    ]
    with detection_path.open("w", newline="", encoding="ascii") as detection_file, frame_path.open(
        "w", newline="", encoding="ascii"
    ) as frame_file:
        detection_writer = csv.DictWriter(detection_file, fieldnames=detection_fields)
        frame_writer = csv.DictWriter(frame_file, fieldnames=frame_fields)
        detection_writer.writeheader()
        frame_writer.writeheader()

        while processed < planned:
            frames: list[np.ndarray] = []
            model_inputs: list[np.ndarray] = []
            transforms = []
            source_indices: list[int] = []
            for _ in range(min(args.batch_size, planned - processed)):
                ok, frame = capture.read()
                if not ok:
                    break
                source_index = args.start_frame + processed + len(frames)
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                model_input, transform = prepare_model_input(
                    roi, image_size, image_size, content_width, content_height
                )
                frames.append(frame)
                model_inputs.append(model_input)
                transforms.append(transform)
                source_indices.append(source_index)
            if not frames:
                break

            synchronize(args.device)
            inference_started = time.perf_counter()
            results = model.predict(
                model_inputs,
                imgsz=image_size,
                conf=minimum_confidence,
                iou=nms_iou,
                max_det=args.max_det,
                device=args.device,
                batch=len(model_inputs),
                verbose=False,
            )
            synchronize(args.device)
            batch_seconds = time.perf_counter() - inference_started
            batch_times.append(batch_seconds)
            per_frame_ms = batch_seconds * 1000.0 / len(frames)

            draw_started = time.perf_counter()
            for frame, model_input, transform, result, source_index in zip(
                frames, model_inputs, transforms, results, source_indices
            ):
                detections: list[ObjectDetection] = []
                if result.boxes is not None:
                    for class_id, confidence, roi_box in zip(
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.xyxy.cpu().numpy(),
                    ):
                        class_id = int(class_id)
                        confidence = float(confidence)
                        if confidence < thresholds[class_id]:
                            continue
                        detections.append(
                            ObjectDetection(
                                class_id=class_id,
                                confidence=confidence,
                                roi_box=roi_box.copy(),
                                frame_box=map_box_to_frame(roi_box, roi_geometry, transform),
                            )
                        )
                tracker.update(detections, source_index, count_line_model_x)
                for detection in detections:
                    if detection.crossed:
                        cumulative[detection.class_id] += 1

                annotated = frame.copy()
                cv2.rectangle(
                    annotated, (roi_x1, roi_y1), (roi_x2, roi_y2),
                    (240, 190, 20), 2, cv2.LINE_AA,
                )
                cv2.line(
                    annotated,
                    (int(round(count_line_frame_x)), roi_y1),
                    (int(round(count_line_frame_x)), roi_y2),
                    (0, 0, 255), 2, cv2.LINE_AA,
                )
                visible = {class_id: 0 for class_id in names}
                for detection in detections:
                    visible[detection.class_id] += 1
                    class_name = names[detection.class_id]
                    color = (30, 205, 40) if class_name == "cell" else (205, 80, 20)
                    draw_box(
                        annotated,
                        detection.frame_box,
                        f"{class_name[0].upper()}{detection.track_id} {detection.confidence:.2f}",
                        color,
                    )
                    detection_writer.writerow(
                        {
                            "frame": source_index,
                            "time_s": f"{source_index / source_fps:.6f}",
                            "track_id": detection.track_id,
                            "class_id": detection.class_id,
                            "class": class_name,
                            "confidence": f"{detection.confidence:.6f}",
                            "frame_x1": int(round(detection.frame_box[0])),
                            "frame_y1": int(round(detection.frame_box[1])),
                            "frame_x2": int(round(detection.frame_box[2])),
                            "frame_y2": int(round(detection.frame_box[3])),
                            "crossed": int(detection.crossed),
                        }
                    )
                add_roi_inset(
                    annotated, model_input, detections, names,
                    count_line_model_x, args.inset_size,
                )
                cell_id = class_ids.get("cell", -1)
                droplet_id = class_ids.get("droplet", -1)
                lines = (
                    f"{RUNTIME_LABEL} | frame {source_index:06d} | batch {args.batch_size}",
                    f"Visible cell {visible.get(cell_id, 0)} droplet {visible.get(droplet_id, 0)}",
                    f"Crossed cell {cumulative.get(cell_id, 0)} droplet {cumulative.get(droplet_id, 0)}",
                )
                for line_index, line in enumerate(lines):
                    y = 27 + 25 * line_index
                    cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 3, cv2.LINE_AA)
                    cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
                writer.write(annotated)
                frame_writer.writerow(
                    {
                        "frame": source_index,
                        "time_s": f"{source_index / source_fps:.6f}",
                        "detected_cell": visible.get(cell_id, 0),
                        "detected_droplet": visible.get(droplet_id, 0),
                        "counted_cell": cumulative.get(cell_id, 0),
                        "counted_droplet": cumulative.get(droplet_id, 0),
                        "batch_inference_ms_per_frame": f"{per_frame_ms:.6f}",
                    }
                )
                if source_index % sample_interval == 0 and len(sheet_frames) < 12:
                    sample_path = samples_dir / f"frame_{source_index:06d}.jpg"
                    cv2.imwrite(str(sample_path), annotated)
                    sheet_frames.append(annotated.copy())
            draw_times.append(time.perf_counter() - draw_started)
            processed += len(frames)
            if args.progress_every and (
                processed == planned or processed % args.progress_every < len(frames)
            ):
                elapsed = time.perf_counter() - wall_started
                print(
                    f"processed={processed}/{planned} export_fps={processed / elapsed:.2f}",
                    flush=True,
                )

    capture.release()
    writer.release()
    wall_seconds = time.perf_counter() - wall_started
    inference_seconds = sum(batch_times)
    draw_seconds = sum(draw_times)
    frame_inference_ms = [
        seconds * 1000.0 / args.batch_size for seconds in batch_times[:-1]
    ]
    if batch_times:
        last_count = processed - args.batch_size * max(0, len(batch_times) - 1)
        frame_inference_ms.append(batch_times[-1] * 1000.0 / max(1, last_count))
    summary = {
        "runtime_label": RUNTIME_LABEL,
        "execution_note": "This is PC GPU teacher inference, not FPGA inference.",
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "processed_frames": processed,
        "source_fps": source_fps,
        "roi_frame_coordinates": list(roi_geometry),
        "model_input": [image_size, image_size],
        "batch_size": args.batch_size,
        "thresholds": {names[key]: value for key, value in thresholds.items()},
        "wall_seconds": wall_seconds,
        "end_to_end_export_fps": processed / wall_seconds if wall_seconds else 0.0,
        "gpu_inference_seconds": inference_seconds,
        "gpu_inference_fps": processed / inference_seconds if inference_seconds else 0.0,
        "gpu_inference_ms_per_frame_mean": statistics.fmean(frame_inference_ms),
        "draw_write_seconds": draw_seconds,
        "crossed_counts": {names[key]: value for key, value in cumulative.items()},
        "artifacts": {
            "video": str(video_path),
            "detections": str(detection_path),
            "frames": str(frame_path),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    make_contact_sheet(sheet_frames, output / "preview_sheet.jpg")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
