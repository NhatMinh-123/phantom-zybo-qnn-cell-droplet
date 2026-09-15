"""Run one fixed video ROI through two QNN detector checkpoints.

The script keeps preprocessing identical for both models and exports one video
per model plus a side-by-side comparison. It also records per-frame latency and
detection counts so the playback result is not confused with measured runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.detection import Detection, decode_predictions
from qnn.model import TinyQuantDetector, config_from_dict


CLASS_COLORS = {
    0: (255, 0, 255),  # cell: magenta in BGR
    1: (255, 160, 0),  # droplet: blue/cyan in BGR
}


@dataclass
class ModelRuntime:
    name: str
    checkpoint_path: Path
    postprocess_path: Path
    model: TinyQuantDetector
    class_names: tuple[str, ...]
    input_width: int
    input_height: int
    thresholds: tuple[float, ...]
    nms_iou: tuple[float, ...]
    box_constraints: tuple[dict[str, float] | None, ...] | None
    box_calibration: tuple[dict[str, float] | None, ...] | None
    anchors: tuple[tuple[float, float], ...]
    slots_per_class: tuple[int, ...]
    checkpoint_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two QNN detectors on a fixed video ROI."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_grouped"
        / "best.pt",
    )
    parser.add_argument(
        "--baseline-postprocess",
        type=Path,
        default=ROOT
        / "reports"
        / "qnn_droplet_postprocess"
        / "w4a6_square192"
        / "postprocess_config.json",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1_rc1"
        / "best.pt",
    )
    parser.add_argument(
        "--candidate-postprocess",
        type=Path,
        default=ROOT
        / "models"
        / "qnn_cell_droplet_v2_w4a6_square192_quality_aug_v1_rc1"
        / "postprocess_config.json",
    )
    parser.add_argument("--roi-x", type=int, default=520)
    parser.add_argument("--roi-y", type=int, default=32)
    parser.add_argument("--roi-width", type=int, default=384)
    parser.add_argument("--roi-height", type=int, default=384)
    parser.add_argument("--display-size", type=int, default=576)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process at most this many frames; 0 processes the whole video.",
    )
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def class_tuple(
    selected: dict[str, Any],
    field: str,
    class_names: tuple[str, ...],
) -> tuple[Any, ...] | None:
    values = selected.get(field)
    if values is None:
        return None
    return tuple(values.get(name) for name in class_names)


def load_runtime(
    name: str,
    checkpoint_path: Path,
    postprocess_path: Path,
    device: torch.device,
) -> ModelRuntime:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    payload = json.loads(postprocess_path.read_text(encoding="utf-8"))
    selected = payload.get("selected", payload)
    thresholds = tuple(
        float(selected["confidence_thresholds"][class_name])
        for class_name in config.class_names
    )
    nms_iou = tuple(
        float(selected["nms_iou"][class_name])
        for class_name in config.class_names
    )
    return ModelRuntime(
        name=name,
        checkpoint_path=checkpoint_path.resolve(),
        postprocess_path=postprocess_path.resolve(),
        model=model,
        class_names=config.class_names,
        input_width=config.image_width,
        input_height=config.image_height,
        thresholds=thresholds,
        nms_iou=nms_iou,
        box_constraints=class_tuple(
            selected, "box_constraints", config.class_names
        ),
        box_calibration=class_tuple(
            selected, "box_calibration", config.class_names
        ),
        anchors=config.anchors,
        slots_per_class=config.slots_per_class,
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
    )


def preprocess_roi(
    roi_bgr: np.ndarray,
    *,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    grayscale = Image.fromarray(rgb).convert("L").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(grayscale, dtype=np.float32) / np.float32(255.0)
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))[None, None]
    return tensor.to(device, non_blocking=device.type == "cuda")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(
    runtime: ModelRuntime,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[list[Detection], float, float]:
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = runtime.model(tensor)
    synchronize(device)
    inference_ms = (time.perf_counter() - start) * 1000.0

    decode_start = time.perf_counter()
    detections = decode_predictions(
        prediction.detach().cpu(),
        confidence_threshold=runtime.thresholds,
        nms_iou=runtime.nms_iou,
        anchors=runtime.anchors,
        slots_per_class=runtime.slots_per_class,
        box_constraints=runtime.box_constraints,
        box_calibration=runtime.box_calibration,
    )[0]
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    return detections, inference_ms, decode_ms


def text_size(
    text: str,
    scale: float,
    thickness: int,
) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    return width, height


def draw_label(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, _, _ = box
    scale = 0.48
    thickness = 1
    width, height = text_size(text, scale, thickness)
    top = max(0, y1 - height - 8)
    right = min(image.shape[1] - 1, x1 + width + 8)
    cv2.rectangle(image, (x1, top), (right, y1), (15, 15, 15), -1)
    cv2.putText(
        image,
        text,
        (x1 + 4, max(height + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    roi: np.ndarray,
    detections: list[Detection],
    class_names: tuple[str, ...],
) -> np.ndarray:
    annotated = roi.copy()
    height, width = annotated.shape[:2]
    for item in detections:
        x1, y1, x2, y2 = item.box
        pixel_box = (
            int(round(x1 * (width - 1))),
            int(round(y1 * (height - 1))),
            int(round(x2 * (width - 1))),
            int(round(y2 * (height - 1))),
        )
        color = CLASS_COLORS.get(item.class_id, (255, 255, 255))
        cv2.rectangle(
            annotated,
            (pixel_box[0], pixel_box[1]),
            (pixel_box[2], pixel_box[3]),
            color,
            2,
            cv2.LINE_AA,
        )
        draw_label(
            annotated,
            pixel_box,
            f"{class_names[item.class_id]} {item.confidence:.2f}",
            color,
        )
    return annotated


def count_classes(
    detections: list[Detection],
    num_classes: int,
) -> list[int]:
    counts = [0] * num_classes
    for item in detections:
        counts[item.class_id] += 1
    return counts


def make_panel(
    roi: np.ndarray,
    detections: list[Detection],
    runtime: ModelRuntime,
    *,
    frame_index: int,
    timestamp: float,
    inference_ms: float,
    decode_ms: float,
    display_size: int,
) -> np.ndarray:
    annotated = draw_detections(roi, detections, runtime.class_names)
    annotated = cv2.resize(
        annotated,
        (display_size, display_size),
        interpolation=cv2.INTER_LINEAR,
    )
    header_height = 96
    panel = np.zeros((display_size + header_height, display_size, 3), dtype=np.uint8)
    panel[header_height:] = annotated
    counts = count_classes(detections, len(runtime.class_names))
    model_fps = 1000.0 / inference_ms if inference_ms > 0.0 else 0.0
    first_line = f"{runtime.name} | frame {frame_index:04d} | t={timestamp:6.2f}s"
    second_line = (
        f"cell={counts[0]:2d}  droplet={counts[1]:2d} | "
        f"infer={inference_ms:6.2f} ms ({model_fps:5.1f} FPS)"
    )
    third_line = f"decode={decode_ms:5.2f} ms | ROI 384x384 -> QNN 192x192"
    for line, y, scale, color in (
        (first_line, 26, 0.56, (245, 245, 245)),
        (second_line, 56, 0.52, (80, 220, 255)),
        (third_line, 82, 0.45, (190, 190, 190)),
    ):
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def open_writer(
    path: Path,
    *,
    codec: str,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    if len(codec) != 4:
        raise ValueError("Codec must contain exactly four characters")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else 0.0,
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "p95_ms": percentile(values, 95),
        "min_ms": min(values, default=0.0),
        "max_ms": max(values, default=0.0),
        "mean_fps": 1000.0 / statistics.fmean(values) if values else 0.0,
    }


def count_summary(values: list[int]) -> dict[str, float | int]:
    differences = [
        abs(current - previous)
        for previous, current in zip(values, values[1:])
    ]
    return {
        "minimum": min(values, default=0),
        "maximum": max(values, default=0),
        "mean": float(statistics.fmean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "mean_absolute_frame_delta": (
            float(statistics.fmean(differences)) if differences else 0.0
        ),
    }


def write_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumbnail_width = 960
    thumbnails = []
    for image in images:
        scale = thumbnail_width / image.shape[1]
        thumbnail = cv2.resize(
            image,
            (thumbnail_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        thumbnails.append(thumbnail)
    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    tile_height = max(image.shape[0] for image in thumbnails)
    sheet = np.full(
        (rows * tile_height, columns * thumbnail_width, 3),
        24,
        dtype=np.uint8,
    )
    for index, image in enumerate(thumbnails):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * thumbnail_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def validate_roi(
    frame_width: int,
    frame_height: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI coordinates and dimensions must be positive")
    if x + width > frame_width or y + height > frame_height:
        raise ValueError(
            f"ROI {(x, y, width, height)} is outside "
            f"frame {(frame_width, frame_height)}"
        )


def main() -> None:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.display_size < 256:
        raise ValueError("--display-size must be at least 256")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output = args.output.resolve()
    samples_dir = output / "samples"
    output.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_runtime(
        "Baseline QNN",
        args.baseline_checkpoint,
        args.baseline_postprocess,
        device,
    )
    candidate = load_runtime(
        "Quality-aug QNN",
        args.candidate_checkpoint,
        args.candidate_postprocess,
        device,
    )
    if (
        baseline.class_names != candidate.class_names
        or baseline.input_width != candidate.input_width
        or baseline.input_height != candidate.input_height
    ):
        raise RuntimeError("The two models do not share the same input contract")

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.source}")
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 30.0
    validate_roi(
        source_width,
        source_height,
        args.roi_x,
        args.roi_y,
        args.roi_width,
        args.roi_height,
    )
    frames_to_process = source_frames
    if args.max_frames:
        frames_to_process = min(source_frames, args.max_frames)
    sample_count = min(max(args.sample_count, 0), frames_to_process)
    sample_indices = set(
        int(value)
        for value in np.linspace(
            0,
            max(0, frames_to_process - 1),
            sample_count,
        )
    )

    ok, first_frame = capture.read()
    if not ok:
        raise RuntimeError("The video contains no readable frames")
    first_roi = first_frame[
        args.roi_y : args.roi_y + args.roi_height,
        args.roi_x : args.roi_x + args.roi_width,
    ]
    cv2.imwrite(str(output / "selected_roi_first_frame.jpg"), first_roi)
    first_tensor = preprocess_roi(
        first_roi,
        width=baseline.input_width,
        height=baseline.input_height,
        device=device,
    )
    with torch.inference_mode():
        for _ in range(max(args.warmup, 0)):
            baseline.model(first_tensor)
            candidate.model(first_tensor)
    synchronize(device)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    panel_size = (args.display_size, args.display_size + 96)
    comparison_size = (panel_size[0] * 2, panel_size[1])
    baseline_path = output / "baseline_qnn_roi.mp4"
    candidate_path = output / "quality_aug_qnn_roi.mp4"
    comparison_path = output / "two_qnn_roi_comparison.mp4"
    baseline_writer = open_writer(
        baseline_path,
        codec=args.codec,
        fps=source_fps,
        size=panel_size,
    )
    candidate_writer = open_writer(
        candidate_path,
        codec=args.codec,
        fps=source_fps,
        size=panel_size,
    )
    comparison_writer = open_writer(
        comparison_path,
        codec=args.codec,
        fps=source_fps,
        size=comparison_size,
    )

    csv_path = output / "per_frame_metrics.csv"
    baseline_inference: list[float] = []
    candidate_inference: list[float] = []
    baseline_decode: list[float] = []
    candidate_decode: list[float] = []
    baseline_cell_counts: list[int] = []
    baseline_droplet_counts: list[int] = []
    candidate_cell_counts: list[int] = []
    candidate_droplet_counts: list[int] = []
    sample_paths: list[Path] = []
    processed = 0
    processing_start = time.perf_counter()

    fieldnames = [
        "frame_index",
        "source_time_s",
        "baseline_inference_ms",
        "baseline_decode_ms",
        "baseline_cell_count",
        "baseline_droplet_count",
        "candidate_inference_ms",
        "candidate_decode_ms",
        "candidate_cell_count",
        "candidate_droplet_count",
    ]
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            while processed < frames_to_process:
                ok, frame = capture.read()
                if not ok:
                    break
                roi = frame[
                    args.roi_y : args.roi_y + args.roi_height,
                    args.roi_x : args.roi_x + args.roi_width,
                ]
                tensor = preprocess_roi(
                    roi,
                    width=baseline.input_width,
                    height=baseline.input_height,
                    device=device,
                )
                baseline_detections, baseline_ms, baseline_decode_ms = infer(
                    baseline, tensor, device
                )
                candidate_detections, candidate_ms, candidate_decode_ms = infer(
                    candidate, tensor, device
                )
                timestamp = processed / source_fps
                baseline_panel = make_panel(
                    roi,
                    baseline_detections,
                    baseline,
                    frame_index=processed,
                    timestamp=timestamp,
                    inference_ms=baseline_ms,
                    decode_ms=baseline_decode_ms,
                    display_size=args.display_size,
                )
                candidate_panel = make_panel(
                    roi,
                    candidate_detections,
                    candidate,
                    frame_index=processed,
                    timestamp=timestamp,
                    inference_ms=candidate_ms,
                    decode_ms=candidate_decode_ms,
                    display_size=args.display_size,
                )
                comparison = np.hstack((baseline_panel, candidate_panel))
                baseline_writer.write(baseline_panel)
                candidate_writer.write(candidate_panel)
                comparison_writer.write(comparison)

                baseline_counts = count_classes(
                    baseline_detections, len(baseline.class_names)
                )
                candidate_counts = count_classes(
                    candidate_detections, len(candidate.class_names)
                )
                writer.writerow(
                    {
                        "frame_index": processed,
                        "source_time_s": f"{timestamp:.6f}",
                        "baseline_inference_ms": f"{baseline_ms:.6f}",
                        "baseline_decode_ms": f"{baseline_decode_ms:.6f}",
                        "baseline_cell_count": baseline_counts[0],
                        "baseline_droplet_count": baseline_counts[1],
                        "candidate_inference_ms": f"{candidate_ms:.6f}",
                        "candidate_decode_ms": f"{candidate_decode_ms:.6f}",
                        "candidate_cell_count": candidate_counts[0],
                        "candidate_droplet_count": candidate_counts[1],
                    }
                )
                baseline_inference.append(baseline_ms)
                candidate_inference.append(candidate_ms)
                baseline_decode.append(baseline_decode_ms)
                candidate_decode.append(candidate_decode_ms)
                baseline_cell_counts.append(baseline_counts[0])
                baseline_droplet_counts.append(baseline_counts[1])
                candidate_cell_counts.append(candidate_counts[0])
                candidate_droplet_counts.append(candidate_counts[1])

                if processed in sample_indices:
                    sample_path = samples_dir / f"frame_{processed:04d}.jpg"
                    cv2.imwrite(
                        str(sample_path),
                        comparison,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                    sample_paths.append(sample_path)
                processed += 1
                if processed % 100 == 0 or processed == frames_to_process:
                    elapsed = time.perf_counter() - processing_start
                    current_fps = processed / elapsed if elapsed > 0.0 else 0.0
                    print(
                        f"Processed {processed}/{frames_to_process} frames "
                        f"({current_fps:.2f} pipeline FPS)",
                        flush=True,
                    )
    finally:
        capture.release()
        baseline_writer.release()
        candidate_writer.release()
        comparison_writer.release()

    elapsed = time.perf_counter() - processing_start
    pipeline_fps = processed / elapsed if elapsed > 0.0 else 0.0
    duration = processed / source_fps if source_fps > 0.0 else 0.0
    write_contact_sheet(sample_paths, output / "comparison_contact_sheet.jpg")

    summary = {
        "source": {
            "path": str(args.source.resolve()),
            "width": source_width,
            "height": source_height,
            "encoded_fps": source_fps,
            "reported_frame_count": source_frames,
            "processed_frames": processed,
            "processed_duration_s": duration,
            "filename_note": (
                "The filename says 100fps, but the container metadata reports "
                f"{source_fps:.3f} FPS."
            ),
        },
        "roi": {
            "x": args.roi_x,
            "y": args.roi_y,
            "width": args.roi_width,
            "height": args.roi_height,
            "qnn_input_width": baseline.input_width,
            "qnn_input_height": baseline.input_height,
        },
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "elapsed_s": elapsed,
            "dual_model_pipeline_fps": pipeline_fps,
            "source_realtime_target_fps": source_fps,
            "meets_source_realtime": pipeline_fps >= source_fps,
            "realtime_ratio": pipeline_fps / source_fps,
            "output_playback_fps": source_fps,
        },
        "models": {
            "baseline": {
                "checkpoint": str(baseline.checkpoint_path),
                "checkpoint_epoch": baseline.checkpoint_epoch,
                "postprocess": str(baseline.postprocess_path),
                "confidence_thresholds": dict(
                    zip(baseline.class_names, baseline.thresholds)
                ),
                "nms_iou": dict(zip(baseline.class_names, baseline.nms_iou)),
                "inference": timing_summary(baseline_inference),
                "decode": timing_summary(baseline_decode),
                "counts": {
                    "cell": count_summary(baseline_cell_counts),
                    "droplet": count_summary(baseline_droplet_counts),
                },
            },
            "candidate": {
                "checkpoint": str(candidate.checkpoint_path),
                "checkpoint_epoch": candidate.checkpoint_epoch,
                "postprocess": str(candidate.postprocess_path),
                "confidence_thresholds": dict(
                    zip(candidate.class_names, candidate.thresholds)
                ),
                "nms_iou": dict(zip(candidate.class_names, candidate.nms_iou)),
                "inference": timing_summary(candidate_inference),
                "decode": timing_summary(candidate_decode),
                "counts": {
                    "cell": count_summary(candidate_cell_counts),
                    "droplet": count_summary(candidate_droplet_counts),
                },
            },
        },
        "outputs": {
            "baseline_video": str(baseline_path.resolve()),
            "candidate_video": str(candidate_path.resolve()),
            "comparison_video": str(comparison_path.resolve()),
            "per_frame_csv": str(csv_path.resolve()),
            "contact_sheet": str(
                (output / "comparison_contact_sheet.jpg").resolve()
            ),
        },
        "accuracy_note": (
            "This video has no frame-level ground truth, so detection accuracy "
            "cannot be measured from it. Counts and temporal count deltas are "
            "diagnostic signals, not precision/recall."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
