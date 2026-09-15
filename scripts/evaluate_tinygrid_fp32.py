#!/usr/bin/env python3
"""Evaluate a TinyGridNet FP32 checkpoint and render auditable predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygrid_qnn.config import TinyGridConfig
from tinygrid_qnn.data import FEATURE_MODES, FeatureGridDataset
from tinygrid_qnn.metrics import binary_metrics, multiclass_report
from tinygrid_qnn.model import TinyGridNetFP32
from tinygrid_qnn.preprocess import crop_gray_roi, feature_maps_from_gray


CLASS_COLORS = ((40, 40, 245), (245, 170, 20))  # cell, droplet in BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "dataset" / "cell_droplet_tinygrid_feature_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-start", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=900)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    return parser.parse_args()


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TinyGridNetFP32, dict[str, object], str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_mode = str(checkpoint["feature_mode"])
    model = TinyGridNetFP32(
        TinyGridConfig(), input_channels=len(FEATURE_MODES[feature_mode])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    thresholds = np.asarray(checkpoint.get("thresholds", (0.5, 0.5)), dtype=np.float32)
    return model, checkpoint, feature_mode, thresholds


@torch.inference_mode()
def infer(
    model: TinyGridNetFP32,
    features: np.ndarray,
    channel_indices: tuple[int, ...],
    device: torch.device,
) -> np.ndarray:
    selected = np.ascontiguousarray(features[list(channel_indices)]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(selected).unsqueeze(0).to(device)
    return torch.sigmoid(model(tensor))[0].cpu().numpy()


def paint_grid(image: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    scale = 4
    canvas = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    overlay = canvas.copy()
    cell_width = canvas.shape[1] / mask.shape[2]
    cell_height = canvas.shape[0] / mask.shape[1]
    for class_id, color in enumerate(CLASS_COLORS):
        ys, xs = np.nonzero(mask[class_id])
        for grid_y, grid_x in zip(ys.tolist(), xs.tolist()):
            x1 = round(grid_x * cell_width)
            y1 = round(grid_y * cell_height)
            x2 = round((grid_x + 1) * cell_width) - 1
            y2 = round((grid_y + 1) * cell_height) - 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
    canvas = cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 30), (16, 20, 24), -1)
    cv2.putText(canvas, title, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def make_pair(gray: np.ndarray, target: np.ndarray, prediction: np.ndarray, caption: str) -> np.ndarray:
    source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    left = paint_grid(source, target.astype(bool), "Ground truth")
    right = paint_grid(source, prediction.astype(bool), "Prediction")
    pair = np.hstack((left, right))
    footer = np.full((34, pair.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(footer, caption, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack((pair, footer))


def sample_score(target: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> tuple[float, list[float]]:
    predictions = probabilities >= thresholds[:, None, None]
    f1_values: list[float] = []
    for index in range(target.shape[0]):
        if not target[index].any() and not predictions[index].any():
            f1_values.append(1.0)
        else:
            f1_values.append(
                binary_metrics(
                    target[index], probabilities[index], float(thresholds[index])
                ).f1
            )
    return float(np.mean(f1_values)), f1_values


def write_contact_sheet(panels: list[np.ndarray], output: Path, columns: int = 2) -> None:
    if not panels:
        return
    rows = (len(panels) + columns - 1) // columns
    height, width = panels[0].shape[:2]
    sheet = np.full((rows * height, columns * width, 3), 12, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = panel
    cv2.imwrite(str(output), sheet)


def evaluate_dataset(
    model: TinyGridNetFP32,
    dataset: FeatureGridDataset,
    thresholds: np.ndarray,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    ranked: list[tuple[float, str, np.ndarray]] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for sample_path in dataset.samples:
            with np.load(sample_path) as sample:
                features = np.ascontiguousarray(sample["features"])
                target = np.ascontiguousarray(sample["target"])
            probabilities = infer(model, features, dataset.channel_indices, device)
            prediction = probabilities >= thresholds[:, None, None]
            macro_f1, class_f1 = sample_score(target, probabilities, thresholds)
            caption = (
                f"{sample_path.stem} | macro F1={macro_f1:.3f} "
                f"cell={class_f1[0]:.3f} droplet={class_f1[1]:.3f}"
            )
            panel = make_pair(features[0], target, prediction, caption)
            ranked.append((macro_f1, sample_path.stem, panel))
            rows.append(
                {
                    "sample": sample_path.stem,
                    "macro_f1": macro_f1,
                    "cell_f1": class_f1[0],
                    "droplet_f1": class_f1[1],
                }
            )
            all_targets.append(target)
            all_probabilities.append(probabilities)
    elapsed = time.perf_counter() - started
    targets = np.stack(all_targets)
    probabilities = np.stack(all_probabilities)
    report = multiclass_report(targets, probabilities, thresholds, TinyGridConfig().class_names)
    report["samples"] = len(dataset)
    report["device"] = str(device)
    report["mean_pipeline_ms_per_roi"] = 1000.0 * elapsed / max(1, len(dataset))

    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["macro_f1"])))
    ranked.sort(key=lambda item: item[0])
    write_contact_sheet([item[2] for item in ranked[:8]], output / "worst_cases.jpg")
    write_contact_sheet([item[2] for item in ranked[-8:]], output / "best_cases.jpg")
    for page_index in range(0, len(ranked), 10):
        write_contact_sheet(
            [item[2] for item in ranked[page_index : page_index + 10]],
            output / f"test_predictions_{page_index // 10 + 1:02d}.jpg",
        )
    return report


def draw_video_grid(
    frame: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    roi: tuple[int, int, int, int],
) -> None:
    x, y, width, height = roi
    config = TinyGridConfig()
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 230, 50), 2)
    for class_id, color in enumerate(CLASS_COLORS):
        ys, xs = np.nonzero(probabilities[class_id] >= thresholds[class_id])
        for grid_y, grid_x in zip(ys.tolist(), xs.tolist()):
            x1 = x + round(grid_x * width / config.grid_width)
            y1 = y + round(grid_y * height / config.grid_height)
            x2 = x + round((grid_x + 1) * width / config.grid_width)
            y2 = y + round((grid_y + 1) * height / config.grid_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def evaluate_video(
    model: TinyGridNetFP32,
    feature_mode: str,
    thresholds: np.ndarray,
    device: torch.device,
    video_path: Path,
    output: Path,
    roi_values: list[int] | None,
    start_frame: int,
    frame_limit: int,
    stride: int,
) -> dict[str, object]:
    config = TinyGridConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    if roi_values is None:
        roi = (611, 419, config.source_roi_width, config.source_roi_height)
    else:
        roi = tuple(roi_values)
    x, y, width, height = roi
    if x < 0 or y < 0 or x + width > source_width or y + height > source_height:
        raise ValueError(f"ROI {roi} is outside video size {source_width}x{source_height}")
    output_path = output / "fp32_validation_preview.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps / max(1, stride),
        (source_width, source_height),
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    ok, previous_frame = capture.read()
    if not ok:
        raise RuntimeError("Could not read the frame before video start")
    processed = 0
    source_index = max(0, start_frame)
    inference_seconds = 0.0
    while processed < frame_limit:
        ok, current_frame = capture.read()
        if not ok:
            break
        if source_index % stride == 0:
            previous_gray = crop_gray_roi(
                previous_frame,
                x=x,
                y=y,
                width=width,
                height=height,
                output_width=config.input_width,
                output_height=config.input_height,
            )
            current_gray = crop_gray_roi(
                current_frame,
                x=x,
                y=y,
                width=width,
                height=height,
                output_width=config.input_width,
                output_height=config.input_height,
            )
            features = feature_maps_from_gray(current_gray, previous_gray)
            tick = time.perf_counter()
            probabilities = infer(model, features, FEATURE_MODES[feature_mode], device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - tick
            draw_video_grid(current_frame, probabilities, thresholds, roi)
            cv2.rectangle(current_frame, (0, 0), (source_width, 42), (12, 16, 20), -1)
            text = (
                f"PC FP32 validation | frame {source_index + 1} | ROI {width}x{height} "
                f"-> {config.input_width}x{config.input_height}"
            )
            cv2.putText(current_frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2, cv2.LINE_AA)
            writer.write(current_frame)
            processed += 1
        previous_frame = current_frame
        source_index += 1
    writer.release()
    capture.release()
    return {
        "source": str(video_path.resolve()),
        "output": str(output_path.resolve()),
        "frames": processed,
        "source_fps": source_fps,
        "roi_xywh": list(roi),
        "mean_model_ms": 1000.0 * inference_seconds / max(1, processed),
        "model_fps_excluding_io_and_render": processed / max(inference_seconds, 1e-9),
        "note": "PC FP32 preview; this is not FPGA execution.",
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint, feature_mode, thresholds = load_model(args.checkpoint.resolve(), device)
    dataset = FeatureGridDataset(args.data.resolve(), args.split, feature_mode=feature_mode)
    dataset_report = evaluate_dataset(model, dataset, thresholds, device, output)
    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "feature_mode": feature_mode,
        "thresholds": {
            name: float(thresholds[index])
            for index, name in enumerate(TinyGridConfig().class_names)
        },
        "dataset": dataset_report,
    }
    if args.video is not None:
        result["video"] = evaluate_video(
            model,
            feature_mode,
            thresholds,
            device,
            args.video.resolve(),
            output,
            args.roi,
            args.video_start,
            args.video_frames,
            max(1, args.video_stride),
        )
    (output / "evaluation_report.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
