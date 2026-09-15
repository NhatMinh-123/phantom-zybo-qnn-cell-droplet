#!/usr/bin/env python3
"""Evaluate Arty S7-25 detections against a labeled YOLO dataset over UART."""

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

from qnn.dataset import YoloDetectionDataset
from qnn.detection import Detection
from qnn.fpga_io import (
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    prepare_image,
    unpack_sparse_detection_axis,
)
from scripts.run_video_finn_uart import transact
from scripts.send_frame_finn_uart_sparse import transact_sparse


DEFAULT_DATA = ROOT / "dataset" / "cell_droplet_roi384_grouped"
DEFAULT_MANIFEST = (
    ROOT
    / "final_results"
    / "arty_s7_25_qnn_detection"
    / "07_50fps_optimized"
    / "fpga_manifest_60fps.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "fpga_accuracy_50fps_20260727"


@dataclass(frozen=True)
class PredictionMatch:
    detection: Detection
    matched: bool
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--protocol",
        choices=("dense", "sparse"),
        default="dense",
        help="UART response protocol emitted by the programmed bitstream",
    )
    parser.add_argument(
        "--clock-hz",
        type=int,
        default=0,
        help="Override the FPGA clock recorded in the manifest",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--frame-id-start", type=int, default=8000)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N images; zero means the full split",
    )
    parser.add_argument(
        "--no-save-tensors",
        action="store_true",
        help="Do not save the FPGA output tensor for each image",
    )
    return parser.parse_args()


def labels_to_boxes(targets: np.ndarray, class_id: int) -> np.ndarray:
    selected = targets[targets[:, 0].astype(np.int64) == class_id]
    if not len(selected):
        return np.empty((0, 4), dtype=np.float32)
    centers = selected[:, 1:3]
    sizes = selected[:, 3:5]
    return np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0.0, None), axis=1)
    second_area = np.prod(
        np.clip(second[:, 2:] - second[:, :2], 0.0, None),
        axis=1,
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def match_image(
    detections: list[Detection],
    targets: np.ndarray,
    *,
    num_classes: int,
    iou_threshold: float,
) -> tuple[list[int], list[int], list[int], list[PredictionMatch], list[float]]:
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes
    matches: list[PredictionMatch] = []
    matched_ious: list[float] = []

    for class_id in range(num_classes):
        class_detections = sorted(
            (item for item in detections if item.class_id == class_id),
            key=lambda item: item.confidence,
            reverse=True,
        )
        ground_truth = labels_to_boxes(targets, class_id)
        used_ground_truth: set[int] = set()
        for detection in class_detections:
            if not len(ground_truth):
                false_positive[class_id] += 1
                matches.append(PredictionMatch(detection, False, 0.0))
                continue
            overlaps = box_iou(
                np.asarray([detection.box], dtype=np.float32),
                ground_truth,
            )[0]
            best_index = int(np.argmax(overlaps))
            best_iou = float(overlaps[best_index])
            matched = (
                best_iou >= iou_threshold and best_index not in used_ground_truth
            )
            if matched:
                used_ground_truth.add(best_index)
                true_positive[class_id] += 1
                matched_ious.append(best_iou)
            else:
                false_positive[class_id] += 1
            matches.append(PredictionMatch(detection, matched, best_iou))
        false_negative[class_id] += len(ground_truth) - len(used_ground_truth)

    return (
        true_positive,
        false_positive,
        false_negative,
        matches,
        matched_ious,
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
    }


def average_precision_50(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, list[np.ndarray]],
    *,
    iou_threshold: float,
) -> tuple[float, list[dict[str, float]]]:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    if total_ground_truth == 0:
        return 0.0, []

    matched = {image_id: set() for image_id in ground_truth}
    sorted_predictions = sorted(
        predictions,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    true_positive: list[float] = []
    false_positive: list[float] = []
    for prediction in sorted_predictions:
        image_id = str(prediction["image_id"])
        target_boxes = ground_truth.get(image_id, [])
        prediction_box = np.asarray([prediction["box"]], dtype=np.float32)
        if target_boxes:
            overlaps = box_iou(
                prediction_box,
                np.asarray(target_boxes, dtype=np.float32),
            )[0]
            best_index = int(np.argmax(overlaps))
            is_match = (
                float(overlaps[best_index]) >= iou_threshold
                and best_index not in matched[image_id]
            )
        else:
            best_index = -1
            is_match = False
        if is_match:
            matched[image_id].add(best_index)
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / total_ground_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    curve = [
        {
            "confidence": float(item["confidence"]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index, item in enumerate(sorted_predictions)
    ]
    return float(np.mean(interpolated)), curve


def draw_evaluation(
    image_path: Path,
    targets: np.ndarray,
    matches: list[PredictionMatch],
    class_names: list[str],
    image_metrics: dict[str, float | int],
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    canvas = cv2.copyMakeBorder(
        image,
        56,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )

    for target in targets:
        class_id = int(target[0])
        center_x, center_y, box_width, box_height = target[1:]
        x1 = int(round((center_x - box_width / 2) * width))
        y1 = int(round((center_y - box_height / 2) * height)) + 56
        x2 = int(round((center_x + box_width / 2) * width))
        y2 = int(round((center_y + box_height / 2) * height)) + 56
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 210, 30), 1)
        cv2.putText(
            canvas,
            f"GT {class_names[class_id][0].upper()}",
            (x1, max(68, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (30, 210, 30),
            1,
            cv2.LINE_AA,
        )

    class_colors = {0: (40, 40, 235), 1: (235, 130, 25)}
    for result in matches:
        detection = result.detection
        x1 = int(round(detection.box[0] * width))
        y1 = int(round(detection.box[1] * height)) + 56
        x2 = int(round(detection.box[2] * width))
        y2 = int(round(detection.box[3] * height)) + 56
        color = (
            class_colors.get(detection.class_id, (220, 180, 30))
            if result.matched
            else (220, 40, 220)
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{class_names[detection.class_id][0].upper()} "
            f"{detection.confidence:.2f} IoU {result.iou:.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (x1, min(canvas.shape[0] - 4, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            color,
            1,
            cv2.LINE_AA,
        )

    title = (
        f"{image_path.name} | TP={image_metrics['true_positive']} "
        f"FP={image_metrics['false_positive']} "
        f"FN={image_metrics['false_negative']} "
        f"F1={float(image_metrics['f1']):.3f}"
    )
    cv2.putText(
        canvas,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GT=green | matched prediction=class color | error=magenta",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(paths: list[Path], destination: Path, columns: int = 3) -> None:
    if not paths:
        return
    thumbnails = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_width = 420
        target_height = round(image.shape[0] * target_width / image.shape[1])
        thumbnails.append(
            cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    if not thumbnails:
        return
    cell_height = max(image.shape[0] for image in thumbnails)
    rows = []
    for index in range(0, len(thumbnails), columns):
        row_images = thumbnails[index : index + columns]
        while len(row_images) < columns:
            row_images.append(np.full_like(thumbnails[0], 245))
        padded = [
            cv2.copyMakeBorder(
                image,
                0,
                cell_height - image.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
            for image in row_images
        ]
        rows.append(np.hstack(padded))
    cv2.imwrite(str(destination), np.vstack(rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(
    output: Path,
    metrics: dict[str, Any],
    fps_values: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np_plot
    except ImportError:
        return

    names = ["Overall", *[name.title() for name in metrics["class_names"]]]
    values = [metrics["overall"], *metrics["classes"].values()]
    precision = [float(item["precision"]) for item in values]
    recall = [float(item["recall"]) for item in values]
    f1 = [float(item["f1"]) for item in values]
    ap50 = [
        float(metrics["map50"]),
        *[float(item["ap50"]) for item in metrics["classes"].values()],
    ]

    x = np_plot.arange(len(names))
    width = 0.2
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - 1.5 * width, precision, width, label="Precision")
    axes[0].bar(x - 0.5 * width, recall, width, label="Recall")
    axes[0].bar(x + 0.5 * width, f1, width, label="F1")
    axes[0].bar(x + 1.5 * width, ap50, width, label="AP50")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("FPGA detection accuracy on labeled test set")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(fps_values, bins=5, color="#16a34a", edgecolor="white")
    axes[1].axvline(50.0, color="#dc2626", linestyle="--", label="50 FPS target")
    timing_label = metrics["accelerator"]["measurement_label"]
    axes[1].set_xlabel(f"{timing_label} FPS")
    axes[1].set_ylabel("Frames")
    axes[1].set_title(f"Measured {timing_label.lower()} throughput per test image")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "accuracy_and_fps.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")
    manifest = load_manifest(args.manifest)
    clock_hz = args.clock_hz or int(manifest["fpga_core"]["clock_hz"])
    class_names = list(manifest["postprocessing"]["decoder"]["class_names"])
    dataset = YoloDetectionDataset(
        args.data,
        args.split,
        input_size=(
            int(manifest["preprocessing"]["resize"]["width"]),
            int(manifest["preprocessing"]["resize"]["height"]),
        ),
        num_classes=len(class_names),
    )
    image_count = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    output = args.output.resolve()
    annotated_dir = output / "annotated"
    tensor_dir = output / "hardware_tensors"
    output.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_tensors:
        tensor_dir.mkdir(parents=True, exist_ok=True)

    aggregate_tp = [0] * len(class_names)
    aggregate_fp = [0] * len(class_names)
    aggregate_fn = [0] * len(class_names)
    predictions_for_ap = {class_id: [] for class_id in range(len(class_names))}
    ground_truth_for_ap = {class_id: {} for class_id in range(len(class_names))}
    per_image_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pr_curve_rows: list[dict[str, Any]] = []
    annotated_paths: list[Path] = []
    fps_values: list[float] = []
    cycles_values: list[int] = []
    all_matched_ious: list[float] = []

    import serial

    started = time.perf_counter()
    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
        write_timeout=10.0,
    ) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        for index in range(image_count):
            image_path = dataset.images[index]
            _, target_tensor = dataset[index]
            targets = target_tensor.numpy()
            image_id = image_path.stem
            for class_id in range(len(class_names)):
                ground_truth_for_ap[class_id][image_id] = [
                    box
                    for box in labels_to_boxes(targets, class_id)
                ]

            codes = prepare_image(image_path, manifest)
            payload = pack_input_axis(codes, manifest)
            frame_id = args.frame_id_start + index
            if args.protocol == "sparse":
                sparse_payload, cycles, uart_seconds = transact_sparse(
                    port,
                    payload,
                    frame_id=frame_id,
                    timeout=args.timeout,
                )
                tensor = unpack_sparse_detection_axis(sparse_payload, manifest)
            else:
                tensor, cycles, uart_seconds = transact(
                    port,
                    payload,
                    manifest,
                    frame_id,
                    args.timeout,
                )
            if not args.no_save_tensors:
                np.save(tensor_dir / f"{image_id}.npy", tensor)
            detections = decode_output_tensor(tensor, manifest)[0]
            tp, fp, fn, matches, matched_ious = match_image(
                detections,
                targets,
                num_classes=len(class_names),
                iou_threshold=args.iou,
            )
            aggregate_tp = [left + right for left, right in zip(aggregate_tp, tp)]
            aggregate_fp = [left + right for left, right in zip(aggregate_fp, fp)]
            aggregate_fn = [left + right for left, right in zip(aggregate_fn, fn)]
            all_matched_ious.extend(matched_ious)

            for detection in detections:
                predictions_for_ap[detection.class_id].append(
                    {
                        "image_id": image_id,
                        "confidence": detection.confidence,
                        "box": detection.box,
                    }
                )
            match_lookup = {id(result.detection): result for result in matches}
            for detection in detections:
                result = match_lookup[id(detection)]
                prediction_rows.append(
                    {
                        "image": image_path.name,
                        "class": class_names[detection.class_id],
                        "confidence": f"{detection.confidence:.8f}",
                        "x1": f"{detection.box[0]:.8f}",
                        "y1": f"{detection.box[1]:.8f}",
                        "x2": f"{detection.box[2]:.8f}",
                        "y2": f"{detection.box[3]:.8f}",
                        "matched": result.matched,
                        "best_iou": f"{result.iou:.8f}",
                    }
                )

            image_counts = score_counts(sum(tp), sum(fp), sum(fn))
            accelerator_fps = clock_hz / cycles
            fps_values.append(accelerator_fps)
            cycles_values.append(cycles)
            annotated = draw_evaluation(
                image_path,
                targets,
                matches,
                class_names,
                image_counts,
            )
            annotated_path = annotated_dir / f"{image_id}.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            annotated_paths.append(annotated_path)
            per_image_rows.append(
                {
                    "image": image_path.name,
                    **image_counts,
                    "matched_mean_iou": (
                        f"{np.mean(matched_ious):.8f}" if matched_ious else ""
                    ),
                    "accelerator_cycles": cycles,
                    "accelerator_ms": f"{cycles / clock_hz * 1000:.8f}",
                    "accelerator_fps": f"{accelerator_fps:.8f}",
                    "uart_seconds": f"{uart_seconds:.8f}",
                    "detections": len(detections),
                    "ground_truth": len(targets),
                    "annotated": annotated_path.relative_to(output).as_posix(),
                }
            )
            print(
                f"[{index + 1:02d}/{image_count:02d}] {image_path.name} "
                f"TP={sum(tp)} FP={sum(fp)} FN={sum(fn)} "
                f"F1={float(image_counts['f1']):.3f} "
                f"FPGA={accelerator_fps:.3f}FPS UART={uart_seconds:.3f}s",
                flush=True,
            )

    classes: dict[str, Any] = {}
    ap_values = []
    for class_id, class_name in enumerate(class_names):
        ap50, curve = average_precision_50(
            predictions_for_ap[class_id],
            ground_truth_for_ap[class_id],
            iou_threshold=args.iou,
        )
        class_metrics = score_counts(
            aggregate_tp[class_id],
            aggregate_fp[class_id],
            aggregate_fn[class_id],
        )
        class_metrics["ap50"] = ap50
        classes[class_name] = class_metrics
        ap_values.append(ap50)
        for point in curve:
            pr_curve_rows.append({"class": class_name, **point})

    overall = score_counts(
        sum(aggregate_tp),
        sum(aggregate_fp),
        sum(aggregate_fn),
    )
    if args.protocol == "sparse":
        measurement_label = "UART-paced sparse stream"
        measurement_scope = (
            "From the first AXI input byte accepted by the FPGA through the final "
            "detector output. This includes UART-paced input arrival but excludes "
            "host serial API overhead and sparse-response transmission."
        )
        interpretation = (
            "Accuracy is measured from FPGA-produced sparse detections against YOLO "
            "ground truth at IoU 0.5. Reported FPS is UART-paced stream throughput, "
            "not the standalone CNN compute-core rate."
        )
    else:
        measurement_label = "buffered accelerator"
        measurement_scope = (
            "From the first AXI input byte accepted by the FPGA through the final "
            "detector output after the complete request has been buffered."
        )
        interpretation = (
            "Accuracy is measured from FPGA-produced tensors against YOLO ground "
            "truth at IoU 0.5. UART request and response transfer time is excluded "
            "from accelerator FPS."
        )

    metrics = {
        "schema_version": 1,
        "source": {
            "dataset": str(args.data.resolve()),
            "split": args.split,
            "images": image_count,
            "manifest": str(args.manifest.resolve()),
            "port": args.port,
            "baud": args.baud,
            "protocol": args.protocol,
            "iou_threshold": args.iou,
        },
        "class_names": class_names,
        "overall": overall,
        "classes": classes,
        "map50": float(np.mean(ap_values)),
        "matched_box_iou": {
            "count": len(all_matched_ious),
            "mean": float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
            "median": float(np.median(all_matched_ious)) if all_matched_ious else 0.0,
            "minimum": float(np.min(all_matched_ious)) if all_matched_ious else 0.0,
            "maximum": float(np.max(all_matched_ious)) if all_matched_ious else 0.0,
        },
        "accelerator": {
            "measurement_label": measurement_label,
            "measurement_scope": measurement_scope,
            "clock_hz": clock_hz,
            "cycles_minimum": min(cycles_values),
            "cycles_mean": float(np.mean(cycles_values)),
            "cycles_maximum": max(cycles_values),
            "fps_minimum": min(fps_values),
            "fps_mean": float(np.mean(fps_values)),
            "fps_maximum": max(fps_values),
            "target_fps": 50.0,
            "target_met_all_images": min(fps_values) >= 50.0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": interpretation,
    }
    write_csv(output / "per_image.csv", per_image_rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "precision_recall_curve.csv", pr_curve_rows)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ranked = sorted(
        zip(per_image_rows, annotated_paths),
        key=lambda item: (
            float(item[0]["f1"]),
            -int(item[0]["false_positive"]) - int(item[0]["false_negative"]),
        ),
    )
    make_contact_sheet(
        [path for _, path in ranked[:12]],
        output / "worst_12_contact_sheet.jpg",
    )
    make_contact_sheet(
        annotated_paths,
        output / "all_30_contact_sheet.jpg",
    )
    plot_metrics(output, metrics, fps_values)

    readme = f"""# FPGA labeled accuracy validation

- Dataset split: `{args.split}` ({image_count} images)
- IoU threshold: {args.iou:.2f}
- Overall precision: {float(overall['precision']) * 100:.2f}%
- Overall recall: {float(overall['recall']) * 100:.2f}%
- Overall F1: {float(overall['f1']) * 100:.2f}%
- mAP50 (101-point): {float(metrics['map50']) * 100:.2f}%
- Cell F1: {float(classes['cell']['f1']) * 100:.2f}%
- Droplet F1: {float(classes['droplet']['f1']) * 100:.2f}%
- Mean matched-box IoU: {float(metrics['matched_box_iou']['mean']):.4f}
- {measurement_label}: {float(metrics['accelerator']['fps_mean']):.3f} FPS
- Every image above 50 FPS: {metrics['accelerator']['target_met_all_images']}

Timing scope: {measurement_scope}

`worst_12_contact_sheet.jpg` contains the highest-priority images for relabeling,
hard-example training and post-processing review. Magenta predictions are false
positives or duplicate/localization failures at IoU {args.iou:.2f}.
"""
    (output / "README.md").write_text(readme, encoding="ascii", newline="\n")
    print(json.dumps(metrics, indent=2))
    print(f"FPGA_ACCURACY_COMPLETE: {output}")


if __name__ == "__main__":
    main()
