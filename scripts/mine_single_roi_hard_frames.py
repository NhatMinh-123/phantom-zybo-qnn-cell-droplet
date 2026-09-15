#!/usr/bin/env python3
"""Mine diverse hard frames for manual review in the compact FPGA ROI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.detection import Detection, decode_predictions
from qnn.fpga_io import (
    decoder_box_calibration,
    decoder_box_constraints,
    decoder_nms_iou,
    load_manifest,
)
from qnn.model import TinyQuantDetector, config_from_dict
from scripts.run_video_finn_uart import prepare_roi_input, scaled_roi


DEFAULT_VIDEO = ROOT / "data" / "raw" / "3.4.mp4"
DEFAULT_CHECKPOINT = (
    ROOT
    / "models"
    / "qnn_cell_droplet_v2_w4a6_square192_grouped"
    / "best.pt"
)
DEFAULT_MANIFEST = (
    ROOT
    / "exports"
    / "qnn_cell_droplet_v2"
    / "tiny_detector_192x192_w4a6_fpga.json"
)
DEFAULT_ROI_CONFIG = (
    ROOT / "models" / "cell_droplet_yolo11n" / "roi384_baseline_config.json"
)
DEFAULT_OUTPUT = ROOT / "dataset" / "single_roi_hard_review"
COLORS = ((40, 40, 235), (235, 130, 25))


@dataclass
class Candidate:
    frame_index: int
    source_time: float
    sharpness: float
    brightness: float
    contrast: float
    detections: list[Detection]
    uncertainty: float
    hard_score: float = 0.0
    reason: str = ""

    @property
    def cells(self) -> int:
        return sum(item.class_id == 0 for item in self.detections)

    @property
    def droplets(self) -> int:
        return sum(item.class_id == 1 for item in self.detections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--hard-count", type=int, default=80)
    parser.add_argument("--minimum-gap", type=int, default=15)
    parser.add_argument("--start-sec", type=float, default=3.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def robust_z(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array)
    deviation = np.median(np.abs(array - median))
    scale = max(1.4826 * deviation, 1e-6)
    return np.abs(array - median) / scale


def reason_for(candidate: Candidate) -> str:
    reasons = []
    if candidate.droplets < 2:
        reasons.append("droplet_miss")
    elif candidate.droplets > 3:
        reasons.append("droplet_extra")
    if candidate.cells >= 8:
        reasons.append("many_cells")
    if candidate.droplets and candidate.cells / candidate.droplets >= 3:
        reasons.append("high_cell_ratio")
    if candidate.uncertainty >= 0.5:
        reasons.append("near_threshold")
    return "+".join(reasons) or "visual_outlier"


def greedy_with_gap(
    candidates: list[Candidate],
    count: int,
    minimum_gap: int,
    used: set[int] | None = None,
) -> list[Candidate]:
    selected: list[Candidate] = []
    blocked = set() if used is None else set(used)
    for candidate in candidates:
        if any(
            abs(candidate.frame_index - frame_index) < minimum_gap
            for frame_index in blocked
        ):
            continue
        selected.append(candidate)
        blocked.add(candidate.frame_index)
        if len(selected) >= count:
            break
    return selected


def draw_preview(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    names = ("cell", "droplet")
    for item in detections:
        x1, y1, x2, y2 = item.box
        points = (
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        )
        color = COLORS[item.class_id]
        cv2.rectangle(output, points[:2], points[2:], color, 2)
        cv2.putText(
            output,
            f"{names[item.class_id]} {item.confidence:.2f}",
            (points[0] + 2, max(15, points[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path)) for path in paths[:30]]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumbnails = [cv2.resize(image, (256, 256)) for image in images]
    while len(thumbnails) % 5:
        thumbnails.append(np.full_like(thumbnails[0], 245))
    rows = [
        np.hstack(thumbnails[index : index + 5])
        for index in range(0, len(thumbnails), 5)
    ]
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    args = parse_args()
    if args.count <= 0 or not 0 <= args.hard_count <= args.count:
        raise ValueError("Require count > 0 and 0 <= hard-count <= count")
    if args.stride <= 0 or args.minimum_gap < 0:
        raise ValueError("stride must be positive and minimum-gap non-negative")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = TinyQuantDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    manifest = load_manifest(args.manifest)
    decoder = manifest["postprocessing"]["decoder"]
    roi_runtime = json.loads(args.roi_config.read_text(encoding="utf-8"))

    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_geometry = scaled_roi(width, height, roi_runtime["roi"])
    first = max(0, int(round(args.start_sec * fps)))
    end_sec = args.end_sec
    if end_sec is None:
        end_sec = max(args.start_sec, total_frames / fps - 3.0)
    last = min(total_frames - 1, int(round(end_sec * fps)))

    pending_tensors: list[torch.Tensor] = []
    pending_metadata: list[tuple[int, float, float, float, float]] = []
    candidates: list[Candidate] = []

    def infer_pending() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device)
        with torch.inference_mode():
            predictions = model(batch).cpu()
        decoded = decode_predictions(
            predictions,
            confidence_threshold=tuple(
                float(value) for value in decoder["confidence_thresholds"]
            ),
            nms_iou=decoder_nms_iou(decoder),
            box_constraints=decoder_box_constraints(decoder),
            box_calibration=decoder_box_calibration(decoder),
            pre_nms_topk=int(decoder["pre_nms_topk"]),
            max_detections=int(decoder["max_detections"]),
            anchors=config.anchors,
            slots_per_class=config.slots_per_class,
        )
        thresholds = tuple(
            float(value) for value in decoder["confidence_thresholds"]
        )
        for metadata, detections in zip(pending_metadata, decoded):
            uncertainty = sum(
                max(
                    0.0,
                    1.0
                    - abs(item.confidence - thresholds[item.class_id]) / 0.12,
                )
                for item in detections
            )
            candidates.append(
                Candidate(
                    frame_index=metadata[0],
                    source_time=metadata[1],
                    sharpness=metadata[2],
                    brightness=metadata[3],
                    contrast=metadata[4],
                    detections=detections,
                    uncertainty=uncertainty,
                )
            )
        pending_tensors.clear()
        pending_metadata.clear()

    x1, y1, x2, y2 = roi_geometry
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < first or frame_index > last or frame_index % args.stride:
            frame_index += 1
            continue
        roi = frame[y1:y2, x1:x2]
        canvas, _ = prepare_roi_input(
            roi,
            config.image_width,
            config.image_height,
            config.image_width,
            round(config.image_height * 0.75),
        )
        grayscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(
            np.ascontiguousarray(grayscale, dtype=np.float32) / 255.0
        ).unsqueeze(0)
        pending_tensors.append(tensor)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        pending_metadata.append(
            (
                frame_index,
                frame_index / fps,
                float(cv2.Laplacian(roi_gray, cv2.CV_64F).var()),
                float(roi_gray.mean()),
                float(roi_gray.std()),
            )
        )
        if len(pending_tensors) >= args.batch_size:
            infer_pending()
        frame_index += 1
    infer_pending()
    capture.release()
    if len(candidates) < args.count:
        raise RuntimeError(
            f"Only {len(candidates)} candidates available for count={args.count}"
        )

    cell_z = robust_z([float(item.cells) for item in candidates])
    droplet_z = robust_z([float(item.droplets) for item in candidates])
    sharpness_z = robust_z([item.sharpness for item in candidates])
    brightness_z = robust_z([item.brightness for item in candidates])
    contrast_z = robust_z([item.contrast for item in candidates])
    for index, candidate in enumerate(candidates):
        candidate.hard_score = float(
            1.4 * cell_z[index]
            + 1.1 * droplet_z[index]
            + 0.35 * sharpness_z[index]
            + 0.25 * brightness_z[index]
            + 0.25 * contrast_z[index]
            + 0.8 * candidate.uncertainty
        )
        candidate.reason = reason_for(candidate)

    hard_ranked = sorted(candidates, key=lambda item: item.hard_score, reverse=True)
    selected_hard = greedy_with_gap(
        hard_ranked,
        args.hard_count,
        args.minimum_gap,
    )
    used = {item.frame_index for item in selected_hard}
    remaining_count = args.count - len(selected_hard)
    representative: list[Candidate] = []
    if remaining_count:
        available = [
            item
            for item in sorted(candidates, key=lambda value: value.frame_index)
            if all(
                abs(item.frame_index - frame_index) >= args.minimum_gap
                for frame_index in used
            )
        ]
        targets = np.linspace(first, last, remaining_count)
        for target in targets:
            choices = [
                item
                for item in available
                if item.frame_index not in used
                and all(
                    abs(item.frame_index - frame_index) >= args.minimum_gap
                    for frame_index in used
                )
            ]
            if not choices:
                break
            chosen = min(choices, key=lambda item: abs(item.frame_index - target))
            chosen.reason = "representative"
            representative.append(chosen)
            used.add(chosen.frame_index)
    selected = sorted(selected_hard + representative, key=lambda item: item.frame_index)
    if len(selected) != args.count:
        raise RuntimeError(
            f"Selected {len(selected)} frames, expected {args.count}; reduce minimum-gap"
        )

    images_dir = output / "train" / "images"
    labels_dir = output / "train" / "labels"
    previews_dir = output / "previews"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    capture = cv2.VideoCapture(str(args.video.resolve()))
    for selection_index, candidate in enumerate(selected, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not reread frame {candidate.frame_index + 1}")
        roi = frame[y1:y2, x1:x2]
        canvas, _ = prepare_roi_input(roi, 384, 384, 384, 288)
        stem = f"3_4_single_roi_src{candidate.frame_index + 1:06d}"
        image_path = images_dir / f"{stem}.jpg"
        cv2.imwrite(str(image_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for item in candidate.detections:
            bx1, by1, bx2, by2 = (
                min(1.0, max(0.0, float(value))) for value in item.box
            )
            if bx2 <= bx1 or by2 <= by1:
                continue
            lines.append(
                f"{item.class_id} {(bx1 + bx2) / 2:.8f} "
                f"{(by1 + by2) / 2:.8f} {bx2 - bx1:.8f} {by2 - by1:.8f}"
            )
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="ascii",
        )
        preview = draw_preview(canvas, candidate.detections)
        preview_path = previews_dir / f"{stem}.jpg"
        cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        preview_paths.append(preview_path)
        rows.append(
            {
                "selection": selection_index,
                "source_frame": candidate.frame_index + 1,
                "source_time_sec": f"{candidate.source_time:.6f}",
                "reason": candidate.reason,
                "hard_score": f"{candidate.hard_score:.6f}",
                "cells_pseudo": candidate.cells,
                "droplets_pseudo": candidate.droplets,
                "sharpness": f"{candidate.sharpness:.6f}",
                "brightness": f"{candidate.brightness:.6f}",
                "contrast": f"{candidate.contrast:.6f}",
                "image": image_path.relative_to(output).as_posix(),
            }
        )
    capture.release()

    with (output / "candidate_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: train/images\n"
        "names:\n  0: cell\n  1: droplet\n",
        encoding="ascii",
    )
    (output / "README.md").write_text(
        "# Single-ROI active-learning review set\n\n"
        "All labels are model-generated suggestions, not ground truth. Review every "
        "cell and droplet box in Roboflow before adding these images to training. "
        "Delete false positives, add missed objects, and tighten loose boxes. Keep "
        "source-frame groups together when creating train/valid/test splits.\n",
        encoding="ascii",
    )
    make_contact_sheet(preview_paths, output / "preview_contact_sheet.jpg")
    summary = {
        "video": str(args.video.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "roi": {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        },
        "candidate_pool": len(candidates),
        "selected": len(selected),
        "hard_selected": len(selected_hard),
        "representative_selected": len(representative),
        "stride": args.stride,
        "minimum_gap_frames": args.minimum_gap,
        "warning": "Pseudo-labels require manual correction before training.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"REVIEW_DATASET_READY: {output}")


if __name__ == "__main__":
    main()
