#!/usr/bin/env python3
"""Run the feature-based weak-supervision model on a real video.

The detector deliberately reuses the exact droplet localization and
candidate-feature functions used to build the training table. The output is
an annotated video plus frame/track tables for visual and timing review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_droplet_microplastic_features import (  # noqa: E402
    CandidateFeatureTracker,
    CandidateTrack,
    DropletObservation,
    DropletSequenceTracker,
    Rect,
    calibrate_dominant_circle,
    centered_crop,
    crop_rect,
    extract_candidates,
    hough_candidates,
)
from scripts.bright_round_feature_gate import (  # noqa: E402
    BrightRoundGateConfig,
    evaluate_bright_round_candidate,
)
from scripts.run_microplastic_one_droplet_hybrid import (  # noqa: E402
    draw_text,
    open_writer,
)
from scripts.train_weak_supervised_feature_models import (  # noqa: E402
    FEATURES,
    prepare_candidate_features,
)


DEFAULT_MODEL = (
    ROOT
    / "models"
    / "bright_round_feature_tree_v1"
    / "recommended_fpga_feature_model.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the feature-based Decision Tree on a real microscope video "
            "and export an annotated review video."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Negative reads the Decision Tree threshold from summary.json.",
    )
    parser.add_argument("--minimum-track-hits", type=int, default=2)
    parser.add_argument(
        "--search-roi-x",
        type=int,
        default=570,
        help="Fixed sampling-gate ROI x coordinate; tuned for droplet 100fps.mp4.",
    )
    parser.add_argument(
        "--search-roi-y",
        type=int,
        default=58,
        help="Fixed sampling-gate ROI y coordinate in the straight channel.",
    )
    parser.add_argument("--search-roi-width", type=int, default=160)
    parser.add_argument("--search-roi-height", type=int, default=160)
    parser.add_argument("--processing-size", type=int, default=128)
    parser.add_argument("--core-radius-ratio", type=float, default=0.72)
    parser.add_argument("--blackhat-kernel", type=int, default=7)
    parser.add_argument("--blackhat-sigma", type=float, default=3.2)
    parser.add_argument("--temporal-sigma", type=float, default=2.5)
    parser.add_argument("--particle-min-area", type=int, default=2)
    parser.add_argument("--particle-max-area", type=int, default=100)
    parser.add_argument("--particle-max-side", type=int, default=18)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--disable-bright-round-gate",
        action="store_true",
        help="Disable the strict black-hat brightness and roundness gate.",
    )
    parser.add_argument("--gate-min-blackhat-peak", type=float, default=20.0)
    parser.add_argument("--gate-min-blackhat-mean", type=float, default=8.0)
    parser.add_argument("--gate-min-axis-ratio", type=float, default=0.55)
    parser.add_argument("--gate-min-circularity", type=float, default=0.42)
    parser.add_argument("--gate-min-solidity", type=float, default=0.55)
    parser.add_argument("--gate-min-extent", type=float, default=0.50)
    parser.add_argument(
        "--candidate-nms-distance",
        type=float,
        default=8.0,
        help="Keep only the highest-probability candidate within this radius.",
    )
    parser.add_argument(
        "--max-kept-candidates",
        type=int,
        default=0,
        help="Maximum candidates per frame after NMS; zero keeps all.",
    )
    parser.add_argument("--track-max-distance", type=float, default=10.0)
    parser.add_argument("--track-max-missed", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--zoom-size", type=int, default=300)
    parser.add_argument(
        "--disable-zoom-inset",
        action="store_true",
        help="Skip the diagnostic processing-patch inset for faster output.",
    )
    parser.add_argument(
        "--show-candidate-labels",
        action="store_true",
        help="Draw per-candidate track/probability text; boxes remain visible.",
    )
    parser.add_argument(
        "--show-localization-guides",
        action="store_true",
        help="Draw the search ROI and droplet/core circles for diagnostics.",
    )
    parser.add_argument("--codec", default="mp4v")
    return parser.parse_args()


def load_threshold(model_path: Path, requested: float) -> float:
    if requested >= 0:
        return float(requested)
    summary_path = model_path.parent / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "No explicit --threshold and summary.json was not found beside "
            f"the model: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["thresholds"]["decision_tree"])


def candidate_probabilities(model, candidates) -> np.ndarray:
    if not candidates:
        return np.empty(0, dtype=np.float64)
    rows = pd.DataFrame([candidate.row for candidate in candidates])
    prepared = prepare_candidate_features(rows)
    return model.predict_proba(prepared[FEATURES])[:, 1]


def filter_bright_round_candidates(
    candidates,
    *,
    config: BrightRoundGateConfig,
):
    """Remove dim, diffuse, elongated, rim-like candidates before ML/tracking."""
    kept = []
    rejected = []
    for candidate in candidates:
        result = evaluate_bright_round_candidate(candidate.row, config)
        candidate.row["bright_round_gate_pass"] = int(result.passed)
        candidate.row["bright_round_gate_score"] = result.score
        candidate.row["bright_round_gate_reasons"] = "|".join(result.reasons)
        candidate.row.update(
            {
                f"bright_round_{name}": value
                for name, value in result.features.items()
            }
        )
        (kept if result.passed else rejected).append(candidate)
    return kept, rejected


def suppress_nearby_candidates(
    candidates,
    probabilities: np.ndarray,
    *,
    minimum_distance: float,
    maximum_candidates: int = 0,
):
    if len(candidates) != len(probabilities):
        raise ValueError("Candidate and probability counts do not match")
    if not candidates:
        return [], probabilities.copy()
    minimum_distance = max(0.0, minimum_distance)
    kept_indices: list[int] = []
    for index in np.argsort(-probabilities):
        x = float(candidates[index].row["center_x_px"])
        y = float(candidates[index].row["center_y_px"])
        if all(
            math.hypot(
                x - float(candidates[kept].row["center_x_px"]),
                y - float(candidates[kept].row["center_y_px"]),
            )
            >= minimum_distance
            for kept in kept_indices
        ):
            kept_indices.append(int(index))
            if maximum_candidates > 0 and len(kept_indices) >= maximum_candidates:
                break
    return (
        [candidates[index] for index in kept_indices],
        probabilities[np.asarray(kept_indices, dtype=np.int64)],
    )


def track_probability(track: CandidateTrack) -> float:
    probabilities = [
        float(observation.get("ml_probability", 0.0))
        for observation in track.observations
    ]
    return float(np.mean(probabilities)) if probabilities else 0.0


def track_record(
    track: CandidateTrack,
    *,
    threshold: float,
    minimum_hits: int,
) -> dict[str, object]:
    probabilities = [
        float(observation.get("ml_probability", 0.0))
        for observation in track.observations
    ]
    mean_probability = (
        float(np.mean(probabilities)) if probabilities else 0.0
    )
    return {
        "track_id": track.track_id,
        "droplet_sequence": track.sequence_id,
        "first_frame": track.first_frame,
        "last_frame": track.last_frame,
        "hits": len(track.observations),
        "duration_frames": track.last_frame - track.first_frame + 1,
        "mean_probability": mean_probability,
        "max_probability": (
            float(np.max(probabilities)) if probabilities else 0.0
        ),
        "predicted_particle": int(
            len(track.observations) >= minimum_hits
            and mean_probability >= threshold
        ),
    }


def candidate_global_box(
    row: dict[str, object],
    *,
    droplet_x: float,
    droplet_y: float,
    processing_size: int,
) -> tuple[int, int, int, int]:
    half = processing_size / 2.0
    x = int(round(droplet_x - half + float(row["bbox_x"])))
    y = int(round(droplet_y - half + float(row["bbox_y"])))
    width = max(1, int(row["bbox_width"]))
    height = max(1, int(row["bbox_height"]))
    return x, y, width, height


def draw_candidate(
    image: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    track_id: int,
    probability: float,
    hits: int,
    threshold: float,
    minimum_hits: int,
    label: bool,
) -> None:
    accepted = probability >= threshold
    confirmed = accepted and hits >= minimum_hits
    if confirmed:
        color = (40, 220, 40)
        thickness = 2
    elif accepted:
        color = (0, 180, 255)
        thickness = 1
    else:
        color = (130, 130, 130)
        thickness = 1
    x, y, width, height = bbox
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        thickness,
    )
    if label and accepted:
        draw_text(
            image,
            f"T{track_id} p={probability:.2f} h={hits}",
            (max(0, x), max(13, y - 3)),
            color=color,
            scale=0.35,
        )


def add_zoom_inset(
    annotated: np.ndarray,
    processing_patch: np.ndarray,
    *,
    candidates,
    assignments: list[int],
    active_tracks: dict[int, CandidateTrack],
    threshold: float,
    minimum_hits: int,
    zoom_size: int,
    show_labels: bool,
) -> None:
    zoom = cv2.cvtColor(processing_patch, cv2.COLOR_GRAY2BGR)
    for candidate, track_id in zip(candidates, assignments):
        track = active_tracks[track_id]
        probability = track_probability(track)
        local_box = (
            int(candidate.row["bbox_x"]),
            int(candidate.row["bbox_y"]),
            int(candidate.row["bbox_width"]),
            int(candidate.row["bbox_height"]),
        )
        draw_candidate(
            zoom,
            bbox=local_box,
            track_id=track_id,
            probability=probability,
            hits=len(track.observations),
            threshold=threshold,
            minimum_hits=minimum_hits,
            label=show_labels,
        )

    zoom = cv2.resize(
        zoom,
        (zoom_size, zoom_size),
        interpolation=cv2.INTER_NEAREST,
    )
    height, width = annotated.shape[:2]
    inset_x = max(8, width - zoom_size - 12)
    inset_y = max(54, height - zoom_size - 12)
    x2 = min(width, inset_x + zoom_size)
    y2 = min(height, inset_y + zoom_size)
    zoom = zoom[: y2 - inset_y, : x2 - inset_x]
    cv2.rectangle(
        annotated,
        (inset_x - 3, inset_y - 24),
        (x2 + 3, y2 + 3),
        (0, 0, 0),
        -1,
    )
    annotated[inset_y:y2, inset_x:x2] = zoom
    draw_text(
        annotated,
        "Processing ROI 128x128",
        (inset_x, inset_y - 7),
        color=(255, 255, 255),
        scale=0.42,
    )


def save_review_samples(
    output: Path,
    samples: list[tuple[int, int, np.ndarray]],
    *,
    sample_count: int,
) -> list[str]:
    sample_dir = output / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(samples, key=lambda item: (-item[0], item[1]))
    selected = selected[:sample_count]
    selected.sort(key=lambda item: item[1])
    paths: list[str] = []
    thumbs: list[np.ndarray] = []
    for score, frame_index, image in selected:
        path = sample_dir / f"frame_{frame_index:06d}_score_{score:03d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(str(path.resolve()))
        thumb_width = 360
        thumb_height = 240
        scale = min(
            thumb_width / image.shape[1],
            (thumb_height - 25) / image.shape[0],
        )
        resized = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
        )
        canvas = np.zeros((thumb_height, thumb_width, 3), dtype=np.uint8)
        x = (thumb_width - resized.shape[1]) // 2
        y = (thumb_height - 25 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        draw_text(
            canvas,
            f"frame={frame_index} review_score={score}",
            (6, thumb_height - 7),
            scale=0.4,
        )
        thumbs.append(canvas)
    if thumbs:
        columns = min(3, len(thumbs))
        rows = int(math.ceil(len(thumbs) / columns))
        blank = np.zeros_like(thumbs[0])
        padded = thumbs + [blank] * (rows * columns - len(thumbs))
        sheet = np.vstack(
            [
                np.hstack(padded[row * columns : (row + 1) * columns])
                for row in range(rows)
            ]
        )
        cv2.imwrite(str(output / "review_contact_sheet.jpg"), sheet)
    return paths


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    model_path = args.model.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    threshold = load_threshold(model_path, args.threshold)
    model = joblib.load(model_path)
    gate_config = BrightRoundGateConfig(
        min_blackhat_peak=args.gate_min_blackhat_peak,
        min_blackhat_mean=args.gate_min_blackhat_mean,
        min_axis_ratio=args.gate_min_axis_ratio,
        min_circularity=args.gate_min_circularity,
        min_solidity=args.gate_min_solidity,
        min_extent=args.gate_min_extent,
    )
    search_roi = Rect(
        args.search_roi_x,
        args.search_roi_y,
        args.search_roi_width,
        args.search_roi_height,
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    search_roi.validate(frame_width, frame_height)

    calibration_start = time.perf_counter()
    calibrated_x, calibrated_radius, calibration_cluster_size = (
        calibrate_dominant_circle(
            capture,
            frame_count=frame_count,
            search_roi=search_roi,
        )
    )
    calibration_seconds = time.perf_counter() - calibration_start
    min_radius = max(18, int(round(calibrated_radius - 11)))
    max_radius = min(94, int(round(calibrated_radius + 11)))

    writer = open_writer(
        output / "feature_ml_annotated.mp4",
        codec=args.codec,
        fps=source_fps if source_fps > 0 else 30.0,
        width=frame_width,
        height=frame_height,
    )
    sequence_tracker = DropletSequenceTracker(
        new_droplet_jump=55.0,
        max_missing=3,
    )
    candidate_tracker = CandidateFeatureTracker(
        max_distance=args.track_max_distance,
        max_missed=args.track_max_missed,
    )

    previous_patch: np.ndarray | None = None
    previous_sequence = 0
    frame_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    algorithm_times: list[float] = []
    model_times: list[float] = []
    frame_times: list[float] = []
    best_sequence_samples: dict[int, tuple[int, int, np.ndarray]] = {}
    uniform_samples: list[tuple[int, int, np.ndarray]] = []
    localized_frames = 0
    candidate_observations = 0
    raw_candidate_observations = 0
    gate_pass_observations = 0
    gate_rejected_observations = 0
    accepted_observations = 0
    confirmed_observations = 0
    processed_frames = 0
    sample_stride = max(
        1,
        min(
            frame_count,
            args.max_frames if args.max_frames else frame_count,
        )
        // max(args.sample_count, 1),
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    wall_start = time.perf_counter()
    frame_index = 0
    while True:
        if args.max_frames and frame_index >= args.max_frames:
            break
        frame_start = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        algorithm_start = time.perf_counter()
        annotated = frame.copy()
        if args.show_localization_guides:
            cv2.rectangle(
                annotated,
                (search_roi.x, search_roi.y),
                (search_roi.x2, search_roi.y2),
                (255, 200, 0),
                1,
            )

        roi = crop_rect(frame, search_roi)
        circles = hough_candidates(
            roi,
            min_radius=min_radius,
            max_radius=max_radius,
            accumulator_threshold=25,
        )
        matches = [
            circle
            for circle in circles
            if abs(circle[0] - calibrated_x) <= 14.0
            and abs(circle[2] - calibrated_radius) <= 10.0
        ]
        selected = (
            min(
                matches,
                key=lambda item: (
                    abs(item[0] - calibrated_x)
                    + 1.1 * abs(item[2] - calibrated_radius)
                ),
            )
            if matches
            else None
        )

        raw_observation = None
        global_circle = None
        if selected is not None:
            local_x, local_y, radius = selected
            global_x = search_roi.x + local_x
            global_y = search_roi.y + local_y
            half = args.processing_size / 2.0
            if (
                global_x >= half
                and global_x < frame_width - half
                and global_y >= half
                and global_y < frame_height - half
            ):
                global_circle = (global_x, global_y, radius)
                raw_observation = DropletObservation(
                    center_x=local_x,
                    center_y=local_y,
                    radius=radius,
                    area=float(math.pi * radius * radius),
                    score=radius,
                    bbox=(
                        int(round(local_x - radius)),
                        int(round(local_y - radius)),
                        int(round(2 * radius)),
                        int(round(2 * radius)),
                    ),
                )

        _, is_new_sequence = sequence_tracker.update(
            raw_observation,
            frame_index,
        )
        if is_new_sequence and previous_sequence:
            for track in candidate_tracker.reset():
                track_rows.append(
                    track_record(
                        track,
                        threshold=threshold,
                        minimum_hits=args.minimum_track_hits,
                    )
                )
            previous_patch = None

        sequence_id = sequence_tracker.sequence_id
        candidates = []
        assignments: list[int] = []
        accepted_current = 0
        confirmed_current = 0
        raw_candidate_count = 0
        gate_pass_count = 0
        gate_rejected_count = 0
        model_ms = 0.0
        if global_circle is not None:
            global_x, global_y, radius = global_circle
            localized_frames += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processing_patch = centered_crop(
                gray_frame,
                global_x,
                global_y,
                args.processing_size,
            )
            core_radius = min(
                args.processing_size // 2 - 4,
                max(10, int(round(radius * args.core_radius_ratio))),
            )
            candidates, _ = extract_candidates(
                processing_patch,
                previous_patch,
                video_slug="practical_test",
                video_name=source.name,
                frame_index=frame_index,
                timestamp_sec=(
                    frame_index / source_fps if source_fps > 0 else 0.0
                ),
                sequence_id=sequence_id,
                droplet_radius_px=radius,
                core_radius_px=core_radius,
                patch_size=32,
                blackhat_kernel=args.blackhat_kernel,
                blackhat_sigma=args.blackhat_sigma,
                temporal_sigma=args.temporal_sigma,
                min_area=args.particle_min_area,
                max_area=args.particle_max_area,
                max_side=args.particle_max_side,
                max_candidates=args.max_candidates,
            )
            raw_candidate_count = len(candidates)
            if args.disable_bright_round_gate:
                gate_pass_count = raw_candidate_count
            else:
                candidates, gate_rejected = filter_bright_round_candidates(
                    candidates,
                    config=gate_config,
                )
                gate_pass_count = len(candidates)
                gate_rejected_count = len(gate_rejected)
            model_start = time.perf_counter()
            probabilities = candidate_probabilities(model, candidates)
            candidates, probabilities = suppress_nearby_candidates(
                candidates,
                probabilities,
                minimum_distance=args.candidate_nms_distance,
                maximum_candidates=args.max_kept_candidates,
            )
            model_ms = (time.perf_counter() - model_start) * 1000.0
            for candidate, probability in zip(candidates, probabilities):
                candidate.row["ml_probability"] = float(probability)
            finished_tracks, assignments = candidate_tracker.update(
                candidates,
                frame_index=frame_index,
                sequence_id=sequence_id,
            )
            for track in finished_tracks:
                track_rows.append(
                    track_record(
                        track,
                        threshold=threshold,
                        minimum_hits=args.minimum_track_hits,
                    )
                )

            active_tracks = {
                track.track_id: track for track in candidate_tracker.active
            }
            if args.show_localization_guides:
                cv2.circle(
                    annotated,
                    (int(round(global_x)), int(round(global_y))),
                    int(round(radius)),
                    (0, 220, 255),
                    2,
                )
                cv2.circle(
                    annotated,
                    (int(round(global_x)), int(round(global_y))),
                    core_radius,
                    (255, 120, 0),
                    1,
                )
            for candidate, track_id in zip(candidates, assignments):
                track = active_tracks[track_id]
                probability = track_probability(track)
                accepted = probability >= threshold
                confirmed = (
                    accepted
                    and len(track.observations) >= args.minimum_track_hits
                )
                accepted_current += int(accepted)
                confirmed_current += int(confirmed)
                draw_candidate(
                    annotated,
                    bbox=candidate_global_box(
                        candidate.row,
                        droplet_x=global_x,
                        droplet_y=global_y,
                        processing_size=args.processing_size,
                    ),
                    track_id=track_id,
                    probability=probability,
                    hits=len(track.observations),
                    threshold=threshold,
                    minimum_hits=args.minimum_track_hits,
                    label=args.show_candidate_labels and confirmed,
                )
            if not args.disable_zoom_inset:
                add_zoom_inset(
                    annotated,
                    processing_patch,
                    candidates=candidates,
                    assignments=assignments,
                    active_tracks=active_tracks,
                    threshold=threshold,
                    minimum_hits=args.minimum_track_hits,
                    zoom_size=min(
                        args.zoom_size,
                        frame_width - 24,
                        frame_height - 70,
                    ),
                    show_labels=args.show_candidate_labels,
                )
            previous_patch = processing_patch
            previous_sequence = sequence_id
        else:
            previous_patch = None
            if frame_index - sequence_tracker.last_frame > 3:
                for track in candidate_tracker.reset():
                    track_rows.append(
                        track_record(
                            track,
                            threshold=threshold,
                            minimum_hits=args.minimum_track_hits,
                        )
                    )

        algorithm_ms = (time.perf_counter() - algorithm_start) * 1000.0
        algorithm_fps = 1000.0 / max(algorithm_ms, 1e-9)
        cv2.rectangle(
            annotated,
            (0, 0),
            (frame_width, 52),
            (8, 12, 12),
            -1,
        )
        draw_text(
            annotated,
            (
                f"Feature ML DecisionTree | frame={frame_index} "
                f"seq={sequence_id} raw={raw_candidate_count} "
                f"gate={gate_pass_count} kept={len(candidates)} "
                f"accepted={accepted_current} confirmed={confirmed_current}"
            ),
            (10, 21),
            color=(255, 255, 255),
            scale=0.48,
        )
        draw_text(
            annotated,
            (
                f"p>={threshold:.2f}, hits>={args.minimum_track_hits} | "
                f"algorithm={algorithm_fps:.1f} FPS | "
                "green=confirmed orange=pending; rejected candidates hidden"
            ),
            (10, 44),
            color=(190, 235, 255),
            scale=0.44,
        )
        writer.write(annotated)

        candidate_observations += len(candidates)
        raw_candidate_observations += raw_candidate_count
        gate_pass_observations += gate_pass_count
        gate_rejected_observations += gate_rejected_count
        accepted_observations += accepted_current
        confirmed_observations += confirmed_current
        algorithm_times.append(algorithm_ms)
        model_times.append(model_ms)
        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        frame_times.append(frame_ms)
        frame_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": (
                    frame_index / source_fps if source_fps > 0 else 0.0
                ),
                "droplet_found": int(global_circle is not None),
                "droplet_sequence": sequence_id,
                "raw_candidate_count": raw_candidate_count,
                "gate_pass_count": gate_pass_count,
                "gate_rejected_count": gate_rejected_count,
                "candidate_count": len(candidates),
                "accepted_count": accepted_current,
                "confirmed_count": confirmed_current,
                "model_inference_ms": model_ms,
                "algorithm_ms": algorithm_ms,
                "frame_total_ms": frame_ms,
            }
        )

        review_score = (
            confirmed_current * 100
            + accepted_current * 10
            + len(candidates)
        )
        if global_circle is not None:
            previous_best = best_sequence_samples.get(sequence_id)
            if previous_best is None or review_score > previous_best[0]:
                best_sequence_samples[sequence_id] = (
                    review_score,
                    frame_index,
                    annotated.copy(),
                )
        if frame_index % sample_stride == 0:
            uniform_samples.append((review_score, frame_index, annotated.copy()))
        processed_frames += 1
        frame_index += 1

    for track in candidate_tracker.reset():
        track_rows.append(
            track_record(
                track,
                threshold=threshold,
                minimum_hits=args.minimum_track_hits,
            )
        )
    wall_seconds = time.perf_counter() - wall_start
    capture.release()
    writer.release()

    sample_pool = list(best_sequence_samples.values()) + uniform_samples
    deduplicated_samples = {
        frame_index: (score, frame_index, image)
        for score, frame_index, image in sample_pool
    }
    sample_paths = save_review_samples(
        output,
        list(deduplicated_samples.values()),
        sample_count=args.sample_count,
    )
    write_csv(output / "per_frame.csv", frame_rows)
    write_csv(output / "particle_tracks.csv", track_rows)

    confirmed_tracks = sum(
        int(row["predicted_particle"]) for row in track_rows
    )
    summary = {
        "experiment": "feature_ml_practical_video_test",
        "source": str(source),
        "model": str(model_path),
        "interpretation": (
            "Candidates must pass the explicit compact high-black-hat-response "
            "gate before Decision Tree scoring and temporal confirmation. "
            "This target definition still requires manual ground-truth review "
            "for biological accuracy."
        ),
        "video": {
            "width": frame_width,
            "height": frame_height,
            "source_fps": source_fps,
            "source_frames": frame_count,
            "processed_frames": processed_frames,
        },
        "calibration": {
            "center_x_in_search_roi": calibrated_x,
            "radius_px": calibrated_radius,
            "cluster_size": calibration_cluster_size,
            "seconds": calibration_seconds,
        },
        "decision": {
            "threshold": threshold,
            "minimum_track_hits": args.minimum_track_hits,
            "zoom_inset_enabled": not args.disable_zoom_inset,
            "candidate_labels_enabled": args.show_candidate_labels,
            "localization_guides_enabled": args.show_localization_guides,
            "bright_round_gate_enabled": not args.disable_bright_round_gate,
            "bright_round_gate": {
                "min_blackhat_peak": gate_config.min_blackhat_peak,
                "min_blackhat_mean": gate_config.min_blackhat_mean,
                "min_peak_ratio": gate_config.min_peak_ratio,
                "min_mean_ratio": gate_config.min_mean_ratio,
                "min_area": gate_config.min_area,
                "max_area": gate_config.max_area,
                "min_axis_ratio": gate_config.min_axis_ratio,
                "min_circularity": gate_config.min_circularity,
                "min_solidity": gate_config.min_solidity,
                "min_extent": gate_config.min_extent,
                "max_radial_distance": gate_config.max_radial_distance,
            },
            "fixed_sampling_roi": {
                "x": search_roi.x,
                "y": search_roi.y,
                "width": search_roi.width,
                "height": search_roi.height,
                "pixels": search_roi.width * search_roi.height,
                "baseline_pixels": 320 * 360,
                "pixel_reduction_fraction": 1.0
                - (search_roi.width * search_roi.height) / (320.0 * 360.0),
            },
        },
        "observations": {
            "localized_droplet_frames": localized_frames,
            "droplet_sequences": sequence_tracker.sequence_id,
            "candidate_observations": candidate_observations,
            "raw_candidate_observations": raw_candidate_observations,
            "gate_pass_observations": gate_pass_observations,
            "gate_rejected_observations": gate_rejected_observations,
            "accepted_observations": accepted_observations,
            "confirmed_observations": confirmed_observations,
            "finished_tracks": len(track_rows),
            "predicted_particle_tracks": confirmed_tracks,
        },
        "timing": {
            "algorithm_mean_ms": float(np.mean(algorithm_times)),
            "algorithm_p95_ms": percentile(algorithm_times, 95),
            "algorithm_fps_from_mean": (
                1000.0 / max(float(np.mean(algorithm_times)), 1e-9)
            ),
            "model_mean_ms": float(np.mean(model_times)),
            "model_p95_ms": percentile(model_times, 95),
            "frame_total_mean_ms_with_encoding": float(np.mean(frame_times)),
            "end_to_end_fps_with_encoding": (
                processed_frames / max(wall_seconds, 1e-9)
            ),
        },
        "artifacts": {
            "annotated_video": str(
                (output / "feature_ml_annotated.mp4").resolve()
            ),
            "per_frame_csv": str((output / "per_frame.csv").resolve()),
            "particle_tracks_csv": str(
                (output / "particle_tracks.csv").resolve()
            ),
            "review_contact_sheet": str(
                (output / "review_contact_sheet.jpg").resolve()
            ),
            "sample_frames": sample_paths,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
