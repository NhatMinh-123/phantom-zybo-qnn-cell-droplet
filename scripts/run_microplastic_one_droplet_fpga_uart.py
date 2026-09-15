#!/usr/bin/env python3
"""Run one-droplet video tracking with patch classification on Arty S7-25."""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_microplastic_one_droplet_hybrid as pipeline
from qnn.patch_preprocess import preprocess_patch
from send_microplastic_patch_uart import (
    decode_response_payload,
    detect_serial_port,
    load_manifest,
    pack_patch,
    quantize_normalized,
    transact,
)


PACKAGED_MANIFEST = ROOT / "manifest.json"
SPATIAL_WORKSPACE_MANIFEST = (
    ROOT
    / "exports"
    / "arty_s7_25_microplastic_patch32_w4a6_spatial_uart"
    / "manifest.json"
)
LEGACY_WORKSPACE_MANIFEST = (
    ROOT
    / "exports"
    / "arty_s7_25_microplastic_patch32_w4a6_uart"
    / "manifest.json"
)
DEFAULT_MANIFEST = (
    PACKAGED_MANIFEST
    if PACKAGED_MANIFEST.is_file()
    else (
        SPATIAL_WORKSPACE_MANIFEST
        if SPATIAL_WORKSPACE_MANIFEST.is_file()
        else LEGACY_WORKSPACE_MANIFEST
    )
)
MANIFEST_PATH = Path(
    os.environ.get("MICROPLASTIC_FPGA_MANIFEST", DEFAULT_MANIFEST)
).expanduser().resolve()
MANIFEST = load_manifest(MANIFEST_PATH)
GATE_THRESHOLD = float(MANIFEST["postprocessing"]["particle_probability_threshold"])
INPUT_THRESHOLDS = MANIFEST["preprocessing"]["input_quantization"]["thresholds"]
INPUT_TRANSFORM = MANIFEST["preprocessing"].get("image_transform", "raw")
CORE_CLOCK_HZ = int(MANIFEST["fpga"]["clock_hz"])

SERIAL_PORT: Any | None = None
PORT_NAME = ""
BAUD = int(MANIFEST["transport"]["baud"])
NEXT_FRAME_ID = 1
OUTPUT_PATH: Path | None = None
FPGA_CALL_ROWS: list[dict[str, Any]] = []
FPGA_ACCEPTED = 0


BaseTracker = pipeline.TemporalParticleTracker
ORIGINAL_DRAW_TEXT = pipeline.draw_text


def use_previous_patch_without_phase_alignment(previous, current, hanning_window):
    del current, hanning_window
    return previous, (0.0, 0.0), 0.0


def draw_fpga_text(image, text, position, **kwargs):
    text = text.replace("Hybrid one-droplet", "Arty S7-25 FPGA")
    text = text.replace("one-droplet ROI", "Arty S7-25 QNN ROI")
    return ORIGINAL_DRAW_TEXT(image, text, position, **kwargs)


def output_from_argv() -> Path | None:
    try:
        index = sys.argv.index("--output")
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1]).expanduser().resolve()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


class EventGatedFpgaTracker(BaseTracker):
    def _fpga_probability(self, track) -> float | None:
        value = getattr(track, "fpga_probability", None)
        return float(value) if value is not None else None

    def _track_confidence(self, track) -> float:
        probability = self._fpga_probability(track)
        if probability is None:
            return super()._track_confidence(track)
        persistence = min(track.hits / max(self.min_hits + 1, 1), 1.0)
        return float(np.clip(0.80 * probability + 0.20 * persistence, 0, 1))

    def is_confirmed(self, track) -> bool:
        probability = self._fpga_probability(track)
        return (
            track.hits >= self.min_hits
            and probability is not None
            and bool(getattr(track, "fpga_accepted", False))
            and self._track_confidence(track) >= self.confidence_threshold
        )

    def _save_patch(self, patch: np.ndarray, frame_index: int, track_id: int) -> str:
        if OUTPUT_PATH is None:
            return ""
        directory = OUTPUT_PATH / "fpga_patches"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"frame_{frame_index:05d}_track_{track_id:05d}.png"
        cv2.imwrite(str(path), patch)
        return str(path.resolve())

    def _classify_track(self, track, frame_index: int) -> None:
        global NEXT_FRAME_ID, FPGA_ACCEPTED
        if SERIAL_PORT is None:
            raise RuntimeError("FPGA serial port is not open")
        patch = np.asarray(track.best_patch, dtype=np.uint8)
        if patch.shape != (32, 32):
            patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
        patch = preprocess_patch(patch, INPUT_TRANSFORM)
        normalized = patch.astype(np.float32) / np.float32(255.0)
        payload = pack_patch(quantize_normalized(normalized, INPUT_THRESHOLDS))
        frame_id = NEXT_FRAME_ID
        NEXT_FRAME_ID = 1 if NEXT_FRAME_ID >= 0xFFFF else NEXT_FRAME_ID + 1

        transaction = None
        last_error: Exception | None = None
        attempts = 0
        for attempts in range(1, 3):
            try:
                transaction = transact(SERIAL_PORT, payload, frame_id, 3.0)
                break
            except Exception as error:  # Retry one transport interruption.
                last_error = error
                SERIAL_PORT.reset_input_buffer()
                SERIAL_PORT.reset_output_buffer()
                time.sleep(0.03)
        if transaction is None:
            assert last_error is not None
            raise RuntimeError(
                f"FPGA classification failed after {attempts} attempts"
            ) from last_error

        decoded = decode_response_payload(transaction.pop("payload"), MANIFEST)
        track.fpga_probability = float(decoded["particle_probability"])
        track.fpga_raw_sum = int(decoded["raw_sum"])
        track.fpga_accepted = bool(decoded["accepted"])
        if track.fpga_accepted:
            FPGA_ACCEPTED += 1
        patch_path = self._save_patch(patch, frame_index, track.track_id)
        cycles = int(transaction["accelerator_cycles"])
        FPGA_CALL_ROWS.append(
            {
                "frame": frame_index,
                "droplet_sequence": getattr(track, "droplet_sequence", ""),
                "track_id": track.track_id,
                "track_hits": track.hits,
                "frame_id": frame_id,
                "attempts": attempts,
                "raw_sum": decoded["raw_sum"],
                "output_code": decoded["output_code"],
                "logit": decoded["logit"],
                "particle_probability": decoded["particle_probability"],
                "accepted": int(decoded["accepted"]),
                "accelerator_cycles": cycles,
                "accelerator_ms": cycles / CORE_CLOCK_HZ * 1000.0,
                "accelerator_patch_rate": CORE_CLOCK_HZ / cycles if cycles else 0.0,
                "uart_round_trip_ms": float(transaction["round_trip_seconds"]) * 1000.0,
                "patch": patch_path,
            }
        )

    def _classify_new_mature_tracks(self, frame_index: int) -> None:
        tracks = [
            track
            for track in self.tracks
            if track.hits >= self.min_hits
            and self._fpga_probability(track) is None
            and track.best_patch is not None
        ]
        for track in tracks:
            self._classify_track(track, frame_index)

    def update(self, candidates, gray_patch, frame_index):
        visible, finished = super().update(candidates, gray_patch, frame_index)
        self._classify_new_mature_tracks(frame_index)
        tracks_by_id = {track.track_id: track for track in self.tracks}
        for item in visible:
            track = tracks_by_id[item.track_id]
            item.confidence = self._track_confidence(track)
            item.confirmed = self.is_confirmed(track)
        return visible, finished


def write_fpga_report(output: Path) -> None:
    calls_path = output / "fpga_calls.csv"
    fields = [
        "frame",
        "droplet_sequence",
        "track_id",
        "track_hits",
        "frame_id",
        "attempts",
        "raw_sum",
        "output_code",
        "logit",
        "particle_probability",
        "accepted",
        "accelerator_cycles",
        "accelerator_ms",
        "accelerator_patch_rate",
        "uart_round_trip_ms",
        "patch",
    ]
    with calls_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(FPGA_CALL_ROWS)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cycles = [float(row["accelerator_cycles"]) for row in FPGA_CALL_ROWS]
    uart_ms = [float(row["uart_round_trip_ms"]) for row in FPGA_CALL_ROWS]
    core_ms = [float(row["accelerator_ms"]) for row in FPGA_CALL_ROWS]
    summary["pipeline"] = "one_droplet_event_gated_arty_s7_qnn_uart_v1"
    summary["accuracy_status"] = (
        "FPGA inference is numerically exact against FINN for the release test "
        "vectors. Video detections still require independent manual ground truth."
    )
    summary["fpga_classifier"] = {
        "board": MANIFEST["board"],
        "device": MANIFEST["device"],
        "manifest": str(MANIFEST_PATH),
        "expected_bitstream": MANIFEST["fpga"]["bitstream"],
        "port": PORT_NAME,
        "baud": BAUD,
        "gate_threshold": GATE_THRESHOLD,
        "invocation_policy": "one cached FPGA inference after temporal min-hits gate",
        "calls": len(FPGA_CALL_ROWS),
        "accepted": FPGA_ACCEPTED,
        "acceptance_ratio": FPGA_ACCEPTED / max(len(FPGA_CALL_ROWS), 1),
        "hardware_release_exactness": MANIFEST["verification"].get(
            "hardware", {"status": "pending"}
        ),
        "mean_accelerator_cycles": statistics.fmean(cycles) if cycles else 0.0,
        "mean_accelerator_ms": statistics.fmean(core_ms) if core_ms else 0.0,
        "mean_accelerator_patch_rate": (
            CORE_CLOCK_HZ / statistics.fmean(cycles) if cycles else 0.0
        ),
        "mean_uart_round_trip_ms": statistics.fmean(uart_ms) if uart_ms else 0.0,
        "p95_uart_round_trip_ms": percentile(uart_ms, 95),
    }
    summary["artifacts"]["fpga_calls_csv"] = str(calls_path.resolve())
    summary["artifacts"]["fpga_patch_directory"] = str(
        (output / "fpga_patches").resolve()
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "FPGA_RUN_README.md").write_text(
        "\n".join(
            [
                "# Arty S7-25 video run",
                "",
                f"- Source: `{summary['source']['path']}`",
                f"- Frames: `{summary['source']['frames_processed']}`",
                f"- FPGA calls: `{len(FPGA_CALL_ROWS)}`",
                f"- Accepted tracks: `{FPGA_ACCEPTED}`",
                f"- Mean core time: `{summary['fpga_classifier']['mean_accelerator_ms']:.3f} ms/patch`",
                f"- Mean UART round trip: `{summary['fpga_classifier']['mean_uart_round_trip_ms']:.3f} ms/patch`",
                f"- End-to-end wall FPS: `{summary['timing']['wall_fps_including_io_and_encoding']:.2f}`",
                "",
                "The source file reports 30 FPS even though its filename contains 100fps.",
                "Video detections are not a ground-truth accuracy measurement.",
            ]
        )
        + "\n",
        encoding="ascii",
    )


def main() -> None:
    global SERIAL_PORT, PORT_NAME, OUTPUT_PATH
    import serial

    OUTPUT_PATH = output_from_argv()
    PORT_NAME = os.environ.get("MICROPLASTIC_FPGA_PORT", "") or detect_serial_port()
    requested_baud = os.environ.get("MICROPLASTIC_FPGA_BAUD")
    baud = int(requested_baud) if requested_baud else BAUD
    print(
        f"Arty S7-25 patch QNN: port={PORT_NAME}, baud={baud}, "
        f"gate={GATE_THRESHOLD:.3f}, manifest={MANIFEST_PATH}"
    )
    with serial.Serial(PORT_NAME, baud, timeout=0.05, write_timeout=3.0) as port:
        SERIAL_PORT = port
        port.reset_input_buffer()
        port.reset_output_buffer()
        time.sleep(0.05)
        pipeline.main()
        SERIAL_PORT = None
    if OUTPUT_PATH is not None:
        write_fpga_report(OUTPUT_PATH)
        print(f"FPGA video results: {OUTPUT_PATH}")


pipeline.align_previous_patch = use_previous_patch_without_phase_alignment
pipeline.draw_text = draw_fpga_text
pipeline.TemporalParticleTracker = EventGatedFpgaTracker


if __name__ == "__main__":
    main()
