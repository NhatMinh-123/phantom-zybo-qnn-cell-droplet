#!/usr/bin/env python3
"""Plot realtime detection, crossing counts, and processing speed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=15)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    with args.frames.resolve().open(newline="", encoding="ascii") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError("Frame summary is empty")

    time_s = np.array([float(row["time_s"]) for row in rows])
    time_s -= time_s[0]
    detected_cell = np.array([float(row["detected_cell"]) for row in rows])
    detected_droplet = np.array([float(row["detected_droplet"]) for row in rows])
    counted_cell = np.array([float(row["counted_cell"]) for row in rows])
    counted_droplet = np.array([float(row["counted_droplet"]) for row in rows])
    processing_fps = np.array([float(row["processing_fps"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(
        time_s,
        moving_average(detected_cell, args.window),
        label="Visible cell",
        color="#2f9e44",
    )
    axes[0].plot(
        time_s,
        moving_average(detected_droplet, args.window),
        label="Visible droplet",
        color="#1971c2",
    )
    axes[0].set_ylabel("Objects in ROI")
    axes[0].set_title("One-ROI cell and droplet realtime pipeline")
    axes[0].legend()

    axes[1].step(time_s, counted_cell, where="post", label="Counted cell", color="#2f9e44")
    axes[1].step(
        time_s,
        counted_droplet,
        where="post",
        label="Counted droplet",
        color="#1971c2",
    )
    axes[1].set_ylabel("Cumulative crossings")
    axes[1].legend()

    steady_fps = processing_fps.copy()
    if len(steady_fps) > 5:
        steady_fps[:5] = np.median(steady_fps[5:])
    axes[2].plot(
        time_s,
        moving_average(steady_fps, args.window),
        color="#e67700",
        label="Processing FPS",
    )
    axes[2].axhline(30.0, color="#c92a2a", linestyle="--", linewidth=1, label="30 FPS source")
    axes[2].set_ylabel("FPS")
    axes[2].set_xlabel("Elapsed video time (s)")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.resolve(), dpi=180)
    plt.close(fig)
    print(f"Plot: {args.output.resolve()}")


if __name__ == "__main__":
    main()
