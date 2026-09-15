#!/usr/bin/env python3
"""Send two-class detection boxes to the Arty S7 over USB-UART."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--port", default="COM12")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--limit", type=int, default=0, help="Frames to send; 0 sends all")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_detections(path: Path) -> dict[int, list[dict[str, str]]]:
    frames: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="ascii") as csv_file:
        for row in csv.DictReader(csv_file):
            frames[int(row["frame"])].append(row)
    return dict(sorted(frames.items()))


def encode_box(row: dict[str, str]) -> str:
    confidence = int(round(float(row["confidence"]) * 1000.0))
    return "{} {} {} {} {} {}".format(
        int(row["class_id"]),
        int(row["frame_x1"]),
        int(row["frame_y1"]),
        int(row["frame_x2"]),
        int(row["frame_y2"]),
        confidence,
    )


def transact(uart: serial.Serial, line: str, expected: bytes) -> bytes:
    uart.write(line.encode("ascii") + b"\n")
    uart.flush()
    response = uart.read(1)
    if response != expected:
        shown = response.decode("ascii", errors="replace") if response else "TIMEOUT"
        raise RuntimeError(
            f"FPGA response {shown!r}, expected {expected.decode()!r}, line={line!r}"
        )
    return response


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    frames = read_detections(args.detections.resolve())
    frame_items = list(frames.items())[:: args.frame_step]
    if args.limit > 0:
        frame_items = frame_items[: args.limit]
    if not frame_items:
        raise RuntimeError("No detections found")

    if args.dry_run:
        box_total = sum(len(rows) for _, rows in frame_items)
        print(f"Dry run: frames={len(frame_items)} boxes={box_total}")
        for frame_index, rows in frame_items[:2]:
            print(f"F  # frame {frame_index}")
            for row in rows[:5]:
                print(encode_box(row))
            if len(rows) > 5:
                print(f"... {len(rows) - 5} more boxes")
            print("E")
        return

    print(f"Opening {args.port} at {args.baud} baud...")
    accepted = {0: 0, 1: 0}
    rejected = 0
    started = time.perf_counter()
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as uart:
        time.sleep(0.15)
        uart.reset_input_buffer()
        uart.reset_output_buffer()
        for frame_index, rows in frame_items:
            transact(uart, "F", b"F")
            frame_accepted = {0: 0, 1: 0}
            frame_rejected = 0
            for row in rows:
                line = encode_box(row)
                uart.write(line.encode("ascii") + b"\n")
                uart.flush()
                response = uart.read(1)
                if response == b"B":
                    class_id = int(row["class_id"])
                    accepted[class_id] += 1
                    frame_accepted[class_id] += 1
                elif response == b"R":
                    rejected += 1
                    frame_rejected += 1
                else:
                    shown = response.decode("ascii", errors="replace") if response else "TIMEOUT"
                    raise RuntimeError(
                        f"Frame {frame_index}: response {shown!r} for line {line!r}"
                    )
            transact(uart, "E", b"D")
            if not args.quiet:
                print(
                    f"frame={frame_index:06d} cell={frame_accepted[0]} "
                    f"droplet={frame_accepted[1]} rejected={frame_rejected}"
                )

    elapsed = time.perf_counter() - started
    print(
        f"FPGA result: frames={len(frame_items)} cell={accepted[0]} "
        f"droplet={accepted[1]} rejected={rejected} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
