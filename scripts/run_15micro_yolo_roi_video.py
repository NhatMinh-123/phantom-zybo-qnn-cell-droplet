#!/usr/bin/env python3
"""Run the 15 um YOLO teacher on one ROI and mark it as a PC reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

import run_cell_droplet_realtime as runner


RUNTIME_LABEL = "PC GPU reference"
ORIGINAL_PUT_TEXT = cv2.putText


def labeled_put_text(image, text, *args, **kwargs):
    if isinstance(text, str) and text.startswith("ROI detector |"):
        text = f"{RUNTIME_LABEL} | {text}"
    return ORIGINAL_PUT_TEXT(image, text, *args, **kwargs)


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError("--output is required") from error


def main() -> None:
    output = output_directory()
    runner.cv2.putText = labeled_put_text
    runner.main()

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    summary["runtime_label"] = RUNTIME_LABEL
    summary["execution_note"] = (
        "YOLO inference in this folder was computed on the PC GPU. It is a "
        "teacher/reference result and is not FPGA inference."
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")


if __name__ == "__main__":
    main()
