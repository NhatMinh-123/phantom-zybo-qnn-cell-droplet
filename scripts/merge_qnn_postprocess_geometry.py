#!/usr/bin/env python3
"""Merge validation-selected box geometry into a QNN postprocess config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", default="droplet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postprocess = json.loads(args.postprocess.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    selected = postprocess.setdefault("selected", {})
    class_names = tuple(selected["confidence_thresholds"])
    identity = {
        "width_scale": 1.0,
        "height_scale": 1.0,
        "center_x_offset": 0.0,
        "center_y_offset": 0.0,
    }
    selected["box_calibration"] = {
        name: dict(identity) for name in class_names
    }
    if geometry.get("accepted", False):
        selected["box_calibration"][args.class_name] = geometry["calibration"]
    selected["box_calibration_source"] = str(args.geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(postprocess, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
