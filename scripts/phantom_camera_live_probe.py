#!/usr/bin/env python3
"""Verify a live Phantom camera stream and optionally keep a preview window open."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from pyphantom import Phantom, utils


ROOT = Path(__file__).resolve().parents[1]


def get_reduce8_image(camera: object) -> np.ndarray:
    """Request the SDK's sensitivity-aware 8-bit conversion."""
    return np.asarray(
        camera._live_cine.get_images(utils.FrameRange(0, 0), Option=1)[0]
    )


def preview_image(image: np.ndarray, color: bool) -> np.ndarray:
    """Convert the Phantom SDK's Reduce8 image to OpenCV channel order."""
    if image.dtype != np.uint8:
        raise TypeError(f"Phantom Reduce8 returned {image.dtype}, expected uint8")
    if color:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "phantom_live")
    args = parser.parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    phantom = Phantom()
    if phantom.camera_count < 1:
        phantom.close()
        raise RuntimeError("No physical Phantom camera is visible to the SDK")
    camera = phantom.Camera(0)
    try:
        color = bool(camera._live_cine.is_color.value)
        hashes: list[str] = []
        first_preview: np.ndarray | None = None
        started = time.perf_counter()
        for _ in range(args.frames):
            image = get_reduce8_image(camera)
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            if first_preview is None:
                first_preview = preview_image(image, color)
        elapsed = time.perf_counter() - started
        if first_preview is None:
            raise RuntimeError("Camera returned no image")
        preview_path = args.output / "phantom_live_preview.png"
        if not cv2.imwrite(str(preview_path), first_preview):
            raise RuntimeError(f"Could not write {preview_path}")
        report = {
            "status": "PASS",
            "camera_model": camera.model,
            "serial": int(camera.serial),
            "ip_address": camera.get_selector_string(utils.CamSelector.gsIPAddress),
            "resolution": [int(camera.resolution.x), int(camera.resolution.y)],
            "configured_camera_fps": float(camera.frame_rate),
            "frames_received": args.frames,
            "unique_frame_hashes": len(set(hashes)),
            "pc_acquisition_fps": args.frames / max(elapsed, 1e-9),
            "preview": str(preview_path.resolve()),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if args.preview:
            while True:
                image = get_reduce8_image(camera)
                display = preview_image(image, color)
                cv2.imshow("Phantom VEO live stream - Esc to close", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()
    finally:
        camera.close()
        phantom.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
