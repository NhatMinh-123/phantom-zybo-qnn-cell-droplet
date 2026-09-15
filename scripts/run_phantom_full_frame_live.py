#!/usr/bin/env python3
"""Show the physical Phantom camera at its uncropped SDK output."""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import time
from pathlib import Path

import cv2
from pyphantom import Phantom, utils
import pyphantom

from phantom_native_stream import NativePhantomReader


ROOT = Path(__file__).resolve().parents[1]


def set_uint(dll: object, handle: ct.c_void_p, selector: int, value: int) -> None:
    data = ct.c_uint32(value)
    status = dll.PhSetCineInfo(handle, selector, ct.byref(data))
    if status < 0:
        raise RuntimeError(f"PhSetCineInfo({selector}) failed: {status}")


def restore_full_frame_processing(camera: object) -> None:
    dll = ct.WinDLL(str(Path(pyphantom.__file__).parent / "data/PhFile.Dll"))
    dll.PhSetCineInfo.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p]
    dll.PhSetCineInfo.restype = ct.c_int32
    handle = ct.c_void_p(camera._live_cine._cine_handle)
    set_uint(dll, handle, 218, 0)  # GCI_CROPACTIVE
    set_uint(dll, handle, 215, 0)  # GCI_RESAMPLEACTIVE
    set_uint(dll, handle, 211, 4)  # Original Phantom interpolation setting


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-serial", type=int, default=25225)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "final_results/phantom_camera_qnn_live/full_frame_sessions",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    phantom = Phantom()
    deadline = time.perf_counter() + args.timeout
    while phantom.camera_count < 1 and time.perf_counter() < deadline:
        time.sleep(0.25)
    if phantom.camera_count < 1:
        phantom.close()
        raise RuntimeError("Physical Phantom camera is not visible to the SDK")

    camera = phantom.Camera(0)
    reader = None
    frame_count = 0
    started = time.perf_counter()
    try:
        if int(camera.serial) != args.camera_serial:
            raise RuntimeError("SDK selected an unexpected camera")
        restore_full_frame_processing(camera)
        reader = NativePhantomReader(camera)
        while True:
            frame = reader.read()
            frame_count += 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            cv2.putText(
                frame,
                f"Phantom full frame {frame.shape[1]}x{frame.shape[0]} | {frame_count / elapsed:.1f} FPS | Esc to close",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 230, 40),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Phantom Full Frame - crop OFF", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        elapsed = time.perf_counter() - started
        report = {
            "camera": camera.model,
            "serial": int(camera.serial),
            "ip": camera.get_selector_string(utils.CamSelector.gsIPAddress),
            "resolution": [int(camera.resolution.x), int(camera.resolution.y)],
            "crop_active": False,
            "resample_active": False,
            "qnn_detection": False,
            "frames": frame_count,
            "elapsed_seconds": elapsed,
            "display_fps": frame_count / max(elapsed, 1e-9),
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    finally:
        if reader is not None:
            reader.close()
        cv2.destroyAllWindows()
        camera.close()
        phantom.close()


if __name__ == "__main__":
    raise SystemExit(main())
