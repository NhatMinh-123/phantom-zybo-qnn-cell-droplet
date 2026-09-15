#!/usr/bin/env python3
"""Recover selected artifacts from a ZIP with shifted central-directory offsets."""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path


SOURCE = Path(r"E:\fpga\15micro_yolo11n_colab_results.zip")
OUTPUT = Path(r"E:\fpga\models\15micro\yolo11n_colab_v1")
TARGETS = {
    "confidence_sweep.csv": 225,
    "metrics_summary.json": 433,
    "yolo11n_baseline/results.csv": 3_597_121,
}
SIGNATURE = b"PK\x03\x04"


def extract_local(data: bytes, expected_offset: int, expected_name: str) -> bytes:
    start = data.rfind(SIGNATURE, max(0, expected_offset - 4096), expected_offset + 1)
    if start < 0:
        raise RuntimeError(f"No local header near {expected_name}")
    fields = struct.unpack_from("<4s5H3L2H", data, start)
    _, _, flags, method, _, _, crc, compressed_size, size, name_len, extra_len = fields
    name_start = start + 30
    name = data[name_start:name_start + name_len].decode("utf-8")
    if name != expected_name:
        raise RuntimeError(f"Expected {expected_name}, found {name} at {start}")
    if flags & 0x08:
        raise RuntimeError(f"Data-descriptor ZIP entry is unsupported: {name}")
    payload_start = name_start + name_len + extra_len
    compressed = data[payload_start:payload_start + compressed_size]
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported compression method {method}: {name}")
    if len(payload) != size:
        raise RuntimeError(f"Size mismatch for {name}: {len(payload)} != {size}")
    actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc != crc:
        raise RuntimeError(f"CRC mismatch for {name}: {actual_crc:#x} != {crc:#x}")
    print(f"Recovered {name} from offset {start}, {len(payload)} bytes, CRC OK")
    return payload


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = SOURCE.read_bytes()
    for name, offset in TARGETS.items():
        payload = extract_local(data, offset, name)
        (OUTPUT / Path(name).name).write_bytes(payload)


if __name__ == "__main__":
    main()
