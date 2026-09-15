#!/usr/bin/env python3
"""UART packet codec for the Arty S7 bright-round feature gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from typing import Mapping


REQUEST_MAGIC = 0xA5
RESPONSE_MAGIC = 0x5A
BAUD = 1_000_000
PAYLOAD_FORMAT = "<BHHBBBHHHH"
PAYLOAD_BYTES = struct.calcsize(PAYLOAD_FORMAT)
REQUEST_BYTES = 1 + PAYLOAD_BYTES + 1
RESPONSE_BYTES = 5

REASON_NAMES = (
    "dim_peak",
    "dim_mean",
    "weak_peak_ratio",
    "weak_mean_ratio",
    "area",
    "elongated",
    "not_circular",
    "not_solid",
    "diffuse",
    "outside_core",
)


def _clamped_integer(value: object, scale: float, maximum: int) -> int:
    return max(0, min(maximum, int(round(float(value) * scale))))


def _clamped_ceiling(value: object, scale: float, maximum: int) -> int:
    return max(0, min(maximum, int(math.ceil(float(value) * scale))))


def candidate_payload(row: Mapping[str, object]) -> bytes:
    return struct.pack(
        PAYLOAD_FORMAT,
        _clamped_integer(row.get("blackhat_max", 0.0), 1.0, 255),
        _clamped_integer(row.get("blackhat_mean", 0.0), 256.0, 65535),
        _clamped_integer(row.get("blackhat_threshold", 0.0), 256.0, 65535),
        _clamped_integer(row.get("pixel_area", 0.0), 1.0, 255),
        _clamped_integer(row.get("bbox_width", 0.0), 1.0, 255),
        _clamped_integer(row.get("bbox_height", 0.0), 1.0, 255),
        _clamped_integer(row.get("circularity", 0.0), 1000.0, 65535),
        _clamped_integer(row.get("solidity", 0.0), 1000.0, 65535),
        _clamped_integer(row.get("extent", 0.0), 1000.0, 65535),
        _clamped_ceiling(row.get("radial_distance_norm", 0.0), 1000.0, 65535),
    )


def encode_request(row: Mapping[str, object]) -> bytes:
    body = bytes((REQUEST_MAGIC,)) + candidate_payload(row)
    checksum = 0
    for value in body:
        checksum ^= value
    return body + bytes((checksum,))


@dataclass(frozen=True)
class FeatureGateResponse:
    decision: bool
    reason_mask: int

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(
            name
            for bit, name in enumerate(REASON_NAMES)
            if self.reason_mask & (1 << bit)
        )


def decode_response(data: bytes) -> FeatureGateResponse:
    if len(data) != RESPONSE_BYTES:
        raise ValueError(f"Expected {RESPONSE_BYTES} response bytes, got {len(data)}")
    if data[0] != RESPONSE_MAGIC:
        raise ValueError(f"Bad response magic 0x{data[0]:02X}")
    checksum = 0
    for value in data[:-1]:
        checksum ^= value
    if checksum != data[-1]:
        raise ValueError(
            f"Bad response checksum 0x{data[-1]:02X}; expected 0x{checksum:02X}"
        )
    return FeatureGateResponse(
        decision=bool(data[1] & 0x01),
        reason_mask=data[2] | ((data[3] & 0x03) << 8),
    )


class BrightRoundFeatureUartClient:
    def __init__(self, serial_port, *, timeout_seconds: float = 0.1) -> None:
        self.serial_port = serial_port
        self.timeout_seconds = timeout_seconds

    def _read_response(self) -> bytes:
        deadline = time.monotonic() + self.timeout_seconds
        response = bytearray()
        synchronized = False
        while time.monotonic() < deadline and len(response) < RESPONSE_BYTES:
            chunk = self.serial_port.read(1)
            if not chunk:
                continue
            if not synchronized:
                if chunk[0] != RESPONSE_MAGIC:
                    continue
                synchronized = True
            response.extend(chunk)
        if len(response) != RESPONSE_BYTES:
            raise TimeoutError(
                f"FPGA UART response timeout: {len(response)}/{RESPONSE_BYTES} bytes"
            )
        return bytes(response)

    def classify(self, row: Mapping[str, object]) -> FeatureGateResponse:
        self.serial_port.write(encode_request(row))
        self.serial_port.flush()
        return decode_response(self._read_response())
