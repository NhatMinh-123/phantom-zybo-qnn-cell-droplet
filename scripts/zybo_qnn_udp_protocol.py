"""UDP transport for the Zybo QNN AXI-DMA runtime."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass


UDP_PORT = 50123
HEADER_BYTES = 24
MAX_PAYLOAD_BYTES = 1400
INPUT_BYTES = 96 * 96
RECORD_BYTES = 13
MAGIC = b"ZQ"
VERSION = 1
MESSAGE_ROI = 1
MESSAGE_RESULT = 2
MESSAGE_HELLO = 3
MESSAGE_HELLO_REPLY = 4
HEADER = struct.Struct("<2sBBIBBBBHHHHI")


@dataclass(frozen=True)
class SparseResult:
    frame_id: int
    roi_id: int
    records: bytes
    record_count: int
    qnn_us: int
    round_trip_seconds: float


def make_packet(
    message_type: int,
    frame_id: int,
    roi_id: int,
    chunk_index: int,
    chunk_count: int,
    total_bytes: int,
    offset: int,
    record_count: int,
    qnn_us: int,
    payload: bytes,
) -> bytes:
    return HEADER.pack(
        MAGIC,
        VERSION,
        message_type,
        frame_id,
        roi_id,
        chunk_index,
        chunk_count,
        0,
        total_bytes,
        offset,
        len(payload),
        record_count,
        qnn_us,
    ) + payload


def unpack_packet(packet: bytes) -> tuple[tuple[object, ...], bytes]:
    if len(packet) < HEADER_BYTES:
        raise ValueError("QNN UDP packet is shorter than its header")
    header = HEADER.unpack(packet[:HEADER_BYTES])
    if header[0] != MAGIC or header[1] != VERSION:
        raise ValueError("Invalid QNN UDP packet signature")
    payload = packet[HEADER_BYTES:]
    if len(payload) != header[10]:
        raise ValueError("QNN UDP payload length mismatch")
    return header, payload


class ZyboQnnUdpClient:
    def __init__(
        self,
        board_ip: str = "100.100.100.2",
        port: int = UDP_PORT,
        timeout: float = 0.5,
        retries: int = 2,
    ) -> None:
        self.address = (board_ip, port)
        self.timeout = timeout
        self.retries = retries
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(timeout)
        self.socket.connect(self.address)

    def close(self) -> None:
        self.socket.close()

    def __enter__(self) -> "ZyboQnnUdpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def hello(self) -> str:
        request = make_packet(MESSAGE_HELLO, 0, 0, 0, 1, 0, 0, 0, 0, b"")
        for _ in range(self.retries + 1):
            self.socket.sendto(request, self.address)
            try:
                packet, _ = self.socket.recvfrom(1500)
            except TimeoutError:
                continue
            header, payload = unpack_packet(packet)
            if header[2] == MESSAGE_HELLO_REPLY:
                return payload.decode("ascii")
        raise TimeoutError(f"No QNN UDP service at {self.address[0]}:{self.address[1]}")

    def infer(self, frame_id: int, roi_id: int, tensor: bytes) -> SparseResult:
        if len(tensor) != INPUT_BYTES:
            raise ValueError(f"Expected {INPUT_BYTES} input bytes, got {len(tensor)}")
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            self._send_tensor(frame_id, roi_id, tensor)
            try:
                return self._receive_result(frame_id, roi_id, started)
            except TimeoutError:
                if attempt == self.retries:
                    raise
        raise AssertionError("unreachable")

    def _send_tensor(self, frame_id: int, roi_id: int, tensor: bytes) -> None:
        chunk_count = (len(tensor) + MAX_PAYLOAD_BYTES - 1) // MAX_PAYLOAD_BYTES
        for chunk_index in range(chunk_count):
            offset = chunk_index * MAX_PAYLOAD_BYTES
            payload = tensor[offset : offset + MAX_PAYLOAD_BYTES]
            packet = make_packet(
                MESSAGE_ROI,
                frame_id,
                roi_id,
                chunk_index,
                chunk_count,
                len(tensor),
                offset,
                0,
                0,
                payload,
            )
            self.socket.sendto(packet, self.address)

    def _receive_result(
        self, frame_id: int, roi_id: int, started: float
    ) -> SparseResult:
        deadline = time.perf_counter() + self.timeout
        output: bytearray | None = None
        chunks_seen: set[int] = set()
        chunk_count = 0
        record_count = 0
        qnn_us = 0
        while time.perf_counter() < deadline:
            self.socket.settimeout(max(0.001, deadline - time.perf_counter()))
            packet, _ = self.socket.recvfrom(1500)
            header, payload = unpack_packet(packet)
            (
                _,
                _,
                message_type,
                reply_frame,
                reply_roi,
                chunk_index,
                reply_chunk_count,
                _,
                total_bytes,
                offset,
                _,
                reply_records,
                reply_qnn_us,
            ) = header
            if (
                message_type != MESSAGE_RESULT
                or reply_frame != frame_id
                or reply_roi != roi_id
            ):
                continue
            expected_chunks = max(1, (total_bytes + MAX_PAYLOAD_BYTES - 1) // MAX_PAYLOAD_BYTES)
            if (
                total_bytes > 576 * 3 * RECORD_BYTES
                or total_bytes != reply_records * RECORD_BYTES
                or reply_chunk_count != expected_chunks
                or chunk_index >= reply_chunk_count
                or offset != chunk_index * MAX_PAYLOAD_BYTES
                or len(payload) != min(MAX_PAYLOAD_BYTES, total_bytes - offset)
            ):
                raise RuntimeError("Invalid QNN result fragment geometry")
            if output is None:
                output = bytearray(total_bytes)
                chunk_count = reply_chunk_count
                record_count = reply_records
                qnn_us = reply_qnn_us
            if (
                total_bytes != len(output)
                or reply_chunk_count != chunk_count
                or reply_records != record_count
                or reply_qnn_us != qnn_us
                or offset + len(payload) > len(output)
            ):
                raise RuntimeError("Inconsistent fragmented response from Zybo")
            if chunk_index in chunks_seen and output[offset : offset + len(payload)] != payload:
                raise RuntimeError("Conflicting duplicate QNN result fragment")
            output[offset : offset + len(payload)] = payload
            chunks_seen.add(chunk_index)
            if len(chunks_seen) == chunk_count:
                if len(output) != record_count * RECORD_BYTES:
                    raise RuntimeError("Sparse QNN response length mismatch")
                return SparseResult(
                    frame_id,
                    roi_id,
                    bytes(output),
                    record_count,
                    qnn_us,
                    time.perf_counter() - started,
                )
        raise TimeoutError(f"QNN result timeout for frame={frame_id} roi={roi_id}")
