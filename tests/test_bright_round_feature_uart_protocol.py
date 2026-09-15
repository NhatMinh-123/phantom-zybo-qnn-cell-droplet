import struct

from scripts.bright_round_feature_uart_protocol import (
    PAYLOAD_BYTES,
    PAYLOAD_FORMAT,
    REQUEST_BYTES,
    RESPONSE_MAGIC,
    candidate_payload,
    decode_response,
    encode_request,
)


def valid_row() -> dict[str, float]:
    return {
        "blackhat_max": 30,
        "blackhat_mean": 12.0,
        "blackhat_threshold": 3.0,
        "pixel_area": 12,
        "bbox_width": 5,
        "bbox_height": 6,
        "circularity": 0.70,
        "solidity": 0.90,
        "extent": 0.70,
        "radial_distance_norm": 0.35,
    }


def test_request_layout_and_checksum() -> None:
    packet = encode_request(valid_row())
    assert len(packet) == REQUEST_BYTES
    assert PAYLOAD_BYTES == 16
    checksum = 0
    for value in packet[:-1]:
        checksum ^= value
    assert packet[-1] == checksum
    values = struct.unpack(PAYLOAD_FORMAT, candidate_payload(valid_row()))
    assert values[:6] == (30, 3072, 768, 12, 5, 6)


def test_response_decode() -> None:
    body = bytes((RESPONSE_MAGIC, 1, 0, 0))
    checksum = 0
    for value in body:
        checksum ^= value
    response = decode_response(body + bytes((checksum,)))
    assert response.decision
    assert response.reason_mask == 0

def test_radial_distance_uses_conservative_ceiling() -> None:
    row = valid_row()
    row["radial_distance_norm"] = 0.900132953528008
    values = struct.unpack(PAYLOAD_FORMAT, candidate_payload(row))
    assert values[-1] == 901

    row["radial_distance_norm"] = 0.9
    values = struct.unpack(PAYLOAD_FORMAT, candidate_payload(row))
    assert values[-1] == 900
