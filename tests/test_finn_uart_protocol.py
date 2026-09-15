from __future__ import annotations

import struct
import unittest

from scripts.send_frame_finn_uart import (
    REQUEST_MAGIC,
    RESPONSE_HEADER_SIZE,
    checksum16,
    make_request,
    parse_response_header,
)


class FinnUartProtocolTests(unittest.TestCase):
    def test_request_layout_and_checksum(self) -> None:
        payload = bytes((1, 2, 3, 250))
        packet = make_request(payload, 0x1234)
        self.assertEqual(packet[:4], REQUEST_MAGIC)
        self.assertEqual(struct.unpack("<H", packet[4:6])[0], 0x1234)
        self.assertEqual(packet[6:-2], payload)
        self.assertEqual(struct.unpack("<H", packet[-2:])[0], checksum16(payload))

    def test_response_header_includes_accelerator_cycles(self) -> None:
        header = b"RDQ1" + struct.pack("<HBII", 9, 0, 51_840, 11_243_700)
        self.assertEqual(len(header), RESPONSE_HEADER_SIZE)
        self.assertEqual(
            parse_response_header(header), (9, 0, 51_840, 11_243_700)
        )

    def test_response_header_rejects_wrong_magic(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bad response magic"):
            parse_response_header(b"BAD!" + bytes(RESPONSE_HEADER_SIZE - 4))


if __name__ == "__main__":
    unittest.main()
