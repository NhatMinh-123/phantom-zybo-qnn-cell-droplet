from __future__ import annotations

import struct
import unittest

import numpy as np

from scripts.send_microplastic_patch_uart import (
    apply_image_transform,
    REQUEST_MAGIC,
    RESPONSE_HEADER_SIZE,
    checksum16,
    decode_raw_sum,
    decode_response_payload,
    make_request,
    parse_response_header,
    quantize_normalized,
)


def manifest_fixture() -> dict:
    return {
        "fpga_boundary": {"uart_output_bytes": 4},
        "postprocessing": {
            "raw_scale": 2.0,
            "raw_bias": 0.5,
            "average_count": 4,
            "output_thresholds": [-1.0, 0.0, 1.0],
            "output_bias": -2,
            "output_scale": 0.25,
            "particle_raw_sum_threshold": 3,
        },
    }



def spatial_manifest_fixture() -> dict:
    return {
        "fpga_boundary": {"uart_output_bytes": 4},
        "postprocessing": {
            "mode": "int8_logit",
            "output_scale": 0.05453263,
            "particle_output_code_threshold": -12,
        },
    }


class MicroplasticPatchUartTests(unittest.TestCase):
    def test_request_layout_and_checksum(self) -> None:
        payload = bytes(range(32))
        request = make_request(payload, 0x1234)
        self.assertEqual(request[:4], REQUEST_MAGIC)
        self.assertEqual(struct.unpack("<H", request[4:6])[0], 0x1234)
        self.assertEqual(request[6:-2], payload)
        self.assertEqual(struct.unpack("<H", request[-2:])[0], checksum16(payload))

    def test_response_header(self) -> None:
        header = b"RDQ1" + struct.pack("<HBII", 7, 0, 4, 55_300)
        self.assertEqual(len(header), RESPONSE_HEADER_SIZE)
        self.assertEqual(parse_response_header(header), (7, 0, 4, 55_300))

    def test_quantization_uses_threshold_crossings(self) -> None:
        thresholds = np.arange(255, dtype=np.float32) / 255.0
        pixels = np.zeros((32, 32), dtype=np.float32)
        pixels[0, 0] = thresholds[0]
        pixels[0, 1] = np.nextafter(thresholds[0], np.float32(-1.0))
        pixels[0, 2] = 1.0
        codes = quantize_normalized(pixels, thresholds)
        self.assertEqual(int(codes[0, 0]), 1)
        self.assertEqual(int(codes[0, 1]), 0)
        self.assertEqual(int(codes[0, 2]), 255)

    def test_dark_residual_transform_emphasizes_dark_center(self) -> None:
        image = np.full((32, 32), 128, dtype=np.uint8)
        image[16, 16] = 96
        manifest = {"preprocessing": {"image_transform": "dark_residual"}}

        transformed = apply_image_transform(image, manifest)

        self.assertGreater(int(transformed[16, 16]), 0)
        self.assertEqual(int(transformed[0, 0]), 0)

    def test_signed_int32_payload_and_decision(self) -> None:
        decoded = decode_response_payload(struct.pack("<i", -5), manifest_fixture())
        self.assertEqual(decoded["raw_sum"], -5)
        self.assertEqual(decoded["class_name"], "background")
        self.assertFalse(decoded["accepted"])

    def test_raw_sum_threshold_is_inclusive(self) -> None:
        decoded = decode_raw_sum(3, manifest_fixture())
        self.assertEqual(decoded["class_name"], "particle")
        self.assertTrue(decoded["accepted"])

    def test_spatial_int8_logit_threshold_is_inclusive(self) -> None:
        decoded = decode_response_payload(
            struct.pack("<i", -12), spatial_manifest_fixture()
        )
        self.assertEqual(decoded["output_code"], -12)
        self.assertEqual(decoded["class_name"], "particle")
        self.assertTrue(decoded["accepted"])

        background = decode_raw_sum(-13, spatial_manifest_fixture())
        self.assertEqual(background["class_name"], "background")
        self.assertFalse(background["accepted"])

    def test_spatial_int8_logit_rejects_out_of_range_code(self) -> None:
        with self.assertRaisesRegex(ValueError, "INT8 output code out of range"):
            decode_raw_sum(128, spatial_manifest_fixture())


if __name__ == "__main__":
    unittest.main()
