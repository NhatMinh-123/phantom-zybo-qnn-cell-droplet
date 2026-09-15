from __future__ import annotations

import unittest

import torch

from scripts.evaluate_packed_input_precision import (
    projected_transport,
    reduce_uint8_codes,
)


class PackedInputPrecisionTests(unittest.TestCase):
    def test_four_bit_codes_expand_to_nibble_multiples(self) -> None:
        codes = torch.arange(256, dtype=torch.float32)
        expanded = reduce_uint8_codes(codes, 4)
        self.assertEqual(int(torch.unique(expanded).numel()), 16)
        self.assertTrue(torch.all((expanded.to(torch.int64) % 17) == 0))
        self.assertEqual(int(expanded[0]), 0)
        self.assertEqual(int(expanded[-1]), 255)

    def test_eight_bit_codes_are_unchanged(self) -> None:
        codes = torch.tensor([0.0, 31.0, 127.0, 255.0])
        self.assertTrue(torch.equal(reduce_uint8_codes(codes, 8), codes))

    def test_four_bit_transport_halves_input_payload(self) -> None:
        eight_bit = projected_transport(
            bits=8,
            pixels=192 * 192,
            baud=12_000_000,
            core_fps=56.1156,
            host_overhead_ms=10.0,
        )
        four_bit = projected_transport(
            bits=4,
            pixels=192 * 192,
            baud=12_000_000,
            core_fps=56.1156,
            host_overhead_ms=10.0,
        )
        self.assertEqual(eight_bit["payload_bytes"], 36_864)
        self.assertEqual(four_bit["payload_bytes"], 18_432)
        self.assertGreater(
            four_bit["projected_update_fps"],
            eight_bit["projected_update_fps"],
        )


if __name__ == "__main__":
    unittest.main()
