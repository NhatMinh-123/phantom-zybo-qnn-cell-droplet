from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qnn.fpga_io import (
    accumulator_to_logits,
    decode_output_tensor,
    load_manifest,
    pack_input_axis,
    pack_output_axis,
    pack_sparse_detection_axis,
    quantize_grayscale,
    sparse_objectness_codes,
    unpack_output_axis,
    unpack_sparse_detection_axis,
)


class FpgaIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_input_quantization_and_packet_length(self) -> None:
        image = np.zeros((144, 192), dtype=np.uint8)
        image[0, 0] = 255
        codes = quantize_grayscale(image, self.manifest)
        self.assertEqual(codes.shape, (144, 192, 1))
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(int(codes[0, 0, 0]), 255)
        self.assertEqual(len(pack_input_axis(codes, self.manifest)), 27_648)

    def test_schema_two_manifest_is_backward_compatible(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["schema_version"] = 2
        manifest["optimized_deployment"] = {"measured_fps": 51.95}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_manifest(path)
        self.assertEqual(loaded["schema_version"], 2)
        self.assertEqual(
            loaded["optimized_deployment"]["measured_fps"],
            51.95,
        )

    def test_output_packet_round_trip(self) -> None:
        values = np.arange(36 * 48 * 15, dtype=np.int16).reshape(36, 48, 15)
        restored = unpack_output_axis(pack_output_axis(values, self.manifest), self.manifest)
        np.testing.assert_array_equal(restored, values)

    def test_int24_output_packet_round_trip(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        stream = manifest["fpga_core"]["output_stream"]
        stream["shape_nhwc"] = [1, 1, 1, 6]
        stream["dtype"] = "INT24"
        stream["axis_tdata_bits"] = 24
        stream["beats_per_frame"] = 6
        stream["bytes_per_frame"] = 18
        values = np.array(
            [-8_388_608, -32_769, -1, 0, 32_768, 8_388_607], dtype=np.int32
        ).reshape(1, 1, 6)
        restored = unpack_output_axis(pack_output_axis(values, manifest), manifest)
        self.assertEqual(restored.dtype, np.int32)
        np.testing.assert_array_equal(restored, values)

    def test_accumulator_conversion_shape_and_quantization(self) -> None:
        accumulator = np.zeros((36, 48, 15), dtype=np.int16)
        logits = accumulator_to_logits(accumulator, self.manifest)
        self.assertEqual(logits.shape, (1, 15, 36, 48))
        scale = self.manifest["postprocessing"]["output_requantization"]["scale"]
        quant_codes = logits / np.float32(scale)
        np.testing.assert_allclose(quant_codes, np.rint(quant_codes), atol=1e-5)

    def test_quantized_logit_output_round_trip(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        stream = manifest["fpga_core"]["output_stream"]
        stream["shape_nhwc"] = [1, 2, 3, 4]
        stream["dtype"] = "INT8"
        stream["axis_tdata_bits"] = 8
        stream["beats_per_frame"] = 24
        stream["bytes_per_frame"] = 24
        manifest["fpga_core"]["output_kind"] = "quantized_logits"
        manifest["fpga_core"]["quantized_logits"] = {
            "dtype": "INT8",
            "bits": 8,
            "scale": 0.125,
            "zero_point": 0,
        }
        values = np.arange(-12, 12, dtype=np.int8).reshape(2, 3, 4)

        restored = unpack_output_axis(pack_output_axis(values, manifest), manifest)
        logits = accumulator_to_logits(restored, manifest)

        np.testing.assert_array_equal(restored, values)
        self.assertEqual(logits.shape, (1, 4, 2, 3))
        np.testing.assert_array_equal(
            logits,
            values.astype(np.float32).transpose(2, 0, 1)[None] * np.float32(0.125),
        )

    def test_sparse_detection_round_trip_preserves_decoder(self) -> None:
        manifest = {
            "fpga_core": {
                "output_stream": {
                    "shape_nhwc": [1, 2, 3, 15],
                    "dtype": "INT8",
                    "axis_tdata_bits": 8,
                    "bytes_per_frame": 90,
                },
                "output_kind": "quantized_logits",
                "quantized_logits": {
                    "scale": 0.1,
                    "zero_point": 0,
                },
            },
            "postprocessing": {
                "decoder": {
                    "class_names": ["cell", "droplet"],
                    "confidence_thresholds": [0.75, 0.8],
                    "slots_per_class": [2, 1],
                    "anchors": [[0.1, 0.1], [0.3, 0.3]],
                    "nms_iou": [0.45, 0.1],
                    "pre_nms_topk": 100,
                    "max_detections": 200,
                }
            },
        }
        tensor = np.zeros((2, 3, 15), dtype=np.int8)
        tensor[..., 0::5] = -128
        cell_code, droplet_code = sparse_objectness_codes(manifest)
        tensor[0, 1, 0:5] = [cell_code, 1, 2, 3, 4]
        tensor[1, 2, 10:15] = [droplet_code, -1, -2, 5, 6]
        tensor[0, 0, 5:10] = [cell_code - 1, 40, 40, 40, 40]

        payload = pack_sparse_detection_axis(tensor, manifest)
        restored = unpack_sparse_detection_axis(payload, manifest)

        self.assertEqual(len(payload), 16)
        np.testing.assert_array_equal(restored[0, 1, 0:5], tensor[0, 1, 0:5])
        np.testing.assert_array_equal(restored[1, 2, 10:15], tensor[1, 2, 10:15])
        self.assertEqual(int(restored[0, 0, 5]), -128)
        self.assertEqual(
            decode_output_tensor(restored, manifest),
            decode_output_tensor(tensor, manifest),
        )


if __name__ == "__main__":
    unittest.main()
