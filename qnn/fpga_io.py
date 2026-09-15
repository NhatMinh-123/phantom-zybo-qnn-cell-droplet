"""Exact image and tensor conversions at the FINN host boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from qnn.detection import Detection


@dataclass(frozen=True)
class _NumpyDetection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "exports"
    / "qnn_cell_droplet_v2"
    / "tiny_detector_192x144_w4a4_rawhead_fpga.json"
)


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported FPGA manifest schema")
    return manifest


def decoder_nms_iou(decoder: dict[str, Any]) -> float | tuple[float, ...]:
    value = decoder["nms_iou"]
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return float(value)


def decoder_box_constraints(
    decoder: dict[str, Any],
) -> tuple[dict[str, float] | None, ...] | None:
    values = decoder.get("box_constraints")
    if values is None:
        return None
    return tuple(
        None if value is None else {key: float(item) for key, item in value.items()}
        for value in values
    )


def decoder_box_calibration(
    decoder: dict[str, Any],
) -> tuple[dict[str, float] | None, ...] | None:
    values = decoder.get("box_calibration")
    if values is None:
        return None
    return tuple(
        None if value is None else {key: float(item) for key, item in value.items()}
        for value in values
    )


def quantize_grayscale(
    grayscale: np.ndarray, manifest: dict[str, Any]
) -> np.ndarray:
    """Convert an HxW uint8 grayscale image into NHWC UINT8 input codes."""

    resize = manifest["preprocessing"]["resize"]
    expected_shape = (int(resize["height"]), int(resize["width"]))
    if grayscale.shape != expected_shape:
        raise ValueError(f"Expected grayscale shape {expected_shape}, got {grayscale.shape}")
    if grayscale.dtype != np.uint8:
        raise TypeError(f"Expected uint8 grayscale pixels, got {grayscale.dtype}")

    quant = manifest["preprocessing"]["input_quantization"]
    scale = np.float32(quant["scale"])
    normalized = grayscale.astype(np.float32) / np.float32(255.0)
    maximum = (1 << int(quant["bits"])) - 1
    codes = np.clip(np.rint(normalized / scale), 0, maximum).astype(np.uint8)
    return np.ascontiguousarray(codes[..., None])


def prepare_image(
    source: Path | str | Image.Image | np.ndarray,
    manifest: dict[str, Any],
    *,
    array_is_bgr: bool = False,
) -> np.ndarray:
    """Resize and quantize a path, PIL image, or RGB/BGR NumPy frame."""

    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            pil_image = image.convert("L")
    elif isinstance(source, Image.Image):
        pil_image = source.convert("L")
    elif isinstance(source, np.ndarray):
        array = source
        if array.ndim == 3 and array.shape[2] == 3 and array_is_bgr:
            array = array[..., ::-1]
        if array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image array, got {array.dtype}")
        pil_image = Image.fromarray(array).convert("L")
    else:
        raise TypeError(f"Unsupported image source type: {type(source)!r}")

    resize = manifest["preprocessing"]["resize"]
    pil_image = pil_image.resize(
        (int(resize["width"]), int(resize["height"])), Image.Resampling.BILINEAR
    )
    grayscale = np.asarray(pil_image, dtype=np.uint8)
    return quantize_grayscale(grayscale, manifest)


def pack_input_axis(codes_nhwc: np.ndarray, manifest: dict[str, Any]) -> bytes:
    stream = manifest["fpga_core"]["input_stream"]
    expected_shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    if codes_nhwc.shape != expected_shape or codes_nhwc.dtype != np.uint8:
        raise ValueError(
            f"Expected UINT8 input shape {expected_shape}, got "
            f"{codes_nhwc.shape} {codes_nhwc.dtype}"
        )
    payload = np.ascontiguousarray(codes_nhwc).tobytes(order="C")
    if len(payload) != int(stream["bytes_per_frame"]):
        raise ValueError("Input payload length does not match stream contract")
    return payload


def integer_dtype_range(dtype: str) -> tuple[bool, int, int, int]:
    match = re.fullmatch(r"(U?INT)(\d+)", dtype.upper())
    if match is None:
        raise ValueError(f"Unsupported integer dtype: {dtype}")
    signed = match.group(1) == "INT"
    bits = int(match.group(2))
    if bits <= 0 or bits > 64:
        raise ValueError(f"Unsupported integer width: {bits}")
    if signed:
        return signed, bits, -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return signed, bits, 0, (1 << bits) - 1


def pack_output_axis(output_nhwc: np.ndarray, manifest: dict[str, Any]) -> bytes:
    """Pack a signed NHWC output tensor into little-endian AXI beats."""

    stream = manifest["fpga_core"]["output_stream"]
    expected_shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    if output_nhwc.shape != expected_shape:
        raise ValueError(
            f"Expected output shape {expected_shape}, got {output_nhwc.shape}"
        )
    signed, bits, minimum, maximum = integer_dtype_range(str(stream["dtype"]))
    if not signed:
        raise ValueError("Output stream must be signed")
    axis_bits = int(stream["axis_tdata_bits"])
    if axis_bits % 8 != 0 or axis_bits < bits or axis_bits > 64:
        raise ValueError(f"Invalid AXI width {axis_bits} for INT{bits}")
    values = np.asarray(output_nhwc, dtype=np.int64).reshape(-1)
    if np.any(values < minimum) or np.any(values > maximum):
        raise OverflowError(f"Output value is outside INT{bits}")
    encoded = values.astype(np.uint64) & np.uint64((1 << axis_bits) - 1)
    byte_count = axis_bits // 8
    packed = np.empty((encoded.size, byte_count), dtype=np.uint8)
    for byte_index in range(byte_count):
        packed[:, byte_index] = (encoded >> (8 * byte_index)).astype(np.uint8)
    payload = packed.tobytes(order="C")
    if len(payload) != int(stream["bytes_per_frame"]):
        raise ValueError("Output payload length does not match stream contract")
    return payload


def unpack_output_axis(payload: bytes, manifest: dict[str, Any]) -> np.ndarray:
    """Unpack low-byte-first UART data into the FPGA's signed NHWC tensor."""

    stream = manifest["fpga_core"]["output_stream"]
    expected_bytes = int(stream["bytes_per_frame"])
    if len(payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} output bytes, got {len(payload)}")
    shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    signed, bits, _, _ = integer_dtype_range(str(stream["dtype"]))
    if not signed:
        raise ValueError("Output stream must be signed")
    axis_bits = int(stream["axis_tdata_bits"])
    if axis_bits % 8 != 0 or axis_bits < bits or axis_bits > 64:
        raise ValueError(f"Invalid AXI width {axis_bits} for INT{bits}")
    byte_count = axis_bits // 8
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, byte_count)
    encoded = np.zeros(raw.shape[0], dtype=np.uint64)
    for byte_index in range(byte_count):
        encoded |= raw[:, byte_index].astype(np.uint64) << (8 * byte_index)
    value_mask = np.uint64((1 << bits) - 1)
    encoded &= value_mask
    sign_bit = np.uint64(1 << (bits - 1))
    values = ((encoded ^ sign_bit).astype(np.int64) - int(sign_bit)).astype(
        np.int16 if bits <= 16 else np.int32 if bits <= 32 else np.int64
    )
    return values.reshape(shape)


SPARSE_DETECTION_RECORD_BYTES = 8


def sparse_objectness_codes(manifest: dict[str, Any]) -> tuple[int, ...]:
    """Return the minimum INT8 objectness code retained for each class."""

    decoder = manifest["postprocessing"]["decoder"]
    thresholds = tuple(float(value) for value in decoder["confidence_thresholds"])
    quantized = manifest["fpga_core"]["quantized_logits"]
    scale = float(quantized["scale"])
    zero_point = int(quantized.get("zero_point", 0))
    if scale <= 0.0:
        raise ValueError("Quantized-logit scale must be positive")
    codes = []
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Sparse confidence thresholds must be in (0, 1)")
        logit = math.log(threshold / (1.0 - threshold))
        codes.append(int(math.ceil(logit / scale + zero_point)))
    return tuple(codes)


def pack_sparse_detection_axis(
    output_nhwc: np.ndarray,
    manifest: dict[str, Any],
) -> bytes:
    """Pack detector slots that can survive the confidence threshold.

    Each record is eight bytes: little-endian grid index, slot index, then
    objectness/x/y/width/height INT8 codes.
    """

    stream = manifest["fpga_core"]["output_stream"]
    expected_shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    if output_nhwc.shape != expected_shape or not np.issubdtype(
        output_nhwc.dtype, np.signedinteger
    ):
        raise ValueError(
            f"Expected signed-integer output shape {expected_shape}, got "
            f"{output_nhwc.shape} {output_nhwc.dtype}"
        )
    if np.any(output_nhwc < -128) or np.any(output_nhwc > 127):
        raise OverflowError("Sparse detector input is outside the INT8 range")
    output_int8 = output_nhwc.astype(np.int8, copy=False)
    grid_height, grid_width, channels = expected_shape
    slots_per_class = tuple(
        int(value)
        for value in manifest["postprocessing"]["decoder"]["slots_per_class"]
    )
    total_slots = sum(slots_per_class)
    if channels != total_slots * 5:
        raise ValueError("Sparse codec requires five channels per detector slot")
    class_codes = sparse_objectness_codes(manifest)
    slot_class_ids = tuple(
        class_id
        for class_id, slot_count in enumerate(slots_per_class)
        for _ in range(slot_count)
    )

    records = bytearray()
    flat = output_int8.reshape(grid_height * grid_width, channels)
    for grid_index, grid_values in enumerate(flat):
        for slot_index, class_id in enumerate(slot_class_ids):
            values = grid_values[slot_index * 5 : (slot_index + 1) * 5]
            if int(values[0]) < class_codes[class_id]:
                continue
            records.extend(int(grid_index).to_bytes(2, "little"))
            records.append(slot_index)
            records.extend(values.view(np.uint8).tobytes())
    return bytes(records)


def unpack_sparse_detection_axis(
    payload: bytes,
    manifest: dict[str, Any],
) -> np.ndarray:
    """Restore a decoder-equivalent dense NHWC tensor from sparse records."""

    if len(payload) % SPARSE_DETECTION_RECORD_BYTES:
        raise ValueError("Sparse detector payload is not a whole number of records")
    stream = manifest["fpga_core"]["output_stream"]
    shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    grid_height, grid_width, channels = shape
    total_slots = channels // 5
    if channels != total_slots * 5:
        raise ValueError("Sparse codec requires five channels per detector slot")

    output = np.zeros(shape, dtype=np.int8)
    output[..., 0::5] = np.int8(-128)
    seen: set[tuple[int, int]] = set()
    for offset in range(0, len(payload), SPARSE_DETECTION_RECORD_BYTES):
        record = payload[offset : offset + SPARSE_DETECTION_RECORD_BYTES]
        grid_index = int.from_bytes(record[0:2], "little")
        slot_index = int(record[2])
        if grid_index >= grid_height * grid_width:
            raise ValueError(f"Sparse grid index is outside the tensor: {grid_index}")
        if slot_index >= total_slots:
            raise ValueError(f"Sparse slot index is outside the tensor: {slot_index}")
        key = (grid_index, slot_index)
        if key in seen:
            raise ValueError(f"Duplicate sparse detector record: {key}")
        seen.add(key)
        grid_y, grid_x = divmod(grid_index, grid_width)
        values = np.frombuffer(record[3:8], dtype=np.int8)
        output[grid_y, grid_x, slot_index * 5 : (slot_index + 1) * 5] = values
    return output


def output_tensor_to_logits(
    output_nhwc: np.ndarray,
    manifest: dict[str, Any],
) -> np.ndarray:
    """Convert the FPGA NHWC integer tensor into decoder-ready NCHW logits."""

    stream = manifest["fpga_core"]["output_stream"]
    expected_shape = tuple(int(value) for value in stream["shape_nhwc"][1:])
    if output_nhwc.shape != expected_shape:
        raise ValueError(
            f"Expected output shape {expected_shape}, got {output_nhwc.shape}"
        )
    if not np.issubdtype(output_nhwc.dtype, np.signedinteger):
        raise TypeError("Output tensor must use a signed integer dtype")

    output_kind = manifest["fpga_core"].get("output_kind", "raw_accumulator")
    if output_kind == "quantized_logits":
        quantized = manifest["fpga_core"]["quantized_logits"]
        scale = np.float32(quantized["scale"])
        zero_point = np.float32(quantized.get("zero_point", 0))
        logits_nhwc = (output_nhwc.astype(np.float32) - zero_point) * scale
        return np.ascontiguousarray(logits_nhwc.transpose(2, 0, 1)[None])
    if output_kind != "raw_accumulator":
        raise ValueError(f"Unsupported FPGA output kind: {output_kind}")

    accumulator = manifest["fpga_core"]["raw_accumulator"]
    bias = np.asarray(accumulator["bias_per_channel"], dtype=np.float32)
    raw_float = (
        output_nhwc.astype(np.float32) * np.float32(accumulator["scale"])
        + bias.reshape(1, 1, -1)
    )

    requant = manifest["postprocessing"]["output_requantization"]
    bits = int(requant["bits"])
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    output_scale = np.float32(requant["scale"])
    output_codes = np.clip(
        np.rint(raw_float / output_scale), minimum, maximum
    ).astype(np.int8)
    logits_nhwc = output_codes.astype(np.float32) * output_scale
    return np.ascontiguousarray(logits_nhwc.transpose(2, 0, 1)[None])


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection_size = np.maximum(bottom_right - top_left, 0.0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        current_area = max(
            0.0,
            float(boxes[current, 2] - boxes[current, 0]),
        ) * max(0.0, float(boxes[current, 3] - boxes[current, 1]))
        remaining_area = np.maximum(boxes[remaining, 2] - boxes[remaining, 0], 0.0) * np.maximum(
            boxes[remaining, 3] - boxes[remaining, 1], 0.0
        )
        overlap = intersection / (current_area + remaining_area - intersection + 1e-7)
        order = remaining[overlap <= iou_threshold]
    return keep


def _decode_predictions_numpy(
    predictions: np.ndarray, decoder: dict[str, Any]
) -> list[list[_NumpyDetection]]:
    batch_size, channels, grid_h, grid_w = predictions.shape
    anchors = tuple(tuple(float(x) for x in pair) for pair in decoder["anchors"])
    slots_per_class = tuple(int(x) for x in decoder["slots_per_class"])
    total_slots = sum(slots_per_class)
    if channels != total_slots * 5:
        raise ValueError("Unexpected detector output shape")

    thresholds = tuple(float(x) for x in decoder["confidence_thresholds"])
    nms_value = decoder_nms_iou(decoder)
    nms_thresholds = (
        nms_value if isinstance(nms_value, tuple) else (nms_value,) * len(anchors)
    )
    constraints = decoder_box_constraints(decoder) or (None,) * len(anchors)
    calibrations = decoder_box_calibration(decoder) or (None,) * len(anchors)
    pre_nms_topk = int(decoder["pre_nms_topk"])
    max_detections = int(decoder["max_detections"])
    values = predictions.reshape(batch_size, total_slots, 5, grid_h, grid_w)
    slot_class_ids = tuple(
        class_id
        for class_id, slot_count in enumerate(slots_per_class)
        for _ in range(slot_count)
    )
    output: list[list[_NumpyDetection]] = []

    for batch_index in range(batch_size):
        image_detections: list[_NumpyDetection] = []
        for class_id, anchor in enumerate(anchors):
            class_slots = np.asarray(
                [slot for slot, owner in enumerate(slot_class_ids) if owner == class_id],
                dtype=np.int64,
            )
            confidence = _sigmoid_numpy(values[batch_index, class_slots, 0])
            padded = np.pad(
                confidence,
                ((0, 0), (1, 1), (1, 1)),
                mode="constant",
                constant_values=-np.inf,
            )
            local_maximum = np.maximum.reduce(
                [
                    padded[:, dy : dy + grid_h, dx : dx + grid_w]
                    for dy in range(3)
                    for dx in range(3)
                ]
            )
            local_slot, grid_y, grid_x = np.nonzero(
                (confidence >= thresholds[class_id])
                & (confidence >= local_maximum - 1e-7)
            )
            if not local_slot.size:
                continue
            slot_ids = class_slots[local_slot]
            scores = confidence[local_slot, grid_y, grid_x]
            if scores.size > pre_nms_topk:
                top = np.argsort(-scores, kind="stable")[:pre_nms_topk]
                grid_y, grid_x, slot_ids, scores = (
                    grid_y[top],
                    grid_x[top],
                    slot_ids[top],
                    scores[top],
                )
            raw_boxes = values[batch_index, slot_ids, 1:5, grid_y, grid_x]
            center_x = (grid_x + _sigmoid_numpy(raw_boxes[:, 0])) / grid_w
            center_y = (grid_y + _sigmoid_numpy(raw_boxes[:, 1])) / grid_h
            width = anchor[0] * np.exp(np.clip(raw_boxes[:, 2], -3.0, 2.0))
            height = anchor[1] * np.exp(np.clip(raw_boxes[:, 3], -3.0, 2.0))
            boxes = np.stack(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
                axis=1,
            ).clip(0.0, 1.0)

            class_constraints = constraints[class_id]
            if class_constraints:
                box_width = boxes[:, 2] - boxes[:, 0]
                box_height = boxes[:, 3] - boxes[:, 1]
                valid = np.ones(boxes.shape[0], dtype=bool)
                values_by_name = {
                    "center_x": (boxes[:, 0] + boxes[:, 2]) / 2,
                    "center_y": (boxes[:, 1] + boxes[:, 3]) / 2,
                    "width": box_width,
                    "height": box_height,
                }
                for name, box_values in values_by_name.items():
                    minimum = class_constraints.get(f"{name}_min")
                    maximum = class_constraints.get(f"{name}_max")
                    if minimum is not None:
                        valid &= box_values >= minimum
                    if maximum is not None:
                        valid &= box_values <= maximum
                boxes, scores = boxes[valid], scores[valid]
                if not scores.size:
                    continue

            selected = _nms_numpy(boxes, scores, nms_thresholds[class_id])
            for index in selected[:max_detections]:
                selected_box = boxes[index].copy()
                calibration = calibrations[class_id]
                if calibration:
                    box_center_x = (selected_box[0] + selected_box[2]) / 2
                    box_center_y = (selected_box[1] + selected_box[3]) / 2
                    box_width = (selected_box[2] - selected_box[0]) * float(
                        calibration.get("width_scale", 1.0)
                    )
                    box_height = (selected_box[3] - selected_box[1]) * float(
                        calibration.get("height_scale", 1.0)
                    )
                    box_center_x += float(calibration.get("center_x_offset", 0.0))
                    box_center_y += float(calibration.get("center_y_offset", 0.0))
                    selected_box = np.asarray(
                        (
                            box_center_x - box_width / 2,
                            box_center_y - box_height / 2,
                            box_center_x + box_width / 2,
                            box_center_y + box_height / 2,
                        ),
                        dtype=np.float32,
                    ).clip(0.0, 1.0)
                image_detections.append(
                    _NumpyDetection(
                        class_id=class_id,
                        confidence=float(scores[index]),
                        box=tuple(float(value) for value in selected_box),
                    )
                )
        image_detections.sort(key=lambda item: item.confidence, reverse=True)
        output.append(image_detections[:max_detections])
    return output


def decode_output_tensor(
    output_nhwc: np.ndarray,
    manifest: dict[str, Any],
) -> list[list["Detection"]]:
    decoder = manifest["postprocessing"]["decoder"]
    logits = output_tensor_to_logits(output_nhwc, manifest)
    try:
        import torch
        from qnn.detection import decode_predictions
    except ModuleNotFoundError:
        return _decode_predictions_numpy(logits, decoder)
    return decode_predictions(
        torch.from_numpy(logits),
        confidence_threshold=tuple(float(x) for x in decoder["confidence_thresholds"]),
        nms_iou=decoder_nms_iou(decoder),
        box_constraints=decoder_box_constraints(decoder),
        box_calibration=decoder_box_calibration(decoder),
        pre_nms_topk=int(decoder["pre_nms_topk"]),
        max_detections=int(decoder["max_detections"]),
        anchors=tuple(tuple(float(x) for x in pair) for pair in decoder["anchors"]),
        slots_per_class=tuple(int(x) for x in decoder["slots_per_class"]),
    )


def accumulator_to_logits(
    accumulator_nhwc: np.ndarray, manifest: dict[str, Any]
) -> np.ndarray:
    """Backward-compatible alias for :func:`output_tensor_to_logits`."""

    return output_tensor_to_logits(accumulator_nhwc, manifest)


def decode_accumulator(
    accumulator_nhwc: np.ndarray, manifest: dict[str, Any]
) -> list[list["Detection"]]:
    """Backward-compatible alias for :func:`decode_output_tensor`."""

    return decode_output_tensor(accumulator_nhwc, manifest)
