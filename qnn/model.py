from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat


@dataclass(frozen=True)
class DetectorConfig:
    input_size: int = 256
    input_width: int = 0
    input_height: int = 0
    num_classes: int = 2
    class_names: tuple[str, ...] = ("cell", "droplet")
    channels: tuple[int, ...] = (8, 12, 16, 16)
    weight_bits: int = 4
    activation_bits: int = 4
    input_bits: int = 8
    output_bits: int = 8
    downsample: int = 4
    downsample_width: int = 0
    downsample_height: int = 0
    slots_per_class: tuple[int, ...] = (1, 1)
    anchors: tuple[tuple[float, float], ...] = (
        (0.030, 0.031),
        (0.210, 0.219),
    )

    @property
    def image_width(self) -> int:
        return self.input_width or self.input_size

    @property
    def image_height(self) -> int:
        return self.input_height or self.input_size

    @property
    def grid_width(self) -> int:
        return self.image_width // (self.downsample_width or self.downsample)

    @property
    def grid_height(self) -> int:
        return self.image_height // (self.downsample_height or self.downsample)

    @property
    def grid_size(self) -> int | tuple[int, int]:
        if self.grid_width == self.grid_height:
            return self.grid_width
        return (self.grid_width, self.grid_height)

    @property
    def output_channels(self) -> int:
        # Each class owns an objectness logit and x/y/w/h box values.
        return sum(self.slots_per_class) * 5

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["grid_size"] = self.grid_size
        return data

    @property
    def slot_class_ids(self) -> tuple[int, ...]:
        return tuple(
            class_id
            for class_id, slot_count in enumerate(self.slots_per_class)
            for _ in range(slot_count)
        )


def config_from_dict(saved: dict[str, object]) -> DetectorConfig:
    return DetectorConfig(
        input_size=int(saved.get("input_size", 256)),
        input_width=int(saved.get("input_width", 0)),
        input_height=int(saved.get("input_height", 0)),
        num_classes=int(saved["num_classes"]),
        class_names=tuple(saved["class_names"]),
        channels=tuple(saved["channels"]),
        weight_bits=int(saved["weight_bits"]),
        activation_bits=int(saved["activation_bits"]),
        input_bits=int(saved["input_bits"]),
        output_bits=int(saved["output_bits"]),
        downsample=int(saved["downsample"]),
        downsample_width=int(saved.get("downsample_width", 0)),
        downsample_height=int(saved.get("downsample_height", 0)),
        slots_per_class=tuple(saved.get("slots_per_class", (1, 1))),
        anchors=tuple(tuple(pair) for pair in saved["anchors"]),
    )


class QuantConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        weight_bits: int,
        activation_bits: int,
    ) -> None:
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            weight_bit_width=weight_bits,
            return_quant_tensor=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=activation_bits,
            return_quant_tensor=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(inputs)))


class TinyQuantDetector(nn.Module):
    """Small class-specific grid detector designed for FINN conversion.

    The model intentionally excludes sigmoid, box decoding and NMS. Those
    operations stay in host software so the exported FPGA graph contains only
    quantized convolutions, batch normalization and quantized activations.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        downsample_width = self.config.downsample_width or self.config.downsample
        downsample_height = self.config.downsample_height or self.config.downsample
        if (
            self.config.image_width % downsample_width != 0
            or self.config.image_height % downsample_height != 0
        ):
            raise ValueError("Input width and height must be divisible by downsample")
        if len(self.config.anchors) != self.config.num_classes:
            raise ValueError("Each class must have one width/height anchor")
        if len(self.config.slots_per_class) != self.config.num_classes:
            raise ValueError("slots_per_class must have one value per class")
        if any(slot_count < 1 for slot_count in self.config.slots_per_class):
            raise ValueError("Each class must own at least one output slot")
        if downsample_width not in (4, 8) or downsample_height not in (4, 8):
            raise ValueError("Supported width/height downsample values are 4 and 8")

        # Inputs are normalized to [0, 1]. QuantReLU preserves that behavior
        # while giving FINN an explicit unsigned-activation predecessor.
        self.input_quant = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=self.config.input_bits,
            return_quant_tensor=False,
        )

        channels = self.config.channels
        third_stride = (
            1 if downsample_height == 4 else 2,
            1 if downsample_width == 4 else 2,
        )
        self.features = nn.Sequential(
            QuantConvBlock(
                1,
                channels[0],
                stride=2,
                weight_bits=self.config.input_bits,
                activation_bits=self.config.activation_bits,
            ),
            QuantConvBlock(
                channels[0],
                channels[1],
                stride=2,
                weight_bits=self.config.weight_bits,
                activation_bits=self.config.activation_bits,
            ),
            QuantConvBlock(
                channels[1],
                channels[2],
                stride=third_stride,
                weight_bits=self.config.weight_bits,
                activation_bits=self.config.activation_bits,
            ),
            QuantConvBlock(
                channels[2],
                channels[3],
                stride=1,
                weight_bits=self.config.weight_bits,
                activation_bits=self.config.activation_bits,
            ),
        )
        self.head = qnn.QuantConv2d(
            channels[-1],
            self.config.output_channels,
            kernel_size=1,
            bias=True,
            weight_bit_width=self.config.output_bits,
            return_quant_tensor=False,
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=self.config.output_bits,
            return_quant_tensor=False,
        )
        self._initialize_head()

    def _initialize_head(self) -> None:
        # A negative objectness bias prevents thousands of initial false boxes.
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
            with torch.no_grad():
                self.head.bias.view(sum(self.config.slots_per_class), 5)[:, 0].fill_(-3.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("Expected input shape [batch, 1, height, width]")
        features = self.features(self.input_quant(inputs))
        return self.output_quant(self.head(features))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
