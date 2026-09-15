"""Tiny quantized patch classifier for candidate-gated FPGA inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import brevitas.nn as qnn
import torch
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat
from torch import nn


@dataclass(frozen=True)
class PatchClassifierConfig:
    input_size: int = 32
    channels: tuple[int, ...] = (8, 12, 16)
    weight_bits: int = 4
    activation_bits: int = 6
    input_bits: int = 8
    output_bits: int = 8
    spatial_head: bool = False
    input_transform: str = "raw"
    class_names: tuple[str, ...] = ("background", "particle")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def patch_config_from_dict(
    saved: dict[str, object],
) -> PatchClassifierConfig:
    return PatchClassifierConfig(
        input_size=int(saved.get("input_size", 32)),
        channels=tuple(int(value) for value in saved["channels"]),
        weight_bits=int(saved["weight_bits"]),
        activation_bits=int(saved["activation_bits"]),
        input_bits=int(saved.get("input_bits", 8)),
        output_bits=int(saved.get("output_bits", 8)),
        spatial_head=bool(saved.get("spatial_head", False)),
        input_transform=str(saved.get("input_transform", "raw")),
        class_names=tuple(saved.get("class_names", ("background", "particle"))),
    )


class QuantPatchBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        weight_bits: int,
        activation_bits: int,
    ) -> None:
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            weight_bit_width=weight_bits,
            return_quant_tensor=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=activation_bits,
            return_quant_tensor=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(inputs)))


class TinyQuantPatchClassifier(nn.Module):
    """Binary W4A6 classifier for one 32x32 grayscale candidate patch.

    The model returns one quantized logit per patch. Sigmoid and the selected
    confidence threshold remain in host/RTL post-processing, which keeps the
    exported neural graph small.
    """

    def __init__(
        self,
        config: PatchClassifierConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PatchClassifierConfig()
        if self.config.input_size < 16 or self.config.input_size % 8:
            raise ValueError("input_size must be at least 16 and divisible by 8")
        if len(self.config.channels) != 3:
            raise ValueError("Exactly three channel stages are required")
        if len(self.config.class_names) != 2:
            raise ValueError("Binary classifier requires two class names")

        self.input_quant = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=self.config.input_bits,
            return_quant_tensor=False,
        )
        channels = self.config.channels
        self.features = nn.Sequential(
            QuantPatchBlock(
                1,
                channels[0],
                weight_bits=self.config.input_bits,
                activation_bits=self.config.activation_bits,
            ),
            QuantPatchBlock(
                channels[0],
                channels[1],
                weight_bits=self.config.weight_bits,
                activation_bits=self.config.activation_bits,
            ),
            QuantPatchBlock(
                channels[1],
                channels[2],
                weight_bits=self.config.weight_bits,
                activation_bits=self.config.activation_bits,
            ),
        )
        pooled_size = self.config.input_size // 8
        self.head = qnn.QuantConv2d(
            channels[-1],
            1,
            kernel_size=pooled_size if self.config.spatial_head else 1,
            bias=True,
            weight_bit_width=self.config.output_bits,
            return_quant_tensor=False,
        )
        self.global_average = (
            nn.Identity()
            if self.config.spatial_head
            else nn.AvgPool2d(kernel_size=pooled_size, stride=pooled_size)
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=self.config.output_bits,
            return_quant_tensor=False,
        )
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            inputs.ndim != 4
            or inputs.shape[1] != 1
            or inputs.shape[2] != self.config.input_size
            or inputs.shape[3] != self.config.input_size
        ):
            raise ValueError(
                "Expected input shape "
                f"[batch, 1, {self.config.input_size}, "
                f"{self.config.input_size}]"
            )
        features = self.features(self.input_quant(inputs))
        logits = self.global_average(self.head(features))
        return self.output_quant(logits).flatten(1)


def count_patch_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
