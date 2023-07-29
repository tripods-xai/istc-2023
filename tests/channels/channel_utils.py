from typing import Type

import torch

from src.channels import NoisyChannel


def create_noiseless_channel(parent_class: Type[NoisyChannel]):
    class NoiselessChannel(parent_class):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    return NoiselessChannel


def create_fixed_noise_channel(
    parent_class: Type[NoisyChannel], additive_noise: torch.Tensor
):
    class FixedNoiseChannel(parent_class):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + additive_noise

    return FixedNoiseChannel
