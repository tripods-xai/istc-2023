from typing import Tuple, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from block_encoder import CodebookEncoder

import abc

import torch
from torch import nn

from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    WithSettings,
    ModuleExtension,
    NamedTensor,
)
from ..channels import NoisyChannel
from ..modulation import Modulator
from ..graphs import InferenceGraph


class Encoder(ModuleExtension, WithSettings):
    @property
    def delay(self) -> int:
        return 0

    @abc.abstractmethod
    def is_codeword(self, y_hat: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def to_codebook(self, dtype=torch.int8) -> "CodebookEncoder":
        pass

    def update(self):
        """When parameters have changed, running this method will update all dependent values."""
        pass

    @property
    def batch_dependent(self) -> bool:
        return False

    def dependency_graph(self) -> InferenceGraph:
        raise NotImplementedError

    def compute_evidence(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
        **kwargs
    ) -> Dict[str, NamedTensor]:
        raise NotImplementedError

    def dummy_input(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError
