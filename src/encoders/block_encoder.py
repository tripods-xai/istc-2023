import abc
from typing import Tuple, Dict, Any
import math

import torch
import torch.nn as nn

from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    bitarray2dec,
    check_int,
    enumerate_binary_inputs,
)

from .encoder import Encoder


class BlockEncoder(Encoder):
    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        pass

    def settings(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

    def to_codebook(self, dtype=torch.int8):
        all_inputs = enumerate_binary_inputs(self.input_size, dtype=torch.int8)
        codebook = self(all_inputs, dtype=dtype)
        return CodebookEncoder(codebook=codebook, device_manager=self.device_manager)


class CodebookEncoder(BlockEncoder):
    def __init__(
        self,
        codebook: torch.CharTensor,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager=device_manager)
        # codebook will have dimension 2^input_size x output_size
        self.codebook = codebook
        self._output_size = self.codebook.shape[-1]
        self._input_size = check_int(math.log2(self.codebook.shape[0]))

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        dec_input = bitarray2dec(x, device=self.device_manager.device)
        return self.codebook[dec_input].to(dtype)

    def long_settings(self) -> Dict[str, Any]:
        return {
            **self.settings(),
            "codebook": "...too big",
        }

    def is_codeword(self, y_hat: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        raise NotImplementedError()


class ParityEncoder(BlockEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager=device_manager)
        self._input_size = input_size
        self._output_size = output_size
        assert self._output_size > self._input_size

        self.weight = nn.Parameter(
            torch.cat(
                [
                    torch.eye(self._input_size, device=self.device_manager.device),
                    torch.randint(
                        0,
                        2,
                        (self._input_size, self._output_size - self._input_size),
                        generator=self.device_manager.generator,
                        device=self.device_manager.device,
                    ).float(),  # Should be a float to be a parameter
                ],
                dim=1,
            )
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def parity_size(self) -> int:
        return self.output_size - self.input_size

    @property
    def syndrome(self):
        parity = self.weight[:, self.input_size :]
        return torch.cat([-parity, torch.eye(self.parity_size)], dim=0) % 2

    def forward(self, x: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        # Overflow is not an issue for int-types since all overflows are modulo even nums.
        return (x.to(dtype) @ self.weight.to(dtype)) % 2

    def is_codeword(self, y_bin: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        return (
            torch.all((y_bin @ self.syndrome) % 2 == 0.0, dim=-1),
            y_bin[:, : self.input_size],
        )

    def long_settings(self) -> Dict[str, Any]:
        return {
            **self.settings(),
            "weight": self.weight.tolist(),
        }
