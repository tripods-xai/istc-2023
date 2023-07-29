from typing import Tuple, Dict, Any, Generic, TypeVar, Union
import math

import torch

from ..constants import INPUT_SYMBOL
from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    enumerate_binary_inputs,
    check_int,
    NamedTensor,
)
from ..interleavers import Interleaver, FixedPermuteInterleaver
from ..graphs import (
    general_turbo_graph,
    InferenceGraph,
    nonrecursive_turbo_graph,
    nonrecursive_dependency_turbo_graph,
)
from ..channels import NoisyChannel
from ..modulation import Modulator

from .encoder import Encoder
from .block_encoder import CodebookEncoder, BlockEncoder
from .convolutional_encoder import (
    TrellisEncoder,
    GeneralizedConvolutionalEncoder,
    AffineConvolutionalEncoder,
)

E = TypeVar("E", bound=BlockEncoder)


class TurboEncoder(Encoder, Generic[E]):
    def __init__(
        self,
        # Typing is ugly here. Issue is because BlockEncoder and TrellisEncoder return different output shapes
        noninterleaved_encoder: E,
        interleaved_encoder: E,
        interleaver: Interleaver,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        assert interleaved_encoder.input_size == noninterleaved_encoder.input_size

        super().__init__(device_manager)
        self.noninterleaved_encoder = noninterleaved_encoder
        self.interleaved_encoder = interleaved_encoder
        self.interleaver = interleaver

    @property
    def input_size(self):
        return self.noninterleaved_encoder.input_size

    def is_codeword(self, y_hat: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        raise NotImplementedError

    def dummy_input(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.input_size), device=self.device_manager.device
        )

    def forward(
        self, data: torch.Tensor, dtype=torch.float, reset_interleaver=True
    ) -> torch.Tensor:
        noni_out = self.noninterleaved_encoder(data, dtype=dtype)
        i_data = (
            self.interleaver(data)
            if reset_interleaver
            else self.interleaver.interleave(data)
        )
        i_out = self.interleaved_encoder(i_data, dtype=dtype)

        return torch.concat([noni_out, i_out], dim=-1)

    def to_codebook(self, dtype=torch.int8) -> CodebookEncoder:
        all_inputs = enumerate_binary_inputs(
            self.input_size, dtype=torch.int8, device=self.device_manager.device
        )
        # BUG: Don't forget to undo this!!!
        # codebook = self(all_inputs, dtype=dtype, reset_interleaver=True)
        codebook = self(all_inputs, dtype=dtype, reset_interleaver=False)
        return CodebookEncoder(codebook=codebook, device_manager=self.device_manager)

    @property
    def batch_dependent(self) -> bool:
        return self.interleaver.batch_dependent

    def settings(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "noninterleaved_encoder": self.noninterleaved_encoder.settings(),
            "interleaved_encoder": self.interleaved_encoder.settings(),
            "interleaver": self.interleaver.settings(),
        }

    def long_settings(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "noninterleaved_encoder": self.noninterleaved_encoder.long_settings(),
            "interleaved_encoder": self.interleaved_encoder.long_settings(),
            "interleaver": self.interleaver.long_settings(),
        }

    def update(self):
        self.interleaved_encoder.update()
        self.noninterleaved_encoder.update()


T = TypeVar("T", bound=TrellisEncoder)


class StreamedTurboEncoder(TurboEncoder[T], Generic[T]):
    def __init__(
        self,
        noninterleaved_encoder: T,
        interleaved_encoder: T,
        interleaver: Interleaver,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(
            noninterleaved_encoder,
            interleaved_encoder,
            interleaver,
            device_manager=device_manager,
        )

    def validate(self):
        assert (
            self.noninterleaved_encoder.input_size
            == self.interleaved_encoder.input_size
        )

    @property
    def num_output_channels(self):
        return (
            self.noninterleaved_encoder.num_output_channels
            + self.interleaved_encoder.num_output_channels
        )

    @property
    def is_nonrecursive(self):
        # print(
        #     f"a: {isinstance(self.noninterleaved_encoder, GeneralizedConvolutionalEncoder)}"
        # )
        # print(
        #     f"b: {isinstance(self.interleaved_encoder, GeneralizedConvolutionalEncoder)}"
        # )
        # print(f"c: {self.noninterleaved_encoder.feedback is None}")
        # print(f"d: {self.interleaved_encoder.feedback is None }")
        return (
            isinstance(self.noninterleaved_encoder, GeneralizedConvolutionalEncoder)
            and isinstance(self.interleaved_encoder, GeneralizedConvolutionalEncoder)
            and self.noninterleaved_encoder.feedback is None
            and self.interleaved_encoder.feedback is None
        )

    @property
    def is_affine(self):
        return isinstance(
            self.noninterleaved_encoder, AffineConvolutionalEncoder
        ) and isinstance(self.interleaved_encoder, AffineConvolutionalEncoder)

    def dependency_graph(self) -> InferenceGraph:
        if self.interleaver.batch_dependent:
            raise ValueError(
                f"Cannot create a dependency graph from a code with a batch dependent interleaver (of type {self.interleaver.__class__})."
            )
        # Right now FixedPermuteInterleaver is the only one that is not batch_dependent
        # If I introduce a new kind, this logic might need to change.
        assert isinstance(self.interleaver, FixedPermuteInterleaver)
        if (
            self.noninterleaved_encoder.num_states
            != self.interleaved_encoder.num_states
        ):
            raise NotImplementedError(
                f"Cannot create turbo dependency graph from two encoders with different number state sizes."
            )
        if self.noninterleaved_encoder.delay != self.interleaved_encoder.delay:
            raise NotImplementedError(
                f"Cannot create turbo dependency graph from two encoders with different delays."
            )
        if self.is_affine:
            return nonrecursive_dependency_turbo_graph(
                self.interleaver.permutation,
                nonrecursive_dependencies_noni=self.noninterleaved_encoder.generator,
                nonrecursive_dependencies_i=self.interleaved_encoder.generator,
                delay=self.noninterleaved_encoder.delay,
            )
        elif self.is_nonrecursive:
            return nonrecursive_turbo_graph(
                self.interleaver.permutation,
                self.noninterleaved_encoder.window,
                delay=self.noninterleaved_encoder.delay,
            )
        else:
            if self.noninterleaved_encoder.delay != 0:
                raise NotImplementedError(
                    "Cannot create a general turbo dependency graph from a constituent encoder with nonzero delay."
                )
            return general_turbo_graph(
                self.interleaver.permutation,
                check_int(math.log2(self.noninterleaved_encoder.num_states)),
            )

    def compute_evidence(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
        input_symbols: list[str] = None,
        state_symbol: str = "s",
        received_factor_symbol: str = "y",
        transition_factor_symbol: str = "st",
        prime_symbol: str = "p",
    ) -> Dict[str, NamedTensor]:
        batch_size, timesteps, channels = received_symbols.shape
        if input_symbols is None:
            input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(timesteps)]

        interleaved_inputs = [
            input_symbols[self.interleaver.permutation[i]] for i in range(timesteps)
        ]

        noninterleaved_evidence = self.noninterleaved_encoder.compute_evidence(
            received_symbols=received_symbols[
                ..., : self.noninterleaved_encoder.num_output_channels
            ],
            channel=channel,
            modulator=modulator,
            input_symbols=input_symbols,
            received_factor_symbol=received_factor_symbol,
            # Recursive codes only
            state_symbol=state_symbol,
            transition_factor_symbol=transition_factor_symbol,
        )

        interleaved_evidence = self.interleaved_encoder.compute_evidence(
            received_symbols=received_symbols[
                ..., self.noninterleaved_encoder.num_output_channels :
            ],
            channel=channel,
            modulator=modulator,
            input_symbols=interleaved_inputs,
            received_factor_symbol=received_factor_symbol + prime_symbol,
            # Recursive codes only
            state_symbol=state_symbol + prime_symbol,
            transition_factor_symbol=transition_factor_symbol + prime_symbol,
        )

        return {**noninterleaved_evidence, **interleaved_evidence}
