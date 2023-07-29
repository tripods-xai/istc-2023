from typing import Optional, Tuple, Dict, Any, List, Union

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..constants import INPUT_SYMBOL
from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    bitarray2dec,
    enumerate_binary_inputs,
    enumerate_binary_inputs_chunked,
    EPSILON,
    check_int,
    base_2_accumulator,
    NamedTensor,
    dynamic_slice,
)
from ..fourier import fourier_to_table
from ..graphs import (
    InferenceGraph,
    general_convolutional_code,
    general_convolutional_factors,
    nonrecursive_convolutional_code,
    nonrecursive_convolutional_factors,
    nonrecursive_dependency_convolutional_code,
    nonrecursive_dependency_convolutional_factors,
)
from ..channels import NoisyChannel
from ..modulation import Modulator
from .trellis import (
    UnrolledTrellis,
    UnrolledStateTransitionGraph,
    Trellis,
    StateTransitionGraph,
)
from .encoder import Encoder
from .block_encoder import CodebookEncoder


class TrellisEncoder(Encoder):
    def __init__(
        self,
        trellises: UnrolledTrellis,
        normalize_output_table=False,
        delay_state_transitions: UnrolledStateTransitionGraph = None,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager)
        self.trellises = trellises
        self.normalize_output_table = normalize_output_table
        self.delay_state_transitions = delay_state_transitions

        # Validation
        if self.delay_state_transitions is not None:
            assert self.delay_state_transitions.num_states == self.trellises.num_states

        (
            self.init_conditional,
            self.delayed_init_state,
        ) = self._construct_init_conditional()
        assert self.trellises.num_inputs == self.num_inputs

    @property
    def num_inputs(self) -> int:
        # Hardcoded
        return 2

    @property
    def num_states(self) -> int:
        return self.trellises.num_states

    @property
    def state_size(self) -> int:
        return check_int(math.log2(self.num_states))

    @property
    def num_output_channels(self):
        return self.trellises.num_outputs

    @property
    def input_size(self):
        return self.trellises.num_steps

    def dummy_input(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.input_size), device=self.device_manager.device
        )

    @property
    def output_tables(self):
        output_tables = self.trellises.output_tables
        if self.normalize_output_table:
            std, mean = torch.std_mean(
                output_tables, dim=[1, 2], keepdim=True, unbiased=False
            )
            output_tables = (output_tables - mean) / (EPSILON + std)
            return output_tables
        else:
            return output_tables

    @property
    def delay(self) -> int:
        return (
            0
            if self.delay_state_transitions is None
            else self.delay_state_transitions.num_steps
        )

    def _construct_init_conditional(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        - init_conditional = States x DelayInputs (2^delay) - This is log p(s0 = s | u[0:d-1]). d := delay.
            if delay == 0: This is just log p(s0 = s) = log [1 if s0 = 0; 0 otherwise]
                                                      = 0 if s0 = 0; -inf otherwise
        - delayed_init_state = DelayInputs (2^delay) - This maps each delay input to a starting state.

        """
        if self.delay_state_transitions is None:
            # delay = 0, so shape is States x 1
            init_conditional = torch.tensor(
                [0.0] + [-np.inf] * (self.num_states - 1),
                device=self.device_manager.device,
            )[:, None]
            # delay = 0 so shape is 2 ^ 0 = 1
            delayed_init_state = torch.tensor(
                [0], device=self.device_manager.device, dtype=torch.long
            )
        else:
            # 2^delay x delay
            binary_inputs = enumerate_binary_inputs(
                self.delay, dtype=torch.long, device=self.device_manager.device
            )
            num_delay_inputs = binary_inputs.shape[0]
            cur_states: torch.LongTensor = torch.zeros(
                num_delay_inputs, dtype=torch.long, device=self.device_manager.device
            )
            for t in range(self.delay):
                cur_states = self.delay_state_transitions.next_states[
                    t, cur_states, binary_inputs[:, t]
                ]
            # 2^delay
            delayed_init_state = cur_states

            # States x 2^delay
            init_conditional_prelog = torch.zeros(
                (self.num_states, num_delay_inputs), device=self.device_manager.device
            )
            init_conditional_prelog[
                delayed_init_state,
                torch.arange(num_delay_inputs, device=self.device_manager.device),
            ] = 1.0
            init_conditional = torch.log(init_conditional_prelog)

        assert init_conditional.shape[0] == self.num_states
        assert init_conditional.shape[1] == 2**self.delay
        assert delayed_init_state.shape[0] == 2**self.delay
        return init_conditional, delayed_init_state

    def forward(self, data: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        """
        Parameters
        ----------
        data (Batch x Time) : torch.Tensor
            Binary input made up of 0,1 only
        """
        batch_size = data.shape[0]
        msg_len = data.shape[1]
        # TensorArray will be Time x Batch x Channels
        output = torch.zeros(
            (batch_size, msg_len, self.trellises.num_outputs),
            dtype=dtype,
            device=self.device_manager.device,
        )
        if self.delay == 0:
            cur_state = self.delayed_init_state[[0] * batch_size]
        else:
            cur_state = self.delayed_init_state[
                bitarray2dec(data[:, : self.delay], device=self.device_manager.device)
            ]
        # Pad the end with 0's to make up for the delay - msg_reduced.shape[0] == msg_len
        data_reduced = F.pad(
            data[:, self.delay :], [0, self.delay], mode="constant"
        ).to(torch.long)
        self.trellises.ensure_length(data_reduced.shape[1])
        # Does the normalization if necessary
        output_tables = self.output_tables
        for t in range(msg_len):
            # print("==========")
            # print(t)
            # print(data_reduced[:, t])
            # print(cur_state)
            output[:, t] = output_tables[t, cur_state, data_reduced[:, t]].to(dtype)
            cur_state = self.trellises.next_states[t, cur_state, data_reduced[:, t]]
        return output

    def is_codeword(self, y_hat: torch.Tensor) -> Tuple[torch.BoolTensor, torch.Tensor]:
        raise NotImplementedError("TODO")

    def with_systematic(self) -> "TrellisEncoder":
        return TrellisEncoder(
            self.trellises.with_systematic(device=self.device_manager.device),
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
            device_manager=self.device_manager,
        )

    def concat_outputs(self, other: "TrellisEncoder") -> "TrellisEncoder":
        assert self.normalize_output_table == other.normalize_output_table
        assert self.delay_state_transitions == other.delay_state_transitions
        if self.device_manager != other.device_manager:
            print(
                "self and other device_managers are different, using the one from self."
            )
        new_trellises = self.trellises.concat_outputs(other.trellises)
        return TrellisEncoder(
            trellises=new_trellises,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
            device_manager=self.device_manager,
        )

    def get_encoder_channels(self, encoder_channels: List[int]):
        return TrellisEncoder(
            trellises=self.trellises.get_output_channels(encoder_channels),
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
            device_manager=self.device_manager,
        )

    def to_codebook(self, dtype=torch.int8, chunk_size=2**18) -> CodebookEncoder:
        all_input_iter = enumerate_binary_inputs_chunked(
            window=self.input_size, dtype=torch.int8, chunk_size=chunk_size
        )
        codebook = torch.empty(
            (2**self.input_size, self.num_output_channels * self.input_size),
            dtype=dtype,
        )
        start = 0
        for inputs in all_input_iter:
            end = start + inputs.shape[0]
            codebook[start:end] = torch.reshape(
                self(inputs, dtype=dtype), (inputs.shape[0], -1)
            )
            start = end

        return CodebookEncoder(codebook=codebook, device_manager=self.device_manager)

    def settings(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "input_size": self.input_size,
            "num_output_channels": self.num_output_channels,
        }

    def dependency_graph(self) -> InferenceGraph:
        return general_convolutional_code(
            self.input_size, check_int(math.log2(self.num_states))
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
    ) -> Dict[str, NamedTensor]:
        """Computes the evidence log-prob tensors for each factor

        Warning
        -------
        This needs to be synced up with `general_convolutional_factors`.
        Things will break otherwise. I've included an assertion to catch
        any future bugs.

        Parameters
        ----------
        received_symbols (Batch x Time x Channels):
            The received message. The factors are assumed to be named of the form
            "y_i" for timestep i.
        factor_groups : Dict[str, Set[str]]
            A mapping of factor group names to a set of names of variables in the factor group.
        [name] ([shape]) : [type]
            [desc]

        Returns
        -------
        [type]
            [desc]

        """
        batch_size, timesteps, channels = received_symbols.shape
        if input_symbols is None:
            input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(timesteps)]
        else:
            assert len(input_symbols) == timesteps
        factor_groups, _ = general_convolutional_factors(
            self.input_size,
            self.state_size,
            input_symbols=input_symbols,
            state_symbol=state_symbol,
            received_factor_symbol=received_factor_symbol,
            transition_factor_symbol=transition_factor_symbol,
        )
        evidence: Dict[str, NamedTensor] = {}

        # Compute evidence for factors of type 1. This comes from output table
        # Batch x Time x States x Inputs
        chi_values = torch.sum(
            channel.log_prob(
                received_symbols[:, :, None, None],
                modulator.modulate(self.output_tables)[None],
            ),
            dim=-1,
        )
        for i in range(timesteps):
            received_factor_i = f"{received_factor_symbol}_{i}"
            input_i = input_symbols[i]
            state_i = f"{state_symbol}_{i}"
            if i == 0:
                # Assume state 0
                evidence_tensor = NamedTensor(
                    chi_values[:, i, 0], dim_names=[input_i], batch_dims=1
                )
            else:
                evidence_tensor = NamedTensor(
                    chi_values[:, i], dim_names=[state_i, input_i], batch_dims=1
                )
            evidence[received_factor_i] = evidence_tensor

        # Compute evidence for factors of type 2. This comes from the `next_states` tensor
        transitions = torch.full(
            (timesteps, self.num_states, self.num_inputs, self.num_states), -np.inf
        )
        next_states_expanded = self.trellises.next_states[..., None]
        transitions.scatter_(
            dim=-1,
            index=next_states_expanded,
            src=torch.zeros(next_states_expanded.shape, dtype=transitions.dtype),
        )
        # Use 1 for batch dim since this is same for all batches.
        # 1 (Batch) x Time x States x States(Next) x Inputs
        transitions = transitions.permute(0, 1, 3, 2)[None]
        for i in range(timesteps - 1):
            state_i = f"{state_symbol}_{i}"
            state_i_p_1 = f"{state_symbol}_{i+1}"
            input_i = input_symbols[i]
            if i == 0:
                evidence_tensor = NamedTensor(
                    transitions[:, i, 0], dim_names=[state_i_p_1, input_i], batch_dims=1
                )
            else:
                evidence_tensor = NamedTensor(
                    transitions[:, i],
                    dim_names=[state_i, state_i_p_1, input_i],
                    batch_dims=1,
                )
            evidence[f"{transition_factor_symbol}_{i}"] = evidence_tensor

        assert evidence.keys() == factor_groups.keys()
        return evidence


class GeneralizedConvolutionalEncoder(TrellisEncoder):
    """Convolutional code represented as a table.

    Attributes
    ----------
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(
        self,
        table: torch.Tensor,
        num_steps: int,
        feedback: torch.LongTensor = None,
        delay=0,
        constraint: str = None,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        """
        Parameters
        ---------
        table (Inputs x Channels) : torch.Tensor
            Maps possible (binary) inputs to the output message.
        feedback = Inputs - feedback for each input
        """
        table = table.to(device=device_manager.device)
        feedback = feedback
        if feedback is not None:
            feedback = feedback.to(device=device_manager.device)
        window = check_int(math.log2(table.shape[0]))
        trellis = self._construct_trellis(
            table=table, window=window, feedback=feedback, device=device_manager.device
        )
        if delay > 0:
            delay_state_transitions = trellis.state_transitions.unroll(delay)
        else:
            delay_state_transitions = None
        super(GeneralizedConvolutionalEncoder, self).__init__(
            trellises=trellis.unroll(num_steps),
            normalize_output_table=False,  # Deprecated
            delay_state_transitions=delay_state_transitions,
            device_manager=device_manager,
        )

        self.table = table
        self.feedback = feedback
        self.window = window
        self.trellis = trellis
        self.constraint = constraint

        # Validation
        if self.feedback is not None:
            assert self.num_possible_windows == self.feedback.shape[0]
            assert self.constraint is None

        self._base_2_accumulator = base_2_accumulator(
            self.window, device=self.device_manager.device
        ).float()

    @property
    def num_possible_windows(self) -> int:
        return self.table.shape[0]

    # def do_table_normalization(self):
    #     std, mean = torch.std_mean(self.table, keepdim=True, unbiased=False)
    #     return (self.table - mean) / (EPSILON + std)

    def get_encoder_channels(self, encoder_channels: List[int]):
        return GeneralizedConvolutionalEncoder(
            table=self.table[:, encoder_channels],
            num_steps=self.input_size,
            feedback=self.feedback,
            delay=self.delay,
            constraint=self.constraint,
            device_manager=self.device_manager,
        )

    def apply_constraint(self) -> torch.Tensor:
        if self.constraint == "opt_unit_power":
            return self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=False,
                shift=True,
            )
        elif self.constraint == "multi_opt_unit_power":
            return self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=True,
                shift=True,
            )
        elif self.constraint == "unit_power":
            return self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=False,
                shift=False,
            )
        elif self.constraint == "multi_unit_power":
            return self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=True,
                shift=False,
            )
        elif self.constraint is None:
            return self.table
        else:
            raise NotImplementedError(f"constraint={self.constraint}")

    def apply_constraint_(self):
        if self.constraint == "opt_unit_power":
            self.table.data = self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=False,
                shift=True,
            )
        elif self.constraint == "multi_opt_unit_power":
            self.table.data = self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=True,
                shift=True,
            )
        elif self.constraint == "unit_power":
            self.table.data = self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=False,
                shift=False,
            )
        elif self.constraint == "multi_unit_power":
            self.table.data = self.opt_unit_power_constraint(
                self.table,
                delay=self.delay,
                window=self.window,
                input_size=self.input_size,
                multi=True,
                shift=False,
            )
        elif self.constraint is None:
            pass
        else:
            raise NotImplementedError(f"constraint={self.constraint}")

    @staticmethod
    def opt_unit_power_constraint(
        table: torch.Tensor, delay: int, window: int, input_size: int, multi, shift
    ):
        # First center the table
        # This constant is the best one to maximize variance
        # while maintaining same power
        if multi:
            meandim = 0
            keepdim = True
        else:
            meandim = None
            keepdim = False
        new_table = table
        if shift:
            table_mean = torch.mean(new_table, dim=meandim, keepdim=keepdim)
            bd_means = [
                torch.mean(new_table[: 2**i], dim=meandim, keepdim=keepdim)
                for i in range(1 + delay, window)
            ] + [
                torch.mean(new_table[:: 2**i], dim=meandim, keepdim=keepdim)
                for i in range(1, delay + 1)
            ]
            shift_constant = (
                1.0
                / input_size
                * ((input_size - window + 1) * table_mean + sum(bd_means))
            )
            new_table = new_table - shift_constant

        # Now we compute the power of the new table
        table_power = torch.mean(new_table**2, dim=meandim, keepdim=keepdim)
        bd_powers = [
            torch.mean(new_table[: 2**i] ** 2, dim=meandim, keepdim=keepdim)
            for i in range(1 + delay, window)
        ] + [
            torch.mean(new_table[:: 2**i] ** 2, dim=meandim, keepdim=keepdim)
            for i in range(1, delay + 1)
        ]
        rescale_constant = torch.sqrt(
            1.0
            / input_size
            * ((input_size - window + 1) * table_power + sum(bd_powers))
        )
        new_table = new_table / (rescale_constant + EPSILON)

        return new_table

    def update(self):
        self.trellis = self._construct_trellis(
            table=self.table,
            window=self.window,
            feedback=self.feedback,
            device=self.device_manager.device,
        )
        self.trellises = self.trellis.unroll(self.input_size)

    @staticmethod
    def _construct_trellis(
        table: torch.Tensor, window: int, feedback: Optional[torch.Tensor], device=None
    ) -> Trellis:
        binary_states = enumerate_binary_inputs(window, device=device)
        if feedback is not None:
            binary_states = torch.concat(
                [binary_states[:, :-1], feedback[:, None]], dim=1
            )

        # Even indicies correspond to the last bit not included in state.
        # (i.e. input is coming from the right when reading data left to right)
        reordered_table = table[bitarray2dec(binary_states, device=device)]
        output_table = torch.stack([reordered_table[::2], reordered_table[1::2]], dim=1)

        next_states = bitarray2dec(binary_states[:, 1:], device=device)
        next_states_table = torch.stack([next_states[::2], next_states[1::2]], dim=1)

        return Trellis(
            state_transitions=StateTransitionGraph.from_next_states(
                next_states_table, device=device
            ),
            output_table=output_table,
        )

    def forward(self, data: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        """
        Parameters
        ----------
        data (Batch x Time) : torch.Tensor
            Binary input made up of 0,1 only
        """
        if self.feedback is None:
            data_padded = F.pad(
                data,
                pad=(self.window - 1 - self.delay, self.delay),
                mode="constant",
                value=0,
            )
            table = self.apply_constraint()
            return self._conv_encode(
                msg=data_padded,
                table=table,
                _base_2_accumulator=self._base_2_accumulator,
                device=self.device_manager.device,
            ).to(dtype)
        else:
            return super().forward(data, dtype=dtype)

    def _check_recursive_condition(self):
        # TODO Figure out how to do rsc for composite trellis codes
        if self.feedback is not None:
            raise ValueError(f"Cannot create recursive code, code already has feedback")
        if not torch.all(
            (self.trellis.output_table[:, :, 0] == 0)
            | (self.trellis.output_table[:, :, 0] == 1)
        ):
            raise ValueError(f"First channel is not binary out.")
        if not torch.all(
            self.trellis.output_table[:, 0, 0] != self.trellis.output_table[:, 1, 0]
        ):
            raise ValueError(
                f"Cannot invert code, some output does not change when input changes: 0 -> {self.trellis.output_table[:, 0, 0]} 1 -> {self.trellis.output_table[:, 1, 0]}"
            )

    def with_systematic(
        self,
    ) -> Union["GeneralizedConvolutionalEncoder", "TrellisEncoder"]:
        # systematic_table = torch.zeros(
        #     ([2] * self.window + [1]),
        #     dtype=self.table.dtype,
        #     device=self.table.device,
        # )
        # systematic_table[
        #     dynamic_slice(systematic_table, dim=self.window - 1, index=1)
        # ] = 1
        # systematic_table = systematic_table.reshape(2**self.window, 1)

        # binary_inputs = enumerate_binary_inputs(self.window, dtype=self.table.dtype, device=self.table.device)
        # systematic_table = binary_inputs[:, -1]
        if self.feedback is not None:
            # Check if the feedback is invertible
            is_invertible = torch.all(self.feedback[::2] != self.feedback[1::2])
            if not is_invertible:
                print("Feedback is not invertible, falling back to trellis code")
                return super().with_systematic()

        if self.feedback is None:
            systematic_table = torch.zeros(
                (2**self.window, 1),
                dtype=self.table.dtype,
                device=self.device_manager.device,
            )
            systematic_table[1::2, 0] = 1
            new_table = torch.cat([systematic_table, self.table], dim=1)
        else:
            new_table = torch.cat([self.feedback[:, None], self.table], dim=1)
            # print(self.feedback)
        # print(new_table)

        return GeneralizedConvolutionalEncoder(
            table=new_table,
            num_steps=self.input_size,
            feedback=self.feedback,
            delay=self.delay,
            constraint=self.constraint,
            device_manager=self.device_manager,
        )

    def to_rc(self) -> "GeneralizedConvolutionalEncoder":
        self._check_recursive_condition()
        feedback = self.table[:, 0].to(torch.int8)
        return GeneralizedConvolutionalEncoder(
            table=self.table[:, 1:], num_steps=self.input_size, feedback=feedback
        )

    def to_rsc(self) -> "GeneralizedConvolutionalEncoder":
        # This feedback is guaranteed to be invertible, so return value
        # will be a GeneralizedConvolutionalEncoder.
        return self.to_rc().with_systematic()

    @staticmethod
    def _conv_encode(
        msg: torch.Tensor,
        table: torch.Tensor,
        _base_2_accumulator: torch.FloatTensor = None,
        device=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        msg (Batch x Time) :  torch.Tensor
            Binary message to encode.
        table (Inputs x Channels) : torch.Tensor
            Maps possible (binary) inputs to the output message.

        Returns
        -------
        (Batch x Time x OutChannels) torch.Tensor
            Encoded binary message

        Logic for encoding when there is no feedback, not responsible for padding with initial state.
        Assumes that the message only has 1 channel (and thus channel dimension is reduced out).
        """
        if device is None:
            device = msg.device
        if _base_2_accumulator is None:
            window = check_int(math.log2(table.shape[0]))
            _base_2_accumulator = base_2_accumulator(window, device=device).float()

        # print(msg)
        # breakpoint()
        # Revisiting this 2022-03-25: It seems like sequences are read left to right into binary.
        # Oldest bit becomes most significant digit.
        index_msg = F.conv1d(
            msg[:, None].float(),  # Batch x InChannels=1 x Time
            _base_2_accumulator[None, None],  # Outchannels=1 x Inchannels=1 x Window
        )[:, 0, :].long()
        return table[index_msg]

    def dependency_graph(self) -> InferenceGraph:
        return nonrecursive_convolutional_code(
            self.input_size, self.window, delay=self.delay
        )

    def compute_evidence(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
        input_symbols: list[str] = None,
        received_factor_symbol: str = "y",
        **kwargs,
    ) -> Dict[str, NamedTensor]:
        """Computes the evidence log-prob tensors for each factor

        Warning
        -------
        This needs to be synced up with `nonrecursive_convolutional_factors`.
        Things will break otherwise. I've included an assertion to catch
        any future bugs.

        Parameters
        ----------
        received_symbols (Batch x Time x Channels):
            The received message. The factors are assumed to be named of the form
            "y_i" for timestep i.
        [name] ([shape]) : [type]
            [desc]

        Returns
        -------
        [type]
            [desc]

        """
        if self.feedback is not None:
            return super().compute_evidence(
                received_symbols=received_symbols,
                channel=channel,
                modulator=modulator,
                input_symbols=input_symbols,
                received_factor_symbol=received_factor_symbol,
                **kwargs,
            )
        batch_size, timesteps, channels = received_symbols.shape
        if input_symbols is None:
            input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(timesteps)]
        else:
            assert len(input_symbols) == timesteps
        factor_groups, _ = nonrecursive_convolutional_factors(
            self.input_size,
            self.window,
            delay=self.delay,
            input_symbols=input_symbols,
            factor_symbol=received_factor_symbol,
        )
        evidence: Dict[str, NamedTensor] = {}

        # Compute evidence for factors of type 1. This comes from code table
        # Batch x Time x 2 x ... (Window times)
        chi_values = torch.sum(
            channel.log_prob(
                received_symbols[:, :, None],  # B x T x 1 x C
                modulator.modulate(self.table)[None, None],  # 1 x 1 x 2^W x C
            ),
            dim=-1,
        ).reshape(batch_size, timesteps, *([2] * self.window))
        for i in range(timesteps):
            factor_i = f"{received_factor_symbol}_{i}"
            low = i - self.window + 1 + self.delay
            high = i + 1 + self.delay
            # Assumption is that indices outside our bounds are taken to be 0 input
            slice_tuple = (
                [slice(None), i]  # Batch, Time
                + [0 for _ in range(low, 0)]
                + [slice(None) for _ in range(max(0, low), min(high, timesteps))]
                + [0 for _ in range(timesteps, high)]
            )
            evidence[factor_i] = NamedTensor(
                chi_values[slice_tuple],
                dim_names=[
                    input_symbols[j] for j in range(max(0, low), min(high, timesteps))
                ],
                batch_dims=1,
            )

        assert {
            factor: evidence_data._dim_name_set
            for factor, evidence_data in evidence.items()
        } == factor_groups
        return evidence


class FourierConvolutionalEncoder(GeneralizedConvolutionalEncoder):
    def __init__(
        self,
        fourier_coefficients: torch.Tensor,
        num_steps: int,
        feedback: torch.LongTensor = None,
        delay=0,
        constraint: str = None,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):

        table = fourier_to_table(fourier_coefficients, device_manager=device_manager)
        super().__init__(
            table=table,
            num_steps=num_steps,
            feedback=feedback,
            delay=delay,
            constraint=constraint,
            device_manager=device_manager,
        )

        self.fourier_coefficients = fourier_coefficients

    def update(self):
        self.table = fourier_to_table(
            self.fourier_coefficients, device_manager=self.device_manager
        )
        super().update()

    def apply_constraint(self) -> torch.Tensor:
        if self.constraint == "opt_unit_power":
            return fourier_to_table(
                self.opt_unit_power_constraint(multi=False, shift=True),
                device_manager=self.device_manager,
            )
        elif self.constraint == "multi_opt_unit_power":
            return fourier_to_table(
                self.opt_unit_power_constraint(multi=True, shift=True),
                device_manager=self.device_manager,
            )
        elif self.constraint == "unit_power":
            return fourier_to_table(
                self.opt_unit_power_constraint(multi=False, shift=False)
            )
        elif self.constraint == "multi_unit_power":
            return fourier_to_table(
                self.opt_unit_power_constraint(multi=True, shift=False)
            )
        elif self.constraint == "table_norm":
            return fourier_to_table(self.table_norm())
        elif self.constraint is None:
            return self.table
        else:
            raise NotImplementedError(f"constraint={self.constraint}")

    def apply_constraint_(self):
        if self.constraint == "opt_unit_power":
            self.fourier_coefficients.data = self.opt_unit_power_constraint(
                multi=False, shift=True
            )
        elif self.constraint == "multi_opt_unit_power":
            self.fourier_coefficients.data = self.opt_unit_power_constraint(
                multi=True, shift=True
            )
        elif self.constraint == "unit_power":
            self.fourier_coefficients.data = self.opt_unit_power_constraint(
                multi=False, shift=False
            )
        elif self.constraint == "multi_unit_power":
            self.fourier_coefficients.data = self.opt_unit_power_constraint(
                multi=True, shift=False
            )
        elif self.constraint == "table_norm":
            self.fourier_coefficients.data = self.table_norm()
        elif self.constraint is None:
            pass
        else:
            raise NotImplementedError(f"constraint={self.constraint}")

    def table_norm(self):
        new_fc = self.fourier_coefficients.clone()
        # Now we compute the power of the new table
        table_power = torch.mean(torch.sum(new_fc**2, dim=0, keepdim=True))
        new_fc = new_fc / (table_power + EPSILON)

        return new_fc

    def opt_unit_power_constraint(self, multi, shift):
        if multi:
            meandim = 0
            keepdim = True
        else:
            meandim = None
            keepdim = False
        new_fc = self.fourier_coefficients.clone()
        if shift:
            # First center the table
            # This constant is the best one to maximize variance
            # while maintaining same power
            table_mean = torch.mean(new_fc[0:1], dim=meandim, keepdim=keepdim)
            bd_means = [
                torch.mean(
                    torch.sum(new_fc[:: 2**i], dim=0, keepdim=True),
                    dim=meandim,
                    keepdim=keepdim,
                )
                for i in range(1, self.window)
            ]
            shift_constant = (
                1.0
                / self.input_size
                * ((self.input_size - self.window + 1) * table_mean + sum(bd_means))
            )
            new_fc = torch.cat([new_fc[0:1] - shift_constant, new_fc[1:]], dim=0)

        # Now we compute the power of the new table
        table_power = torch.mean(
            torch.sum(new_fc**2, dim=0, keepdim=True), dim=meandim, keepdim=keepdim
        )
        bd_powers = [
            torch.mean(
                sum(
                    torch.sum(new_fc[j :: 2**i], dim=0, keepdim=True) ** 2
                    for j in range(2**i)
                ),
                dim=meandim,
                keepdim=keepdim,
            )
            for i in range(1, self.window)
        ]
        rescale_constant = torch.sqrt(
            1.0
            / self.input_size
            * ((self.input_size - self.window + 1) * table_power + sum(bd_powers))
        )
        new_fc = new_fc / (rescale_constant + EPSILON)

        return new_fc


class AffineConvolutionalEncoder(GeneralizedConvolutionalEncoder):
    """Convolutional code represented by a single boolean affine function"""

    def __init__(
        self,
        generator: torch.Tensor,
        bias: torch.Tensor,
        num_steps: int,
        delay=0,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        """
        Parameters
        ----------
        generator (OutChannels x Window) : torch.Tensor
            The binary matrix that represents the convolved linear transform
            of the affine function defining the code.
        bias (OutChannels) : torch.Tensor
            The binary vector that represents the binary convolved binary bias
            of the affine function defining the code.
        """

        assert generator.shape[0] == bias.shape[0]

        generator = generator.to(device=device_manager.device)
        bias = bias.to(device=device_manager.device)

        # Create the table
        table = self._affine_encode(
            enumerate_binary_inputs(
                generator.shape[1],
                dtype=generator.dtype,
                device=device_manager.device,
            ),
            generator,
            bias,
        )[
            :, 0, :
        ]  # Inputs x Channels

        super().__init__(
            table, num_steps=num_steps, delay=delay, device_manager=device_manager
        )

        self.generator = generator
        self.bias = bias

    def forward(self, data: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        """
        Parameters
        ----------
        data (Batch x Time) : torch.Tensor
            Binary input made up of 0,1 only
        """
        data_padded = F.pad(data, pad=(self.window - 1, 0), mode="constant", value=0)
        return self._affine_encode(
            msg=data_padded,
            generator=self.generator,
            bias=self.bias,
        ).to(dtype)

    def get_encoder_channels(self, encoder_channels: List[int]):
        return AffineConvolutionalEncoder(
            generator=self.generator[encoder_channels],
            bias=self.bias[encoder_channels],
            num_steps=self.input_size,
            delay=self.delay,
            device_manager=self.device_manager,
        )

    def dependency_graph(self) -> InferenceGraph:
        return nonrecursive_dependency_convolutional_code(
            self.input_size, dependencies=self.generator, delay=self.delay
        )

    def compute_evidence(
        self,
        received_symbols: torch.Tensor,
        channel: NoisyChannel,
        modulator: Modulator,
        input_symbols: list[str] = None,
        received_factor_symbol: str = "y",
        **kwargs,
    ) -> Dict[str, NamedTensor]:
        """Computes the evidence log-prob tensors for each factor

        Warning
        -------
        This needs to be synced up with `affine_convolutional_factors`.
        Things will break otherwise. I've included an assertion to catch
        any future bugs.

        Parameters
        ----------
        received_symbols (Batch x Time x Channels):
            The received message. The factors are assumed to be named of the form
            "y_cj_i" for timestep i and channel j.
        [name] ([shape]) : [type]
            [desc]

        Returns
        -------
        [type]
            [desc]

        """
        batch_size, timesteps, channels = received_symbols.shape
        _channels, window = self.generator.shape
        assert _channels == channels
        if input_symbols is None:
            input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(timesteps)]
        else:
            assert len(input_symbols) == timesteps
        factor_groups, _ = nonrecursive_dependency_convolutional_factors(
            self.input_size,
            dependencies=self.generator,
            delay=self.delay,
            input_symbols=input_symbols,
            factor_symbol=received_factor_symbol,
        )
        evidence: Dict[str, NamedTensor] = {}

        # Compute evidence for factors of type 1. This comes from code table
        # Batch x Time x 2 x ... (Window times) x Channels
        channel_chi_values = channel.log_prob(
            received_symbols[:, :, None],  # B x T x 1 x C
            modulator.modulate(self.table)[None, None],  # 1 x 1 x 2^W x C
        ).reshape(batch_size, timesteps, *([2] * self.window), channels)
        for channel in range(channels):
            for i in range(timesteps):
                factor_i = f"{received_factor_symbol}_c{channel}_{i}"
                low = i - window + 1 + self.delay
                high = i + 1 + self.delay
                input_range = torch.arange(low, high)
                relevant_mask = (
                    (input_range >= 0)
                    & (input_range < timesteps)
                    & (self.generator[channel].bool())
                )
                relevant = input_range[relevant_mask]
                # Assumption is that indices outside our bounds are taken to be 0 input
                # For bits the generator doesn't consider, it doesn't matter which value we choose
                slice_tuple = (
                    [slice(None), i]  # Batch, Time
                    + [
                        slice(None) if is_rel else 0
                        for is_rel in relevant_mask.tolist()
                    ]
                    + [channel]  # Channel
                )
                evidence[factor_i] = NamedTensor(
                    channel_chi_values[slice_tuple],
                    dim_names=[input_symbols[j] for j in relevant.tolist()],
                    batch_dims=1,
                )

        assert {
            factor: evidence_data._dim_name_set
            for factor, evidence_data in evidence.items()
        } == factor_groups
        return evidence

    @staticmethod
    def _affine_encode(
        msg: torch.Tensor, generator: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        msg (Batch x Time) :  torch.Tensor
            Binary message to encode.

        Returns
        -------
        (Batch x Time x OutChannels) torch.Tensor
            Encoded binary message

        Actual logic for encoding, is not responsible for padding with initial state.
        Assumes that the message only has 1 channel (and thus channel dimension is reduced out).
        """
        return (
            (
                F.conv1d(
                    msg[:, None].to(torch.float),  # Batch x InChannels=1 x Time
                    generator[:, None, :].to(
                        torch.float
                    ),  # Outchannels x Inchannels=1 x Window
                    bias=bias.to(torch.float),
                )
                % 2
            )
            .transpose(1, 2)
            .to(msg.dtype)
        )
