from typing import List, Sequence

import torch
import torch.nn as nn

from ..utils import MaskedTensor


class StateTransitionGraph:
    def __init__(
        self, next_states: torch.LongTensor, previous_states: MaskedTensor
    ) -> None:
        self.next_states = next_states
        # MaskedTensor of  |States| x |PrevStates| (Ragged) x 2. Last dimension is pair previous state and transition input
        self.previous_states = previous_states

        self.validate()

    def validate(self):
        assert self.next_states.ndim == 2
        assert self.previous_states.tensor.ndim == 3
        assert self.previous_states.tensor.shape[0] == self.num_states

    @property
    def num_states(self):
        return self.next_states.shape[0]

    @property
    def num_inputs(self):
        return self.next_states.shape[1]

    @staticmethod
    def from_next_states(
        next_states: torch.LongTensor, device=None
    ) -> "StateTransitionGraph":
        if device is None:
            device = next_states.device
        num_states = next_states.shape[0]
        num_inputs = next_states.shape[1]
        previous_states_accum: List[List[List[int]]] = [[] for _ in range(num_states)]
        previous_states_mask: List[List[List[bool]]] = [[] for _ in range(num_states)]
        for state in range(num_states):
            for input_sym in range(num_inputs):
                next_state = next_states[state, input_sym]
                previous_states_accum[next_state].append([state, input_sym])
                previous_states_mask[next_state].append([False, False])
        # Add the masked entries
        most_prev_states = max(len(ps) for ps in previous_states_accum)
        for state in range(num_states):
            cur_prev_states = len(previous_states_accum[state])
            previous_states_accum[state] += [
                [-1, -1] for _ in range(most_prev_states - cur_prev_states)
            ]
            previous_states_mask[state] += [
                [True, True] for _ in range(most_prev_states - cur_prev_states)
            ]

        previous_states_tensor = torch.tensor(
            previous_states_accum, device=device, dtype=torch.long
        )
        previous_states_tensor_mask = torch.tensor(
            previous_states_mask, device=device, dtype=torch.bool
        )

        previous_states = MaskedTensor(
            previous_states_tensor, previous_states_tensor_mask
        )

        return StateTransitionGraph(
            next_states=next_states, previous_states=previous_states
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateTransitionGraph):
            return (
                torch.all(self.next_states == other.next_states)
                and (self.previous_states == other.previous_states).all()
            )
        else:
            return NotImplemented

    def unroll(self, num_steps: int):
        return UnrolledStateTransitionGraph(
            next_states=torch.tile(
                self.next_states,
                dims=[num_steps] + [1] * int(self.next_states.ndim),
            ),
            previous_states=MaskedTensor(
                tensor=torch.tile(
                    self.previous_states.tensor,
                    dims=[num_steps] + [1] * int(self.previous_states.tensor.ndim),
                ),
                mask=torch.tile(
                    self.previous_states.mask,
                    dims=[num_steps] + [1] * int(self.previous_states.mask.ndim),
                ),
            ),
        )


class UnrolledStateTransitionGraph:
    def __init__(
        self, next_states: torch.LongTensor, previous_states: MaskedTensor
    ) -> None:
        # Tensor of Time x |States| x |Inputs|
        self.next_states = next_states
        # MaskedTensor of Time x |States| x |PrevStates| x 2. Last dimension is pair previous state and transition input
        self.previous_states = previous_states

    def validate(self):
        assert self.next_states.ndim == 3
        assert self.previous_states.tensor.shape[1] == self.num_states
        assert self.previous_states.tensor.shape[0] == self.num_steps

    @property
    def num_states(self) -> int:
        return self.next_states.shape[1]

    @property
    def num_inputs(self) -> int:
        return self.next_states.shape[2]

    @property
    def num_steps(self) -> int:
        return self.next_states.shape[0]

    @staticmethod
    def from_next_states(
        next_states: torch.LongTensor,
    ) -> "UnrolledStateTransitionGraph":
        num_steps = next_states.shape[0]
        num_states = next_states.shape[1]
        num_inputs = next_states.shape[2]
        previous_states_accum: List[List[List[List[int]]]] = [
            [[] for _ in range(num_states)] for _ in range(num_steps)
        ]
        previous_states_mask: List[List[List[int]]] = [
            [[] for _ in range(num_states)] for _ in range(num_steps)
        ]
        for step in range(num_steps):
            for state in range(num_states):
                for input_sym in range(num_inputs):
                    next_state = next_states[step, state, input_sym]
                    previous_states_accum[step][next_state].append([state, input_sym])
                    previous_states_mask[step][next_state].append([False, False])
        # Add the masked entries
        most_prev_states = max(
            len(ps) for ps_step in previous_states_accum for ps in ps_step
        )
        for step in range(num_steps):
            for state in range(num_states):
                cur_prev_states = len(previous_states_accum[next_state])
                previous_states_accum[step][next_state] += [
                    [-1, -1] for _ in range(most_prev_states - cur_prev_states)
                ]
                previous_states_mask[step][next_state] += [
                    [True, True] for _ in range(most_prev_states - cur_prev_states)
                ]
        previous_states_tensor = torch.LongTensor(previous_states_accum)
        previous_states_tensor_mask = torch.BoolTensor(previous_states_mask)

        previous_states = MaskedTensor(
            previous_states_tensor, previous_states_tensor_mask
        )
        return UnrolledStateTransitionGraph(
            next_states=next_states, previous_states=previous_states
        )

    @staticmethod
    def concat_unrolled_state_transitions(
        state_transitions: Sequence["UnrolledStateTransitionGraph"],
    ) -> "UnrolledStateTransitionGraph":
        next_states = torch.concat([st.next_states for st in state_transitions], dim=0)

        return UnrolledStateTransitionGraph.from_next_states(next_states=next_states)

    def concat(
        self, other: "UnrolledStateTransitionGraph"
    ) -> "UnrolledStateTransitionGraph":
        return self.concat_unrolled_state_transitions([self, other])

    def ensure_length(self, msg_length: int) -> "UnrolledStateTransitionGraph":
        assert self.next_states.shape[0] == msg_length
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnrolledStateTransitionGraph):
            return (
                torch.all(self.next_states == other.next_states)
                and (self.previous_states == other.previous_states).all()
            )
        else:
            return NotImplemented


class Trellis(nn.Module):
    def __init__(
        self,
        state_transitions: StateTransitionGraph,
        output_table: torch.Tensor,
    ) -> None:
        super(Trellis, self).__init__()
        self.state_transitions = state_transitions
        self.output_table = output_table
        self.validate()

    def validate(self):
        assert self.output_table.ndim == 3
        assert self.output_table.shape[0] == self.num_states
        assert self.output_table.shape[1] == self.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.output_table.shape[2]

    @property
    def num_inputs(self) -> int:
        return self.state_transitions.num_inputs

    @property
    def num_states(self) -> int:
        return self.state_transitions.num_states

    @property
    def next_states(self) -> int:
        return self.state_transitions.next_states

    def concat_outputs(
        self, trellis2: "Trellis", check_compatibility=True
    ) -> "Trellis":
        if (not check_compatibility) or self.check_state_table_compatibility(trellis2):
            return Trellis(
                state_transitions=self.state_transitions,
                output_table=torch.concat(
                    [self.output_table, trellis2.output_table], dim=2
                ),
            )
        else:
            raise ValueError("Input trellis is not compatible with source trellis")

    def with_systematic(self) -> "Trellis":
        id_output_table = torch.zeros(
            (self.num_states, self.num_inputs, 1),
            dtype=self.output_table.dtype,
        )
        id_output_table[:, 1] = 1
        return Trellis(self.state_transitions, id_output_table).concat_outputs(self)

    def check_state_table_compatibility(self, trellis2: "Trellis") -> bool:
        return self.state_transitions == trellis2.state_transitions

    def is_same(self, other: "Trellis") -> bool:
        if isinstance(other, Trellis):
            return self.state_transitions == other.state_transitions and torch.all(
                self.output_table == other.output_table
            )
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return Trellis(self.state_transitions, self.output_table * other)

    def __add__(self, other):
        return Trellis(self.state_transitions, self.output_table + other)

    def __sub__(self, other):
        return Trellis(self.state_transitions, self.output_table - other)

    def __truediv__(self, other):
        return Trellis(self.state_transitions, self.output_table / other)

    def unroll(self, num_steps: int) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions.unroll(num_steps),
            output_tables=torch.tile(
                self.output_table[None],
                [num_steps] + [1] * int(self.output_table.ndim),
            ),
        )


class UnrolledTrellis(nn.Module):
    def __init__(
        self,
        state_transitions: UnrolledStateTransitionGraph,
        output_tables: torch.Tensor,
    ) -> None:
        super(UnrolledTrellis, self).__init__()
        self.state_transitions = state_transitions
        # Time x States x Inputs x Outputs
        self.output_tables = output_tables
        self.validate()

    def validate(self):
        assert self.output_tables.ndim == 4
        assert self.output_tables.shape[0] == self.num_steps
        assert self.output_tables.shape[1] == self.num_states
        assert self.output_tables.shape[2] == self.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.output_tables.shape[3]

    @property
    def num_inputs(self) -> int:
        return self.state_transitions.num_inputs

    @property
    def num_states(self) -> int:
        return self.state_transitions.num_states

    @property
    def next_states(self) -> int:
        return self.state_transitions.next_states

    @property
    def num_steps(self) -> int:
        return self.state_transitions.num_steps

    def get_output_channels(self, channels: List[int]):
        return UnrolledTrellis(
            self.state_transitions, self.output_tables[..., channels]
        )

    @staticmethod
    def concat_unrolled_trellises(
        trellises: Sequence["UnrolledTrellis"],
    ):
        return UnrolledTrellis(
            state_transitions=UnrolledStateTransitionGraph.concat_unrolled_state_transitions(
                [trellis.state_transitions for trellis in trellises]
            ),
            output_tables=torch.concat(
                [trellis.output_tables for trellis in trellises], dim=0
            ),
        )

    def concat_time(self, other: "UnrolledTrellis"):
        return self.concat_unrolled_trellises([self, other])

    def check_state_table_compatibility(self, trellis2: "UnrolledTrellis") -> bool:
        return self.state_transitions == trellis2.state_transitions

    def concat_outputs(
        self, other: "UnrolledTrellis", check_compatibility=True
    ) -> "UnrolledTrellis":
        if (not check_compatibility) or self.check_state_table_compatibility(other):
            return UnrolledTrellis(
                state_transitions=self.state_transitions,
                output_tables=torch.concat(
                    [self.output_tables, other.output_tables], dim=3
                ),
            )
        else:
            raise ValueError("Input trellis is not compatible with source trellis")

    def with_systematic(self, device=None) -> "UnrolledTrellis":
        id_output_table = torch.zeros(
            (self.num_steps, self.num_states, self.num_inputs, 1),
            dtype=self.output_tables.dtype,
            device=device,
        )
        id_output_table[:, :, 1] = 1
        return UnrolledTrellis(self.state_transitions, id_output_table).concat_outputs(
            self
        )

    def ensure_length(self, msg_length: int) -> "UnrolledTrellis":
        assert self.output_tables.shape[0] == msg_length
        self.state_transitions.ensure_length(msg_length)
        return self

    def __mul__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables * other,
        )

    def __add__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables + other,
        )

    def __sub__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables - other,
        )

    def __truediv__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables / other,
        )
