import torch
import numpy as np
import math

from ..constants import INPUT_SYMBOL
from ..utils import check_int, dynamic_get, DeviceManager, DEFAULT_DEVICE_MANAGER

from .graph import InferenceGraph


def general_convolutional_factors(
    length: int,
    state_size: int,
    input_symbols: list[str] = None,
    state_symbol: str = "s",
    received_factor_symbol: str = "y",
    transition_factor_symbol: str = "st",
):
    """Create the factor groups for a general (recursive) convolutional code.

    Note
    ----
    We have 2 different types of factors
    1. Factors associated with the corrupted received value. For received value
        $y_i$ ($i \\in \\{0, \\dots, k-1\\}$, for $k$ timesteps), this factor is
        $P(y_i|s_i, u_i)$. If $i=0$, we omit $s_0$ as it is always $0$ in the
        absence of delay.
    2. Factors associated with the transition between states. These factors are
        of the form $P(s_{i+1} | u_i, s_i)$ ($i \\in \\{0, \\dots, k-2\\}$, for
        $k$ timesteps). If $i=0$, we omit $s_0$ as it is always $0$ in the
        absence of delay. We also omitted $i=k-1$ because that state is not relevant
        to Y.
    With these factors we get,
    $$P(Y,S,U) = P(U) [\\prod_{i=0}^{k-1} P(y_i | u_i, s_i)] [\\prod_{i=0}^{k-2} P(s_{i+1} | u_i, s_i)]$$
    $P(U)$ is a constant, so this will cancel out when we compute our likelihood ratios.

    Parameters
    ----------
    [name] ([shape]) : [type]
        [desc]

    Returns
    -------
    Dict[str, Set[str]]
        A mapping of factor group names to variable names.
    Dict[str, int]
        A mapping of variable names to their log (base 2) state size.
    [type]
        [desc]


    """

    factor_groups = {}
    if input_symbols is None:
        input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(length)]
    else:
        assert length == len(input_symbols)

    # Factors of type 1 (See Note)
    for i, input_node in enumerate(input_symbols):
        state_i = f"{state_symbol}_{i}"
        factor = set([input_node] + ([state_i] if i > 0 else []))
        factor_groups[f"{received_factor_symbol}_{i}"] = factor

    # Factors of type 2 (See Note)
    for i, input_node in enumerate(input_symbols[:-1]):
        state_i = f"{state_symbol}_{i}"
        state_i_p_1 = f"{state_symbol}_{i+1}"
        factor = set([input_node] + ([state_i] if i > 0 else []) + [state_i_p_1])
        factor_groups[f"{transition_factor_symbol}_{i}"] = factor

    variables = set.union(*factor_groups.values())
    variable_state_sizes = {
        name: (1 if name[0] == "u" else state_size) for name in variables
    }

    return factor_groups, variable_state_sizes


def general_convolutional_code(length: int, state_size: int):
    factor_groups, variable_state_sizes = general_convolutional_factors(
        length, state_size=state_size
    )
    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def general_turbo_graph(interleaver_p: torch.LongTensor, state_size):
    assert interleaver_p.ndim == 1
    length = interleaver_p.shape[0]
    ni_factor_groups, ni_variable_state_sizes = general_convolutional_factors(
        length, state_size=state_size
    )

    # Could potentially lead to odd behavior using a torch tensor for formatting.
    interleaved_inputs = [f"u_{interleaver_p[i]}" for i in range(length)]
    i_factor_groups, i_variable_state_sizes = general_convolutional_factors(
        length,
        state_size=state_size,
        input_symbols=interleaved_inputs,
        state_symbol="sp",
        received_factor_symbol="yp",
        transition_factor_symbol="stp",
    )

    factor_groups = {**ni_factor_groups, **i_factor_groups}
    variable_state_sizes = {**ni_variable_state_sizes, **i_variable_state_sizes}

    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def nonrecursive_convolutional_factors(
    length: int,
    window: int,
    delay: int,
    input_symbols: list[str] = None,
    factor_symbol: str = "y",
):
    factor_groups = {}
    if input_symbols is None:
        input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(length)]
    else:
        assert length == len(input_symbols)

    for i in range(length):
        factor = set(input_symbols[max(0, i - window + 1 + delay) : (i + 1 + delay)])
        factor_groups[f"{factor_symbol}_{i}"] = factor

    variables = set.union(*factor_groups.values())
    variable_state_sizes = {name: 1 for name in variables}

    return factor_groups, variable_state_sizes


def nonrecursive_convolutional_code(length: int, window: int, delay: int = 0):
    factor_groups, variable_state_sizes = nonrecursive_convolutional_factors(
        length, window=window, delay=delay
    )
    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def nonrecursive_turbo_graph(
    interleaver_p: torch.LongTensor, window: int, delay: int = 0
):
    assert interleaver_p.ndim == 1
    length = interleaver_p.shape[0]
    ni_factor_groups, ni_variable_state_sizes = nonrecursive_convolutional_factors(
        length, window=window, delay=delay
    )

    # Could potentially lead to odd behavior using a torch tensor for formatting.
    interleaved_inputs = [f"{INPUT_SYMBOL}_{interleaver_p[i]}" for i in range(length)]
    i_factor_groups, i_variable_state_sizes = nonrecursive_convolutional_factors(
        length,
        window=window,
        delay=delay,
        input_symbols=interleaved_inputs,
        factor_symbol="yp",
    )

    factor_groups = {**ni_factor_groups, **i_factor_groups}
    variable_state_sizes = {**ni_variable_state_sizes, **i_variable_state_sizes}

    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def recursive_dependency_convolutional_factors(
    length: int,
    nonrecursive_dependencies: torch.LongTensor,
    recursive_dependencies: torch.LongTensor,
    delay: int = 0,
    input_symbols: list[str] = None,
    received_factor_symbol: str = "y",
    feedback_symbol: str = "f",
    feedback_factor_symbol: str = "ff",
    systematic=False,
):
    """Create the factor groups for a recursive convolutional code with known dependencies.
    Equivalent to `general_convolutional_factors` with the dependency tensors all 1s.

    Note
    ----
    We have 2 different types of factors. Let $w$ be the window.
    1. Factors associated with the corrupted received value. For received value
        $y_i$ ($i \\in \\{0, \\dots, k-1\\}$, for $k$ timesteps), this factor is
        $P(y_i|f_{i-w+1},...f_i)$ with indices omitted if we have no dependence.
        For indices < 0, they are omitted as they are 0 w/o delay. Factors w/
        delay are not implemented.
    2. Factors associated with application of the feedback. These factors are
        of the form $P(f_{i} | u_i, f_{i-w+1},..., f_{i-1})$ ($i \\in
        \\{0, \\dots, k-2\\}$, for $k$ timesteps) with indices omitted if we
        have no dependence. For indices < 0, they are omitted as they are 0 w/o delay. Factors w/
        delay are not implemented.
    With these factors we get,
    $$P(Y,F,U) = P(U) [\\prod_{i=0}^{k-1} P(y_i | f_{i-w+1},...f_i)] [\\prod_{i=0}^{k-2} P(f_{i} | u_i, f_{i-w+1},..., f_{i-1})]$$
    $P(U)$ is a constant, so this will cancel out when we compute our likelihood ratios.

    Parameters
    ----------
    [name] ([shape]) : [type]
        [desc]

    Returns
    -------
    Dict[str, Set[str]]
        A mapping of factor group names to variable names.
    Dict[str, int]
        A mapping of variable names to their log (base 2) state size.
    [type]
        [desc]


    """
    if input_symbols is None:
        input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(length)]
    else:
        assert length == len(input_symbols)

    channels, window = nonrecursive_dependencies.shape
    assert recursive_dependencies.ndim == 1
    assert recursive_dependencies.shape[0] == window

    if delay != 0:
        raise NotImplementedError("Nonzero delay is not implemented.")

    factor_groups = {}

    # Factor Type 1
    for channel in range(channels):
        for i in range(length):
            if systematic and channel == 0:
                factor_groups[f"{received_factor_symbol}_c{channel}_{i}"] = {
                    input_symbols[i]
                }
            else:
                relevant = torch.arange(i - window + 1, i + 1)[
                    nonrecursive_dependencies[channel].bool()
                ]
                relevant = relevant[(relevant >= 0) & (relevant < length)].tolist()

                factor = {f"{feedback_symbol}_{j}" for j in relevant}
                factor_groups[f"{received_factor_symbol}_c{channel}_{i}"] = factor

    # Factor Type 2
    for i in range(length):
        relevant = torch.arange(i - window + 1, i + 1)[recursive_dependencies.bool()]
        relevant = relevant[(relevant >= 0) & (relevant < length)].tolist()
        # Feedback should always depend on the input
        assert relevant[-1] == i

        factor = (
            {f"{feedback_symbol}_{j}" for j in relevant[:-1]}
            | {input_symbols[i]}
            | {f"{feedback_symbol}_{i}"}
        )
        factor_groups[f"{feedback_factor_symbol}_{i}"] = factor

    variables = set.union(*factor_groups.values())
    variable_state_sizes = {name: 1 for name in variables}

    return factor_groups, variable_state_sizes


def recursive_dependency_convolutional_code(
    length: int,
    nonrecursive_dependencies: torch.LongTensor,
    recursive_dependencies: torch.LongTensor,
    delay: int = 0,
    systematic=False,
):
    # We have a factor type for each channel
    factor_groups, variable_state_sizes = recursive_dependency_convolutional_factors(
        length,
        nonrecursive_dependencies=nonrecursive_dependencies,
        recursive_dependencies=recursive_dependencies,
        delay=delay,
        systematic=systematic,
    )
    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def recursive_dependency_turbo_graph(
    interleaver_p: torch.LongTensor,
    nonrecursive_dependencies_noni: torch.LongTensor,
    recursive_dependencies_noni: torch.LongTensor,
    nonrecursive_dependencies_i: torch.LongTensor,
    recursive_dependencies_i: torch.LongTensor,
    delay: int = 0,
    systematic=False,
):
    assert interleaver_p.ndim == 1
    length = interleaver_p.shape[0]
    (
        ni_factor_groups,
        ni_variable_state_sizes,
    ) = recursive_dependency_convolutional_factors(
        length,
        nonrecursive_dependencies=nonrecursive_dependencies_noni,
        recursive_dependencies=recursive_dependencies_noni,
        delay=delay,
        systematic=systematic,
    )

    # Could potentially lead to odd behavior using a torch tensor for formatting.
    interleaved_inputs = [f"{INPUT_SYMBOL}_{interleaver_p[i]}" for i in range(length)]
    (
        i_factor_groups,
        i_variable_state_sizes,
    ) = recursive_dependency_convolutional_factors(
        length,
        nonrecursive_dependencies=nonrecursive_dependencies_i,
        recursive_dependencies=recursive_dependencies_i,
        delay=delay,
        input_symbols=interleaved_inputs,
        received_factor_symbol="yp",
        feedback_factor_symbol="ffp",
        feedback_symbol="fp",
        systematic=systematic,
    )

    factor_groups = {**ni_factor_groups, **i_factor_groups}
    variable_state_sizes = {**ni_variable_state_sizes, **i_variable_state_sizes}

    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def nonrecursive_dependency_convolutional_factors(
    length: int,
    dependencies: torch.LongTensor,
    delay: int = 0,
    input_symbols: list[str] = None,
    factor_symbol: str = "y",
):
    if input_symbols is None:
        input_symbols = [f"{INPUT_SYMBOL}_{i}" for i in range(length)]
    else:
        assert length == len(input_symbols)

    # We have a factor type for each channel
    channels, window = dependencies.shape
    factor_groups = {}
    for channel in range(channels):
        for i in range(length):
            relevant = torch.arange(i - window + 1 + delay, i + 1 + delay)[
                dependencies[channel].bool()
            ]
            relevant = relevant[(relevant >= 0) & (relevant < length)].tolist()

            factor = set(input_symbols[j] for j in relevant)
            factor_groups[f"{factor_symbol}_c{channel}_{i}"] = factor

    variables = set.union(*factor_groups.values())
    variable_state_sizes = {name: 1 for name in variables}

    return factor_groups, variable_state_sizes


def nonrecursive_dependency_convolutional_code(
    length: int, dependencies: torch.LongTensor, delay: int = 0
):
    # We have a factor type for each channel
    factor_groups, variable_state_sizes = nonrecursive_dependency_convolutional_factors(
        length, dependencies=dependencies, delay=delay
    )
    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)


def infer_depencies(
    table: torch.Tensor,
    tol=None,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    num_inputs, channels = table.shape
    window = check_int(math.log2(num_inputs))
    table = table.reshape(*([2] * window), channels)
    # We have no dependence on bit if changing the bit does
    # not change output. This occurs iff the two entries in
    # the bit's respective dimension are equal.
    def check_equality(t1: torch.Tensor, t2: torch.Tensor) -> bool:
        if tol is None:
            return torch.all(t1 == t2)
        else:
            return torch.allclose(t1, t2, rtol=0, atol=tol)

    # 2 x ...(Window) x Channels
    dim_deps = torch.tensor(
        [
            [
                int(
                    not check_equality(
                        dynamic_get(table, dim=[i, -1], index=[0, c]),
                        dynamic_get(table, dim=[i, -1], index=[1, c]),
                    )
                )
                for i in range(window)
            ]
            for c in range(channels)
        ],
        dtype=torch.int8,
        device=device_manager.device,
    )
    assert dim_deps.shape == (channels, window)

    return dim_deps


def nonrecursive_dependency_turbo_graph(
    interleaver_p: torch.LongTensor,
    nonrecursive_dependencies_noni: torch.LongTensor,
    nonrecursive_dependencies_i: torch.LongTensor,
    delay: int = 0,
):
    assert interleaver_p.ndim == 1
    length = interleaver_p.shape[0]
    (
        ni_factor_groups,
        ni_variable_state_sizes,
    ) = nonrecursive_dependency_convolutional_factors(
        length, dependencies=nonrecursive_dependencies_noni, delay=delay
    )

    # Could potentially lead to odd behavior using a torch tensor for formatting.
    interleaved_inputs = [f"{INPUT_SYMBOL}_{interleaver_p[i]}" for i in range(length)]
    (
        i_factor_groups,
        i_variable_state_sizes,
    ) = nonrecursive_dependency_convolutional_factors(
        length,
        dependencies=nonrecursive_dependencies_i,
        delay=delay,
        input_symbols=interleaved_inputs,
        factor_symbol="yp",
    )

    factor_groups = {**ni_factor_groups, **i_factor_groups}
    variable_state_sizes = {**ni_variable_state_sizes, **i_variable_state_sizes}

    return InferenceGraph.from_factor_groups(factor_groups, variable_state_sizes)
