from typing import Callable
import numpy as np

import torch
import torch.nn.functional as F

from ..utils import (
    MaskedTensor,
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    get_reducer,
    enumerate_binary_inputs,
)
from ..encoders import TrellisEncoder
from ..modulation import Modulator
from ..channels import NoisyChannel

from .decoder import SoftDecoder


def compute_bitwise_delay_llr_validate_tensors(
    L_init_ext: torch.FloatTensor,
    L_init_int: torch.FloatTensor,
    delay: int,
    batch_size: int,
):
    assert L_init_ext.shape == torch.Size([batch_size, 2**delay])
    assert L_init_int.shape == torch.Size([batch_size, 2**delay])


def compute_bitwise_delay_llr(
    L_init_ext: torch.FloatTensor,
    L_init_int: torch.FloatTensor,
    delay: int,
    use_max=False,
    validate_tensors=False,
    device=None,
):
    reducer = get_reducer(use_max=use_max)

    batch_size = L_init_ext.shape[0]
    if validate_tensors:
        compute_bitwise_delay_llr_validate_tensors(
            L_init_ext, L_init_int, delay, batch_size
        )
    L_init = L_init_int + L_init_ext  # B x 2^d

    # 2^d x d
    delay_inputs = enumerate_binary_inputs(delay, dtype=L_init_int.dtype, device=device)

    # B x 2^d x 1 + 1 x 2^d x d -> B x 2^d x d ->(reducer) B x d
    L_init_posterior = reducer(
        F.log_softmax(L_init[..., None], dim=1) + torch.log(delay_inputs)[None], dim=1
    )

    return L_init_posterior


def backward_recursion_step(
    next_B: torch.FloatTensor,
    gamma_value_slice: torch.FloatTensor,
    next_states_slice: torch.LongTensor,
    reducer: Callable[..., torch.Tensor],
):
    """
    Parameters
    ----------
    - next_B (Batch x States): excluding batch, for timestep k, next_B[i] = log p(Y[k+2:K-1] | s[k+2] = i)
    - gamma_value_slice (Batch x States x 2)
    - next_states_slice (States x 2)
    """
    # B x |S| x 2 + B x |S| x 2 -> B x |S|
    beta = reducer(
        gamma_value_slice + next_B[:, next_states_slice],
        dim=2,
    )
    return beta - reducer(beta, dim=1, keepdims=True)


def backward_recursion(
    gamma_values: torch.FloatTensor,
    next_states: torch.LongTensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., torch.Tensor],
    device=None,
):
    """
    Parameters
    ----------
    - gamma_values (Batch x Time x States x 2): excluding batch, gamma_values[k, i, b] = log p(Y[k], s[k+1] = next_states[k, i, b] | s[k] = i)
    - next_states (Time x States x 2): next_states[k, i, b] is the state we transition to if we are in state i at timestep k and receive b.
    """
    # (excluding batch)
    # B[k, i] = log p(Y[k+1:K-1] | s[k+1] = i)
    #         = log( Sum over t[ p(Y[k+2:K-1] | s[k+2] = next_states[k+1, i, t]) * p(Y[k+1], s[k+2] = next_states[k+1, i, t] | s[k+1] = i) ] )
    #         = logsumexp over t[ B[k+1, next_states[k+1, i, t]] + gamma_values[k+1, i, t] ]
    B = torch.zeros((batch_size, K, S), device=device, dtype=gamma_values.dtype)
    for k in range(K - 2, -1, -1):
        # B x S
        B[:, k] = backward_recursion_step(
            B[:, k + 1], gamma_values[:, k + 1], next_states[k + 1], reducer
        )
    return B


def forward_recursion(
    gamma_values: torch.FloatTensor,
    previous_states: MaskedTensor,
    forward_init: torch.FloatTensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., torch.Tensor],
    device=None,
):
    """
    Parameters
    ----------
    - gamma_values (Batch x Time x States x 2): excluding batch, gamma_values[k, i, b] = log p(Y[k], s[k+1] = next_states[k, i, b] | s[k] = i).
    - previous_states (Time x States x PrevStates(Ragged) x 2): previous_states[k, i, j] is a pair of
        - the jth previous state of state i at timestep k.
        - the received bit that transfers the previous state to state i at timestep k.
    - forward_init (Batch x States): excluding batch, forward_init[i] = log p(s_0=i)
    """

    # previous_gamma_values (B x K x |S| x |Prev|): previous_gamma_values[:, k, i, t] is the gamma for received k
    # from prev_states[i, t] to state i. |Prev| is ragged.
    # excluding batch, previous_gamma_values[k, i, t] = gamma_values[k, previous_states[k, i, t, 0], previous_states[k, i, t, 1]], t is ragged
    # B x K x S x I ->(transpose) K x S x I x B ->(gather_nd) K x S x P (ragged) x B
    previous_gamma_values_tensor = gamma_values[
        :,
        torch.arange(K, device=device)[:, None, None],
        previous_states.tensor[..., 0],
        previous_states.tensor[..., 1],
    ]
    previous_gamma_values = MaskedTensor(
        tensor=previous_gamma_values_tensor,
        mask=torch.tile(previous_states.mask[None, ..., 0], (batch_size, 1, 1, 1)),
        fill_value=-np.inf,  # Because log(0) = -inf
    )

    A = torch.zeros((batch_size, K, S), device=device, dtype=gamma_values.dtype)
    # forward_init is B x S
    A[:, 0] = forward_init
    for k in range(1, K):
        # B x S x P (ragged)
        previous_alphas = MaskedTensor(
            A[:, k - 1, previous_states.tensor[k - 1, :, :, 0]],
            previous_states.mask[None, k - 1, :, :, 0].tile([batch_size, 1, 1]),
            fill_value=-np.inf,
        )
        # B x S x P (ragged)
        previous_gammas = previous_gamma_values[:, k - 1]

        # B x S
        alpha = reducer(previous_gammas.tensor + previous_alphas.tensor, dim=2)

        A[:, k] = alpha - reducer(alpha, dim=1, keepdims=True)
    return A  # B x K x |S|


def map_decode_no_delay_validate_tensors(
    L_int: torch.FloatTensor,
    chi_values: torch.FloatTensor,
    next_states: torch.LongTensor,
    previous_states: MaskedTensor,
    forward_init: torch.FloatTensor,
    batch_size: int,
    K: int,
    S: int,
):
    assert L_int.shape == torch.Size([batch_size, K])
    assert chi_values.shape == torch.Size([batch_size, K, S, 2])
    assert next_states.shape == torch.Size([K, S, 2])
    assert previous_states.tensor.shape[0] == K
    assert previous_states.tensor.shape[1] == S
    assert previous_states.tensor.shape[3] == 2
    assert torch.all(previous_states.mask[..., 0] == previous_states.mask[..., 1])
    assert forward_init.shape == torch.Size([batch_size, S])


def map_decode_no_delay(
    L_int: torch.FloatTensor,
    chi_values: torch.FloatTensor,
    next_states: torch.LongTensor,
    previous_states: MaskedTensor,
    forward_init: torch.FloatTensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., torch.Tensor],
    validate_tensors=False,
    device=None,
):
    """
    Note
    ----
    Below we use the following variables:
    - Y (Batch x Time x Channels) : Received corrupted sequence

    Parameters
    ----------
    - L_int (Batch x Time): the logit intrinsic information (prior) on the value of each bit
    - chi_values (Batch x Time x States x 2): excluding batch, chi_values[k, i, b] = log p(Y[k] | s[k+1] = next_states[k, i, b], s[k] = i)
    - next_states (Time x States x 2): next_states[k, i, b] is the state we transition to if we are in state i at timestep k and receive b.
    - previous_states (Time x States x PrevStates(Ragged) x 2): previous_states[k, i, j] is a pair of
        - the jth previous state of state i at timestep k.
        - the received bit that transfers the previous state to state i at timestep k.
    - forward_init (Batch x States): excluding batch, forward_init[i] = log p(s_0=i)
    """
    if validate_tensors:
        map_decode_no_delay_validate_tensors(
            L_int,
            chi_values,
            next_states,
            previous_states,
            forward_init,
            batch_size,
            K,
            S,
        )

    # B x K x 2
    transition_prob_values = torch.stack(
        [F.logsigmoid(-L_int), F.logsigmoid(L_int)], dim=2
    )

    # Compute ln(Gamma) values
    # gamma_values[k, i, t] = log p(Y[k], s[k+1] = next_states[k, i, t] | s[k] = i) = log p(s[k+1] = next_states[k, i, t] | s[k] = i) + chi_values[k, i, t]
    # B x K x |S| x 2
    gamma_values = chi_values + transition_prob_values[:, :, None, :]

    # Compute ln(B)
    # B x K x |S|
    B = backward_recursion(
        gamma_values,
        next_states,
        batch_size,
        K,
        S,
        reducer,
        device=device,
    )

    # B x K x |S|
    A = forward_recursion(
        gamma_values,
        previous_states,
        forward_init,
        batch_size,
        K,
        S,
        reducer,
        device=device,
    )

    # Compute L_ext
    # L = log Sum over i[ p(Y[0:K-1], s_k=i, s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], Y[k], Y[k+1:K-1], s_k=i, s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k], s_k+1=next_states[k, i, 1] | s_k=i) * P(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(s_k+1=next_states[k, i, 1] | s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(x_k=1) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log( p(x_k=1) / p(x_k=0) ) * log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = L_int + logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[k, i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[k, i, 0]] ]
    # = L_int + L_ext
    # -> L_ext = logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[k, i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[k, i, 0]] ]

    # Should be shape B x K x |S| x 2
    B_next_states = B[:, torch.arange(K, device=device)[:, None, None], next_states]

    # This L_ext includes the padded bits - B x K
    L_ext = reducer(
        A + chi_values[:, :, :, 1] + B_next_states[:, :, :, 1], dim=2
    ) - reducer(A + chi_values[:, :, :, 0] + B_next_states[:, :, :, 0], dim=2)
    return L_ext, A, B, gamma_values


def map_decode_validate_tensors(
    L_int: torch.FloatTensor,
    chi_values: torch.FloatTensor,
    next_states: torch.LongTensor,
    previous_states: MaskedTensor,
    init_conditional: torch.FloatTensor,
    L_init_int: torch.FloatTensor,
    delay: int,
    batch_size: int,
    K: int,
    S: int,
):
    assert L_int.shape == torch.Size([batch_size, K])
    assert chi_values.shape == torch.Size([batch_size, K, S, 2])
    assert next_states.shape == torch.Size([K, S, 2])
    assert previous_states.tensor.shape[0] == K
    assert previous_states.tensor.shape[1] == S
    assert previous_states.tensor.shape[3] == 2
    assert torch.all(previous_states.mask[..., 0] == previous_states.mask[..., 1])
    assert init_conditional.shape == torch.Size([S, 2**delay])
    assert L_init_int.shape == torch.Size([S, 2**delay])


def map_decode(
    L_int: torch.FloatTensor,
    chi_values: torch.FloatTensor,
    next_states: torch.LongTensor,
    previous_states: MaskedTensor,
    init_conditional: torch.FloatTensor,
    L_init_int: torch.FloatTensor,
    delay: int,
    use_max: bool = False,
    validate_tensors=False,
    device=None,
) -> torch.Tensor:
    """
    Note
    ----
    Below we use the following variables:
    - Y (Batch x Time x Channels) : Received corrupted sequence. Usually used in
        probabilities as a random variable (excluding batch).
    - u (Batch x Time) : Source binary sequence. Usually used in
        probabilities as a random variable (excluding batch).

    Parameters
    -----------
    - L_int (Batch x Time-delay): the logit intrinsic information (prior) on the value of each bit
    - chi_values (Batch x Time x States x 2): excluding batch, chi_values[k, i, b] = log p(Y[k] | s[k+1] = next_states[k, i, b], s[k] = i)
    - next_states (Time x States x 2): next_states[k, i, b] is the state we transition to if we are in state i at timestep k and receive b.
    - previous_states (Time x States x PrevStates(Ragged) x 2): previous_states[k, i, j] is a pair of
        - the jth previous state of state i at timestep k.
        - the received bit that transfers the previous state to state i at timestep k.
    - init_conditional (States x DelayInputs (2^delay)): log p(s0 = s | u[0:d-1]). d := delay
    - L_init_int (Batch x DelayInputs(2^delay)): p(u[0:d-1]). d := delay.
    - delay: int - the number of delay timesteps.
    - use_max: bool - if True, then use the hardmax approximation of logexpsum
    """
    reducer = get_reducer(use_max=use_max)

    batch_size = chi_values.shape[0]
    K = chi_values.shape[1]
    S = next_states.shape[1]
    if validate_tensors:
        map_decode_validate_tensors(
            L_int,
            chi_values,
            next_states,
            previous_states,
            init_conditional,
            L_init_int,
            delay,
            batch_size,
            K,
            S,
        )

    # Softmax over DelayInputs axis
    logprob_prior_int = F.log_softmax(L_init_int, dim=1)
    # Broadcast
    init_conditional = init_conditional[None]  # S x 2^d -> 1 x S x 2^d
    logprob_prior_int = logprob_prior_int[:, None]  # B x 2^d -> B x 1 x 2^d
    # 1 x S x 2^d + B x 1 x 2^d -> B x S x 2^d
    forward_init = reducer(init_conditional + logprob_prior_int, dim=2)  # B x S

    # First we need to add in extra entries corresponding to 0 bits padded on because of delay
    L_int_no_delay = F.pad(L_int, pad=[0, delay], value=-np.inf)

    L_ext_no_delay, A, B, gamma_values = map_decode_no_delay(
        L_int=L_int_no_delay,
        chi_values=chi_values,
        next_states=next_states,
        previous_states=previous_states,
        forward_init=forward_init,
        batch_size=batch_size,
        K=K,
        S=S,
        reducer=reducer,
        device=device,
    )

    # Removing the paded bits
    if delay > 0:
        L_ext = L_ext_no_delay[:, :-delay]
    else:
        L_ext = L_ext_no_delay

    # Now we compute the log posterior on the delay bits
    # This is (excluding batch) beta_minus1[i] = log p(Y[0:K] | s[0] = i)
    beta_minus1 = backward_recursion_step(
        next_B=B[:, 0],
        gamma_value_slice=gamma_values[:, 0],
        next_states_slice=next_states[0],
        reducer=reducer,
    )  # B x S
    # B x S x 1 + 1 x S x 2^d -> B x S x 2^d ->(reducer) B x 2^d
    # This is log p(Y[0:k]|u[0:d-1])
    L_init_ext = F.log_softmax(
        reducer(beta_minus1[..., None] + init_conditional, dim=1), dim=1
    )

    return L_ext, L_init_ext


class BCJRDecoder(SoftDecoder):
    """A decoder for a TrellisEncoder that uses the BCJR algorithm.

    Attributes
    ----------
    use_max : bool
        A flag that determines whether or not to use the hardmax approximation
        of logsumexp.

    """

    def __init__(
        self,
        encoder: TrellisEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        use_max: bool = False,
        use_float16: bool = False,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super(BCJRDecoder, self).__init__(device_manager=device_manager)
        self.trellis_code = encoder
        self.modulator = modulator
        self.channel = channel
        self.use_max = use_max
        self.use_float16 = use_float16
        self.dtype = torch.float16 if use_float16 else torch.float32

        self.validate()

    def validate(self):
        pass

    @property
    def num_streams(self):
        return self.trellis_code.num_output_channels

    @property
    def num_output_channels(self):
        return 1

    @property
    def delay(self) -> int:
        return self.trellis_code.delay

    @property
    def source_data_len(self) -> int:
        return self.trellis_code.input_size

    def forward(
        self,
        received_symbols: torch.Tensor,
        L_int: torch.FloatTensor = None,
        L_init_int: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        received_symbols (Batch x Time x Channels) : torch.Tensor
            The received corrupted stream. A manifestation of the random variable Y.
            Usually a torch.FloatTensor, but may be other types as well.
        L_int (Batch x Time-delay) : torch.FloatTensor
            The logit intrinsic information (prior) on the value of each (non-delayed) bit.
        L_init_int (Batch x 2^self.delay) : torch.FloatTensor
            The logit intrinsic information (prior) on each delayed unreceived binary sequence.

        Returns
        -------
        (Batch x Time) torch.FloatTensor
            The posterior probability on each bit of the source sequence.

        """
        batch_size = received_symbols.shape[0]
        if L_int is None:
            L_int = torch.zeros(
                (batch_size, self.source_data_len - self.delay),
                device=self.device_manager.device,
                dtype=self.dtype,
            )
        if L_init_int is None:
            L_init_int = torch.zeros(
                (batch_size, 2**self.delay),
                device=self.device_manager.device,
                dtype=self.dtype,
            )

        # K x S x 2 x Channels
        output_tables = self.modulator.modulate(self.trellis_code.output_tables)
        # B x K x Channels (log_likelihood) K x S x 2 x Channels -(sum dim=-1)> B x K x S x 2
        chi_values = torch.sum(
            self.channel.log_prob(
                received_symbols[:, :, None, None],
                output_tables[None],
                dtype=self.dtype,
            ),
            dim=-1,
        )

        L_ext, L_init_ext = map_decode(
            next_states=self.trellis_code.trellises.state_transitions.next_states,
            previous_states=self.trellis_code.trellises.state_transitions.previous_states,
            L_int=L_int,
            chi_values=chi_values,
            init_conditional=self.trellis_code.init_conditional.to(dtype=self.dtype),
            L_init_int=L_init_int,
            delay=self.trellis_code.delay,
            use_max=self.use_max,
            device=self.device_manager.device,
        )

        L_transmitted_posterior = L_int + L_ext  # B x K-delay
        L_init_bitwise_posterior = compute_bitwise_delay_llr(
            L_init_ext=L_init_ext,
            L_init_int=L_init_int,
            delay=self.trellis_code.delay,
            use_max=self.use_max,
            device=self.device_manager.device,
        )  # B x delay
        # B x K
        L_posterior = torch.concat(
            [L_init_bitwise_posterior, L_transmitted_posterior], dim=1
        )

        return L_posterior

    def settings(self):
        return {
            "trellis_encoder": self.trellis_code.settings(),
            "modulator": self.modulator.settings(),
            "channel": self.channel.settings(),
            "use_max": self.use_max,
            "num_streams": self.num_streams,
        }

    def long_settings(self):
        return {
            "trellis_encoder": self.trellis_code.long_settings(),
            "modulator": self.modulator.long_settings(),
            "channel": self.channel.long_settings(),
            "use_max": self.use_max,
            "num_streams": self.num_streams,
        }
