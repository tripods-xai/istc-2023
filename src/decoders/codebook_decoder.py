import torch

from ..encoders import CodebookEncoder
from ..modulation import Modulator
from ..channels import NoisyChannel
from ..utils import DEFAULT_DEVICE_MANAGER, DeviceManager

from .decoder import SoftDecoder


def get_joint_log_probs(
    y: torch.Tensor,
    codebook: torch.Tensor,
    modulator: Modulator,
    channel: NoisyChannel,
    chunk_size: int,
    joint_log_probs_out: torch.FloatTensor = None,
):
    """
    Parameters
    -----------
    joint_log_probs_out (Batch x num_input_seqs) : torch.FloatTensor
        A FloatTensor that will hold the output.
    """
    batch_size = y.shape[0]
    num_input_seqs = codebook.shape[0]

    if joint_log_probs_out is None:
        joint_log_probs_out = torch.empty(
            (batch_size, num_input_seqs), dtype=torch.float, device=y.device
        )

    y = torch.reshape(y, (batch_size, -1))
    codebook = torch.reshape(codebook, (num_input_seqs, -1))

    for i in range(0, num_input_seqs, chunk_size):
        top = min(num_input_seqs, i + chunk_size)
        modulated_codebook_chunk = modulator(codebook[i:top])
        joint_log_probs_out[:, i:top] = torch.sum(
            channel.log_prob(y[:, None], modulated_codebook_chunk[None]),
            dim=-1,
        )

    return joint_log_probs_out


class CodebookDecoder(SoftDecoder):
    def __init__(
        self,
        encoder: CodebookEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        chunk_size: int = 2**20,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(device_manager)
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.chunk_size = chunk_size

        assert self.encoder.codebook.shape[0] == 2**self.source_data_len

    @property
    def source_data_len(self) -> int:
        return self.encoder.input_size

    # TODO: This logic can be pulled out into a special brute-force decoder.
    # Then we can merge this with the class below
    def conditional_logits(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y (Batch x source_data_len / rate) : torch.Tensor
            The corrupted received data.
        chunk_size : int
            The size of the chunks to use in computing
        """
        batch_size = y.shape[0]

        # Get the log probs for each input
        # log P(y | x = codeword)
        joint_log_probs = get_joint_log_probs(
            y,
            self.encoder.codebook,
            modulator=self.modulator,
            channel=self.channel,
            chunk_size=self.chunk_size,
        )

        # Naively do a sum for each ind of input. This is n2^n ops for block_len n
        joint_log_probs = joint_log_probs.reshape(
            (batch_size,) + ((2,) * self.source_data_len)
        )
        ui_is_one_cond_log_prob = torch.empty(
            (batch_size, self.source_data_len), device=self.device_manager.device
        )
        ui_is_zero_cond_log_prob = torch.empty(
            (batch_size, self.source_data_len), device=self.device_manager.device
        )
        # When drawing input samples from the uniform distribution, P(U) is
        # a constant and will get cancelled out.
        for i in range(self.source_data_len):
            ui_is_one = torch.select(joint_log_probs, i + 1, 1)
            ui_is_one_cond_log_prob[:, i] = torch.logsumexp(
                ui_is_one.reshape(batch_size, -1), dim=-1
            )
            ui_is_zero = torch.select(joint_log_probs, i + 1, 0)
            ui_is_zero_cond_log_prob[:, i] = torch.logsumexp(
                ui_is_zero.reshape(batch_size, -1), dim=-1
            )

        logits = ui_is_one_cond_log_prob - ui_is_zero_cond_log_prob
        return logits

    def forward(self, received_symbols: torch.Tensor):
        return self.conditional_logits(received_symbols)

    def settings(self):
        return {
            "codebook_encoder": self.encoder.settings(),
            "modulator": self.modulator.settings(),
            "channel": self.channel.settings(),
        }

    def long_settings(self):
        return {
            "codebook_encoder": self.encoder.long_settings(),
            "modulator": self.modulator.long_settings(),
            "channel": self.channel.long_settings(),
        }
