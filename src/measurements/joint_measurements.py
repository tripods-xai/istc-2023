from typing import Union

import torch
import torch.nn as nn
import gc

from ..encoders import SizedEncoder, ENC_interCNN, StreamedTurboEncoder
from ..modulation import Modulator
from ..decoders import SoftDecoder, JunctionTreeDecoder, IteratedBCJRTurboDecoder
from ..channels import NoisyChannel
from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER, filter_state_dict, Precision

from .encoder_measurements import (
    DecoderConditionalEntropySampler,
    DecoderCrossEntropySampler,
)


class TurboAESampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: ENC_interCNN,
        modulator: Modulator,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        encoder_as_conv: bool = False,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(
            encoder.to_conv_code(constrain=True) if encoder_as_conv else encoder,
            modulator,
            channel,
            decoder,
            device_manager,
        )

        self.encoder_as_conv = encoder_as_conv
        self.neural_encoder = encoder
        self._encoder_decoder = nn.ModuleDict(
            {"encoder": self.neural_encoder, "decoder": self.decoder}
        )

    def module(self):
        return self._encoder_decoder

    def load_state_dict(self, state_dict):
        self._encoder_decoder.load_state_dict(state_dict=state_dict)
        self.encoder = (
            self.neural_encoder.to_conv_code(constrain=True)
            if self.encoder_as_conv
            else self.neural_encoder
        )


class DecompositionSampler(
    DecoderConditionalEntropySampler, DecoderCrossEntropySampler
):
    def __init__(
        self,
        encoder: SizedEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        elimination_seed: Union[int, None] = None,
        dtype: torch.dtype = torch.float16,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(encoder, modulator, channel, decoder, device_manager)

        self.elimination_seed = elimination_seed
        self.junction_tree_decoder = JunctionTreeDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            device_manager=device_manager,
            cluster_tree=None,
            elimination_seed=elimination_seed,
            dtype=dtype,
        )

    def instance_errors(  # type: ignore[override]
        self,
        u: torch.Tensor,
        approx_logits: torch.FloatTensor,
        true_logits: torch.FloatTensor,
    ) -> dict[str, torch.Tensor]:
        approx_errors = DecoderCrossEntropySampler.instance_errors(
            self, u, approx_logits
        )
        true_errors = DecoderCrossEntropySampler.instance_errors(self, u, true_logits)

        kl_divergence = approx_errors["xe"] - true_errors["xe"]

        return {
            **{f"true_{k}": v for k, v in true_errors.items()},
            **{f"approx_{k}": v for k, v in approx_errors.items()},
            "kl": kl_divergence,
        }

    def sample(self, batch_size: int):
        input_data, corrupted_data, sample_stats = self.sample_data(batch_size)
        true_logits = self.junction_tree_decoder(corrupted_data)
        approx_logits = self.decoder(corrupted_data)

        return {
            **self.instance_errors(
                input_data,
                approx_logits=approx_logits,
                true_logits=true_logits,
            ),
            **sample_stats,
        }


class TurboAEDecompositionSampler(DecompositionSampler):
    def __init__(
        self,
        encoder: ENC_interCNN,
        modulator: Modulator,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        encoder_as_conv: bool = False,
        elimination_seed: Union[int, None] = None,
        dtype: torch.dtype = torch.float16,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        conv_encoder = (
            encoder.to_conv_code(constrain=True) if encoder_as_conv else encoder
        )

        super().__init__(
            conv_encoder,
            modulator,
            channel,
            decoder,
            elimination_seed=elimination_seed,
            dtype=dtype,
            device_manager=device_manager,
        )
        self.encoder_as_conv = encoder_as_conv

        # Overwrite the encoder to use the original
        self.encoder = encoder
        self._encoder_decoder = nn.ModuleDict(
            {"encoder": self.encoder, "decoder": self.decoder}
        )

    def module(self):
        return self._encoder_decoder

    def load_state_dict(self, state_dict, rebuild_jtree=False):
        self._encoder_decoder.load_state_dict(state_dict=state_dict)
        ctree = None if rebuild_jtree else self.junction_tree_decoder.cluster_tree
        encoder = (
            self.encoder.to_conv_code(constrain=True)
            if self.encoder_as_conv
            else self.encoder
        )
        self.junction_tree_decoder = JunctionTreeDecoder(
            encoder=encoder,
            modulator=self.modulator,
            channel=self.channel,
            device_manager=self.device_manager,
            cluster_tree=ctree,
            elimination_seed=self.elimination_seed,
            dtype=self.junction_tree_decoder.dtype,
        )


class BCJRDecompositionSampler(
    DecoderConditionalEntropySampler, DecoderCrossEntropySampler
):
    def __init__(
        self,
        encoder: StreamedTurboEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        num_iter=6,
        elimination_seed: int = None,
        dtype: torch.dtype = torch.float16,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = IteratedBCJRTurboDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            use_max=False,
            num_iter=num_iter,
            device_manager=device_manager,
        )

        super().__init__(
            encoder,
            modulator,
            channel,
            decoder,
            elimination_seed=elimination_seed,
            dtype=dtype,
            device_manager=device_manager,
        )

        self.num_iter = num_iter
