import abc
import torch
import torch.nn.functional as F

from ..utils import (
    binary_entropy_with_logits,
    DEFAULT_DEVICE_MANAGER,
)
from ..encoders import (
    CodebookEncoder,
    TrellisEncoder,
    BlockEncoder,
    TurboEncoder,
    StreamedTurboEncoder,
    SizedEncoder,
)
from ..modulation import Modulator
from ..channels import NoisyChannel
from ..decoders import (
    BCJRDecoder,
    SoftDecoder,
    CodebookDecoder,
    IteratedBCJRTurboDecoder,
    IteratedBCJRSystematicTurboDecoder,
    HazzysTurboDecoder,
    TurboAEDecoder,
    JunctionTreeDecoder,
)


class Sampler:
    def __init__(self, device_manager=DEFAULT_DEVICE_MANAGER) -> None:
        self.device_manager = device_manager

    @abc.abstractmethod
    def sample(self, batch_size: int):
        pass


class DecoderCrossEntropySampler(Sampler):
    def __init__(
        self,
        encoder: SizedEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(device_manager)
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.decoder = decoder

    @property
    def block_len(self) -> int:
        return self.encoder.input_size

    def sample_data(self, batch_size: int):
        input_data = torch.randint(
            0,
            2,
            (batch_size, self.block_len),
            generator=self.device_manager.generator,
            device=self.device_manager.device,
            dtype=torch.int8,
        )
        encoded_data = self.encoder(input_data)
        modulated_data = self.modulator(encoded_data)
        corrupted_data = self.channel(modulated_data)
        return (
            input_data,
            corrupted_data,
            {
                "modulated_power": torch.mean(modulated_data**2, dim=[-1, -2]),
                "modulated_mean": torch.mean(modulated_data, dim=[-1, -2]),
            },
        )

    def instance_errors(
        self, u: torch.Tensor, logits: torch.FloatTensor
    ) -> dict[str, torch.Tensor]:
        cross_ent = F.binary_cross_entropy_with_logits(
            logits, u.float(), reduction="none"
        )
        mismatch = (logits > 0) != u.bool()

        return {
            "xe": torch.mean(cross_ent, dim=-1),
            "ber": torch.mean(mismatch.float(), dim=-1),
            "bler": torch.mean(
                torch.any(mismatch, dim=-1, keepdim=True).float(), dim=-1
            ),
        }

    def sample(self, batch_size: int):
        input_data, corrupted_data, sample_stats = self.sample_data(batch_size)
        logits = self.decoder(corrupted_data)
        return {**self.instance_errors(input_data, logits), **sample_stats}


class NeuralDecoderCrossEntropySampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: StreamedTurboEncoder,
        decoder_path: str,
        modulator: Modulator,
        channel: NoisyChannel,
        num_iteration: int = 6,
        num_iter_ft: int = 5,
        dec_num_layer: int = 5,
        dec_num_unit: int = 100,
        dec_kernel_size: int = 5,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = TurboAEDecoder(
            num_iteration=num_iteration,
            num_iter_ft=num_iter_ft,
            dec_num_layer=dec_num_layer,
            dec_num_unit=dec_num_unit,
            dec_kernel_size=dec_kernel_size,
            interleaver=encoder.interleaver,
            device_manager=device_manager,
        )

        print(f"Initializing decoder from path {decoder_path}")
        s_dict = torch.load(decoder_path, map_location=device_manager.device)
        decoder.pre_initialize(s_dict)

        super().__init__(encoder, modulator, channel, decoder, device_manager)

        self.decoder_path = decoder_path


class IteratedTurboCrossEntropySampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: StreamedTurboEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        num_iter: int = 6,
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
        super().__init__(encoder, modulator, channel, decoder, device_manager)


class IteratedSystematicTurboCrossEntropySampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: StreamedTurboEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        num_iter: int = 6,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = IteratedBCJRSystematicTurboDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            use_max=False,
            num_iter=num_iter,
            device_manager=device_manager,
        )
        super().__init__(encoder, modulator, channel, decoder, device_manager)


class HazzysTurboCrossEntropySampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: StreamedTurboEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        num_iter: int = 6,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = HazzysTurboDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            use_max=False,
            num_iter=num_iter,
            device_manager=device_manager,
        )
        super().__init__(encoder, modulator, channel, decoder, device_manager)


class DecoderConditionalEntropySampler(DecoderCrossEntropySampler):
    def __init__(
        self,
        encoder: SizedEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(encoder, modulator, channel, decoder, device_manager)

    @property
    def block_len(self) -> int:
        return self.encoder.input_size

    def instance_entropies(self, cond_logits: torch.FloatTensor) -> torch.Tensor:
        cond_ent = binary_entropy_with_logits(cond_logits)
        return torch.mean(cond_ent, dim=-1)

    def instance_true_bers(self, cond_logits: torch.FloatTensor) -> torch.Tensor:
        probs = torch.sigmoid(cond_logits)
        return torch.mean(torch.minimum(probs, 1 - probs))

    def sample(self, batch_size: int):
        input_data, corrupted_data, sample_stats = self.sample_data(batch_size)
        cond_logits = self.decoder(corrupted_data)
        return {
            "ce": self.instance_entropies(cond_logits),
            "true_ber": self.instance_true_bers(cond_logits),
            **self.instance_errors(u=input_data, logits=cond_logits),
            **sample_stats,
        }


class TrellisConditionalEntropySampler(DecoderConditionalEntropySampler):
    def __init__(
        self,
        encoder: TrellisEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = BCJRDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            use_max=False,
            device_manager=device_manager,
        )

        super().__init__(
            encoder, modulator, channel, decoder, device_manager=device_manager
        )


class CodebookConditionalEntropySampler(DecoderConditionalEntropySampler):
    def __init__(
        self,
        encoder: CodebookEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        chunk_size: int = 2**24,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = CodebookDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            chunk_size=chunk_size,
            device_manager=device_manager,
        )

        super().__init__(
            encoder, modulator, channel, decoder, device_manager=device_manager
        )

    @property
    def chunk_size(self) -> int:
        return self.decoder.chunk_size


class JunctionTreeConditionalEntropySampler(DecoderConditionalEntropySampler):
    def __init__(
        self,
        encoder: SizedEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        cluster_tree=None,
        elimination_seed: int = None,
        dtype: torch.dtype = torch.float32,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        decoder = JunctionTreeDecoder(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            cluster_tree=cluster_tree,
            elimination_seed=elimination_seed,
            dtype=dtype,
            device_manager=device_manager,
        )

        super().__init__(
            encoder, modulator, channel, decoder, device_manager=device_manager
        )
