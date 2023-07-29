import torch
import torch.nn.functional as F

from ..utils import (
    MaskedTensor,
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    get_reducer,
    enumerate_binary_inputs,
)
from ..encoders import StreamedTurboEncoder, TrellisEncoder
from ..modulation import Modulator
from ..channels import NoisyChannel

from .bcjr import BCJRDecoder
from .decoder import SoftDecoder


class IteratedBCJRTurboDecoder(SoftDecoder):
    """A decoder for a TrellisTurboEncoder that uses the BCJR algorithm.

    Attributes
    ----------
    use_max : bool
        A flag that determines whether or not to use the hardmax approximation
        of logsumexp.

    """

    def __init__(
        self,
        encoder: StreamedTurboEncoder[TrellisEncoder],
        modulator: Modulator,
        channel: NoisyChannel,
        use_max: bool = False,
        num_iter: int = 6,
        use_float16=False,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super(IteratedBCJRTurboDecoder, self).__init__(device_manager=device_manager)
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.use_max = use_max
        self.num_iter = num_iter
        self.use_float16 = use_float16
        self.dtype = torch.float16 if self.use_float16 else torch.float32

        self.decoder1 = BCJRDecoder(
            encoder=self.encoder.noninterleaved_encoder,
            modulator=modulator,
            channel=channel,
            use_max=use_max,
            use_float16=use_float16,
            device_manager=device_manager,
        )
        self.decoder2 = BCJRDecoder(
            encoder=self.encoder.interleaved_encoder,
            modulator=modulator,
            channel=channel,
            use_max=use_max,
            use_float16=use_float16,
            device_manager=device_manager,
        )
        assert self.encoder.delay == 0

        self.validate()

    def validate(self):
        pass

    @property
    def num_streams(self):
        return self.encoder.num_output_channels

    @property
    def num_output_channels(self):
        return 1

    @property
    def delay(self) -> int:
        return self.encoder.delay

    @property
    def source_data_len(self) -> int:
        return self.encoder.input_size

    def compute_known_information(self, received_symbols: torch.Tensor):
        batch_size = received_symbols.shape[0]
        return torch.zeros(
            (batch_size, self.source_data_len),
            device=self.device_manager.device,
            dtype=self.dtype,
        )

    def forward(
        self,
        received_symbols: torch.Tensor,
        L_int: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        received_symbols (Batch x Time x Channels) : torch.Tensor
            The received corrupted stream. A manifestation of the random variable Y.
            Usually a torch.FloatTensor, but may be other types as well.
        L_int (Batch x Time-delay) : torch.FloatTensor
            The logit intrinsic information (prior) on the value of each (non-delayed) bit.

        Returns
        -------
        (Batch x Time) torch.FloatTensor
            The approximated posterior probability on each bit of the source sequence.

        """

        batch_size = received_symbols.shape[0]
        if L_int is None:
            L_int = torch.zeros(
                (batch_size, self.source_data_len - self.delay),
                device=self.device_manager.device,
                dtype=self.dtype,
            )

        y_noni = received_symbols[
            :, :, : self.encoder.noninterleaved_encoder.num_output_channels
        ]
        y_i = received_symbols[
            :, :, self.encoder.noninterleaved_encoder.num_output_channels :
        ]

        return self.decode(
            y_noni=y_noni,
            y_i=y_i,
            L_int=L_int,
            known_information=self.compute_known_information(received_symbols),
        )

    def decode(
        self,
        y_noni: torch.Tensor,
        y_i: torch.Tensor,
        L_int: torch.Tensor,
        known_information: torch.Tensor,
    ):
        L_int_1 = L_int
        for i in range(self.num_iter):
            L_ext_1 = self.decoder1(y_noni, L_int=L_int_1) - L_int_1 - known_information

            L_int_2 = self.encoder.interleaver.interleave(L_ext_1)
            L_ext_2 = self.decoder2(y_i, L_int=L_int_2) - L_int_2

            L_int_1 = self.encoder.interleaver.deinterleave(L_ext_2) - known_information

        return L_int_1 + L_ext_1 + known_information

    def settings(self):
        return {
            "encoder": self.encoder.settings(),
            "modulator": self.modulator.settings(),
            "channel": self.channel.settings(),
            "use_max": self.use_max,
            "num_streams": self.num_streams,
            "num_iter": self.num_iter,
        }

    def long_settings(self):
        return {
            "trellis_encoder": self.trellis_code.long_settings(),
            "modulator": self.modulator.long_settings(),
            "channel": self.channel.long_settings(),
            "use_max": self.use_max,
            "num_streams": self.num_streams,
            "num_iter": self.num_iter,
        }


class IteratedBCJRSystematicTurboDecoder(IteratedBCJRTurboDecoder):
    """A decoder for a TrellisTurboEncoder that uses the BCJR algorithm.

    Attributes
    ----------
    use_max : bool
        A flag that determines whether or not to use the hardmax approximation
        of logsumexp.

    """

    def __init__(
        self,
        encoder: StreamedTurboEncoder[TrellisEncoder],
        modulator: Modulator,
        channel: NoisyChannel,
        use_max: bool = False,
        num_iter: int = 6,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            use_max=use_max,
            num_iter=num_iter,
            device_manager=device_manager,
        )
        self.interleaved_encoder = self.encoder.interleaved_encoder.with_systematic()
        self.decoder2 = BCJRDecoder(
            encoder=self.interleaved_encoder,
            modulator=modulator,
            channel=channel,
            use_max=use_max,
            device_manager=device_manager,
        )

    def forward(
        self,
        received_symbols: torch.Tensor,
        L_int: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        received_symbols (Batch x Time x Channels) : torch.Tensor
            The received corrupted stream. A manifestation of the random variable Y.
            Usually a torch.FloatTensor, but may be other types as well.
        L_int (Batch x Time-delay) : torch.FloatTensor
            The logit intrinsic information (prior) on the value of each (non-delayed) bit.

        Returns
        -------
        (Batch x Time) torch.FloatTensor
            The approximated posterior probability on each bit of the source sequence.

        """
        batch_size = received_symbols.shape[0]
        if L_int is None:
            L_int = torch.zeros(
                (batch_size, self.source_data_len - self.delay),
                device=self.device_manager.device,
            )

        y_noni = received_symbols[
            :, :, : self.encoder.noninterleaved_encoder.num_output_channels
        ]
        y_i = torch.cat(
            [
                self.encoder.interleaver.interleave(received_symbols[:, :, :1]),
                received_symbols[
                    :, :, self.encoder.noninterleaved_encoder.num_output_channels :
                ],
            ],
            dim=-1,
        )

        return self.decode(
            y_noni=y_noni,
            y_i=y_i,
            L_int=L_int,
            known_information=self.compute_known_information(received_symbols),
        )


class HazzysTurboDecoder(IteratedBCJRSystematicTurboDecoder):
    """A decoder for a systematic TrellisTurboEncoder that uses the BCJR algorithm.

    Attributes
    ----------
    use_max : bool
        A flag that determines whether or not to use the hardmax approximation
        of logsumexp.

    """

    def compute_known_information(self, received_symbols: torch.Tensor):
        sys_stream = received_symbols[..., 0]
        ones = self.modulator.modulate(
            torch.ones_like(sys_stream, device=self.device_manager.device)
        )
        zeros = self.modulator.modulate(
            torch.zeros_like(sys_stream, device=self.device_manager.device)
        )
        return self.channel.log_prob(sys_stream, ones) - self.channel.log_prob(
            sys_stream, zeros
        )
