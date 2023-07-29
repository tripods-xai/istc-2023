import torch
import torch.nn.functional as F

from src.encoders import AffineConvolutionalEncoder
from src.channels import AWGN
from src.modulation import BPSK
from src.decoders import BCJRDecoder, CodebookDecoder
from src.utils import sigma2snr, DeviceManager


def test_757_codebook_decoder_with_bcjr():
    manager = DeviceManager(no_cuda=True, seed=1234)

    num_samples = 1000
    batch_size = 100
    msg_length = 18

    input_bits = torch.randint(0, 2, (2, msg_length))

    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    ).to_rsc()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    codebook_encoder = encoder.to_codebook()

    codebook_decoder = CodebookDecoder(
        codebook_encoder, modulator=modulator, channel=channel, device_manager=manager
    )

    bcjr_decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    received_msg = channel(modulator(encoder(input_bits)))
    codebook_confidence = codebook_decoder(
        received_msg,
    )
    bcjr_confidence = bcjr_decoder(
        received_msg,
    )

    torch.testing.assert_close(codebook_confidence, bcjr_confidence)
