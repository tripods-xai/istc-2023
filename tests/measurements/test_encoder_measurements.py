import torch
import torch.nn.functional as F

from src.encoders import AffineConvolutionalEncoder
from src.channels import AWGN
from src.modulation import BPSK
from src.decoders import BCJRDecoder
from src.utils import sigma2snr, DeviceManager

from src.measurements import (
    CodebookConditionalEntropySampler,
    TrellisConditionalEntropySampler,
)


def test_757_cross_entropy_with_bcjr():
    manager = DeviceManager(no_cuda=True, seed=1234)

    num_samples = 1000
    batch_size = 100
    msg_length = 18
    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    ).to_rsc()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    codebook_encoder = encoder.to_codebook()
    entropy_sampler = CodebookConditionalEntropySampler(
        codebook_encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    bcjr_entropy_sampler = TrellisConditionalEntropySampler(
        encoder, modulator, channel, device_manager=manager
    )

    codebook_est_ce = 0.0
    bcjr_est_ce = 0.0
    prev_samples = 0
    for i in range(0, num_samples, batch_size):
        cur_batch_size = min(num_samples - i, batch_size)
        codebook_ce_samples = entropy_sampler.sample(cur_batch_size)["xe"]
        bcjr_ce_samples = bcjr_entropy_sampler.sample(cur_batch_size)["xe"]

        # TODO: This can be pulled out into an object that keeps track of the running mean
        cur_samples = prev_samples + cur_batch_size
        codebook_est_ce = (
            codebook_est_ce * prev_samples + torch.mean(codebook_ce_samples).numpy()
        ) / (cur_samples)
        bcjr_est_ce = (
            bcjr_est_ce * prev_samples + torch.mean(bcjr_ce_samples).numpy()
        ) / (cur_samples)

    print(f"BCJR estimate: {bcjr_est_ce}")
    print(f"General estimate: {codebook_est_ce}")
    assert True  # Change this to false to see the print statements
