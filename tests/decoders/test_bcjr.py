import numpy as np
from numpy.testing import assert_array_almost_equal
import commpy.channelcoding as cc

import torch

from src.encoders import AffineConvolutionalEncoder
from src.decoders import BCJRDecoder
from src.channels import AWGN
from src.utils import DeviceManager, sigma2snr
from src.modulation import BPSK

from ..channels import create_noiseless_channel


def test_compare_torch_map_decode_to_commpy_map_decode_no_noise():
    manager = DeviceManager(no_cuda=True, seed=1234)
    # Two messages of time 20 and 1 channel
    msg_len = 20
    input_bits = torch.randint(0, 2, (2, msg_len))

    gen = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias = torch.FloatTensor([0, 0])
    encoder = AffineConvolutionalEncoder(gen, bias, num_steps=msg_len)

    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )

    modulator = BPSK(device_manager=manager)

    decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    received_msg = channel(modulator(encoder(input_bits)))
    torch_confidence = decoder(
        received_msg,
    )

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    commpy_received = (
        2.0
        * np.stack(
            [
                cc.conv_encode(
                    input_bits.numpy()[0, :], commpy_trellis, termination="cont"
                ).reshape(20, 2),
                cc.conv_encode(
                    input_bits.numpy()[1, :], commpy_trellis, termination="cont"
                ).reshape(20, 2),
            ],
            axis=0,
        )
        - 1.0
    )
    np_received = received_msg.numpy()
    assert_array_almost_equal(np_received, commpy_received)

    L_int = np.zeros(input_bits.shape[1])
    L = np.stack(
        [
            cc.map_decode(
                np_received[0, :, 0],
                np_received[0, :, 1],
                commpy_trellis,
                sigma**2,
                L_int,
                mode="compute",
            )[0],
            cc.map_decode(
                np_received[1, :, 0],
                np_received[1, :, 1],
                commpy_trellis,
                sigma**2,
                L_int,
                mode="compute",
            )[0],
        ],
        axis=0,
    )

    assert_array_almost_equal(L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_map_decode_to_commpy_map_decode_with_noise():
    manager = DeviceManager(no_cuda=True, seed=1234)
    # Two messages of time 20 and 1 channel
    msg_len = 20
    input_bits = torch.randint(0, 2, (2, msg_len))

    gen = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias = torch.FloatTensor([0, 0])
    encoder = AffineConvolutionalEncoder(gen, bias, num_steps=msg_len)

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    received_msg = channel(modulator(encoder(input_bits)))
    torch_confidence = decoder(
        received_msg,
    )

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()

    L_int = np.zeros(input_bits.shape[1])
    L = np.stack(
        [
            cc.map_decode(
                np_received[0, :, 0],
                np_received[0, :, 1],
                commpy_trellis,
                sigma**2,
                L_int,
                mode="compute",
            )[0],
            cc.map_decode(
                np_received[1, :, 0],
                np_received[1, :, 1],
                commpy_trellis,
                sigma**2,
                L_int,
                mode="compute",
            )[0],
        ],
        axis=0,
    )

    assert_array_almost_equal(L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_map_decode_to_commpy_map_decode_no_noise_nonzero_L_int():
    manager = DeviceManager(no_cuda=True, seed=1234)
    # Two messages of time 20 and 1 channel
    msg_len = 20
    input_bits = torch.randint(0, 2, (2, msg_len))

    gen = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias = torch.FloatTensor([0, 0])
    encoder = AffineConvolutionalEncoder(gen, bias, num_steps=msg_len)

    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )

    modulator = BPSK(device_manager=manager)

    decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    received_msg = channel(modulator(encoder(input_bits)))
    L_int = torch.randn(input_bits.shape, dtype=torch.float)
    torch_confidence = decoder(received_msg, L_int=L_int)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()
    np_L_int = L_int.numpy()
    L = np.stack(
        [
            cc.map_decode(
                np_received[0, :, 0],
                np_received[0, :, 1],
                commpy_trellis,
                sigma**2,
                np_L_int[0],
                mode="compute",
            )[0],
            cc.map_decode(
                np_received[1, :, 0],
                np_received[1, :, 1],
                commpy_trellis,
                sigma**2,
                np_L_int[1],
                mode="compute",
            )[0],
        ],
        axis=0,
    )

    assert_array_almost_equal(L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_map_decode_to_commpy_map_decode_with_noise_nonzero_L_int():
    manager = DeviceManager(no_cuda=True, seed=1234)
    # Two messages of time 20 and 1 channel
    msg_len = 20
    input_bits = torch.randint(0, 2, (2, msg_len))

    gen = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias = torch.FloatTensor([0, 0])
    encoder = AffineConvolutionalEncoder(gen, bias, num_steps=msg_len)

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    received_msg = channel(modulator(encoder(input_bits)))
    L_int = torch.randn(input_bits.shape, dtype=torch.float)
    torch_confidence = decoder(received_msg, L_int=L_int)

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    np_received = received_msg.numpy()
    np_L_int = L_int.numpy()
    L = np.stack(
        [
            cc.map_decode(
                np_received[0, :, 0],
                np_received[0, :, 1],
                commpy_trellis,
                sigma**2,
                np_L_int[0],
                mode="compute",
            )[0],
            cc.map_decode(
                np_received[1, :, 0],
                np_received[1, :, 1],
                commpy_trellis,
                sigma**2,
                np_L_int[1],
                mode="compute",
            )[0],
        ],
        axis=0,
    )

    assert_array_almost_equal(L, torch_confidence.numpy(), decimal=5)
