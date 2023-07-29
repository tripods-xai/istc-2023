import numpy as np
from numpy.testing import assert_array_almost_equal
import commpy.channelcoding as cc

import torch

from src.encoders import AffineConvolutionalEncoder, StreamedTurboEncoder
from src.interleavers import FixedPermuteInterleaver
from src.decoders import IteratedBCJRTurboDecoder, IteratedBCJRSystematicTurboDecoder
from src.channels import AWGN
from src.utils import DeviceManager, sigma2snr
from src.modulation import BPSK

from ..channels import create_noiseless_channel
from ..encoders import interleaver_to_commpy
from .decoder_utils import vturbo_decode, vsturbo_decode


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_without_noise_one_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 1
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg: torch.Tensor = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_without_noise_two_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 2
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_without_noise_six_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 6
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_with_noise_one_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 1
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_with_noise_two_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 2
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_turbo_decode_to_commpy_turbo_decode_with_noise_six_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    code1 = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    gen2 = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
    bias2 = torch.FloatTensor([0, 0])
    code2 = AffineConvolutionalEncoder(
        gen2, bias2, num_steps=msg_length, device_manager=manager
    )

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 6
    decoder = IteratedBCJRTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 4]]))
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_without_noise_one_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 1
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg: torch.Tensor = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_without_noise_two_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 2
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_without_noise_six_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = create_noiseless_channel(AWGN)(
        snr=sigma2snr(sigma), device_manager=manager
    )
    modulator = BPSK(device_manager=manager)

    num_iter = 6
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_with_noise_one_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 1
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_with_noise_two_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 2
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)


def test_compare_torch_systematic_turbo_decode_to_commpy_turbo_decode_with_noise_six_iter():
    manager = DeviceManager(seed=3859)
    msg_length = 20

    gen1 = torch.FloatTensor([[1, 1, 1], [1, 0, 1]])
    bias1 = torch.FloatTensor([0, 0])
    base_code = AffineConvolutionalEncoder(
        gen1, bias1, num_steps=msg_length, device_manager=manager
    )

    code2 = base_code.to_rc()
    code1 = code2.with_systematic()

    batch_size = 2
    input_bits = torch.randint(0, 2, size=(batch_size, msg_length))

    interleaver = FixedPermuteInterleaver(msg_length)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=manager,
    )
    sigma = 1.0
    channel = AWGN(sigma2snr(sigma))
    modulator = BPSK(device_manager=manager)

    num_iter = 6
    decoder = IteratedBCJRSystematicTurboDecoder(
        encoder=turbo_encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        num_iter=num_iter,
        device_manager=manager,
    )

    msg = turbo_encoder(input_bits)
    received_msg = channel(modulator(msg))
    torch_confidence = decoder(received_msg)

    trellis1 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    trellis2 = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(interleaver)

    np_received = received_msg.numpy()
    commpy_L = vsturbo_decode(
        np_received, trellis1, trellis2, sigma**2, num_iter, commpy_interleaver
    )

    assert_array_almost_equal(commpy_L, torch_confidence.numpy(), decimal=5)
