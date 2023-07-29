import pytest

import torch
from tqdm import trange

from src.encoders import (
    AffineConvolutionalEncoder,
    turbo_755_1_00,
    turbo_155_7_00,
    turboae_binary_exact_nobd,
)
from src.channels import AWGN
from src.modulation import BPSK, IdentityModulation
from src.decoders import BCJRDecoder, JunctionTreeDecoder, CodebookDecoder
from src.utils import sigma2snr, DeviceManager
from src.graphs import (
    nonrecursive_turbo_graph,
    general_turbo_graph,
    nonrecursive_dependency_turbo_graph,
    nonrecursive_dependency_convolutional_factors,
    general_convolutional_factors,
)
from src.experiments import load_original_turboe_encoder, load_interleaver

from ..utils import test_manager


@pytest.mark.recursive
def test_757_jt_decoder_with_bcjr_short():
    manager = test_manager

    num_samples = 100
    batch_size = 100
    msg_length = 10

    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    ).to_rsc()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == general_convolutional_factors(msg_length, state_size=2)[0]
    )

    bcjr_decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        bcjr_confidence = bcjr_decoder(
            received_msg,
        )

        torch.testing.assert_close(jt_confidence, bcjr_confidence, atol=5e-5, rtol=0)


@pytest.mark.recursive
def test_757_jt_decoder_with_bcjr_long():
    manager = test_manager

    num_samples = 100
    batch_size = 100
    msg_length = 100

    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    ).to_rsc()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == general_convolutional_factors(msg_length, state_size=2)[0]
    )

    bcjr_decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        bcjr_confidence = bcjr_decoder(
            received_msg,
        )

        torch.testing.assert_close(jt_confidence, bcjr_confidence, atol=2e-4, rtol=0)


def test_751_jt_decoder_with_bcjr_short():
    manager = test_manager

    num_samples = 100
    batch_size = 100
    msg_length = 10

    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    )

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_dependency_convolutional_factors(
            msg_length, dependencies=encoder.generator, delay=encoder.delay
        )[0]
    )

    bcjr_decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        bcjr_confidence = bcjr_decoder(
            received_msg,
        )

        torch.testing.assert_close(jt_confidence, bcjr_confidence, atol=2e-4, rtol=0)


def test_751_jt_decoder_with_bcjr_long():
    manager = test_manager

    num_samples = 100
    batch_size = 100
    msg_length = 100

    encoder = AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=msg_length,
    )

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_dependency_convolutional_factors(
            msg_length, dependencies=encoder.generator, delay=encoder.delay
        )[0]
    )

    bcjr_decoder = BCJRDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        use_max=False,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        bcjr_confidence = bcjr_decoder(
            received_msg,
        )

        torch.testing.assert_close(jt_confidence, bcjr_confidence, atol=2e-4, rtol=0)


def test_turbo_755_1_jt_decoder_with_codebook_small():
    manager = test_manager

    num_samples = 10
    batch_size = 100
    msg_length = 8

    encoder = turbo_755_1_00(num_steps=msg_length, device_manager=manager)
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_dependency_turbo_graph(
            encoder.interleaver.permutation,
            nonrecursive_dependencies_noni=encoder.noninterleaved_encoder.generator,
            nonrecursive_dependencies_i=encoder.interleaved_encoder.generator,
            delay=encoder.noninterleaved_encoder.delay,
        ).factor_groups
    )

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        codebook_confidence = codebook_decoder(
            received_msg,
        )

        torch.testing.assert_close(
            jt_confidence, codebook_confidence, atol=2e-4, rtol=0
        )


@pytest.mark.slow
def test_turbo_755_1_jt_decoder_with_codebook():
    manager = test_manager

    num_samples = 10
    batch_size = 100
    msg_length = 16

    encoder = turbo_755_1_00(num_steps=msg_length, device_manager=manager)
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_dependency_turbo_graph(
            encoder.interleaver.permutation,
            nonrecursive_dependencies_noni=encoder.noninterleaved_encoder.generator,
            nonrecursive_dependencies_i=encoder.interleaved_encoder.generator,
            delay=encoder.noninterleaved_encoder.delay,
        ).factor_groups
    )

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        codebook_confidence = codebook_decoder(
            received_msg,
        )

        torch.testing.assert_close(
            jt_confidence, codebook_confidence, atol=2e-4, rtol=0
        )


@pytest.mark.recursive
@pytest.mark.slow
def test_turbo_155_7_jt_decoder_with_codebook():
    manager = test_manager

    num_samples = 10
    batch_size = 100
    msg_length = 16

    encoder = turbo_155_7_00(num_steps=msg_length, device_manager=manager)
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == general_turbo_graph(encoder.interleaver.permutation, 2).factor_groups
    )

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for _ in range(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        codebook_confidence = codebook_decoder(
            received_msg,
        )

        torch.testing.assert_close(
            jt_confidence, codebook_confidence, atol=2e-4, rtol=0
        )


@pytest.mark.skip(
    reason="This is a benchmarking test that I only want to run explicitly."
)
def test_turbo_755_1_jt_decoder_limits():
    manager = test_manager

    # Rough measurements
    ## Batch Size 1 : 75 is possible, takes ~3.5 min/it. 70 is better at ~20s/it
    ## Batch size 100
    ### 50 - 15.3s/it
    ### 60 - 72.7s/it
    ## Batch size 1000
    ### 45 - 30s/it
    ## Batch size 10000
    ### 35 - 17.56s/it
    num_samples = 10
    batch_size = 10
    msg_length = 10

    encoder = turbo_755_1_00(num_steps=msg_length, device_manager=manager)
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(encoder.interleaver.permutation, 3).factor_groups
    )

    for _ in trange(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )


@pytest.mark.slow
def test_turboae_binary_nobd_jt_decoder_with_codebook():
    manager = test_manager

    num_samples = 10
    batch_size = 100
    msg_length = 16
    delay = 0

    encoder = turboae_binary_exact_nobd(
        num_steps=msg_length, device_manager=manager, delay=delay
    )
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder, modulator=modulator, channel=channel, device_manager=manager
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(
            encoder.interleaver.permutation,
            window=5,
            delay=delay,
        ).factor_groups
    )

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for _ in trange(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
        codebook_confidence = codebook_decoder(
            received_msg,
        )

        torch.testing.assert_close(
            jt_confidence, codebook_confidence, atol=2e-4, rtol=0
        )


@pytest.mark.slow
def test_turboae_binary_nobd_delay_jt_decoder_with_codebook():
    manager = test_manager

    num_samples = 10
    batch_size = 1
    msg_length = 23
    delay = 2

    encoder = turboae_binary_exact_nobd(
        num_steps=msg_length, device_manager=manager, delay=delay
    )
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
        dtype=torch.float32,
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(
            encoder.interleaver.permutation,
            window=5,
            delay=delay,
        ).factor_groups
    )

    print(encoder.interleaver)
    input_msgs = []
    received_msgs = []
    for _ in range(num_samples):
        input_bits = torch.randint(
            0, 2, (batch_size, msg_length), device=test_manager.device
        )
        input_msgs.append(input_bits)
        received_msg = channel(modulator(encoder(input_bits)))
        received_msgs.append(received_msg)

    jt_confidences = []
    for i in trange(num_samples):
        jt_confidence = junction_tree_decoder(
            received_msgs[i],
        )
        jt_confidences.append(jt_confidence)

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )
    for i in trange(num_samples):
        codebook_confidence = codebook_decoder(
            received_msgs[i],
        )

        torch.testing.assert_close(
            jt_confidences[i].to(dtype=codebook_confidence.dtype),
            codebook_confidence,
            atol=2e-4,
            rtol=0,
        )


@pytest.mark.slow
def test_turboae_original_cont_jt_decoder_with_codebook_small():
    manager = test_manager

    num_samples = 10
    batch_size = 1
    msg_length = 10
    delay = 4

    interleaver = load_interleaver(
        interleaver_type="fixed",
        block_len=msg_length,
        manager=manager,
        interleaver_base_seed=15023,
    )
    encoder = load_original_turboe_encoder(
        interleaver=interleaver, turboae_type="continuous", device_manager=manager
    )
    encoder.compute_mean_std_()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = IdentityModulation(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
        dtype=torch.float32,
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(
            encoder.interleaver.permutation,
            window=9,
            delay=delay,
        ).factor_groups
    )

    print(encoder.interleaver)
    input_msgs = []
    received_msgs = []
    for _ in range(num_samples):
        input_bits = torch.randint(
            0, 2, (batch_size, msg_length), device=test_manager.device
        )
        input_msgs.append(input_bits)
        received_msg = channel(modulator(encoder(input_bits)))
        received_msgs.append(received_msg)

    jt_confidences = []
    for i in trange(num_samples):
        jt_confidence = junction_tree_decoder(
            received_msgs[i],
        )
        jt_confidences.append(jt_confidence)

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for i in trange(num_samples):
        codebook_confidence = codebook_decoder(
            received_msgs[i],
        )

        torch.testing.assert_close(
            jt_confidences[i].to(dtype=codebook_confidence.dtype),
            codebook_confidence,
            atol=2e-4,
            rtol=0,
        )


@pytest.mark.slow
def test_turboae_original_cont_jt_decoder_with_codebook_long():
    manager = test_manager

    num_samples = 10
    batch_size = 1
    msg_length = 16
    delay = 4

    interleaver = load_interleaver(
        interleaver_type="fixed",
        block_len=msg_length,
        manager=manager,
        interleaver_base_seed=15023,
    )
    encoder = load_original_turboe_encoder(
        interleaver=interleaver, turboae_type="continuous", device_manager=manager
    )
    encoder.compute_mean_std_()

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = IdentityModulation(device_manager=manager)

    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
        dtype=torch.float32,
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(
            encoder.interleaver.permutation,
            window=9,
            delay=delay,
        ).factor_groups
    )

    print(encoder.interleaver)
    input_msgs = []
    received_msgs = []
    for _ in range(num_samples):
        input_bits = torch.randint(
            0, 2, (batch_size, msg_length), device=test_manager.device
        )
        input_msgs.append(input_bits)
        received_msg = channel(modulator(encoder(input_bits)))
        received_msgs.append(received_msg)

    jt_confidences = []
    for i in trange(num_samples):
        jt_confidence = junction_tree_decoder(
            received_msgs[i],
        )
        jt_confidences.append(jt_confidence)

    codebook_decoder = CodebookDecoder(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for i in trange(num_samples):
        codebook_confidence = codebook_decoder(
            received_msgs[i],
        )

        torch.testing.assert_close(
            jt_confidences[i].to(dtype=codebook_confidence.dtype),
            codebook_confidence,
            atol=2e-4,
            rtol=0,
        )


@pytest.mark.skip(
    reason="This is a benchmarking test that I only want to run explicitly."
)
def test_turboae_binary_jt_decoder_limits():
    manager = test_manager

    # Rough measurements
    ## Batch size 10
    ### 50 - 154.11s/it - size = 27
    ## Batch size 200
    ### 40 - 120s/it - size = 22
    ## Batch size 10000
    ### 30 - 131.87s/it
    ## Batch size 100000
    ### 20 - 97s/it
    num_samples = 10
    batch_size = 100000
    msg_length = 20

    encoder = turboae_binary_exact_nobd(num_steps=msg_length, device_manager=manager)
    print(f"Encoder is nonrecursive: {encoder.is_nonrecursive}")

    sigma = 1.0
    channel = AWGN(snr=sigma2snr(sigma), device_manager=manager)

    modulator = BPSK(device_manager=manager)

    cluster_tree = (
        encoder.dependency_graph()
        .with_elimination_ordering(
            seed=manager.generate_seed(), sample_thresh=3, tries=300
        )
        .as_cluster_tree()
    )
    junction_tree_decoder = JunctionTreeDecoder(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
        cluster_tree=cluster_tree,
    )
    assert (
        junction_tree_decoder.cluster_tree.factor_groups
        == nonrecursive_turbo_graph(encoder.interleaver.permutation, 5).factor_groups
    )

    for _ in trange(num_samples):
        input_bits = torch.randint(0, 2, (batch_size, msg_length))

        received_msg = channel(modulator(encoder(input_bits)))
        jt_confidence = junction_tree_decoder(
            received_msg,
        )
