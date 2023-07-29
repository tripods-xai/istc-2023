import torch
import math
import pytest

import numpy as np
from commpy import channelcoding as cc

from src.encoders import (
    TrellisEncoder,
    StateTransitionGraph,
    Trellis,
    GeneralizedConvolutionalEncoder,
    FourierConvolutionalEncoder,
    AffineConvolutionalEncoder,
    StreamedTurboEncoder,
    turboae_cont_exact_nobd,
)
from src.interleavers import (
    FixedPermuteInterleaver,
    BatchRandomPermuteInterleaver,
    RandomPermuteInterleaver,
)
from src.utils import DeviceManager
from src.fourier import fourier_to_table

from .encoder_utils import interleaver_to_commpy, vsystematic_turbo_encode


def test_trellis_code_construct_basic():
    next_states = torch.LongTensor([[1, 2], [0, 1], [1, 2]])
    state_transitions = StateTransitionGraph.from_next_states(next_states)
    output_table = torch.FloatTensor(
        [
            [
                [-0.12057394, 2.1424615, 0.98776376],
                [-0.03558801, -0.9341358, -0.78986055],
            ],
            [[1.1309315, 1.2084178, -1.0634259], [-0.44598797, 1.0388252, 0.03602624]],
            [[0.28023955, 0.6486982, 1.1790744], [1.8008167, 1.3370351, -0.1548117]],
        ]
    )
    trellis = Trellis(state_transitions, output_table)
    num_steps = 100
    trellises = trellis.unroll(num_steps)

    code = TrellisEncoder(trellises=trellises)
    assert code.trellises is trellises
    assert code.num_states == next_states.shape[0]
    assert code.input_size == num_steps
    assert code.num_output_channels == output_table.shape[2]


def test_trellis_code_call_basic():
    next_states = torch.LongTensor([[1, 2], [0, 1], [1, 2]])
    state_transitions = StateTransitionGraph.from_next_states(next_states)
    output_table = torch.FloatTensor(
        [
            [
                [-0.12057394, 2.1424615, 0.98776376],
                [-0.03558801, -0.9341358, -0.78986055],
            ],
            [[1.1309315, 1.2084178, -1.0634259], [-0.44598797, 1.0388252, 0.03602624]],
            [[0.28023955, 0.6486982, 1.1790744], [1.8008167, 1.3370351, -0.1548117]],
        ]
    )
    trellis = Trellis(state_transitions, output_table)

    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    code = TrellisEncoder(trellis.unroll(num_steps=msg.shape[1]))
    expected_output = torch.FloatTensor(
        [
            [
                [-0.03558801, -0.9341358, -0.78986055],
                [0.28023955, 0.6486982, 1.1790744],
                [-0.44598797, 1.0388252, 0.03602624],
                [-0.44598797, 1.0388252, 0.03602624],
            ],
            [
                [-0.12057394, 2.1424615, 0.98776376],
                [-0.44598797, 1.0388252, 0.03602624],
                [-0.44598797, 1.0388252, 0.03602624],
                [1.1309315, 1.2084178, -1.0634259],
            ],
            [
                [-0.12057394, 2.1424615, 0.98776376],
                [1.1309315, 1.2084178, -1.0634259],
                [-0.03558801, -0.9341358, -0.78986055],
                [1.8008167, 1.3370351, -0.1548117],
            ],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_gen_conv_code_construct_no_feedback_basic():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    code = GeneralizedConvolutionalEncoder(table, num_steps=4)
    assert torch.all(code.table == table)
    assert code.feedback is None
    assert code.num_output_channels == 2
    assert code.num_possible_windows == 8
    assert code.window == 3

    expected_next_states = torch.LongTensor([[0, 1], [2, 3], [0, 1], [2, 3]])
    expected_output_table = torch.FloatTensor(
        [[[1, 1], [1, 1]], [[2, 0], [0, 0]], [[2, 1], [2, 1]], [[2, 0], [2, 2]]]
    )
    assert code.trellis.is_same(
        Trellis(
            StateTransitionGraph.from_next_states(expected_next_states),
            expected_output_table,
        )
    )


def test_gen_conv_code_call_no_feedback_basic():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    code = GeneralizedConvolutionalEncoder(table, num_steps=4)
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    expected_output = torch.FloatTensor(
        [
            [[1, 1], [2, 0], [2, 1], [0, 0]],
            [[1, 1], [1, 1], [0, 0], [2, 0]],
            [[1, 1], [1, 1], [1, 1], [0, 0]],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_gen_conv_code_with_systematic_no_feedback():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    code = GeneralizedConvolutionalEncoder(table, num_steps=4).with_systematic()
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    expected_output = torch.FloatTensor(
        [
            [[1, 1, 1], [0, 2, 0], [1, 2, 1], [1, 0, 0]],
            [[0, 1, 1], [1, 1, 1], [1, 0, 0], [0, 2, 0]],
            [[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0]],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_gen_conv_code_call_trellis_equiv():
    manager = DeviceManager(seed=1090)

    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]],
        device=manager.device,
    )
    msg_len = 100
    code = GeneralizedConvolutionalEncoder(
        table, num_steps=msg_len, device_manager=manager
    )
    msg = torch.randint(
        0,
        2,
        size=(100, msg_len),
        dtype=torch.float,
        device=manager.device,
        generator=manager.generator,
    )

    trellis_code = TrellisEncoder(code.trellises, device_manager=manager)

    assert torch.all(code(msg) == trellis_code(msg))

    table2 = torch.FloatTensor(
        [[1], [1], [2], [0], [2], [2], [2], [2]], device=manager.device
    )
    code2 = GeneralizedConvolutionalEncoder(
        table2, num_steps=msg_len, device_manager=manager
    )
    msg2 = torch.randint(
        0,
        2,
        size=(100, msg_len),
        dtype=torch.float,
        device=manager.device,
        generator=manager.generator,
    )

    trellis_code2 = TrellisEncoder(code2.trellises, device_manager=manager)

    assert torch.all(code2(msg2) == trellis_code2(msg2))


def test_affine_conv_code_construct_basic():
    gen = torch.FloatTensor([[1, 0, 1], [1, 1, 1]])
    bias = torch.FloatTensor([1, 0])
    code = AffineConvolutionalEncoder(gen, bias, num_steps=4)

    expected_table = torch.FloatTensor(
        [[1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0], [0, 0], [1, 1]]
    )

    assert code.num_output_channels == 2
    assert torch.all(code.bias == bias)
    assert torch.all(code.generator == gen)
    assert torch.all(code.table == expected_table)


def test_affine_conv_code_call_basic():
    gen = torch.FloatTensor([[1, 0, 1], [1, 1, 1]])
    bias = torch.FloatTensor([1, 0])
    code = AffineConvolutionalEncoder(gen, bias, num_steps=4)
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    expected_output = torch.FloatTensor(
        [
            [[0, 1], [1, 1], [1, 0], [0, 0]],
            [[1, 0], [0, 1], [0, 0], [0, 0]],
            [[1, 0], [1, 0], [0, 1], [0, 0]],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_affine_conv_code_call_gen_conv_code_equiv():
    manager = DeviceManager(seed=1090)

    msg_len = 100

    gen = torch.FloatTensor([[1, 0, 1], [1, 1, 1]], device=manager.device)
    bias = torch.FloatTensor([1, 0], device=manager.device)
    code = AffineConvolutionalEncoder(
        gen, bias, num_steps=msg_len, device_manager=manager
    )
    code2 = GeneralizedConvolutionalEncoder(
        code.table, num_steps=msg_len, device_manager=manager
    )

    msg = torch.randint(
        0,
        2,
        size=(100, msg_len),
        dtype=torch.float,
        device=manager.device,
        generator=manager.generator,
    )

    assert torch.all(code(msg) == code2(msg))


def test_gen_conv_code_construct_with_feedback_basic():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    feedback = torch.CharTensor([1, 1, 1, 0, 0, 0, 1, 1])
    code = GeneralizedConvolutionalEncoder(table, feedback=feedback, num_steps=4)
    assert torch.all(code.table == table)
    assert torch.all(code.feedback == feedback)
    assert code.num_output_channels == 2
    assert code.num_possible_windows == 8
    assert code.window == 3

    expected_next_states = torch.LongTensor([[1, 1], [3, 2], [0, 0], [3, 3]])
    expected_output_table = torch.FloatTensor(
        [[[1, 1], [1, 1]], [[0, 0], [2, 0]], [[2, 1], [2, 1]], [[2, 2], [2, 2]]]
    )
    assert code.trellis.is_same(
        Trellis(
            StateTransitionGraph.from_next_states(expected_next_states),
            expected_output_table,
        )
    )


def test_gen_conv_code_call_with_feedback_basic():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    feedback = torch.CharTensor([1, 1, 1, 0, 0, 0, 1, 1])
    code = GeneralizedConvolutionalEncoder(table, feedback=feedback, num_steps=4)
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    expected_output = torch.FloatTensor(
        [
            [[1, 1], [0, 0], [2, 2], [2, 2]],
            [[1, 1], [2, 0], [2, 1], [1, 1]],
            [[1, 1], [0, 0], [2, 2], [2, 2]],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_gen_conv_code_with_systematic_non_invert_feedback():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    feedback = torch.CharTensor([1, 1, 1, 0, 0, 0, 1, 1])
    code = GeneralizedConvolutionalEncoder(table, feedback=feedback, num_steps=4)
    code = code.with_systematic()
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    expected_output = torch.FloatTensor(
        [
            [[1, 1, 1], [0, 0, 0], [1, 2, 2], [1, 2, 2]],
            [[0, 1, 1], [1, 2, 0], [1, 2, 1], [0, 1, 1]],
            [[0, 1, 1], [0, 0, 0], [1, 2, 2], [1, 2, 2]],
        ]
    )
    assert torch.all(code(msg) == expected_output)


def test_gen_conv_code_with_systematic_invert_feedback():
    table = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    feedback = torch.CharTensor([1, 0, 1, 0, 0, 1, 1, 0])
    code = GeneralizedConvolutionalEncoder(
        table, feedback=feedback, num_steps=4
    ).with_systematic()
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0]])
    # [(001)000 -> (000)001 -> (011)010 -> (101)101]
    # [(000)001 -> (011)010 -> (101)101 -> (010)011]
    expected_output = torch.FloatTensor(
        [
            [[1, 1, 1], [0, 1, 1], [1, 2, 0], [1, 2, 1]],
            [[0, 1, 1], [1, 2, 0], [1, 2, 1], [0, 0, 0]],
        ]
    )

    assert torch.all(code(msg) == expected_output)


def test_affine_nonsys_rsc_equivalence():
    gen = torch.FloatTensor([[1, 0, 1], [1, 1, 1]])
    bias = torch.FloatTensor([1, 0])
    nonsys_code = AffineConvolutionalEncoder(gen, bias, num_steps=4)
    rsc_code = nonsys_code.to_rsc()
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])

    nonsys_out = nonsys_code(msg)
    rsc_out = rsc_code(nonsys_out[:, :, 0])

    torch.all(nonsys_out == rsc_out)


def test_gen_conv_nonsys_rsc_equivalence():
    table = torch.FloatTensor(
        [[1, 1], [0, 1], [0, 3], [1, 0], [0, 1], [1, 2], [1, 0], [0, 2]]
    )
    nonsys_code = GeneralizedConvolutionalEncoder(table, feedback=None, num_steps=4)
    rsc_code = nonsys_code.to_rsc()
    msg = torch.FloatTensor([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])

    nonsys_out = nonsys_code(msg)
    rsc_out = rsc_code(nonsys_out[:, :, 0])

    torch.all(nonsys_out == rsc_out)


def test_gen_conv_nonsys_rc_conversion_exceptions():
    table_not_bin = torch.FloatTensor(
        [[1, 1], [1, 1], [2, 0], [0, 0], [2, 1], [2, 1], [2, 0], [2, 2]]
    )
    table_not_invert = torch.FloatTensor(
        [[1, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 2]]
    )
    feedback = torch.CharTensor([1, 1, 1, 0, 0, 0, 1, 1])
    table_good = torch.FloatTensor(
        [[1, 1], [0, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 2]]
    )
    feedback = torch.CharTensor([1, 1, 1, 0, 0, 0, 1, 1])

    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalEncoder(
            table_not_bin, feedback=None, num_steps=4
        ).to_rc()
    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalEncoder(
            table_not_invert, feedback=None, num_steps=4
        ).to_rc()
    with pytest.raises(ValueError) as e_info:
        GeneralizedConvolutionalEncoder(
            table_good, feedback=feedback, num_steps=4
        ).to_rc()

    GeneralizedConvolutionalEncoder(table_good, feedback=None, num_steps=4).to_rc()


def test_compare_757_with_commpy():
    manager = DeviceManager(seed=1090)

    msg_length = 100
    torch_code = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]),
        torch.FloatTensor([0, 0]),
        num_steps=msg_length,
    ).to_rsc()
    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    np_msg = msg.to(torch.long).numpy()

    torch_out = torch_code(msg)
    commpy_out = np.apply_along_axis(
        cc.conv_encode, axis=1, arr=np_msg, trellis=commpy_trellis, termination="cont"
    )

    np.testing.assert_array_equal(torch_out.numpy().reshape((100, 200)), commpy_out)


def test_compare_turbo_757_with_commpy():
    manager = DeviceManager(seed=2090)

    msg_length = 100
    torch_base_encoder = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]),
        torch.FloatTensor([0, 0]),
        num_steps=msg_length,
    ).to_rc()
    torch_interleaver = FixedPermuteInterleaver(msg_length, device_manager=manager)
    torch_encoder = StreamedTurboEncoder(
        torch_base_encoder.with_systematic(),
        torch_base_encoder,
        interleaver=torch_interleaver,
        device_manager=manager,
    )

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]), feedback=7)
    commpy_interleaver = interleaver_to_commpy(torch_interleaver)

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    np_msg = msg.numpy().astype(int)

    torch_out = torch_encoder(msg)
    commpy_out = vsystematic_turbo_encode(
        np_msg, commpy_trellis, commpy_trellis, commpy_interleaver
    )

    np.testing.assert_array_equal(torch_out.numpy(), commpy_out)


def test_compare_turbo_755_0_with_commpy():
    manager = DeviceManager(seed=2090)

    msg_length = 100
    torch_base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]),
        torch.FloatTensor([0, 0]),
        num_steps=msg_length,
    )
    torch_base_encoder2 = torch_base_encoder1.get_encoder_channels([1])
    torch_interleaver = FixedPermuteInterleaver(msg_length, device_manager=manager)
    torch_encoder = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver,
        device_manager=manager,
    )

    commpy_trellis = cc.Trellis(np.array([2]), np.array([[7, 5]]))
    commpy_interleaver = interleaver_to_commpy(torch_interleaver)

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    np_msg = msg.numpy().astype(int)

    torch_out = torch_encoder(msg)
    commpy_out = vsystematic_turbo_encode(
        np_msg, commpy_trellis, commpy_trellis, commpy_interleaver
    )

    np.testing.assert_array_equal(torch_out.numpy(), commpy_out)


def test_reset_interleaver_flag_works():
    manager = DeviceManager(seed=2090)

    msg_length = 100
    torch_base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]).to(manager.device),
        torch.FloatTensor([0, 0]).to(manager.device),
        num_steps=msg_length,
        device_manager=manager,
    )
    torch_base_encoder2 = torch_base_encoder1.get_encoder_channels([1])
    torch_interleaver = BatchRandomPermuteInterleaver(
        msg_length, device_manager=manager
    )
    torch_encoder = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    torch_out = torch_encoder(msg)
    torch_out_again = torch_encoder(msg, reset_interleaver=False)
    assert (torch_out == torch_out_again).all()

    torch_out_2 = torch_encoder(msg)
    assert not (torch_out == torch_out_2).all()


def test_turbo_codebook_creation_batch_random_interleaver():
    manager = DeviceManager(seed=2090)

    msg_length = 10
    torch_base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]).to(manager.device),
        torch.FloatTensor([0, 0]).to(manager.device),
        num_steps=msg_length,
        device_manager=manager,
    )
    torch_base_encoder2 = torch_base_encoder1.get_encoder_channels([1])
    torch_interleaver = BatchRandomPermuteInterleaver(
        msg_length, device_manager=manager
    )
    torch_encoder = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver,
        device_manager=manager,
    )

    with pytest.raises(ValueError):
        codebook_encoder = torch_encoder.to_codebook()

    msg = torch.randint(
        0,
        2,
        size=(100, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )
    _ = torch_encoder(msg)
    codebook_encoder = torch_encoder.to_codebook()
    codebook_encoder_again = torch_encoder.to_codebook()
    assert (codebook_encoder.codebook == codebook_encoder_again.codebook).all()
    codebook_encoder_again = None
    _ = torch_encoder(msg)
    codebook_encoder_2 = torch_encoder.to_codebook()
    assert not (codebook_encoder.codebook == codebook_encoder_2.codebook).all()


def test_turbo_batch_dependent_flag():
    manager = DeviceManager(seed=2090)

    msg_length = 10
    torch_base_encoder1 = AffineConvolutionalEncoder(
        torch.FloatTensor([[1, 1, 1], [1, 0, 1]]).to(manager.device),
        torch.FloatTensor([0, 0]).to(manager.device),
        num_steps=msg_length,
        device_manager=manager,
    )
    torch_base_encoder2 = torch_base_encoder1.get_encoder_channels([1])
    torch_interleaver_batch_random = BatchRandomPermuteInterleaver(
        msg_length, device_manager=manager
    )
    torch_encoder_batch_random = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver_batch_random,
        device_manager=manager,
    )
    assert torch_encoder_batch_random.batch_dependent
    torch_interleaver_fixed_batch = FixedPermuteInterleaver(
        msg_length, device_manager=manager
    )
    torch_encoder_batch_fixed = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver_fixed_batch,
        device_manager=manager,
    )
    assert not torch_encoder_batch_fixed.batch_dependent
    torch_interleaver_random_sample = RandomPermuteInterleaver(
        msg_length, device_manager=manager
    )
    torch_encoder_random_sample = StreamedTurboEncoder(
        torch_base_encoder1,
        torch_base_encoder2,
        interleaver=torch_interleaver_random_sample,
        device_manager=manager,
    )
    assert torch_encoder_random_sample.batch_dependent


@pytest.mark.slow
def test_table_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)

    ratio = table / nonsys_code.apply_constraint()
    torch.testing.assert_close(ratio, ratio[0, 0].broadcast_to(ratio.size()))


@pytest.mark.slow
def test_table_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)

    ratio = table / nonsys_code.apply_constraint()
    torch.testing.assert_close(ratio, ratio[0, 0].broadcast_to(ratio.size()))


@pytest.mark.slow
def test_table_multi_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )

    ratio = table / nonsys_code.apply_constraint()
    for i in range(table.shape[1]):
        torch.testing.assert_close(
            ratio[:, i], ratio[0, i].broadcast_to(ratio[:, i].size())
        )
    assert not torch.allclose(
        ratio[0, 1:], ratio[0, 0].broadcast_to(ratio[0, 1:].size())
    )


@pytest.mark.slow
def test_table_multi_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )

    ratio = table / nonsys_code.apply_constraint()
    for i in range(table.shape[1]):
        torch.testing.assert_close(
            ratio[:, i], ratio[0, i].broadcast_to(ratio[:, i].size())
        )
    assert not torch.allclose(
        ratio[0, 1:], ratio[0, 0].broadcast_to(ratio[0, 1:].size())
    )


@pytest.mark.slow
def test_table_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_table_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_table_multi_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_table_multi_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    nonsys_code = GeneralizedConvolutionalEncoder(
        table,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_fc_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.# Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)

    ratio = (
        fourier_to_table(fc, device_manager=manager) / nonsys_code.apply_constraint()
    )
    torch.testing.assert_close(
        ratio, ratio[0, 0].broadcast_to(ratio.size()), atol=1e-4, rtol=1.3e-6
    )


@pytest.mark.slow
def test_fc_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)

    ratio = (
        fourier_to_table(fc, device_manager=manager) / nonsys_code.apply_constraint()
    )
    torch.testing.assert_close(
        ratio, ratio[0, 0].broadcast_to(ratio.size()), atol=1e-4, rtol=1.3e-6
    )


@pytest.mark.slow
def test_fc_multi_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )

    ratio = (
        fourier_to_table(fc, device_manager=manager) / nonsys_code.apply_constraint()
    )
    for i in range(fc.shape[1]):
        torch.testing.assert_close(
            ratio[:, i],
            ratio[0, i].broadcast_to(ratio[:, i].size()),
            atol=1e-4,
            rtol=1.3e-6,
        )
    assert not torch.allclose(
        ratio[0, 1:],
        ratio[0, 0].broadcast_to(ratio[0, 1:].size()),
        atol=1e-4,
        rtol=1.3e-6,
    )


@pytest.mark.slow
def test_fc_multi_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )

    ratio = (
        fourier_to_table(fc, device_manager=manager) / nonsys_code.apply_constraint()
    )
    for i in range(fc.shape[1]):
        torch.testing.assert_close(
            ratio[:, i],
            ratio[0, i].broadcast_to(ratio[:, i].size()),
            atol=1e-4,
            rtol=1.3e-6,
        )
    assert not torch.allclose(
        ratio[0, 1:],
        ratio[0, 0].broadcast_to(ratio[0, 1:].size()),
        atol=1e-4,
        rtol=1.3e-6,
    )


@pytest.mark.slow
def test_fc_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_fc_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_fc_multi_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_fc_multi_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    nonsys_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc,
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_table_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_multi_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_table_multi_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_table_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, table.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_opt_unit_power_delay_short_message():
    manager = DeviceManager(seed=1412)
    msg_length = 5
    nonsys_code = turboae_cont_exact_nobd(
        num_steps=msg_length,
        device_manager=manager,
        delay=4,
        interleaver="fixed",
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(100000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    ).item()
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    ).item()

    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, nonsys_code.num_output_channels):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_opt_unit_power_delay_long_message():
    manager = DeviceManager(seed=2090)
    msg_length = 100
    nonsys_code = turboae_cont_exact_nobd(
        num_steps=msg_length,
        device_manager=manager,
        delay=4,
        interleaver="turboae",
        constraint="opt_unit_power",
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    ).item()
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    ).item()

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, nonsys_code.num_output_channels):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_table_multi_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 5
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_table_multi_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    table = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[15.0, 13.0, 0.0]])
    msg_length = 100
    noninterleaved_code = GeneralizedConvolutionalEncoder(
        table[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaved_code = GeneralizedConvolutionalEncoder(
        table[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(table.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_fc_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.# Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_fc_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_fc_multi_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_fc_multi_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    assert not torch.allclose(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        assert not torch.allclose(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_fc_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_fc_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    center0 = torch.mean(encoded[:, :, 0])
    power0 = torch.mean(encoded[:, :, 0] ** 2)
    for i in range(1, fc.shape[1]):
        assert not torch.allclose(torch.mean(encoded[:, :, i]), center0)
        assert not torch.allclose(torch.mean(encoded[:, :, i] ** 2), power0)


@pytest.mark.slow
def test_turbo_fc_multi_opt_unit_power_short_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 5
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(10000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )


@pytest.mark.slow
def test_turbo_fc_multi_opt_unit_power_long_message():
    manager = DeviceManager(seed=2090)
    # Computation of a function using fourier coefficients
    # is highly sensitive to floating point error. They should
    # be kept fairly close to 0 to avoid this.
    fc = torch.randn(
        (32, 3), generator=manager.generator, device=manager.device
    ) + torch.FloatTensor([[0.4, -0.5, 0.1]])
    msg_length = 100
    noninterleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, :2],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaved_code = FourierConvolutionalEncoder(
        fourier_coefficients=fc[:, 2:],
        feedback=None,
        num_steps=msg_length,
        device_manager=manager,
        constraint="multi_opt_unit_power",
    )
    interleaver = FixedPermuteInterleaver(input_size=msg_length, device_manager=manager)
    nonsys_code = StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_code,
        interleaved_encoder=interleaved_code,
        interleaver=interleaver,
        device_manager=manager,
    )

    msg = torch.randint(
        0,
        2,
        size=(1000000, msg_length),
        dtype=torch.float,
        generator=manager.generator,
        device=manager.device,
    )

    encoded = nonsys_code(msg)
    power = torch.mean(encoded**2)
    power_err = torch.mean(
        2 * torch.std(encoded**2, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )
    center = torch.mean(encoded)
    center_err = torch.mean(
        2 * torch.std(encoded, dim=0, unbiased=True) / math.sqrt(msg.shape[0])
    )

    torch.testing.assert_close(power, torch.tensor(1.0), atol=power_err, rtol=0)
    torch.testing.assert_close(center, torch.tensor(0.0), atol=center_err, rtol=0)
    for i in range(fc.shape[1]):
        power_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i] ** 2, dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        center_i_err = torch.mean(
            2
            * torch.std(encoded[:, :, i], dim=0, unbiased=True)
            / math.sqrt(msg.shape[0])
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i] ** 2),
            torch.tensor(1.0),
            atol=power_i_err,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.mean(encoded[:, :, i]), torch.tensor(0.0), atol=center_i_err, rtol=0
        )
