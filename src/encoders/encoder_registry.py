from pathlib import Path
from pprint import pprint
import torch

from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER
from ..constants import (
    TURBOAE_EXACT_TABLE1_BITS_3_98,
    TURBOAE_EXACT_TABLE2_BITS_3_98,
    TURBOAE_ENCODER_CONT_PATH,
    MODELS_DIR,
    TURBOAE_ENCODER_CONT_MEAN,
    TURBOAE_ENCODER_CONT_STD,
)
from ..interleavers import (
    FixedPermuteInterleaver,
    RandomPermuteInterleaver,
    TurboAEInterleaver,
    Interleaver,
)

from .convolutional_encoder import (
    AffineConvolutionalEncoder,
    GeneralizedConvolutionalEncoder,
    TrellisEncoder,
)
from .turbo import StreamedTurboEncoder
from .turboae import ENC_interCNN


def load_interleaver(
    interleaver, num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER
):
    if interleaver == "turboae":
        interleaver = (TurboAEInterleaver(device_manager=device_manager),)
        assert num_steps == len(interleaver)
    elif interleaver == "batch_random":
        interleaver = RandomPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )
    elif interleaver is None:
        interleaver = FixedPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )

    return interleaver


def is_systematic(name: str):
    if name in _SYSTEMATIC_CODES:
        return True
    else:
        print("Could not find code in systematic set, falling back to name parsing.")
        return name.split("_")[1][0] == "1" or name == conv_identity.__name__


def conv_75_1_00(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
) -> AffineConvolutionalEncoder:
    return AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 1]]),
        torch.CharTensor([0, 0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_15_7_00(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return conv_75_1_00(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    ).to_rsc()


# Random 1
def conv_56_1_11(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[1, 0, 1], [1, 1, 0]]),
        torch.CharTensor([1, 1]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_16_5_11(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return conv_56_1_11(num_steps, device_manager=device_manager, delay=delay).to_rsc()


# Random 2
def conv_15_1_10(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[0, 0, 1], [1, 0, 1]]),
        torch.CharTensor([1, 0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


# Random 3
def conv_73_1_10(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [0, 1, 1]]),
        torch.CharTensor([1, 0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_13_7_10(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return conv_73_1_10(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    ).to_rsc()


# Random 4
def conv_30_1_10(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[0, 1, 1], [0, 0, 0]]),
        torch.CharTensor([1, 0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_10_3_10(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return conv_30_1_10(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    ).to_rsc()


# Random 5
def conv_74_1_01(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[1, 1, 1], [1, 0, 0]]),
        torch.CharTensor([1, 0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_14_7_01(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return conv_74_1_01(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    ).to_rsc()


def conv_identity(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    return AffineConvolutionalEncoder(
        torch.CharTensor([[0, 1]]),
        torch.CharTensor([0]),
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def convert_to_turbo_2_stream(
    noninterleaved_encoder, interleaver, device_manager: DeviceManager
):
    assert noninterleaved_encoder.num_output_channels == 2
    interleaved_encoder = noninterleaved_encoder.get_encoder_channels([-1])
    return StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
        device_manager=device_manager,
    )


def convert_to_turbo_3_stream(
    encoder: TrellisEncoder, interleaver, device_manager: DeviceManager
):
    assert encoder.num_output_channels == 3
    interleaved_encoder = encoder.get_encoder_channels([-1])
    noninterleaved_encoder = encoder.get_encoder_channels([0, 1])
    return StreamedTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
        device_manager=device_manager,
    )


def turbo_755_1_00(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    if interleaver == "turboae":
        interleaver = (TurboAEInterleaver(device_manager=device_manager),)
        assert num_steps == len(interleaver)
    elif interleaver == "batch_random":
        interleaver = RandomPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )
    elif interleaver is None:
        interleaver = FixedPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )

    noninterleaved_encoder = conv_75_1_00(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_2_stream(
        noninterleaved_encoder, interleaver, device_manager=device_manager
    )


def turbo_155_7_00(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[GeneralizedConvolutionalEncoder]:
    if interleaver == "turboae":
        interleaver = (TurboAEInterleaver(device_manager=device_manager),)
        assert num_steps == len(interleaver)
    elif interleaver == "batch_random":
        interleaver = RandomPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )
    elif interleaver is None:
        interleaver = FixedPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )

    noninterleaved_encoder = conv_15_7_00(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_2_stream(
        noninterleaved_encoder, interleaver, device_manager=device_manager
    )


# RANDOM_1 = AffineTurboSpec(
#     weights=tf.constant(
#         [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1]], dtype=tf.int32
#     ),
#     bias=tf.constant([1, 0, 1], dtype=tf.int32),
# )
# RANDOM_2 = AffineTurboSpec(
#     weights=tf.constant(
#         [[1, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]], dtype=tf.int32
#     ),
#     bias=tf.constant([0, 1, 0], dtype=tf.int32),
# )
# RANDOM_3 = AffineTurboSpec(
#     weights=tf.constant(
#         [[1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=tf.int32
#     ),
#     bias=tf.constant([0, 1, 0], dtype=tf.int32),
# )
# RANDOM_4 = AffineTurboSpec(
#     weights=tf.constant(
#         [[0, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 1]], dtype=tf.int32
#     ),
#     bias=tf.constant([0, 0, 0], dtype=tf.int32),
# )
# RANDOM_5 = AffineTurboSpec(
#     weights=tf.constant(
#         [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=tf.int32
#     ),
#     bias=tf.constant([1, 0, 0], dtype=tf.int32),
# )


def conv_random1_nonsys(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 1], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_random2_nonsys(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([0, 1, 0], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_random3_nonsys(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([0, 1, 0], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_random4_nonsys(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[0, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([0, 0, 0], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def conv_random5_nonsys(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 0], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def turbo_random1_nonsys(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = conv_random1_nonsys(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def turbo_random2_nonsys(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = conv_random2_nonsys(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def turbo_random3_nonsys(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = conv_random3_nonsys(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def turbo_random4_nonsys(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = conv_random4_nonsys(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def turbo_random5_nonsys(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = conv_random5_nonsys(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def tae_conv_approximated_nonsys1(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [0, 1, 0, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 0], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def tae_conv_approximated_nonsys2(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 1], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def tae_conv_approximated_nonsys3(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 0, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 1], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def tae_conv_approximated_nonsys4(
    num_steps: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER, delay=0
):
    generator = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]],
        dtype=torch.int8,
        device=device_manager.device,
    )
    bias = torch.tensor([1, 0, 1], dtype=torch.int8, device=device_manager.device)
    return AffineConvolutionalEncoder(
        generator=generator,
        bias=bias,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )


def tae_turbo_approximated_nonsys1(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = tae_conv_approximated_nonsys1(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def tae_turbo_approximated_nonsys2(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = tae_conv_approximated_nonsys2(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def tae_turbo_approximated_nonsys3(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = tae_conv_approximated_nonsys3(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def tae_turbo_approximated_nonsys4(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[AffineConvolutionalEncoder]:
    interleaver = load_interleaver(
        interleaver=interleaver, num_steps=num_steps, device_manager=device_manager
    )
    encoder = tae_conv_approximated_nonsys4(
        num_steps=num_steps, device_manager=device_manager, delay=delay
    )
    return convert_to_turbo_3_stream(
        encoder, interleaver, device_manager=device_manager
    )


def turboae_binary_exact_nobd(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
) -> StreamedTurboEncoder[GeneralizedConvolutionalEncoder]:
    if interleaver == "turboae":
        interleaver = (TurboAEInterleaver(device_manager=device_manager),)
        assert num_steps == len(interleaver)
    elif interleaver == "batch_random":
        interleaver = RandomPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )
    else:
        interleaver = FixedPermuteInterleaver(
            input_size=num_steps, device_manager=device_manager
        )

    code1 = GeneralizedConvolutionalEncoder(
        table=TURBOAE_EXACT_TABLE1_BITS_3_98,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )
    code2 = GeneralizedConvolutionalEncoder(
        table=TURBOAE_EXACT_TABLE2_BITS_3_98,
        num_steps=num_steps,
        delay=delay,
        device_manager=device_manager,
    )

    return StreamedTurboEncoder(
        noninterleaved_encoder=code1,
        interleaved_encoder=code2,
        interleaver=interleaver,
        device_manager=device_manager,
    )


def turboae_cont_exact_nn(
    interleaver: Interleaver,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    precomputed_stats=True,
):
    state_dict = torch.load(
        TURBOAE_ENCODER_CONT_PATH, map_location=device_manager.device
    )
    base_cnn = ENC_interCNN(
        enc_num_layer=state_dict["enc_num_layer"].item(),
        enc_num_unit=state_dict["enc_num_unit"].item(),
        enc_kernel_size=state_dict["enc_kernel_size"].item(),
        interleaver=interleaver,
        device_manager=device_manager,
    )
    base_cnn.pre_initialize(state_dict)
    if precomputed_stats:
        base_cnn.mean = TURBOAE_ENCODER_CONT_MEAN
        base_cnn.std = TURBOAE_ENCODER_CONT_STD

    return base_cnn


def turboae_cont_exact_nobd(
    interleaver: Interleaver,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    constraint="opt_unit_power",
    **kwargs,
):
    print(f"Unused arguments to {turboae_cont_exact_nobd.__name__}: {kwargs}")
    if constraint != "opt_unit_power":
        constrain = False
        raise NotImplementedError()
    else:
        constrain = True

    base_cnn = turboae_cont_exact_nn(
        device_manager=device_manager,
        interleaver=interleaver,
    )
    encoder = base_cnn.to_conv_code(constrain=constrain, no_delay=True)
    del base_cnn
    return encoder


def trained_encoder(
    num_steps: int,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    delay=0,
    interleaver=None,
    constraint="opt_unit_power",
):
    assert delay == 0
    encoder_path = (
        MODELS_DIR
        / "train_turbo_table_block_len_16_7f7a61dd4c93cfaeb103d0093fcc98f5959ca30b.pt"
    )
    encoder_state_dict = torch.load(encoder_path, map_location=device_manager.device)
    table_noninterleaved = encoder_state_dict["noninterleaved_encoder.table"].to(
        device_manager.device
    )
    table_interleaved = encoder_state_dict["interleaved_encoder.table"].to(
        device_manager.device
    )
    noninterleaved_encoder = GeneralizedConvolutionalEncoder(
        num_steps=num_steps,
        table=table_noninterleaved,
        constraint=constraint,
        device_manager=device_manager,
    )
    interleaved_encoder = GeneralizedConvolutionalEncoder(
        num_steps=num_steps,
        table=table_interleaved,
        constraint=constraint,
        device_manager=device_manager,
    )
    noninterleaved_encoder.apply_constraint_()
    noninterleaved_encoder.update()
    interleaved_encoder.apply_constraint_()
    interleaved_encoder.update()
    return StreamedTurboEncoder[GeneralizedConvolutionalEncoder](
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
        device_manager=device_manager,
    )


def get_info_from_checkpoint(filepath: Path, device_manager: DeviceManager):
    state_dict = torch.load(filepath, map_location=device_manager.device)
    interleaver_permutation = state_dict["encoder.interleaver.permutation"]
    block_len = len(interleaver_permutation)
    interleaver = FixedPermuteInterleaver(
        input_size=block_len,
        permutation=interleaver_permutation,
        device_manager=device_manager,
    )

    # Encoder settings
    enc_num_layer = state_dict["encoder.enc_num_layer"].item()
    enc_num_unit = state_dict["encoder.enc_num_unit"].item()
    enc_kernel_size = state_dict["encoder.enc_kernel_size"].item()

    return {
        "interleaver": interleaver,
        "enc_num_layer": enc_num_layer,
        "enc_num_unit": enc_num_unit,
        "enc_kernel_size": enc_kernel_size,
        "block_len": block_len,
    }


_ENCODER_FACTORIES = {
    # Convcodes
    conv_15_7_00.__name__: conv_15_7_00,
    conv_75_1_00.__name__: conv_75_1_00,
    conv_56_1_11.__name__: conv_56_1_11,
    conv_16_5_11.__name__: conv_16_5_11,
    conv_15_1_10.__name__: conv_15_1_10,
    conv_73_1_10.__name__: conv_73_1_10,
    conv_13_7_10.__name__: conv_13_7_10,
    conv_30_1_10.__name__: conv_30_1_10,
    conv_10_3_10.__name__: conv_10_3_10,
    conv_74_1_01.__name__: conv_74_1_01,
    conv_14_7_01.__name__: conv_14_7_01,
    conv_identity.__name__: conv_identity,
    turboae_binary_exact_nobd.__name__: turboae_binary_exact_nobd,
    turboae_cont_exact_nobd.__name__: turboae_cont_exact_nobd,
    turboae_cont_exact_nn.__name__: turboae_cont_exact_nn,
    trained_encoder.__name__: trained_encoder,
    turbo_755_1_00.__name__: turbo_755_1_00,
    turbo_155_7_00.__name__: turbo_155_7_00,
    turbo_random1_nonsys.__name__: turbo_random1_nonsys,
    turbo_random2_nonsys.__name__: turbo_random2_nonsys,
    turbo_random3_nonsys.__name__: turbo_random3_nonsys,
    turbo_random4_nonsys.__name__: turbo_random4_nonsys,
    turbo_random5_nonsys.__name__: turbo_random5_nonsys,
    conv_random1_nonsys.__name__: conv_random1_nonsys,
    conv_random2_nonsys.__name__: conv_random2_nonsys,
    conv_random3_nonsys.__name__: conv_random3_nonsys,
    conv_random4_nonsys.__name__: conv_random4_nonsys,
    conv_random5_nonsys.__name__: conv_random5_nonsys,
    tae_conv_approximated_nonsys1.__name__: tae_conv_approximated_nonsys1,
    tae_conv_approximated_nonsys2.__name__: tae_conv_approximated_nonsys2,
    tae_conv_approximated_nonsys3.__name__: tae_conv_approximated_nonsys3,
    tae_conv_approximated_nonsys4.__name__: tae_conv_approximated_nonsys4,
    tae_turbo_approximated_nonsys1.__name__: tae_turbo_approximated_nonsys1,
    tae_turbo_approximated_nonsys2.__name__: tae_turbo_approximated_nonsys2,
    tae_turbo_approximated_nonsys3.__name__: tae_turbo_approximated_nonsys3,
    tae_turbo_approximated_nonsys4.__name__: tae_turbo_approximated_nonsys4,
}
_SYSTEMATIC_CODES = {
    conv_15_7_00.__name__,
    conv_16_5_11.__name__,
    conv_15_1_10.__name__,
    conv_13_7_10.__name__,
    conv_10_3_10.__name__,
    conv_14_7_01.__name__,
    conv_identity.__name__,
    turbo_155_7_00.__name__,
}


def get_encoder(name: str):
    return _ENCODER_FACTORIES[name]
