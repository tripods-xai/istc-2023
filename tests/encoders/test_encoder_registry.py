from src.utils import DeviceManager
from src.interleavers import FixedPermuteInterleaver

from src.encoders import *

# Random Generators
# [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1]]
# [[0, 1, 1, 0, 1]]
# [[1, 0, 0, 0, 1], [0, 0, 1, 1, 1]]
# [[1, 0, 0, 1, 1]]
# [[1, 0, 1, 0, 1], [1, 0, 0, 1, 1]]
# [[1, 0, 0, 1, 1]]
# [[0, 1, 0, 1, 1], [1, 1, 1, 0, 1]]
# [[0, 0, 0, 1, 1]]
# [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]]
# [[0, 0, 1, 1, 1]]

# Random biases
# [1, 0]
# [1]
# [0, 1]
# [0]
# [0, 1]
# [0]
# [0, 0]
# [0]
# [1, 0]
# [0]


def test_turbo_random_nonsys_no_delay():
    manager = DeviceManager(no_cuda=True, seed=1235)
    num_steps = 100
    batch_size = 1000
    delay = 0

    interleaver = FixedPermuteInterleaver(num_steps, device_manager=manager)

    random1 = turbo_random1_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random2 = turbo_random2_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random3 = turbo_random3_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random4 = turbo_random4_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random5 = turbo_random5_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )

    random1_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([1, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 1, 0, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random2_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 0, 1], [0, 0, 1, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random3_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 1, 0, 1], [1, 0, 0, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random4_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 0, 1, 1], [1, 1, 1, 0, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random5_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([1, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 0, 1, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )

    pairs = [
        (random1, random1_expected),
        (random2, random2_expected),
        (random3, random3_expected),
        (random4, random4_expected),
        (random5, random5_expected),
    ]

    for actual, expected in pairs:
        samples = torch.randint(
            0, 2, size=(batch_size, num_steps), device=manager.device, dtype=torch.int8
        )
        result_actual = actual(samples, dtype=torch.int8)
        result_expected = expected(samples, dtype=torch.int8)
        assert torch.all(result_actual == result_expected)


def test_turbo_random_nonsys_with_delay():
    manager = DeviceManager(no_cuda=True, seed=1235)
    num_steps = 100
    batch_size = 1000
    delay = 2

    interleaver = FixedPermuteInterleaver(num_steps, device_manager=manager)

    random1 = turbo_random1_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random2 = turbo_random2_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random3 = turbo_random3_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random4 = turbo_random4_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    random5 = turbo_random5_nonsys(
        num_steps=num_steps,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )

    random1_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 0, 0, 1], [0, 1, 1, 0, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([1, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 1, 0, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random2_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 0, 1], [0, 0, 1, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random3_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 1, 0, 1], [1, 0, 0, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 1], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random4_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 1, 0, 1, 1], [1, 1, 1, 0, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([0, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 0, 0, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )
    random5_expected = StreamedTurboEncoder(
        noninterleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]],
                dtype=torch.int8,
                device=manager.device,
            ),
            bias=torch.tensor([1, 0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaved_encoder=AffineConvolutionalEncoder(
            generator=torch.tensor(
                [[0, 0, 1, 1, 1]], dtype=torch.int8, device=manager.device
            ),
            bias=torch.tensor([0], dtype=torch.int8, device=manager.device),
            num_steps=num_steps,
            delay=delay,
            device_manager=manager,
        ),
        interleaver=interleaver,
        device_manager=manager,
    )

    pairs = [
        (random1, random1_expected),
        (random2, random2_expected),
        (random3, random3_expected),
        (random4, random4_expected),
        (random5, random5_expected),
    ]

    for actual, expected in pairs:
        samples = torch.randint(
            0, 2, size=(batch_size, num_steps), device=manager.device, dtype=torch.int8
        )
        result_actual = actual(samples, dtype=torch.int8)
        result_expected = expected(samples, dtype=torch.int8)
        assert torch.all(result_actual == result_expected)
