import torch

from src.utils import DeviceManager, enumerate_binary_inputs
from src.fourier import *
from .raj_fourier import fourier as raj_fourier


def test_fourier_conversion_basic():
    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    actual_fc = compute_fourier_coefficients(f=torch_max, window=2)
    expected_fc = torch.FloatTensor([0.5, 0.5, 0.5, -0.5])

    torch.testing.assert_close(actual=actual_fc, expected=expected_fc)


def test_fourier_conversion_multioutput():
    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    def torch_min(inputs):
        return torch.min(1 - 2 * inputs, dim=-1).values

    def torch_max_min(inputs):
        return torch.stack(
            [
                torch_max(inputs),
                torch_min(inputs),
            ],
            dim=-1,
        )

    actual_fc = compute_fourier_coefficients(f=torch_max_min, window=2)
    expected_fc = torch.stack(
        [
            raj_fourier(lambda x: torch.max(x, dim=-1).values, N=2),
            raj_fourier(lambda x: torch.min(x, dim=-1).values, N=2),
        ],
        dim=-1,
    )

    torch.testing.assert_close(actual=actual_fc, expected=expected_fc)


def test_fourier_conversion_from_table():
    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    def torch_min(inputs):
        return torch.min(1 - 2 * inputs, dim=-1).values

    def torch_max_min(inputs):
        return torch.stack(
            [
                torch_max(inputs),
                torch_min(inputs),
            ],
            dim=-1,
        )

    table = torch_max_min(enumerate_binary_inputs(window=5, dtype=torch.float))

    actual_fc = table_to_fourier(table=table)
    expected_fc = torch.stack(
        [
            raj_fourier(lambda x: torch.max(x, dim=-1).values, N=5),
            raj_fourier(lambda x: torch.min(x, dim=-1).values, N=5),
        ],
        dim=-1,
    )

    torch.testing.assert_close(actual=actual_fc, expected=expected_fc)


def test_inverse_fourier_basic():
    manager = DeviceManager(seed=1312)

    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    fc = compute_fourier_coefficients(f=torch_max, window=2, device_manager=manager)
    input_data = torch.randint(
        0, 2, size=(13, 2), dtype=torch.float, generator=manager.generator
    )
    output_data = compute_fourier_function(fc, input_data, device_manager=manager)

    torch.testing.assert_close(actual=output_data, expected=torch_max(input_data))


def test_inverse_fourier_multioutput():
    manager = DeviceManager(seed=1312)

    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    def torch_min(inputs):
        return torch.min(1 - 2 * inputs, dim=-1).values

    def torch_max_min(inputs):
        return torch.stack(
            [
                torch_max(inputs),
                torch_min(inputs),
            ],
            dim=-1,
        )

    fc = compute_fourier_coefficients(f=torch_max_min, window=2, device_manager=manager)
    input_data = torch.randint(
        0, 2, size=(13, 2), dtype=torch.float, generator=manager.generator
    )
    output_data = compute_fourier_function(fc, input_data, device_manager=manager)

    torch.testing.assert_close(actual=output_data, expected=torch_max_min(input_data))


def test_fourier_to_table():
    manager = DeviceManager(seed=1312)

    def torch_max(inputs):
        return torch.max(1 - 2 * inputs, dim=-1).values

    def torch_min(inputs):
        return torch.min(1 - 2 * inputs, dim=-1).values

    def torch_max_min(inputs):
        return torch.stack(
            [
                torch_max(inputs),
                torch_min(inputs),
            ],
            dim=-1,
        )

    fc = compute_fourier_coefficients(f=torch_max_min, window=5, device_manager=manager)
    actual_table = fourier_to_table(fc, device_manager=manager)

    torch.testing.assert_close(
        actual=actual_table,
        expected=torch_max_min(
            enumerate_binary_inputs(window=5, device=manager.device, dtype=torch.float)
        ),
    )
