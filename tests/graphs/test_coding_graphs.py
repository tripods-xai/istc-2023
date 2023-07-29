from src.encoders import (
    conv_75_1_00,
    turboae_binary_exact_nobd,
    conv_15_7_00,
    conv_16_5_11,
)
from src.utils import DeviceManager

from ..utils import test_manager
from src.graphs import *


def test_infer_dependencies_nonrecursive_conv_code():
    encoder = conv_75_1_00(10, device_manager=test_manager)
    generator = encoder.generator
    inferred_generator = infer_depencies(encoder.table)

    assert torch.all(inferred_generator == generator)


def test_infer_dependencies_tae_binary_exact():
    encoder = turboae_binary_exact_nobd(10, device_manager=test_manager)
    noninterleaved_inferred_generator = infer_depencies(
        encoder.noninterleaved_encoder.table
    )
    interleaved_inferred_generator = infer_depencies(encoder.interleaved_encoder.table)

    expected_noninterleaved = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.int8, device=test_manager.device
    )
    expected_interleaved = torch.tensor(
        [[1, 1, 1, 1, 1]], dtype=torch.int8, device=test_manager.device
    )
    assert torch.all(noninterleaved_inferred_generator == expected_noninterleaved)
    assert torch.all(interleaved_inferred_generator == expected_interleaved)


def test_infer_dependencies_recursive_conv_code_feedback():
    encoder = conv_15_7_00(10, device_manager=test_manager)
    feedback_deps = torch.tensor([[1, 1, 1]])
    inferred_feedback_deps = infer_depencies(encoder.feedback[:, None])

    assert torch.all(inferred_feedback_deps == feedback_deps)

    encoder = conv_16_5_11(10, device_manager=test_manager)
    feedback_deps = torch.tensor([[1, 0, 1]])
    inferred_feedback_deps = infer_depencies(encoder.feedback[:, None])

    assert torch.all(inferred_feedback_deps == feedback_deps)
