import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neural_utils import *


def test_create():
    num_layer = 3
    in_channels = 10
    out_channels = 20
    kernel_size = 3

    layer = SameShapeConv1d(
        num_layer=num_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )

    assert layer.num_layer == num_layer
    assert not layer.no_act
    assert layer.kernel_size == kernel_size
    assert not layer.first_pad
    assert not layer.front_pad

    assert len(layer.cnns) == num_layer
    for i, cnn in enumerate(layer.cnns):
        if i != 0:
            cnn.in_channels == out_channels
        else:
            cnn.in_channels == in_channels
        cnn.out_channels == out_channels
        cnn.kernel_size == kernel_size
    assert layer.first_padding_layer is None
    assert len(layer.padding_layers) == num_layer
    for padding in layer.padding_layers:
        pad = (kernel_size - 1) // 2
        assert padding.padding == (pad, pad)
    assert layer.activation == F.elu


def test_forward_basic():
    num_layer = 2
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    layer = SameShapeConv1d(
        num_layer=num_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )

    compare_cnn_1 = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="same",
    )
    compare_cnn_1.weight = nn.Parameter(layer.cnns[0].weight.clone())
    compare_cnn_1.bias = nn.Parameter(layer.cnns[0].bias.clone())

    compare_cnn_2 = nn.Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="same",
    )
    compare_cnn_2.weight = nn.Parameter(layer.cnns[1].weight.clone())
    compare_cnn_2.bias = nn.Parameter(layer.cnns[1].bias.clone())

    test_input = torch.randn((3, 2, 1))

    expected_output = F.elu(
        compare_cnn_2(F.elu(compare_cnn_1(test_input.transpose(1, 2))))
    ).transpose(1, 2)
    actual_output = layer(test_input)

    assert torch.allclose(actual_output, expected_output)


def test_forward_first_pad():
    num_layer = 2
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    layer = SameShapeConv1d(
        num_layer=num_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        first_pad=True,
    )
    assert layer.first_pad

    compare_cnn_1 = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=2,
    )
    compare_cnn_1.weight = nn.Parameter(layer.cnns[0].weight.clone())
    compare_cnn_1.bias = nn.Parameter(layer.cnns[0].bias.clone())

    compare_cnn_2 = nn.Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="valid",
    )
    compare_cnn_2.weight = nn.Parameter(layer.cnns[1].weight.clone())
    compare_cnn_2.bias = nn.Parameter(layer.cnns[1].bias.clone())

    test_input = torch.randn((3, 2, 1))

    expected_output = F.elu(
        compare_cnn_2(F.elu(compare_cnn_1(test_input.transpose(1, 2))))
    ).transpose(1, 2)
    actual_output = layer(test_input)

    assert torch.allclose(actual_output, expected_output)


def test_forward_front_pad():
    num_layer = 2
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    layer = SameShapeConv1d(
        num_layer=num_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        front_pad=True,
    )
    assert layer.front_pad

    padder = nn.ConstantPad1d((2, 0), 0.0)
    compare_cnn_1 = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="valid",
    )
    compare_cnn_1.weight = nn.Parameter(layer.cnns[0].weight.clone())
    compare_cnn_1.bias = nn.Parameter(layer.cnns[0].bias.clone())

    compare_cnn_2 = nn.Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="valid",
    )
    compare_cnn_2.weight = nn.Parameter(layer.cnns[1].weight.clone())
    compare_cnn_2.bias = nn.Parameter(layer.cnns[1].bias.clone())

    test_input = torch.randn((3, 2, 1))
    assert torch.all(padder(test_input.transpose(1, 2))[:, :, :2] == 0.0)

    expected_output = F.elu(
        compare_cnn_2(padder(F.elu(compare_cnn_1(padder(test_input.transpose(1, 2))))))
    ).transpose(1, 2)
    actual_output = layer(test_input)

    assert torch.allclose(actual_output, expected_output)


def test_forward_first_front_pad():
    num_layer = 2
    in_channels = 1
    out_channels = 2
    kernel_size = 3
    layer = SameShapeConv1d(
        num_layer=num_layer,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        front_pad=True,
        first_pad=True,
    )
    assert layer.front_pad
    assert layer.first_pad

    padder = nn.ConstantPad1d((4, 0), 0.0)
    compare_cnn_1 = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="valid",
    )
    compare_cnn_1.weight = nn.Parameter(layer.cnns[0].weight.clone())
    compare_cnn_1.bias = nn.Parameter(layer.cnns[0].bias.clone())

    compare_cnn_2 = nn.Conv1d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="valid",
    )
    compare_cnn_2.weight = nn.Parameter(layer.cnns[1].weight.clone())
    compare_cnn_2.bias = nn.Parameter(layer.cnns[1].bias.clone())

    test_input = torch.randn((3, 2, 1))
    assert torch.all(padder(test_input.transpose(1, 2))[:, :, :2] == 0.0)

    expected_output = F.elu(
        compare_cnn_2(F.elu(compare_cnn_1(padder(test_input.transpose(1, 2)))))
    ).transpose(1, 2)
    actual_output = layer(test_input)

    assert torch.allclose(actual_output, expected_output)
