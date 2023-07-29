import torch

from src.interleavers import TurboAEInterleaver
from src.encoders.turboae import *

from ..utils import test_manager


def test_table_forward():
    encoder = ENC_interCNN(
        enc_num_layer=2,
        enc_num_unit=100,
        enc_kernel_size=3,
        interleaver=TurboAEInterleaver(device_manager=test_manager),
        first_pad=True,
        front_pad=False,
        device_manager=test_manager,
    )
    block_len = 100
    batch_size = 1

    test_input = torch.randint(
        low=0,
        high=2,
        size=(batch_size, block_len),
        generator=test_manager.generator,
        device=test_manager.device,
    )
    # test_input = torch.zeros((batch_size, block_len), device=test_manager.device)
    # test_input[:, 0] = 1.0

    without_table = encoder(test_input, table=False)
    with_table = encoder(test_input, table=True, constrain=False)

    assert torch.allclose(without_table, with_table, atol=1e-5, rtol=0)

    # This shouldn't fail
    scalar = torch.var(with_table)
    scalar.backward()


def test_table_normalize():
    encoder = ENC_interCNN(
        enc_num_layer=2,
        enc_num_unit=100,
        enc_kernel_size=3,
        interleaver=TurboAEInterleaver(device_manager=test_manager),
        first_pad=True,
        front_pad=False,
        device_manager=test_manager,
    )
    block_len = 100
    batch_size = 100000
    with torch.no_grad():
        test_input = torch.randint(
            low=0,
            high=2,
            size=(batch_size, block_len),
            generator=test_manager.generator,
            device=test_manager.device,
        )
        # test_input = torch.zeros((batch_size, block_len), device=test_manager.device)
        # test_input[:, 0] = 1.0

        with_table = encoder(test_input, table=True, constrain=True)

        assert torch.allclose(
            torch.mean(with_table), torch.tensor(0.0), atol=1e-4, rtol=0
        )

        assert torch.allclose(
            torch.var(with_table), torch.tensor(1.0), atol=1e-4, rtol=0
        )
