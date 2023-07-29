import torch

from src.decoders import TurboAEDecoder
from src.constants import TURBOAE_DECODER_BINARY_PATH

from ..utils import test_manager

"""
For comparing against the original TurboAE code, see dev/turboae. I have modified
the original code to load the model below and test that the same inputs produce
the same outputs.
"""


def test_turboae_decoder_construct():
    # FIXME: This test is out of date
    manager = test_manager
    state_dict = torch.load(TURBOAE_DECODER_BINARY_PATH, map_location=manager.device)
    from pprint import pprint

    pprint(state_dict.keys())

    decoder = TurboAEDecoder(state_dict=state_dict, device_manager=manager)

    sample_received = torch.randn(size=(30, decoder.source_data_len, 3))

    decoder(sample_received)
