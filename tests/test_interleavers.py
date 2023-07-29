import torch

from src.interleavers import BatchRandomPermuteInterleaver
from src.utils import DeviceManager

def test_batch_variable_interleaver():
    manager = DeviceManager(seed=1234)
    msg_length = 10
    interleaver = BatchRandomPermuteInterleaver(input_size=msg_length, device_manager=manager)
    
    assert len(interleaver) == msg_length
    
    test_msg_1 = torch.stack([torch.arange(msg_length), torch.arange(msg_length)], dim=0)
    # The batch was interleaved with one permutation.
    outs = interleaver(test_msg_1)
    assert (outs[0] == outs[1]).all()
    # Successive calls to interleave use the same permutation.
    out_again = interleaver.interleave(test_msg_1)
    assert (out_again == outs).all()
    # deinterleave undoes the used permutation
    deinterleaved = interleaver.deinterleave(out_again)
    assert (deinterleaved == test_msg_1).all()
    
    # Calling interleaver again uses a different permutation
    outs_2 = interleaver(test_msg_1)
    assert not (outs == outs_2).all()
    
    
    