import numpy as np

from src.encoders import Encoder
from src.interleavers import FixedPermuteInterleaver
from src.utils import enumerate_binary_inputs

from commpy.channelcoding.interleavers import RandInterlv
from commpy.channelcoding.convcode import conv_encode


def get_codebook(encoder: Encoder, block_size: int):
    inputs = enumerate_binary_inputs(block_size).float()
    return encoder(inputs)


def interleaver_to_commpy(interleaver: FixedPermuteInterleaver):
    commpy_interleaver = RandInterlv(len(interleaver), seed=0)
    commpy_interleaver.p_array = interleaver.permutation.numpy()
    return commpy_interleaver


def turbo_encode(msg_bits, trellis1, trellis2, interleaver: RandInterlv):
    """Turbo Encoder.

    Encode Bits using a parallel concatenated rate-1/3
    turbo code consisting of two rate-1/2 systematic
    convolutional component codes.

    Parameters
    ----------
    msg_bits : 1D ndarray containing {0, 1}
        Stream of bits to be turbo encoded.

    trellis1 : Trellis object
        Trellis representation of the
        first two codes in the parallel concatenation.

    trellis2 : Trellis object
        Trellis representation of the
        first and interleaved code in the parallel concatenation.

    interleaver : Interleaver object
        Interleaver used in the turbo code.

    Returns
    -------
    [stream1, stream2, stream3] : list of 1D ndarrays
        Encoded bit streams corresponding
        to two non-interleaved outputs and one interleaved output.
    """
    stream = conv_encode(msg_bits, trellis1, termination="cont")
    stream1 = stream[::2]
    stream2 = stream[1::2]

    interlv_msg_bits = interleaver.interlv(msg_bits)
    interlv_stream = conv_encode(interlv_msg_bits, trellis2, termination="cont")
    stream1_i = interlv_stream[::2]
    stream3 = interlv_stream[1::2]

    assert len(stream1) == len(stream2) == len(stream3) == len(msg_bits)

    return [stream1, stream2, stream1_i, stream3]


def vsystematic_turbo_encode(msg, trellis1, trellis2, interleaver):
    def _turbo_encode(single_msg):
        outputs = turbo_encode(single_msg, trellis1, trellis2, interleaver)
        return np.stack([outputs[0], outputs[1], outputs[3]], axis=1)

    return np.apply_along_axis(_turbo_encode, axis=1, arr=msg)
