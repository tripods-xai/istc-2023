try:
    import commpy.channelcoding as cc
    from commpy.channelcoding.interleavers import RandInterlv
except:
    print("Could not load commpy")
    RandInterlv = None


def turbo_decode(
    stream1,
    stream2,
    stream1_i,
    stream3,
    trellis1,
    trellis2,
    noise_variance,
    number_iterations,
    interleaver: RandInterlv,
    L_int=None,
):
    """Turbo Decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/3) bits using the BCJR algorithm.

    Parameters
    ----------
    stream1 : 1D ndarray
        Received symbols corresponding to
        the first output bits in the codeword.

    stream2 : 1D ndarray
        Received symbols corresponding to
        the second output bits in the codeword.

    stream3 : 1D ndarray
        Received symbols corresponding to the
        third output bits (interleaved) in the codeword.

    trellis1 : Trellis object
        Trellis representation of the non-interleaved convolutional codes
        used in the Turbo code.

    trellis2 : Trellis object
        Trellis representation of the interleaved convolutional codes
        used in the Turbo code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    number_iterations : int
        Number of the iterations of the
        BCJR algorithm used in turbo decoding.

    interleaver : Interleaver object.
        Interleaver used in the turbo code.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    Returns
    -------
    decoded_bits : 1D ndarray of ints containing {0, 1}
        Decoded bit stream.

    """
    if L_int is None:
        L_int = np.zeros(len(stream1))

    L_int_1 = L_int

    # # Interleave stream 1 for input to second decoder
    # stream1_i = interleaver.interlv(stream1)

    iteration_count = 0
    max_iters = number_iterations
    while iteration_count < max_iters:
        [L_1, decoded_bits] = cc.map_decode(
            stream1, stream2, trellis1, noise_variance, L_int_1, "compute"
        )

        L_ext_1 = L_1 - L_int_1
        L_int_2 = interleaver.interlv(L_ext_1)

        # MAP Decoder - 2
        [L_2, decoded_bits] = cc.map_decode(
            stream1_i, stream3, trellis2, noise_variance, L_int_2, "compute"
        )
        L_ext_2 = L_2 - L_int_2
        L_int_1 = interleaver.deinterlv(L_ext_2)

        iteration_count += 1

    L_2_deinterleaved = interleaver.deinterlv(L_2)
    decoded_bits = (L_2_deinterleaved > 0).astype(int)

    return L_2_deinterleaved, decoded_bits


def vturbo_decode(
    msg, trellis1, trellis2, noise_variance, number_iterations, interleaver
):
    def _turbo_decode(single_msg):
        str1, str2, str1_i, str3 = [single_msg[:, i] for i in range(4)]
        L_2_deinterleaved, decoded_bits = turbo_decode(
            str1,
            str2,
            str1_i,
            str3,
            trellis1,
            trellis2,
            noise_variance,
            number_iterations,
            interleaver,
        )
        return L_2_deinterleaved

    return np.stack([_turbo_decode(msg[i]) for i in range(msg.shape[0])], axis=0)


def vsturbo_decode(
    msg, trellis1, trellis2, noise_variance, number_iterations, interleaver
):
    def _sturbo_decode(single_msg):
        str1, str2, str3 = [single_msg[:, i] for i in range(3)]
        str1_i = interleaver.interlv(str1)
        L_2_deinterleaved, decoded_bits = turbo_decode(
            str1,
            str2,
            str1_i,
            str3,
            trellis1,
            trellis2,
            noise_variance,
            number_iterations,
            interleaver,
        )
        return L_2_deinterleaved

    return np.stack([_sturbo_decode(msg[i]) for i in range(msg.shape[0])], axis=0)
