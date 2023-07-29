"""
Adapted from code by Raj Shekar
"""

import math

import torch

from .utils import DEFAULT_DEVICE_MANAGER, enumerate_binary_inputs, check_int


def boolean_to_fourier(X):
    """
    converts boolean to fourier i.e. 0/1's -> 1/-1's
    X : must be boolean i.e. 0/1's
    """
    X = 1 - 2 * X
    return X


def table_to_fourier(table: torch.FloatTensor, parities=None, device_manager=DEFAULT_DEVICE_MANAGER):
    """Computes the fourier coefficients of an N-bit binary input function

    Parameters
    ----------
    table (2^Window(BinaryInputs) x Streams): torch.FloatTensor
        input-output table for a binary function with `Streams` number of outputs.
    parities (Parities x Window): torch.FloatTensor 
        List of parities for which to compute Fourier Coefficients. 
        If None, all FC's are computed.
    
    Returns
    -------
    (2^Window(FourierInputs) x Streams) torch.FloatTensor
        The fourier coefficients of corresponding to the function computed
        by the table
    """
    window = check_int(math.log2(table.shape[0]))
    binary_inputs = enumerate_binary_inputs(
        window=window, dtype=torch.float, device=device_manager.device
    )
    if parities is None:
        parities = binary_inputs.clone()
    else:
        parities = parities.float()
    fourier_inputs = boolean_to_fourier(binary_inputs)
    
    # Parities x FourierInputs x N
    XParS = fourier_inputs[None] * parities[:, None]
    XParS = torch.where(XParS == 0, 1, XParS)
    # Parities x FourierInputs
    XParS = torch.prod(XParS, dim=-1)
    fc = 1 / binary_inputs.shape[0] * torch.tensordot(XParS, table, dims=1)

    return fc

def compute_fourier_coefficients(
    f, window, parities=None, device_manager=DEFAULT_DEVICE_MANAGER
):
    """Computes the fourier coefficients of an N-bit binary input function

    Args:
        f: `window`-bit Binary function, which takes a batch of B inputs at once
            and returns a batch of B outputs for all the inputs.
            f: {0, 1}^NxB -> R^B, B can be in the range : 1 ≤ B ≤ 2^N
        window: Input Width
        parities: List of parities for which to compute Fourier Coefficients.
            If None, all FC's are computed.
    Returns:
        The fourier coefficients of f
    Usage:
        >>> # Computing fourier coefficients of max of two binary inputs.
        >>> import torch
        >>> from fourier import fourier
        >>> def max(X):
        >>>     '''max takes a batch of inputs, and returns the batch of outputs for all inputs.'''
        >>>     return torch.max(X,dim=1).values
        >>> fc = fourier(f=max, N=2)
        >>> # fc = tensor([ 0.5000,  0.5000,  0.5000, -0.5000], dtype=torch.float16)
        >>> # max must be able to operate on a batch of inputs as following example.
        >>> X = torch.randint(low=0, high=2, size=(4,2))*2-1
        >>> # X = tensor([[-1, -1],
        >>> #             [ 1,  1],
        >>> #             [ 1, -1],
        >>> #             [-1, -1]])
        >>> y = max(X)
        >>> # y = tensor([-1,  1,  1, -1])
        >>> fc = fourier(f=max, N=2)
        >>> # fc = tensor([ 0.5000,  0.5000,  0.5000, -0.5000], dtype=torch.float16)
        >>> parities = torch.tensor([[0, 0],[1, 1]], dtype=torch.int, device=device)
        >>> fc = fourier(f=max, N=2, parities=parities)
        >>> # fc = tensor([0.5000, -0.5000], dtype=torch.float16)
    """
    # import pdb; pdb.set_trace
    binary_inputs = enumerate_binary_inputs(
        window=window, dtype=torch.float, device=device_manager.device
    )
    table = f(binary_inputs)
    return table_to_fourier(table, parities=parities, device_manager=device_manager)

def compute_fourier_function(
    fc: torch.FloatTensor, input_data: torch.Tensor, device_manager=DEFAULT_DEVICE_MANAGER
):
    """evaluates the function with given fourier coefficients on the given inputs

    Parameters
    ----------
    fc (2^Window(Parities) x ...) : torch.FloatTensor
        The fourier coefficients of the pseudoboolean functions. There
        are `Streams` number of functions
    input_data (Batch x Window) : torch.Tensor
        A tensor of {0,1} to apply the pseudoboolean function to.

    Returns
    -------
    torch.FloatTensor (Batch x Streams)
        The result of applying each of the `Streams` number of pseudoboolean
        functions to the batch of inputs.

    """
    window = input_data.shape[1]
    assert window == check_int(math.log2(fc.shape[0]))
    # 2^Window(Parities) x Window
    parities = enumerate_binary_inputs(
        window=window, dtype=torch.int8, device=device_manager.device
    )  # parities
    input_data = boolean_to_fourier(input_data.char())
    # Batch x Parities x Window
    output_data = input_data[:, None] * parities[None]
    output_data = torch.where(output_data == 0, 1, output_data)
    # Batch x Parities
    output_data = torch.prod(output_data, dim=-1)
    output_data = torch.tensordot(output_data.float(), fc, dims=1)

    return output_data


def fourier_to_table(fc: torch.FloatTensor, device_manager=DEFAULT_DEVICE_MANAGER):
    window = check_int(math.log2(fc.shape[0]))
    input_data = enumerate_binary_inputs(
        window=window, dtype=torch.int8, device=device_manager.device
    )
    return compute_fourier_function(fc, input_data, device_manager=device_manager)
        