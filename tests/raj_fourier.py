import numpy as np
import torch


def toFourier(X):
    """
    converts boolean to fourier i.e. 0/1's -> 1/-1's
    X : must be boolean i.e. 0/1's
    """
    # X = X.to(torch.long)
    X = 1 - 2 * X
    # X = torch.where(X == 1, -1, X)
    # X = torch.where(X == 0, 1, X)
    return X


def allBitVectors(length=5):
    # if length==1:
    #     return np.array([[0], [1]])
    # v = allBitVectors(length=length-1)
    # v0 = np.pad(v, ((0,0),(1,0)))
    # v1 = np.pad(v, ((0,0),(1,0)))
    # v1[:,0] = 1
    # v = np.concatenate((v0, v1), axis=0)

    v = np.array([[0], [1]])
    currLen = 1
    # for i in range(1, length):
    while currLen < length:
        print(f"currLen : {currLen}")
        v0 = np.pad(v, ((0, 0), (1, 0)))
        v1 = np.pad(v, ((0, 0), (1, 0)))
        v1[:, 0] = 1
        v = np.concatenate((v0, v1), axis=0)
        currLen += 1
    return v


def fourier(f, N, parities=None):
    """Computes the fourier coefficients of an N-bit binary input function

    Args:
        f: N-bit Binary function, which takes a batch of B inputs at once
            and returns a batch of B outputs for all the inputs.
            f: {-1, 1}^NxB -> R^B, B can be in the range : 1 ≤ B ≤ 2^N
        N: Input Width
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    X = torch.from_numpy(allBitVectors(length=N)).to(device)
    if parities is None:
        parities = X.clone()
    X = toFourier(X)
    fx = f(X)
    # coeffs = torch.zeros(2**N)
    fc = []
    for i, s in enumerate(parities):
        # print(i)
        XParS = X * s
        XParS = torch.where(XParS == 0, 1, XParS)
        XParS = torch.prod(XParS, dim=1)
        # coeffs[i] = torch.mean(f(X).type(torch.float16)*XParS)
        # coeffs[i] = torch.mean(fx.type(torch.float16)*XParS)
        fc.append(torch.mean(fx.type(torch.float16) * XParS))
    return torch.tensor(fc, device=device)


import math


def inverseFourier(fc, X):
    """evaluates the function with given fourier coefficients on the given inputs
    Args:
        fc:     fourier coefficients
        X:        inputs
    Returns:
        output of function with fc as fourier coefficients on inputs X.
    """
    C, N = X.shape
    assert N == int(math.log2(len(fc)))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    P = torch.from_numpy(allBitVectors(length=N)).to(device)  # parities
    Y = []
    for c in range(C):
        Yc = X[c] * P
        Yc = torch.where(Yc == 0, 1, Yc)
        Yc.to(device)
        Yc = torch.prod(Yc, dim=1)
        Yc = torch.sum(Yc * fc)
        Y.append(Yc)
    return torch.tensor(Y, device=device)
