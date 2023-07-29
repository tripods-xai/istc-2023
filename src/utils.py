from typing import (
    Tuple,
    Union,
    Dict,
    Any,
    Iterator,
    Sequence,
    Collection,
    TypeVar,
    Generic,
    Iterable,
    Literal,
)
from numbers import Number

from pathlib import Path
import math
import os
from datetime import datetime
import abc

import numpy as np
import numpy_indexed as npi
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import RELATIVE_ROOT, TMP_DIR

Precision = Literal["half", "single", "double"]
PRECISION_DTYPE_DICT: Dict[Precision, torch.dtype] = {
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
}

EPSILON = 1e-12


# TODO: make this so that we can get new seeds for all the stuff we need
class DeviceManager:
    CUDA_NAME = "cuda:0"
    CPU_NAME = "cpu"

    def __init__(self, no_cuda=False, seed=None) -> None:
        self.no_cuda = no_cuda
        self.seed = seed

        self._generator = None

    @property
    def use_cuda(self):
        return (not self.no_cuda) and torch.cuda.is_available()

    @property
    def device(self):
        device_name = self.CUDA_NAME if self.use_cuda else self.CPU_NAME
        return device_name

    @property
    def generator(self):
        if self._generator is None:
            self._generator = torch.Generator(device=self.device)
            if self.seed is not None:
                self._generator.manual_seed(self.seed)
        return self._generator

    def generate_seed(self):
        return torch.randint(
            99999, size=(1,), device=self.device, generator=self.generator
        )[0].item()

    def clone(self):
        new_device_manager = DeviceManager(no_cuda=self.no_cuda, seed=self.seed)
        new_device_manager._generator = torch.Generator(
            device=new_device_manager.device
        ).set_state(self.generator.get_state().clone())
        return new_device_manager


DEFAULT_DEVICE_MANAGER = DeviceManager()


class ModuleExtension(nn.Module):
    def __init__(self, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER) -> None:
        super().__init__()
        self.device_manager = device_manager

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @device_manager.setter
    def device_manager(self, value):
        self._device_manager = value


class WithSettings(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def settings(self) -> Dict[str, Any]:
        pass

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


# Stand in until torch.nested_tensor and torch.masked_tensor is
# on the stable release and is well supported. Designed to imitate
# numpy.ma.
class MaskedTensor:
    def __init__(
        self, tensor: torch.Tensor, mask: torch.BoolTensor, fill_value=0, no_fill=False
    ) -> None:
        """
        Parameters
        ----------
        mask : torch.BoolTensor
            Invalid entries are marked with True.
        """
        self.mask = mask
        self.fill_value = fill_value
        if no_fill:
            self.tensor = tensor
        else:
            self.tensor = torch.where(self.mask, self.fill_value, tensor)
        assert self.tensor.shape == self.mask.shape

    def __eq__(self, other: object) -> "MaskedTensor":
        if isinstance(other, MaskedTensor):
            return MaskedTensor(
                tensor=(self.tensor == other.tensor), mask=(self.mask | other.mask)
            )
        if isinstance(other, torch.Tensor):
            return MaskedTensor(tensor=(self.tensor == other), mask=self.mask)
        else:
            return NotImplemented

    def all(self, dim: int = None, keepdim=False) -> Union[bool, "MaskedTensor"]:
        if dim is None:
            return torch.all(self.tensor | self.mask)
        else:
            tensor = torch.all(self.tensor | self.mask, dim=dim, keepdim=keepdim)
            # New mask is set to True (masked) if all entries along the reduced dim
            # were true (i.e. masked).
            mask = torch.all(self.mask, dim=dim, keepdim=keepdim)
            return MaskedTensor(tensor=tensor, mask=mask, no_fill=True)

    def sum(self, dim: Union[int, Sequence[int]] = None, keepdim=False):
        if dim is None:
            return torch.sum(self.tensor * (~self.mask))
        else:
            tensor = torch.sum(self.tensor * (~self.mask), dim=dim, keepdim=keepdim)
            # New mask is set to True (masked) if all entries along the reduced dim
            # were true (i.e. masked).
            mask = torch.all(self.mask, dim=dim, keepdim=keepdim)
            return MaskedTensor(tensor=tensor, mask=mask, no_fill=True)

    def __getitem__(self, ind) -> Union[torch.Tensor, "MaskedTensor"]:
        tensor_select = self.tensor[ind]
        if tensor_select.ndim == 0:
            # Here we deviate from numpy and just return the filled value
            return tensor_select
        else:
            return MaskedTensor(
                tensor=tensor_select,
                mask=self.mask[ind],
                fill_value=self.fill_value,
                no_fill=True,
            )

    def fill(self, fill_value) -> "MaskedTensor":
        return MaskedTensor(
            tensor=self.tensor, mask=self.mask, fill_value=fill_value, no_fill=False
        )

    def numel(self) -> int:
        return (~self.mask).count_nonzero().item()

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def to(self, dtype: torch.dtype):
        return MaskedTensor(
            self.tensor.to(dtype),
            mask=self.mask,
            fill_value=self.fill_value,
            no_fill=True,
        )

    def reshape(self, *new_shape):
        return MaskedTensor(
            tensor=self.tensor.reshape(*new_shape),
            mask=self.mask.reshape(*new_shape),
            fill_value=self.fill_value,
            no_fill=True,
        )

    def transpose(self, dim0: int, dim1: int):
        return MaskedTensor(
            tensor=self.tensor.transpose(dim0=dim0, dim1=dim1),
            mask=self.mask.transpose(dim0=dim0, dim1=dim1),
            fill_value=self.fill_value,
            no_fill=True,
        )

    @staticmethod
    def tensor_to_masked(tensor: torch.Tensor, fill_value: int = 0):
        return MaskedTensor(
            tensor=tensor,
            mask=torch.zeros(
                tensor.shape,
                dtype=torch.bool,
                device=tensor.device,
            ),
            fill_value=fill_value,
            no_fill=True,
        )

    @staticmethod
    def cat(
        tensors: Union["MaskedTensor", torch.Tensor], dim: int = 0
    ) -> Union["MaskedTensor", torch.Tensor]:
        masked_tensors = [t for t in tensors if isinstance(t, MaskedTensor)]
        if len(masked_tensors) == 0:
            # All just regular tensors
            return torch.cat(tensors, dim=dim)
        else:
            fill_value = masked_tensors[0].fill_value
            assert all(
                mt.fill_value == fill_value for mt in masked_tensors
            ), "All masked tensors must have the same fill value to concatenate"
            # Turn the tensors into masked tensors
            new_tensors = [
                (
                    t
                    if isinstance(t, MaskedTensor)
                    else MaskedTensor.tensor_to_masked(t, fill_value=fill_value)
                )
                for t in tensors
            ]
            tensor_cat = torch.cat([mt.tensor for mt in new_tensors], dim=dim)
            mask_cat = torch.cat([mt.mask for mt in new_tensors], dim=dim)
            return MaskedTensor(
                tensor=tensor_cat, mask=mask_cat, fill_value=fill_value, no_fill=True
            )


def all_distinct(X: np.array):
    return len(np.unique(X)) == len(X)


class NamedTensor:
    """
    Rolling out my own basic version of this until
    PyTorch's API on NamedTensors stabilizes.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        dim_names: Sequence[str],
        batch_dims=0,
        check=True,
        dim_name_set=None,
    ) -> None:
        """
        For now enforcing that dim_names need to be unique
        """
        self.tensor = tensor
        self.dim_names = np.asarray(dim_names)
        self.batch_dims = batch_dims
        if check:
            assert self.dim_names.ndim == 1
            assert self.dim_names.shape[0] + self.batch_dims == self.tensor.ndim

        self._dim_name_set = (
            dim_name_set if dim_name_set is not None else {*self.dim_names}
        )
        if check:
            assert len(self._dim_name_set) == len(self.dim_names)

    def to(self, device: str = None, dtype=None):
        return NamedTensor(
            tensor=self.tensor.to(device=device, dtype=dtype),
            dim_names=self.dim_names,
            batch_dims=self.batch_dims,
            check=False,
            dim_name_set=self._dim_name_set,
        )

    def permute(self, dim_name_permutation: Sequence[str], check=True):
        if check:
            assert len(dim_name_permutation) == len(self.dim_names)
            dim_name_permutation_set = {*dim_name_permutation}
            assert dim_name_permutation_set == self._dim_name_set
        new_dim_inds = self.get_dim_inds(dim_name_permutation, check=False)
        # print([*range(self.batch_dims), *new_dim_inds])
        new_tensor = self.tensor.permute(*range(self.batch_dims), *new_dim_inds)
        return NamedTensor(
            new_tensor,
            dim_names=dim_name_permutation,
            batch_dims=self.batch_dims,
            check=False,
            dim_name_set=self._dim_name_set,
        )

    def canonicalize_dim_order(self) -> "NamedTensor":
        return self.permute(sorted(self._dim_name_set))

    @property
    def shape(self):
        return self.tensor.shape

    @staticmethod
    def sum(named_tensors: Sequence["NamedTensor"]) -> "NamedTensor":
        if len(named_tensors) == 0:
            raise ValueError("Empty sum of NamedTensors is not defined.")

        return sum(named_tensors[1:], start=named_tensors[0])

    def unsqueeze(
        self, new_dim_names: Sequence[str], check=True, new_dim_names_set=None
    ):
        """Unsqueezes the NamedTensor to include additional dims of shape 1

        Parameters
        ----------
        new_dim_names : Sequence[str]
            A superset of the current dim_names. The returned NamedTensor
            will have dimensions corresponding to this sequence. Any dims
            from the original NamedTensor will use the original data.

        Returns
        -------
        NamedTensor
            A NamedTensor with dimension names given by `new_dim_names`.
            If a name in `new_dim_names` was not in the original NamedTensor,
            then it will be included with a shape of 1. If the name was in
            the original NamedTensor, then it will be included with its original
            shape and data.
        """
        new_dim_names = np.asarray(new_dim_names)
        new_dim_names_set = (
            {*new_dim_names} if new_dim_names_set is None else new_dim_names_set
        )
        if check:
            assert len(new_dim_names) == len(new_dim_names_set)
            assert self._dim_name_set.issubset(new_dim_names_set)

        current_dims_in_new_mask = np.in1d(
            new_dim_names, self.dim_names, assume_unique=True
        )
        # First permute our tensor to match the arrangement of the dimensions in
        # the new dim names
        new_named_tensor = self.permute(
            new_dim_names[current_dims_in_new_mask], check=False
        )

        # Now build the shape
        new_shape = np.ones(len(new_dim_names), dtype=int)
        new_shape[current_dims_in_new_mask] = np.array(
            new_named_tensor.shape[self.batch_dims :]
        )
        new_shape = np.concatenate(
            [np.array(new_named_tensor.shape[: self.batch_dims]), new_shape]
        )

        # Finally put it together to get the expanded tensor
        return NamedTensor(
            new_named_tensor.tensor.reshape(*new_shape),
            dim_names=new_dim_names,
            batch_dims=new_named_tensor.batch_dims,
            check=False,
            dim_name_set=new_dim_names_set,
        )

    def get_dim_inds(self, dim_names: Sequence[str], check=True):
        if len(dim_names) == 0:
            # Need this case because npi.indices throws an error.
            return np.array([])
        dim_names = np.asarray(dim_names)
        return (
            npi.indices(
                self.dim_names, dim_names, missing=("raise" if check else "ignore")
            )
            + self.batch_dims
        )

    @staticmethod
    def align_tensors(
        named_tensors: Sequence["NamedTensor"],
    ) -> Sequence["NamedTensor"]:
        aligned_dim_names_set = set.union(
            *[named_tensor._dim_name_set for named_tensor in named_tensors]
        )
        return [
            nt.unsqueeze(
                list(aligned_dim_names_set),
                new_dim_names_set=aligned_dim_names_set,
                check=False,
            )
            for nt in named_tensors
        ]

    def active_dims(self) -> Sequence[str]:
        """Get the dims that have non-unit shape.

        Returns
        -------
        np.array
            An array of the dim names that have non-unit shape.

        """
        active_no_batch_dim_inds = [
            i
            for i, dim_shape in enumerate(self.tensor.shape[self.batch_dims :])
            if dim_shape > 1
        ]
        return self.dim_names[active_no_batch_dim_inds]

    def __add__(self, other: "NamedTensor") -> "NamedTensor":
        # First align the tensors
        self_aligned, other = NamedTensor.align_tensors([self, other])
        assert self_aligned.batch_dims == other.batch_dims

        # Now add
        new_tensor = self_aligned.tensor + other.tensor
        return NamedTensor(
            tensor=new_tensor,
            dim_names=self_aligned.dim_names,
            batch_dims=self_aligned.batch_dims,
            check=False,
            dim_name_set=self_aligned._dim_name_set,
        )

    def logsumexp(self, dim: Sequence[str] = None, keepdim=False, check=True):
        if dim is None:
            # Keep the batch dimensions
            return torch.logsumexp(
                self.tensor, dim=list(range(self.batch_dims, self.tensor.ndim))
            )

        reduce_dim_set = {*dim}
        if check:
            assert len(reduce_dim_set) == len(dim)
        new_tensor = torch.logsumexp(
            self.tensor,
            dim=list(self.get_dim_inds(dim, check=check)),
            keepdim=keepdim,
        )
        if not keepdim:
            # Remove the reduced axis names
            new_dim_names = self.dim_names[~np.in1d(self.dim_names, dim)]
            new_dim_names_set = self._dim_name_set - reduce_dim_set
        else:
            new_dim_names = self.dim_names
            new_dim_names_set = self._dim_name_set
        return NamedTensor(
            tensor=new_tensor,
            dim_names=new_dim_names,
            batch_dims=self.batch_dims,
            dim_name_set=new_dim_names_set,
        )


M = TypeVar("M", bound=nn.Module)


class BatchChunker(nn.Module, Generic[M]):
    def __init__(self, module: M, chunk_size: int) -> None:
        super().__init__()
        self.module = module
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor):
        # Assumes first dimension is batch dimension
        batch_size = x.shape[0]
        res = []
        for i in range(0, batch_size, self.chunk_size):
            res.append(self.module(x[i : i + self.chunk_size]))

        return torch.cat(res, dim=0)


def get_reducer(use_max: bool):
    return (
        (
            lambda tensor, dim=None, keepdims=False: torch.max(
                tensor, dim=dim, keepdims=keepdims
            ).values
        )
        if use_max
        else torch.logsumexp
    )


def snr2sigma(snr):
    return 10 ** (-snr / 20)


def sigma2snr(sigma):
    return -20 * math.log10(sigma)


def snr2sigma_torch(snr):
    return 10 ** (-snr / 20)


def sigma2snr_torch(sigma):
    return -20 * torch.log10(sigma)


def get_dummy(shape: Tuple[int, ...], dim: int):
    unsqueeze_shape = [1] * len(shape)
    unsqueeze_shape[dim] = shape[dim]
    unsqueeze_shape = tuple(unsqueeze_shape)
    return torch.arange(shape[dim]).reshape(unsqueeze_shape).expand(*shape)


def gen_affine_convcode_generator(
    window: int, num_output_channels: int, device_manager=DEFAULT_DEVICE_MANAGER
):
    m = torch.zeros(
        (num_output_channels, window), dtype=torch.int8, device=device_manager.device
    )
    m[0, -1] = 1
    return torch.randint(
        0,
        2,
        (num_output_channels, window),
        dtype=torch.int8,
        device=device_manager.device,
        generator=device_manager.generator,
    ) | m, torch.randint(
        0,
        2,
        (num_output_channels,),
        dtype=torch.int8,
        device=device_manager.device,
        generator=device_manager.generator,
    )


def dec2bitarray(
    arr: torch.Tensor,
    num_bits: int,
    little_endian: bool = False,
    dtype=torch.int8,
    device: torch.device = None,
) -> torch.Tensor:
    if isinstance(arr, int):
        arr = torch.tensor(arr, device=device)

    if little_endian:
        shift_arr = torch.arange(num_bits, device=arr.device)
    else:
        shift_arr = torch.flip(torch.arange(num_bits, device=arr.device), dims=(0,))

    return (torch.bitwise_right_shift(arr[..., None].to(torch.long), shift_arr) % 2).to(
        dtype
    )


def enumerate_binary_inputs_chunked(
    window: int,
    dtype=torch.int8,
    constraint: "MaskedTensor" = None,
    chunk_size=None,
    device: torch.device = None,
) -> Iterator[torch.Tensor]:
    """
    Parameters
    ----------
    constraint (window) : MaskedTensor
        Unmasked values are fixed when enumerating inputs.
    """
    if constraint is None:
        constraint = MaskedTensor(
            torch.zeros(window, device=device, dtype=dtype),
            mask=torch.ones(window, device=device, dtype=torch.bool),
            no_fill=True,
        )
    else:
        assert constraint.shape == (window,)
        constraint = constraint.to(dtype)

    constrained = constraint.numel()
    num_inputs = 2 ** (window - constrained)

    if chunk_size is None:
        chunk_size = num_inputs

    for i in range(0, num_inputs, chunk_size):
        top = min(i + chunk_size, num_inputs)
        res = torch.zeros((top - i, window), dtype=dtype, device=device)
        res[:, ~constraint.mask] = constraint.tensor[None, ~constraint.mask]
        # Fill in the nonconstrained values
        res[:, constraint.mask] = dec2bitarray(
            torch.arange(i, top, device=device),
            window - constrained,
            dtype=dtype,
            device=device,
        )
        yield res


def enumerate_binary_inputs(
    window: int,
    dtype=torch.int8,
    constraint: "MaskedTensor" = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Returns tensor of dimension 2 of all 2^n binary sequences of length `window`"""
    return next(
        enumerate_binary_inputs_chunked(
            window=window, dtype=dtype, constraint=constraint, device=device
        )
    )


def base_2_accumulator(
    length: int, little_endian: bool = False, device: torch.device = None
) -> torch.LongTensor:
    """
    Returns
    -------
    (length) torch.LongTensor
        An array of powers of 2. The order is decreasing if little_endian = False
    """
    powers_of_2 = torch.bitwise_left_shift(
        1, torch.arange(length, dtype=torch.long, device=device)
    )
    if little_endian:
        return powers_of_2
    else:
        return torch.flip(powers_of_2, dims=(0,))


def bitarray2dec(
    arr: torch.Tensor, little_endian=False, dim=-1, dtype=torch.long, device=None
) -> torch.Tensor:
    # CUDA only supports tensordot for double,float, half.
    if DeviceManager.CUDA_NAME == device:
        intermediate_dtype = torch.float
    else:
        intermediate_dtype = dtype
    if device is None:
        device = arr.device
    if arr.shape[dim] == 0:
        reduced_shape = arr.shape[:dim] + arr.shape[dim + 1 :]
        return torch.zeros(reduced_shape, dtype=intermediate_dtype, device=device)
    base_2 = base_2_accumulator(
        arr.shape[dim], little_endian=little_endian, device=device
    )
    return torch.tensordot(
        arr.to(intermediate_dtype), base_2.to(intermediate_dtype), dims=[[dim], [0]]
    ).to(dtype)


def check_int(x: Number) -> int:
    if isinstance(x, int):
        return x
    else:
        assert isinstance(x, float) and x.is_integer()
        return int(x)


FLOATING_TYPES = [torch.float16, torch.float32, torch.float64]
SIGNED_TYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
UNSIGNED_TYPES = [torch.uint8, torch.bool]


def check_signed(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in FLOATING_TYPES + SIGNED_TYPES:
        return x
    else:
        raise AssertionError(f"Tensor {x} of type {x.dtype} is not a signed tensor.")


def get_smallest_signed_int_type(min_val: int, max_val: int):
    for itype in SIGNED_TYPES:
        iinfo = torch.iinfo(itype)
        if iinfo.min <= min_val and iinfo.max >= max_val:
            return itype
    raise ValueError(
        f"No datatype can represent signed ints with max={max_val} and min={min_val}."
    )


# metric utils
LOG2 = math.log(2)


def binary_entropy_with_log(log1: torch.Tensor, log0: torch.Tensor):
    return (-log1 * torch.exp(log1) - log0 * torch.exp(log0)) / LOG2


def binary_entropy_with_logits(logits: torch.FloatTensor):
    return (
        F.binary_cross_entropy_with_logits(
            input=logits, target=torch.sigmoid(logits), reduction="none"
        )
        / LOG2
    )


# File utils
def safe_open_dir(dirpath: str) -> str:
    if not os.path.isdir(dirpath):
        print(f"Directory {dirpath} does not exist, creating it")
        os.makedirs(dirpath)
    return dirpath


def safe_create_file(filepath: str) -> str:
    dirpath = os.path.dirname(filepath)
    dirpath = safe_open_dir(dirpath)
    return filepath


def tmp_if_debug(filepath: Path, debug: bool) -> Path:
    if debug:
        return TMP_DIR / filepath.relative_to(RELATIVE_ROOT)
    else:
        return filepath


TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"


def get_timestamp():
    return datetime.now().strftime(TIME_FORMAT)


def parse_timestamp(dt: datetime):
    return datetime.strptime(dt, TIME_FORMAT)


def peek(collection: Collection):
    return next(iter(collection))


def dynamic_slice(
    t: torch.Tensor,
    dim: Union[int, Sequence],
    index: Union[int, Sequence, torch.Tensor],
):
    if isinstance(dim, int):
        dim = [dim]
        index = [index]
    slicer = [slice(None)] * t.ndim
    for i, dim_ind in enumerate(dim):
        slicer[dim_ind] = index[i]
    return tuple(slicer)


def dynamic_get(
    t: torch.Tensor,
    dim: Union[int, Sequence],
    index: Union[int, Sequence, torch.Tensor],
):
    return t[dynamic_slice(t, dim, index)]


def filter_state_dict(state_dict: Dict[str, torch.Tensor], key: str):
    new_state_dict = {}
    for k in state_dict.keys():
        k_terms = k.split(".")
        if k_terms[0] == key:
            new_state_dict[".".join(k_terms[1:])] = state_dict[k]
    return new_state_dict


T = TypeVar("T")


class RepeatableGenerator(Iterable[T]):
    def __init__(self, gen: Iterator[T]):
        self.gen = gen
        self.data = []

    def __iter__(self) -> Iterator[T]:
        if len(self.data) == 0:
            for d in self.gen:
                self.data.append(d)
                yield d
        else:
            for d in self.data:
                yield d


class GeneratorWithLength(Generic[T]):
    def __init__(self, gen, length):
        self.gen_container = RepeatableGenerator(gen)
        self.length = length
        self.captured = [None] * self.length

    def __len__(self):
        return self.length

    def __iter__(self) -> Iterator[T]:
        return iter(self.gen_container)


def data_gen(
    input_size: int,
    num_steps: int,
    batch_size: int,
    num_batches: int = 1,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
) -> Iterator[GeneratorWithLength[torch.LongTensor]]:
    for _ in range(num_steps):
        yield GeneratorWithLength(
            gen=(
                torch.randint(
                    0,
                    2,
                    size=(batch_size, input_size),
                    device=device_manager.device,
                    generator=device_manager.generator,
                )
                for _ in range(num_batches)
            ),
            length=num_batches,
        )


DEFAULT_SEED = 1234
