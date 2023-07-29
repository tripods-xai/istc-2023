import torch
import numpy as np
import pytest

from src.utils import *


def test_enumerate_binary_inputs_basic():
    window = 3

    actual = enumerate_binary_inputs(window=window)
    expected = torch.CharTensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    print(actual)
    print(expected)
    assert torch.all(actual == expected)


def test_enumerate_binary_inputs_constrained():
    window = 5

    constraint = MaskedTensor(
        tensor=torch.CharTensor([0, 1, 0, 0, 0]),
        mask=torch.BoolTensor([True, False, False, True, True]),
    )
    actual = enumerate_binary_inputs(window=window, constraint=constraint)
    expected = torch.CharTensor(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
        ]
    )

    print(actual)
    print(expected)
    assert torch.all(actual == expected)


def test_named_tensor_create():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)
    assert torch.all(named_tensor.tensor == tensor)
    assert np.all(named_tensor.dim_names == np.asarray(dim_names))


def test_named_tensor_create_fail():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    # Wrong batch_dims
    with pytest.raises(AssertionError):
        named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=2)
    # Wrong dim_names
    with pytest.raises(AssertionError):
        named_tensor = NamedTensor(
            tensor=tensor, dim_names=dim_names[:-1], batch_dims=1
        )
    # Nonunique dim_names
    with pytest.raises(AssertionError):
        named_tensor = NamedTensor(
            tensor=tensor, dim_names=["A", "B", "A"], batch_dims=1
        )
    # Wrong dim names shape
    with pytest.raises(AssertionError):
        named_tensor = NamedTensor(tensor=tensor, dim_names=[dim_names], batch_dims=1)


def test_named_tensor_permute():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    dim_name_permutation = ["C", "A", "B"]

    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)
    permuted_named_tensor = named_tensor.permute(dim_name_permutation)

    assert np.all(permuted_named_tensor.dim_names == np.asarray(dim_name_permutation))
    assert torch.all(permuted_named_tensor.tensor == tensor.permute(0, 3, 1, 2))


def test_named_tensor_permute_fail():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)
    # Missing dim name
    with pytest.raises(AssertionError):
        permuted_named_tensor = named_tensor.permute(["B", "A"])
    # Extra dim name
    with pytest.raises(AssertionError):
        permuted_named_tensor = named_tensor.permute(["B", "A", "C", "D"])
    # Duplicated dim name
    with pytest.raises(AssertionError):
        permuted_named_tensor = named_tensor.permute(["B", "A", "C", "C"])


def test_named_tensor_get_dim_inds():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    assert np.all(named_tensor.get_dim_inds(["C", "A"]) == np.array([3, 1]))
    assert np.all(named_tensor.get_dim_inds(["C", "C", "A"]) == np.array([3, 3, 1]))
    assert np.all(named_tensor.get_dim_inds([]) == np.array([]))


def test_named_tensor_get_dim_inds_fail():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    # Extra dim name
    with pytest.raises(KeyError):
        dim_inds = named_tensor.get_dim_inds(["C", "D"])


def test_named_tensor_active_dims():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 1, 3, 4, 1)
    dim_names = ["A", "**", "B", "C", "*"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)
    assert np.all(np.asarray(["A", "B", "C"]) == named_tensor.active_dims())


def test_named_tensor_add_basic():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    new_named_tensor = named_tensor + named_tensor
    assert torch.all(
        new_named_tensor.permute(dim_names).tensor == named_tensor.tensor * 2
    )
    assert {*new_named_tensor.dim_names} == {*named_tensor.dim_names}
    assert new_named_tensor.batch_dims == named_tensor.batch_dims


def test_named_tensor_add_broadcast():
    dim_names = ["A", "B", "C"]
    tensor1 = torch.arange(5 * 6, dtype=torch.float).reshape(5, 2, 3)
    tensor2 = torch.arange(5 * 8, dtype=torch.float).reshape(5, 2, 4)

    named_tensor1 = NamedTensor(tensor=tensor1, dim_names=dim_names[:2], batch_dims=1)
    named_tensor2 = NamedTensor(
        tensor=tensor2, dim_names=[dim_names[0], dim_names[2]], batch_dims=1
    )
    new_named_tensor = (named_tensor1 + named_tensor2).permute(dim_names)

    assert new_named_tensor.shape == (5, 2, 3, 4)
    assert torch.all(
        new_named_tensor.tensor == tensor1[..., None] + tensor2[..., None, :]
    )
    assert set(new_named_tensor.dim_names) == set(dim_names)
    assert new_named_tensor.batch_dims == named_tensor1.batch_dims


def test_named_tensor_add_fail():
    tensor = torch.arange(5 * 6, dtype=torch.float).reshape(5, 2, 3)
    # Differing number of batch dims
    with pytest.raises(AssertionError):
        ntensor1 = NamedTensor(tensor=tensor, dim_names=["A", "B"], batch_dims=1)
        ntensor2 = NamedTensor(tensor=tensor[None], dim_names=["A", "B"], batch_dims=2)
        ntensor1 + ntensor2


def test_named_tensor_logsumexp():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4) + 1
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    # No dim provided
    full_reduced = named_tensor.logsumexp()
    assert torch.all(full_reduced == torch.logsumexp(tensor, dim=[1, 2, 3]))

    # Some dims provided, do not keep reduced dims
    reduced = named_tensor.logsumexp(dim=["A", "C"])
    assert torch.all(reduced.tensor == torch.logsumexp(tensor, dim=[1, 3]))
    assert np.all(reduced.dim_names == np.asarray(["B"]))
    assert reduced.batch_dims == named_tensor.batch_dims

    # Some dims provided, keep reduced dims
    reduced = named_tensor.logsumexp(dim=["A", "C"], keepdim=True)
    assert torch.all(
        reduced.tensor == torch.logsumexp(tensor, dim=[1, 3], keepdim=True)
    )
    assert np.all(reduced.dim_names == named_tensor.dim_names)
    assert reduced.batch_dims == named_tensor.batch_dims


def test_named_tensor_unsqueeze():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    # Same order
    new_dims = ["A", "Z", "X", "B", "C", "Y"]
    new_named_tensor = named_tensor.unsqueeze(new_dims)
    assert torch.all(
        new_named_tensor.tensor == named_tensor.tensor.reshape(5, 2, 1, 1, 3, 4, 1)
    )
    assert np.all(new_named_tensor.dim_names == np.asarray(new_dims))
    assert new_named_tensor.batch_dims == named_tensor.batch_dims

    # Different order
    new_dims = ["B", "Z", "X", "A", "C", "Y"]
    new_named_tensor = named_tensor.unsqueeze(new_dims)
    assert torch.all(
        new_named_tensor.tensor
        == named_tensor.tensor.permute(0, 2, 1, 3).reshape(5, 3, 1, 1, 2, 4, 1)
    )
    assert np.all(new_named_tensor.dim_names == np.asarray(new_dims))
    assert new_named_tensor.batch_dims == named_tensor.batch_dims


def test_named_tensor_unsqueeze_fail():
    tensor = torch.arange(5 * 24, dtype=torch.float).reshape(5, 2, 3, 4)
    dim_names = ["A", "B", "C"]
    named_tensor = NamedTensor(tensor=tensor, dim_names=dim_names, batch_dims=1)

    # Repeated dim name
    with pytest.raises(AssertionError):
        new_dims = ["A", "Z", "Z", "B", "C", "Y"]
        new_named_tensor = named_tensor.unsqueeze(new_dims)

    # Missing original dimensions
    with pytest.raises(AssertionError):
        new_dims = ["B", "Z", "X", "C", "Y"]
        new_named_tensor = named_tensor.unsqueeze(new_dims)


def test_dynamic_get():
    tensor = torch.randn(2, 3, 4)
    assert torch.all(dynamic_get(tensor, 1, 2) == tensor[:, 2, :])
    assert torch.all(dynamic_get(tensor, 1, [0, 2]) == tensor[:, [0, 2], :])
    assert torch.all(dynamic_get(tensor, [0, 2], [0, 2]) == tensor[0, :, 2])


def test_device_manager_default():
    manager = DeviceManager()
    assert not manager.no_cuda
    assert manager.seed is None


@pytest.mark.parametrize("no_cuda", [True, False], ids=["cpu", "cuda"])
def test_device_manager_no_cuda(no_cuda):
    manager = DeviceManager(no_cuda=no_cuda)
    assert (not manager.use_cuda) == (no_cuda or (not torch.cuda.is_available()))
    assert manager.device == ("cuda:0" if manager.use_cuda else "cpu")


@pytest.mark.parametrize(
    "no_cuda,seed", [[True, 1234], [True, None], [False, 1234], [False, None]]
)
def test_device_manager_seed(no_cuda, seed):
    manager = DeviceManager(no_cuda=no_cuda, seed=seed)
    assert manager.seed == seed

    other_manager = DeviceManager(no_cuda=no_cuda, seed=seed)
    if seed is not None:
        assert torch.all(
            manager.generator.get_state() == other_manager.generator.get_state()
        )


@pytest.mark.parametrize("no_cuda", [True, False], ids=["cpu", "cuda"])
def test_device_manager_generate_seed(no_cuda):
    seed = 1234

    manager = DeviceManager(no_cuda=no_cuda, seed=seed)
    other_manager = DeviceManager(no_cuda=no_cuda, seed=seed)

    assert manager.generate_seed() == other_manager.generate_seed()
    assert manager.generate_seed() != manager.generate_seed()


@pytest.mark.parametrize("no_cuda", [True, False], ids=["cpu", "cuda"])
def test_device_manager_clone(no_cuda):
    seed = 1234
    manager = DeviceManager(no_cuda=no_cuda, seed=seed)
    new_manager = manager.clone()

    assert manager.device == new_manager.device
    assert new_manager.generator is not manager.generator
    assert torch.all(new_manager.generator.get_state() == manager.generator.get_state())
    assert new_manager.generate_seed() == manager.generate_seed()
