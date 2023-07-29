import abc
from typing import Dict, Any

import torch

from .utils import ModuleExtension, WithSettings, DeviceManager, DEFAULT_DEVICE_MANAGER
from .constants import TURBOAE_INTERLEAVER_PERMUTATION


class Interleaver(ModuleExtension, WithSettings):
    @abc.abstractmethod
    def deinterleave(self, msg: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def interleave(self, msg: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def interleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        pass

    @abc.abstractmethod
    def deinterleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        pass

    @property
    @abc.abstractmethod
    def batch_dependent(self) -> bool:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class FixedPermuteInterleaver(Interleaver):
    def __init__(
        self,
        input_size: int,
        permutation: torch.LongTensor = None,
        depermutation: torch.LongTensor = None,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(device_manager=device_manager)

        self.input_size = input_size
        if permutation is None:
            permutation = torch.randperm(
                self.input_size,
                generator=self.device_manager.generator,
                device=self.device_manager.device,
            )
        self.register_buffer("permutation", permutation)
        if depermutation is None:
            depermutation = torch.argsort(self.permutation)
        self.register_buffer("depermutation", depermutation)

        self.batch_size = None
        self.validate()

    def validate(self):
        assert len(self.permutation) == self.input_size == len(self.depermutation)
        assert torch.all(
            self.permutation[self.depermutation]
            == torch.arange(self.input_size, device=self.device_manager.device)
        )

    def __len__(self):
        return self.input_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        `data` is a Nd tensor with shape Batch x Time x ...
        We interleave over the time dimension
        """
        self.batch_size = data.shape[0]
        return self.interleave(data)

    def deinterleave(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, self.depermutation]

    def interleave(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, self.permutation]

    def interleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes : torch.LongTensor
            The indexes to call the permutation function on. The same
            fixed permutation will be used on each input.
        """
        if isinstance(indexes, int):
            indexes = torch.tensor(
                indexes, dtype=torch.long, device=self.device_manager.device
            )
        return self.permutation[indexes]

    def deinterleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes (Batch_size x ...) : torch.LongTensor
            The indexes to call the depermutation function on. The same
            fixed depermutation will be used on each batch. If it is an int, it will
            be expanded to be a LongTensor of shape (Batch_size)
        """
        if isinstance(indexes, int):
            indexes = torch.tensor(
                indexes, dtype=torch.long, device=self.device_manager.device
            )
        return self.depermutation[indexes]

    def settings(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
        }

    def long_settings(self) -> Dict[str, Any]:
        return {
            **self.settings(),
            "permutation": self.permutation.tolist(),
            "depermutation": self.depermutation.tolist(),
        }

    @property
    def batch_dependent(self) -> bool:
        return False


class TurboAEInterleaver(FixedPermuteInterleaver):
    def __init__(
        self,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):

        super().__init__(
            input_size=len(TURBOAE_INTERLEAVER_PERMUTATION),
            permutation=TURBOAE_INTERLEAVER_PERMUTATION.to(
                device=device_manager.device
            ),
            device_manager=device_manager,
        )


class BatchRandomPermuteInterleaver(Interleaver):
    def __init__(
        self, input_size: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER
    ):
        super().__init__(device_manager=device_manager)
        self.input_size = input_size
        # Put the underscore to mark that these should not be accessed.
        # They are ephemeral.
        self._permutation = None
        self._depermutation = None

    def __len__(self):
        return self.input_size

    def generate_permutation(self):
        permutation = torch.randperm(
            self.input_size,
            generator=self.device_manager.generator,
            device=self.device_manager.device,
        )
        depermutation = torch.argsort(permutation)

        return permutation, depermutation

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        `data` is a Nd tensor with shape Batch x Time x ...
        We interleave over the time dimension.
        `call` will generate new permutations. If we need to repeat
        the interleaving or deinterleave, we should use the `interleave`
        or `deinterleave` methods.
        """
        self._permutation, self._depermutation = self.generate_permutation()
        return self.interleave(data)

    def deinterleave(self, data: torch.Tensor) -> torch.Tensor:
        if self._depermutation is None:
            raise ValueError(
                "Depermutation is not defined, you must run forward first."
            )
        return data[:, self._depermutation]

    def interleave(self, data: torch.Tensor) -> torch.Tensor:
        if self._permutation is None:
            raise ValueError("Permutation is not defined, you must run forward first.")
        return data[:, self._permutation]

    def interleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes (Batch_size x ...) : torch.LongTensor
            The indexes to call the permutation function on. The same
            fixed permutation will be used on each input.
        """
        if isinstance(indexes, int):
            indexes = torch.tensor(
                indexes, dtype=torch.long, device=self.device_manager.device
            )
        return self._permutation[indexes]

    def deinterleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes (Batch_size x ...) : torch.LongTensor
            The indexes to call the depermutation function on. The same
            fixed depermutation will be used on each batch. If it is an int, it will
            be expanded to be a LongTensor of shape (Batch_size)
        """
        if isinstance(indexes, int):
            indexes = torch.tensor(
                indexes, dtype=torch.long, device=self.device_manager.device
            )
        return self._depermutation[indexes]

    def settings(self) -> Dict[str, Any]:
        return {"input_size": self.input_size}

    @property
    def batch_dependent(self) -> bool:
        return True


class RandomPermuteInterleaver(Interleaver):
    def __init__(
        self, input_size: int, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER
    ):
        super().__init__(device_manager=device_manager)
        self.input_size = input_size
        # Put the underscore to mark that these should not be accessed.
        # They are ephemeral.
        self._permutation = None
        self._depermutation = None

    def __len__(self):
        return self.input_size

    def generate_permutations(self, batch_size):
        perms = torch.empty(
            (batch_size, self.input_size),
            dtype=torch.long,
            device=self.device_manager.device,
        )
        for i in range(batch_size):
            torch.randperm(
                self.input_size, generator=self.device_manager.generator, out=perms[i]
            )

        deperms = torch.argsort(perms, dim=-1)

        return perms, deperms

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        `data` is a Nd tensor with shape Batch x Time x ...
        We interleave over the time dimension.
        `call` will generate new permutations. If we need to repeat
        the interleaving or deinterleave, we should use the `interleave`
        or `deinterleave` methods.
        """
        batch_size = data.shape[0]
        self._permutation, self._depermutation = self.generate_permutations(batch_size)
        return self.interleave(data)

    def deinterleave(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = self._depermutation.shape[0]
        return data[
            torch.arange(batch_size, device=self.device_manager.device)[:, None],
            self._depermutation,
        ]

    def interleave(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = self._permutation.shape[0]
        return data[
            torch.arange(batch_size, device=self.device_manager.device)[:, None],
            self._permutation,
        ]

    def interleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes (Batch_size x ...) : torch.LongTensor
            The indexes to call the permutation function on. The permutation
            corresponding to the particular batch will be used. If `indexes`
            is an `int`, then the output will be of shape (Batch_size).
        """
        if isinstance(indexes, int):
            return self._permutation[:, indexes]
        else:
            batch_size = self._permutation.shape[0]
            return self._permutation[
                torch.arange(batch_size, device=self.device_manager.device)[:, None],
                indexes.reshape(batch_size, -1),
            ].reshape(indexes.shape)

    def deinterleave_index(self, indexes: torch.LongTensor) -> torch.LongTensor:
        """
        Parameters
        ----------
        indexes (Batch_size x ...) : torch.LongTensor
            The indexes to call the depermutation function on. The depermutation
            corresponding to the particular batch will be used. If `indexes`
            is an `int`, then the output will be of shape (Batch_size).
        """
        if isinstance(indexes, int):
            return self._depermutation[:, indexes]
        else:
            batch_size = self._depermutation.shape[0]
            return self._depermutation[
                torch.arange(batch_size, device=self.device_manager.device)[:, None],
                indexes.reshape(batch_size, -1),
            ].reshape(indexes.shape)

    def settings(self) -> Dict[str, Any]:
        return {"input_size": self.input_size}

    @property
    def batch_dependent(self) -> bool:
        return True
