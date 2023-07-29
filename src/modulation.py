from typing import Dict, Any
import abc

import torch

from .utils import (
    check_signed,
    WithSettings,
    ModuleExtension,
)
from .utils import EPSILON, DeviceManager, DEFAULT_DEVICE_MANAGER


class Modulator(ModuleExtension, WithSettings, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def demodulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        pass

    def modulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        return self(y, dtype=dtype)


class BPSK(Modulator):
    def forward(self, x: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        return 2 * x.to(dtype) - 1

    def demodulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        # Needs to be a signed input
        return ((torch.sign(check_signed(y)) + 1) / 2).to(dtype)

    def settings(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


class Normalization(Modulator):
    def __init__(self, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER) -> None:
        super().__init__(device_manager)
        self._std = None
        self._mean = None

    def forward(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        assert dtype in [torch.float, torch.double]
        self._std, self._mean = torch.std_mean(y.to(dtype))
        return self.modulate(y, dtype=dtype)

    def modulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        if (self._std is None) or (self._mean is None):
            raise ValueError(
                "You must run the Normalizer over a batch first to determine std and mean."
            )
        return ((y - self._mean) / (self._std + EPSILON)).to(dtype)

    def demodulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        return self.forward(y, dtype=dtype)

    def settings(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


class IdentityModulation(Modulator):
    def forward(self, x: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        return x.to(dtype)

    def demodulate(self, y: torch.Tensor, dtype=torch.float) -> torch.Tensor:
        return self.forward(y, dtype=dtype)

    def settings(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


_MODULATOR_REGISTRY = {
    "bpsk": BPSK,
    "normalizer": Normalization,
    "identity": IdentityModulation,
}


def get_modulator(modulator_type: str):
    return _MODULATOR_REGISTRY[modulator_type]()
