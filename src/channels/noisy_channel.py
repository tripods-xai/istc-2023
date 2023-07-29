from typing import Dict, Any
import abc

import torch
from torch.distributions import Normal, Bernoulli

from ..utils import (
    ModuleExtension,
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    snr2sigma,
    WithSettings,
    snr2sigma_torch,
)


class NoisyChannel(ModuleExtension, WithSettings, metaclass=abc.ABCMeta):
    def __init__(self, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER) -> None:
        super().__init__(device_manager=device_manager)

    @abc.abstractmethod
    def demodulate(self, y: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def apply_error_vector(
        self, y_hat: torch.Tensor, error_vector: torch.BoolTensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def corrupt(self, x: torch.Tensor) -> torch.FloatTensor:
        pass

    @abc.abstractmethod
    def log_prob(
        self, y: torch.Tensor, x_prime: torch.Tensor, dtype=torch.float
    ) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            A tensor of same shape as y and x_prime with entry log P(y | x_prime)
        """
        pass


class AWGN(NoisyChannel):
    def __init__(
        self, snr: float, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER
    ) -> None:
        super().__init__(device_manager)
        self.snr = snr
        self.sigma = snr2sigma(self.snr)
        self.distribution = Normal(0.0, self.sigma)
        self._noise = None

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        self._noise = (
            self.distribution.stddev
            * torch.randn(
                x.size(),
                generator=self.device_manager.generator,
                device=self.device_manager.device,
            )
            + self.distribution.mean
        )
        return self.corrupt(x)

    def corrupt(self, x: torch.Tensor) -> torch.FloatTensor:
        if self._noise is None:
            raise ValueError("Need to call forward first.")
        return x.to(torch.float) + self._noise

    # TODO: This should be moved to a Modulator
    def demodulate(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sign(y)

    def apply_error_vector(
        self, y_hat: torch.Tensor, error_vector: torch.BoolTensor
    ) -> torch.Tensor:
        return torch.where(error_vector, -1.0, 1.0) * y_hat

    def log_prob(
        self, y: torch.Tensor, x_prime: torch.Tensor, dtype=torch.float
    ) -> torch.Tensor:
        """
        Returns log f(Z = y-x_prime), where Z is RV for noise and f
        is pdf for Gaussian.
        """
        # print(y.shape)
        # print(x_prime.shape)
        noise = y.to(dtype) - x_prime.to(dtype)
        return self.distribution.log_prob(noise).to(dtype)

    def settings(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "snr": self.snr, "sigma": self.sigma}

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


### NOTE: Implementation Change 2023-05-03.
### No longer sampling uniformly between snr, but
### now between sigma like Jiang.
class VariableAWGN(NoisyChannel):
    def __init__(
        self,
        snr_low: float,
        snr_high: float,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager)
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.sigma_low = snr2sigma(self.snr_low)
        self.sigma_high = snr2sigma(self.snr_high)

        self.batch_sigma = None
        self._noise = None

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        batch_size = x.shape[0]
        self.batch_sigma = (self.sigma_low - self.sigma_high) * torch.rand(
            batch_size,
            generator=self.device_manager.generator,
            device=self.device_manager.device,
        ) + self.sigma_high

        # Expanding batch_sigma so dims match x.
        # self.batch_sigma[:, None...(x.ndim-1)]
        self._noise = self.batch_sigma[
            (slice(None),) + (None,) * (x.ndim - 1)
        ] * torch.randn(
            x.size(),  # Pytorch version of .shape
            generator=self.device_manager.generator,
            device=self.device_manager.device,
        )
        return self.corrupt(x)

    def corrupt(self, x: torch.Tensor) -> torch.FloatTensor:
        if self._noise is None:
            raise ValueError("Need to call forward first.")
        return x.to(torch.float) + self._noise

    # TODO: This should be moved to a Modulator
    def demodulate(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sign(y)

    def apply_error_vector(
        self, y_hat: torch.Tensor, error_vector: torch.BoolTensor
    ) -> torch.Tensor:
        return torch.where(error_vector, -1.0, 1.0) * y_hat

    def log_prob(self, y: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Returns log f(Z = y-x_prime), where Z is RV for noise and f
        is pdf for Gaussian.
        """
        raise NotImplementedError

    def settings(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "snr_low": self.snr_low,
            "sigma_low": self.sigma_low,
            "snr_high": self.snr_high,
            "sigma_high": self.sigma_high,
        }

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()


class BinarySymmetric(NoisyChannel):
    def __init__(
        self, p: float, device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER
    ) -> None:
        super().__init__(device_manager)
        self.p = p
        self.distribution = Bernoulli(probs=self.p)
        self._noise = None

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        self._noise = (
            2
            * torch.bernoulli(
                torch.full(
                    size=x.shape,
                    fill_value=self.p,
                    dtype=torch.float,
                    device=self.device_manager.device,
                ),
                generator=self.device_manager.generator,
            )
            - 1
        )
        return self.corrupt(x)

    def corrupt(self, x: torch.Tensor) -> torch.FloatTensor:
        if self._noise is None:
            raise ValueError("Need to call forward first.")
        return x.to(torch.float) * self._noise

    # TODO: This should be moved to a Modulator
    def demodulate(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def apply_error_vector(
        self, y_hat: torch.Tensor, error_vector: torch.BoolTensor
    ) -> torch.Tensor:
        return torch.where(error_vector, -1.0, 1.0) * y_hat

    def log_prob(self, y: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Returns log f(Z = (y != x_prime)), where Z is RV for noise and f
        is pmf for Bernoulli.
        """

        noise = (y.long() != x_prime.long()).float()
        return self.distribution.log_prob(noise)

    def settings(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "p": self.p}

    def long_settings(self) -> Dict[str, Any]:
        return self.settings()
