import abc

from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER, ModuleExtension, WithSettings


class HardDecoder(ModuleExtension, WithSettings, metaclass=abc.ABCMeta):
    def __init__(
        self,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager=device_manager)

    @property
    @abc.abstractclassmethod
    def source_data_len(self):
        pass


class SoftDecoder(ModuleExtension, WithSettings, metaclass=abc.ABCMeta):
    def __init__(
        self,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(device_manager=device_manager)

    @property
    @abc.abstractclassmethod
    def source_data_len(self):
        pass
