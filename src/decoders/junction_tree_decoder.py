from typing import Dict, Any, Union
import gc

import torch

from ..constants import INPUT_SYMBOL
from ..encoders import SizedEncoder
from ..modulation import Modulator
from ..channels import NoisyChannel
from ..utils import DEFAULT_DEVICE_MANAGER, DeviceManager

from .decoder import SoftDecoder


class JunctionTreeDecoder(SoftDecoder):
    def __init__(
        self,
        encoder: SizedEncoder,
        modulator: Modulator,
        channel: NoisyChannel,
        cluster_tree=None,
        elimination_seed: Union[int, None] = None,
        dtype=torch.float16,
        device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    ):
        super().__init__(device_manager)
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        if dtype is torch.float16 and device_manager.device == device_manager.CPU_NAME:
            raise ValueError(
                f"Could not use half floats because the device is {device_manager.device}."
            )
        self.dtype = dtype
        if cluster_tree is None:
            if elimination_seed is None:
                elimination_seed = self.device_manager.generate_seed()
            print(f"Elimination Seed: {elimination_seed}")
            self.cluster_tree = (
                self.encoder.dependency_graph()
                .with_elimination_ordering(seed=elimination_seed)
                .as_cluster_tree()
            )
        else:
            self.cluster_tree = cluster_tree

    @property
    def source_data_len(self) -> int:
        return self.encoder.input_size

    @property
    def num_output_channels(self):
        return 1

    def forward(
        self, received_symbols: torch.Tensor, input_symbol=INPUT_SYMBOL, check=True
    ):
        batch_size, timesteps, channels = received_symbols.shape
        # Run the rest on CPU since it is memory intensive:
        evidence = self.encoder.compute_evidence(
            received_symbols, self.channel, self.modulator
        )
        # evidence = {k: v.to(device=DeviceManager.CPU_NAME) for k,v in evidence.items()}
        evidence = {k: v.to(dtype=self.dtype) for k, v in evidence.items()}
        msg_digraph = self.cluster_tree.propogate_evidence(evidence)
        input_variables = [f"{input_symbol}_{i}" for i in range(timesteps)]
        posterior_dict = self.cluster_tree.compute_posteriors(
            msg_digraph, variables=input_variables, evidence=evidence, delete_data=True
        )
        del msg_digraph
        del evidence
        gc.collect()

        if check:
            for variable in input_variables:
                assert (
                    len(posterior_dict[variable].dim_names) == 1
                ), f"Expeced only one variable, had variables {posterior_dict[variable].dim_names}"
                assert posterior_dict[variable].dim_names[0] == variable
                assert posterior_dict[variable].shape == (batch_size, 2)
        # Batch x Time x 2 (Inputs)
        posterior_tensor = torch.stack(
            [posterior_dict[variable].tensor for variable in input_variables], 1
        )
        return posterior_tensor[:, :, 1] - posterior_tensor[:, :, 0]

    def settings(self) -> Dict[str, Any]:
        # TODO: Add more details on the ClusterTree
        return {
            "encoder": self.encoder.settings(),
            "modulator": self.modulator.settings(),
            "channel": self.channel.settings(),
        }
