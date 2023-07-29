from typing import Callable, Union
import time
from pprint import pprint

import torch
from torch.optim import Optimizer
import torch.nn as nn
from torchinfo import summary

from ..utils import DEFAULT_DEVICE_MANAGER, safe_create_file
from ..decoders import TurboAEDecoder
from ..encoders import Encoder
from ..modulation import Modulator
from ..channels import NoisyChannel

from .trainer import CodingTrainer, canonicalize_schedules


class DecoderTrainer(CodingTrainer):
    def __init__(
        self,
        decoder: TurboAEDecoder,
        encoder: Encoder,
        modulator: Modulator,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ) -> None:
        self.device_manager = device_manager
        self.decoder = decoder
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.validation_channel = validation_channel

        self.output_path = output_path

    def forward(self, input_batch: torch.Tensor, validate=False):
        channel = self.validation_channel if validate else self.channel
        with torch.no_grad():
            modulated = self.modulator(self.encoder(input_batch))
            meandim = list(range(1, modulated.ndim)) if validate else None
            metrics = {
                "modulated_power": torch.mean(modulated**2, dim=meandim),
                "modulated_center": torch.mean(modulated, dim=meandim),
            }
            received = channel(modulated)

        return self.decoder(received), metrics

    @property
    def trainable_module(self) -> nn.Module:
        return self.decoder

    @property
    def input_size(self) -> int:
        return self.encoder.input_size

    def summary(self) -> None:
        print("Decoder:")
        pprint(self.decoder.settings())
        summary(self.decoder)

        print("Encoder:")
        pprint(self.encoder.settings())
        summary(self.encoder)

        print("Channel:")
        pprint(self.channel.settings())
        print("Modulator:")
        pprint(self.modulator.settings())
        print("Validation Channel:")
        pprint(self.validation_channel.settings())

    def train(
        self,
        optimizer_factory: Callable[..., Optimizer],
        num_steps: Union[int, list],
        batch_size: Union[int, list],
        batches_per_update: Union[int, list] = 1,
        save_every=10,
        validate_every=10,
        num_validation_steps=5,
        save_optimizer=True,
    ):
        num_steps, batch_size, batches_per_update = canonicalize_schedules(
            num_steps, batch_size, batches_per_update
        )
        total_steps = sum(num_steps)

        self.apply_constraint()
        self.summary()

        optimizer = optimizer_factory(self.parameters())

        i = -1
        for cur_num_steps, cur_batch_size, cur_batches_per_update in zip(
            num_steps, batch_size, batches_per_update
        ):
            print(
                f"Running {cur_batches_per_update} batches per update of batch size {cur_batch_size} for {cur_num_steps} steps"
            )
            start = i + 1

            for i, input_batches in enumerate(
                self.data_gen(
                    num_steps=cur_num_steps,
                    batch_size=cur_batch_size,
                    num_batches=cur_batches_per_update,
                ),
                start=start,
            ):
                ####################
                start = time.time()
                prev_params = {
                    k: v.detach().clone() for k, v in self.named_parameters()
                }
                prev_grads = {
                    k: v.grad.detach().clone()
                    for k, v in self.named_parameters()
                    if v.grad is not None
                }
                ##########################

                if i % validate_every == 0 and self.validation_channel is not None:
                    validation_results = self.validate(
                        batch_size=cur_batch_size,
                        num_validation_steps=num_validation_steps,
                    )
                    yield {
                        **validation_results,
                        "step": i,
                        "num_validation_steps": num_validation_steps,
                        "type": "validation",
                    }

                if i % save_every == 0 and i != 0 and self.output_path is not None:
                    print("Saving neural decoder")
                    torch.save(
                        self.trainable_module.state_dict(),
                        safe_create_file(self.output_path),
                    )
                    if save_optimizer:
                        print("Saving optimizer")
                        torch.save(
                            optimizer.state_dict(),
                            safe_create_file(str(self.output_path) + ".opt"),
                        )

                #########################
                res = self.train_step(
                    input_batches=input_batches,
                    optimizer=optimizer,
                    prev_grads=prev_grads,
                    prev_params=prev_params,
                )

                param_metrics = self.param_metrics()

                yield {
                    **res,
                    **{k: v.item() for k, v in param_metrics.items()},
                    "step": i,
                    "total": total_steps,
                    "time": time.time() - start,
                }
                ###################################

            if self.validation_channel is not None:
                validation_results = self.validate(
                    batch_size=cur_batch_size,
                    num_validation_steps=num_validation_steps,
                )
                yield {
                    **validation_results,
                    "step": i,
                    "num_validation_steps": num_validation_steps,
                    "type": "validation",
                }

            if self.output_path is not None:
                print("Saving neural decoder")
                torch.save(
                    self.trainable_module.state_dict(),
                    safe_create_file(self.output_path),
                )
                if save_optimizer:
                    print("Saving optimizer")
                    torch.save(
                        optimizer.state_dict(),
                        safe_create_file(str(self.output_path) + ".opt"),
                    )
