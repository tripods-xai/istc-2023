from typing import Literal, Tuple, Dict, Callable, Union
from pprint import pprint
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchinfo import summary

from ..encoders import Encoder
from ..decoders import SoftDecoder, JunctionTreeDecoder
from ..channels import NoisyChannel
from ..modulation import Normalization, IdentityModulation
from ..engine import ResultsProcessor
from ..utils import (
    DEFAULT_DEVICE_MANAGER,
    DeviceManager,
    safe_create_file,
    BatchChunker,
    get_timestamp,
)
from ..constants import CHECKPOINTS_DIR

from .trainer import CodingTrainer, canonicalize_schedules

TrainType = Literal["joint", "encoder", "decoder"]


class TurboAETrainer(CodingTrainer):
    def __init__(
        self,
        input_size: int,
        encoder: Encoder,
        decoder: SoftDecoder,
        encoder_channel: NoisyChannel,
        decoder_channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        jtree_decoder: Union[
            JunctionTreeDecoder, BatchChunker[JunctionTreeDecoder]
        ] = None,
        output_path=None,
        batch_normalization=False,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ) -> None:
        self._input_size = input_size
        self.modulator = (
            Normalization(device_manager=device_manager)
            if batch_normalization
            else IdentityModulation(device_manager=device_manager)
        )
        self.encoder_channel = encoder_channel
        self.decoder_channel = decoder_channel
        self.validation_channel = validation_channel
        self.encoder = encoder
        self.decoder = decoder
        self._encoder_decoder = nn.ModuleDict(
            {"encoder": self.encoder, "decoder": self.decoder}
        )
        self.jtree_decoder = jtree_decoder

        self.output_path = output_path

        super().__init__(device_manager)

    def sync_generator(self, other_device_manager: DeviceManager):
        raise NotImplementedError()

    @torch.no_grad()
    def run_jtree_decoding(
        self,
        num_jtree_steps: int,
        batch_size: int,
        num_validation_steps=20,
        # Other details for forward
        **forward_kwargs,
    ):
        results_processor = ResultsProcessor([])

        for i, input_batches in enumerate(
            self.data_gen(
                num_steps=num_jtree_steps, batch_size=batch_size, num_batches=1
            )
        ):
            input_batch = list(input_batches)[0]
            print(f"JTree Step {i+1}/{num_validation_steps}")
            start = time.time()
            modulated = self.run_encoder(
                input_batch,
                with_grad=False,
            )
            corrupted = self.validation_channel(modulated)
            decoded = self.run_decoder(corrupted, with_grad=False)

            # Run the Jtree on cpu
            jtree_decoded = self.jtree_decoder(corrupted)

            ### TODO: Pick up work here. We need to
            # - compute the XE of the jtree decoded
            # - compare against XE of decoded
            # - add early stopping to sampling for when we reach some threshold (e.g. error bar of width 0.2x mean)
            _, metrics = self.metrics(input_batch, logits, no_mean=True)
            results_processor.update({**metrics, **forward_metrics})
            print(f"Took {time.time() - start}s")

        return results_processor.results

    @property
    def input_size(self):
        return self._input_size

    def run_encoder(
        self, input_batch: torch.Tensor, with_grad=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if with_grad:
            modulated = self.modulator(self.encoder(input_batch))
        else:
            with torch.no_grad():
                modulated = self.modulator(self.encoder(input_batch))

        return modulated

    def run_decoder(
        self, corrupted_batch: torch.Tensor, with_grad=False
    ) -> torch.Tensor:
        if with_grad:
            decoded_batch = self.decoder(corrupted_batch)
        else:
            with torch.no_grad():
                decoded_batch = self.decoder(corrupted_batch)

        return decoded_batch

    def forward(
        self,
        input_batch: torch.Tensor,
        train_type: TrainType = None,
        validate=False,
    ):
        if train_type is None and (not validate):
            raise ValueError(
                "If train type is not provided, validate must be set to true."
            )
        modulated = self.run_encoder(
            input_batch,
            with_grad=(not validate) and (train_type in ["joint", "encoder"]),
        )

        meandim = list(range(1, modulated.ndim)) if validate else None
        metrics = {
            "modulated_power": torch.mean(modulated**2, dim=meandim),
            "modulated_center": torch.mean(modulated, dim=meandim),
        }
        if validate:
            channel = self.validation_channel
        elif train_type == "joint":
            raise NotImplementedError
        elif train_type == "decoder":
            channel = self.decoder_channel
        elif train_type == "encoder":
            channel = self.encoder_channel
        else:
            raise NotImplementedError

        corrupted = channel(modulated)
        decoded = self.run_decoder(corrupted, with_grad=not validate)

        return decoded, metrics

    @property
    def trainable_module(self):
        return self._encoder_decoder

    def reevaluate(self, inputs: torch.Tensor, validate: bool = False):
        raise NotImplementedError

    def apply_constraint(self):
        # For now we will not apply a constraint.
        # Ideally with more work, we would want to
        # get the table from the neural network, then
        # apply the constraint. This needs some experimentation
        # to make sure performance is not degraded by using front and
        # first padding.
        pass

    def summary(self) -> None:
        print("Encoder:")
        summary(self.encoder, input_size=(1, self.input_size), depth=5)
        print("Encoder Settings:")
        pprint(self.encoder.settings())
        print("Decoder:")
        summary(self.decoder, input_size=(1, self.input_size, 3), depth=5)
        print("Decoder Settings:")
        pprint(self.decoder.settings())
        print("Modulator:")
        pprint(self.modulator.settings())
        print("Encoder Channel:")
        pprint(self.encoder_channel.settings())
        print("Decoder Channel:")
        pprint(self.decoder_channel.settings())

    def run_epoch(
        self,
        optimizer: Optimizer,
        train_type: TrainType,
        cur_steps_per_epoch: int,
        cur_batch_size: int,
        cur_batches_per_update: int,
        param_names,
    ):
        for _, input_batches in enumerate(
            self.data_gen(
                num_steps=cur_steps_per_epoch,
                batch_size=cur_batch_size,
                num_batches=cur_batches_per_update,
            ),
        ):
            start = time.time()

            prev_params = {
                k: v.detach().clone()
                for k, v in self.named_parameters()
                if k in param_names
            }
            prev_grads = {
                k: v.grad.detach().clone()
                for k, v in self.named_parameters()
                if (v.grad is not None) and (k in param_names)
            }

            res = self.train_step(
                input_batches=input_batches,
                optimizer=optimizer,
                prev_grads=prev_grads,
                prev_params=prev_params,
                param_filter=param_names,
                back_tracking=False,
                train_type=train_type,
            )

            param_metrics = self.param_metrics()

            yield {
                **res,
                **{k: v.item() for k, v in param_metrics.items()},
                "train_type": train_type,
                "total_steps_per_epoch": cur_steps_per_epoch,
                "time": time.time() - start,
            }
            ###################################
            # TODO: Consider creating a convergence criterion that looks at history
            # if res["converged"]:
            #     print("Training converged.")
            #     break

    def write_checkpoint(self, epoch: int):
        filename = f"turboae_trainer_ep{epoch}_{get_timestamp()}.pt"
        checkpoint_path = CHECKPOINTS_DIR / filename
        print(f"Writing checkpoint to: {checkpoint_path}")
        torch.save(
            self._encoder_decoder.state_dict(),
            safe_create_file(checkpoint_path),
        )

    def train(
        self,
        encoder_optimizer_factory: Callable[..., Optimizer],
        decoder_optimizer_factory: Callable[..., Optimizer],
        num_epochs: Union[int, list],
        batch_size: Union[int, list],
        batches_per_update: Union[int, list] = 1,
        # TODO: Verify these are the correct defaults
        encoder_steps_per_epoch: Union[int, list] = 1,
        decoder_steps_per_epoch: Union[int, list] = 5,
        save_every=10,  # # of epochs between saving
        validate_every=10,  # # of epochs between validation
        num_validation_steps=5,
        save_optimizer=True,
        mode: Literal["alternating", "joint"] = "alternating",
        write_checkpoints=False,
    ):
        if mode != "alternating":
            raise NotImplementedError("Only `alternating` training is implemented.")

        (
            num_epochs,
            batch_size,
            batches_per_update,
            encoder_steps_per_epoch,
            decoder_steps_per_epoch,
        ) = canonicalize_schedules(
            num_epochs,
            batch_size,
            batches_per_update,
            encoder_steps_per_epoch,
            decoder_steps_per_epoch,
        )

        self.apply_constraint()
        self.summary()

        encoder_optimizer = encoder_optimizer_factory(self.encoder.parameters())
        decoder_optimizer = decoder_optimizer_factory(self.decoder.parameters())
        param_names = list(zip(*self.named_parameters()))[0]
        encoder_param_names = [
            name for name in param_names if name.split(".")[0] == "encoder"
        ]
        decoder_param_names = [
            name for name in param_names if name.split(".")[0] == "decoder"
        ]

        epoch = 0
        if write_checkpoints:
            self.write_checkpoint(epoch)

        for (
            cur_num_epochs,
            cur_batch_size,
            cur_batches_per_update,
            cur_encoder_steps_per_epoch,
            cur_decoder_steps_per_epoch,
        ) in zip(
            num_epochs,
            batch_size,
            batches_per_update,
            encoder_steps_per_epoch,
            decoder_steps_per_epoch,
        ):
            print(
                f"""
                Running {cur_batches_per_update} batches per update of batch size {cur_batch_size} 
                for {cur_num_epochs} epochs with {cur_encoder_steps_per_epoch} encoder steps and 
                {cur_decoder_steps_per_epoch} decoder steps.
                """
            )
            # TODO: Need to do different channel behavior for encoder and decoder
            epoch_time = 0
            for cur_epoch, epoch in enumerate(
                range(epoch + 1, epoch + 1 + cur_num_epochs)
            ):
                if (self.validation_channel is not None) and (
                    epoch % validate_every == 0
                ):
                    validation_results = self.validate(
                        batch_size=cur_batch_size,
                        num_validation_steps=num_validation_steps,
                    )
                    yield {
                        **validation_results,
                        "epoch": epoch,
                        "num_validation_steps": num_validation_steps,
                        "type": "validation",
                    }
                # JTree Measurements
                if ...:

                    ...
                if (self.output_path is not None) and (epoch % save_every == 0):
                    print("Saving turboae encoder-decoder pair")
                    torch.save(
                        self._encoder_decoder.state_dict(),
                        safe_create_file(self.output_path),
                    )
                    if save_optimizer:
                        print("Saving optimizer")
                        torch.save(
                            {
                                "encoder_optimizer": encoder_optimizer.state_dict(),
                                "decoder_optimizer": decoder_optimizer.state_dict(),
                            },
                            safe_create_file(str(self.output_path) + ".opt"),
                        )

                epoch_start = time.time()
                # Encoder
                for i, res in enumerate(
                    self.run_epoch(
                        optimizer=encoder_optimizer,
                        train_type="encoder",
                        cur_steps_per_epoch=cur_encoder_steps_per_epoch,
                        cur_batch_size=cur_batch_size,
                        cur_batches_per_update=cur_batches_per_update,
                        param_names=encoder_param_names,
                    )
                ):
                    yield {
                        **res,
                        "epoch": epoch,
                        "step": i,
                        "total": cur_encoder_steps_per_epoch,
                        "total_steps_per_epoch": cur_num_epochs,
                        "avg_epoch_time": epoch_time,
                    }
                # Decoder
                for i, res in enumerate(
                    self.run_epoch(
                        optimizer=decoder_optimizer,
                        train_type="decoder",
                        cur_steps_per_epoch=cur_decoder_steps_per_epoch,
                        cur_batch_size=cur_batch_size,
                        cur_batches_per_update=cur_batches_per_update,
                        param_names=decoder_param_names,
                    )
                ):
                    yield {
                        **res,
                        "epoch": epoch,
                        "step": i,
                        "total_epochs": cur_num_epochs,
                        "total": cur_decoder_steps_per_epoch,
                        "avg_epoch_time": epoch_time,
                    }
                this_epoch_time = time.time() - epoch_start
                epoch_time = (epoch_time * cur_epoch + this_epoch_time) / (
                    cur_epoch + 1
                )
                print(f"This epoch time: {this_epoch_time}.")
                print(f"Average Epoch Time: {epoch_time}.")
                if write_checkpoints:
                    self.write_checkpoint(epoch)

            if self.validation_channel is not None:
                validation_results = self.validate(
                    batch_size=cur_batch_size,
                    num_validation_steps=num_validation_steps,
                )
                yield {
                    **validation_results,
                    "epoch": epoch,
                    "num_validation_steps": num_validation_steps,
                    "type": "validation",
                }
            if self.output_path is not None:
                print("Saving turboae encoder-decoder pair")
                torch.save(
                    self._encoder_decoder.state_dict(),
                    safe_create_file(self.output_path),
                )
                if save_optimizer:
                    print("Saving optimizer")
                    torch.save(
                        {
                            "encoder_optimizer": encoder_optimizer.state_dict(),
                            "decoder_optimizer": decoder_optimizer.state_dict(),
                        },
                        safe_create_file(str(self.output_path) + ".opt"),
                    )
