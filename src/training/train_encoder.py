import abc
from typing import Callable, Union, List
import time
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..utils import (
    DEFAULT_DEVICE_MANAGER,
    safe_create_file,
    binary_entropy_with_logits,
    gen_affine_convcode_generator,
    enumerate_binary_inputs,
    DeviceManager,
    peek,
)
from ..fourier import table_to_fourier, boolean_to_fourier
from ..encoders import (
    Encoder,
    StreamedTurboEncoder,
    GeneralizedConvolutionalEncoder,
    FourierConvolutionalEncoder,
    AffineConvolutionalEncoder,
)
from ..modulation import Normalization, IdentityModulation
from ..channels import NoisyChannel
from .trainer import CodingTrainer, SwarmTrainer, canonicalize_schedules
from ..decoders import CodebookDecoder, IteratedBCJRTurboDecoder
from ..interleavers import Interleaver


class DecoderBasedSwarmEncoderTrainer(SwarmTrainer["DecoderBasedEncoderTrainer"]):
    def summary(self) -> None:
        pprint(self.swarm.settings())
        print("Summaries for first 3 agents:")
        for name, agent in zip(self.swarm.agent_names[:3], self.swarm.agents[:3]):
            print(f"Agent {name}")
            agent.summary()

    def train(
        self,
        optimizer_factory: Callable[..., Optimizer],
        num_steps: Union[int, list],
        batch_size: Union[int, list],
        batches_per_update: Union[int, list] = 1,
        save_every=10,
        validate_every=10,
        num_validation_steps=5,
        save_optimizers=True,
    ):
        num_steps, batch_size, batches_per_update = canonicalize_schedules(
            num_steps,
            batch_size,
            batches_per_update,
        )

        self.swarm.apply_constraint()
        self.summary()

        optimizer_swarm = [
            optimizer_factory(agent.parameters()) for agent in self.swarm.agents
        ]

        i = 0
        for cur_num_steps, cur_batch_size, cur_batches_per_update in zip(
            num_steps, batch_size, batches_per_update
        ):
            print(
                f"Running {cur_batches_per_update} batches per update of batch size {cur_batch_size} for {cur_num_steps} steps"
            )
            for i, input_batches in enumerate(
                self.data_gen(
                    num_steps=cur_num_steps,
                    batch_size=cur_batch_size,
                    num_batches=cur_batches_per_update,
                ),
                start=i,
            ):
                ###################################
                start = time.time()
                #########################
                heaviest_agent_i = self.swarm.heaviest_agent_i()
                heaviest_agent = self.swarm.agents[heaviest_agent_i]
                if (
                    i % validate_every == 0
                    and heaviest_agent.validation_channel is not None
                ):
                    validation_results = heaviest_agent.validate(
                        batch_size=cur_batch_size,
                        num_validation_steps=num_validation_steps,
                    )
                    agent_summary = {
                        "agent": self.swarm.agent_names[heaviest_agent_i],
                        "mass": self.swarm.masses[heaviest_agent_i].item(),
                    }
                    yield {
                        **validation_results,
                        "agent": agent_summary,
                        "step": i,
                        "num_validation_steps": num_validation_steps,
                        "type": "validation",
                    }

                if i % save_every == 0 and i != 0 and self.output_path is not None:
                    print("Saving swarm")
                    torch.save(
                        [
                            module.state_dict()
                            for module in self.swarm.trainable_modules
                        ],
                        safe_create_file(self.output_path),
                    )
                    if save_optimizers:
                        print("Saving swarm optimizers")
                        torch.save(
                            [optimizer.state_dict() for optimizer in optimizer_swarm],
                            safe_create_file(str(self.output_path) + ".opts"),
                        )

                #########################
                self.swarm, res = self.swarm.update(
                    input_batches=input_batches,
                    optimizer_swarm=optimizer_swarm,
                )

                param_metrics_swarm = [
                    {k: v.item() for k, v in agent.param_metrics().items()}
                    for agent in self.swarm.agents
                ]

                agent_summaries = {
                    f"Agent {name}": {
                        "mass": self.swarm.masses[i].item(),
                        "results": res[i],
                        "param_metrics": param_metrics_swarm[i],
                    }
                    for i, name in enumerate(self.swarm.agent_names)
                }

                yield {
                    "total_mass": torch.sum(self.swarm.masses).item(),
                    "num_agents": len(self.swarm),
                    "agents": agent_summaries,
                    "type": "training",
                    "step": i,
                    "total": cur_num_steps,
                    "time": time.time() - start,
                }
                ###################################

                if (
                    len(self.swarm) == 1
                    and agent_summaries[peek(agent_summaries)]["results"]["converged"]
                ):
                    print("Training converged.")
                    break

            heaviest_agent_i = self.swarm.heaviest_agent_i()
            heaviest_agent = self.swarm.agents[heaviest_agent_i]
            if heaviest_agent.validation_channel is not None:
                validation_results = heaviest_agent.validate(
                    batch_size=cur_batch_size,
                    num_validation_steps=num_validation_steps,
                )
                agent_summary = {
                    "agent": self.swarm.agent_names[heaviest_agent_i],
                    "mass": self.swarm.masses[heaviest_agent_i].item(),
                }
                yield {
                    "agent": agent_summary,
                    **validation_results,
                    "step": i,
                    "num_validation_steps": num_validation_steps,
                    "type": "validation",
                }

            if self.output_path is not None:
                print("Saving swarm")
                torch.save(
                    [module.state_dict() for module in self.swarm.trainable_modules],
                    safe_create_file(self.output_path),
                )
                if save_optimizers:
                    print("Saving swarm optimizers")
                    torch.save(
                        [optimizer.state_dict() for optimizer in optimizer_swarm],
                        safe_create_file(str(self.output_path) + ".opts"),
                    )


class DecoderBasedEncoderTrainer(CodingTrainer):
    def __init__(
        self,
        input_size: int,
        encoder: Encoder,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
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
        self.channel = channel
        self.validation_channel = validation_channel
        self.encoder = encoder

        self.output_path = output_path

        super().__init__(device_manager)

    def sync_generator(self, other_device_manager: DeviceManager):
        seed = other_device_manager.generate_seed()
        self.channel.device_manager.generator.manual_seed(seed)
        if self.validation_channel is not None:
            self.validation_channel.device_manager.generator.manual_seed(seed)

    @property
    def input_size(self):
        return self._input_size

    @abc.abstractmethod
    def forward(self, input_batch: torch.Tensor, validate=False):
        pass

    @property
    def trainable_module(self):
        return self.encoder

    @abc.abstractmethod
    def reevaluate(self, inputs: torch.Tensor, validate: bool = False):
        pass

    # def sync_generator(other_device_manager: DeviceManager):

    def train(
        self,
        optimizer_factory: Callable[..., Optimizer],
        num_steps: Union[int, list],
        batch_size: Union[int, list],
        batches_per_update: Union[int, list] = 1,
        save_every=10,
        validate_every=10,
        num_validation_steps=5,
        fourier_every=10,
        save_optimizer=True,
        back_tracking=False,
        descent=0.2,
        shrinkage=0.9,
        failure=10,
    ):
        num_steps, batch_size, batches_per_update = canonicalize_schedules(
            num_steps,
            batch_size,
            batches_per_update,
        )

        self.apply_constraint()
        self.summary()

        optimizer = optimizer_factory(self.parameters())
        fourier_coefs = []
        prev_params = None

        i = 0
        for cur_num_steps, cur_batch_size, cur_batches_per_update in zip(
            num_steps, batch_size, batches_per_update
        ):
            print(
                f"Running {cur_batches_per_update} batches per update of batch size {cur_batch_size} for {cur_num_steps} steps"
            )
            start = i

            for i, input_batches in enumerate(
                self.data_gen(
                    num_steps=cur_num_steps,
                    batch_size=cur_batch_size,
                    num_batches=cur_batches_per_update,
                ),
                start=i,
            ):
                ####################
                start = time.time()

                ##################
                # DEBUG
                # if prev_params is not None:
                #     _named_params_dict = {k: v for k, v in self.encoder.named_parameters()}
                #     _diffs = {
                #         name: (_named_params_dict[name].detach() - prev_param)
                #         for name, prev_param in prev_params.items()
                #     }
                #     pprint(_diffs)
                ###################################
                prev_params = {
                    k: v.detach().clone() for k, v in self.named_parameters()
                }
                prev_grads = {
                    k: v.grad.detach().clone()
                    for k, v in self.named_parameters()
                    if v.grad is not None
                }
                ##########################

                if i % fourier_every == 0:
                    coef_dict = self.fourier()
                    ##############################
                    # DEBUG
                    # print("=====================")
                    # print(coef_dict)
                    # pprint(coef_dict)
                    #########################
                    if len(coef_dict) > 0:
                        fourier_coefs.append({**coef_dict, "step": i})

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
                    print("Saving turbo encoder")
                    torch.save(
                        self.trainable_module.state_dict(),
                        safe_create_file(self.output_path),
                    )
                    if len(fourier_coefs) > 0:
                        print("Saving fourier coefs")
                        ####################
                        # DEBUG
                        # print("=====================")
                        # pprint(fourier_coefs)
                        # print("=====================")
                        ################################
                        torch.save(
                            fourier_coefs,
                            safe_create_file(str(self.output_path)) + ".fourier",
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
                    back_tracking=back_tracking,
                    descent=descent,
                    shrinkage=shrinkage,
                    failure=failure,
                )

                param_metrics = self.param_metrics()

                yield {
                    **res,
                    **{k: v.item() for k, v in param_metrics.items()},
                    "step": i,
                    "total": cur_num_steps,
                    "time": time.time() - start,
                }
                ###################################

                if res["converged"]:
                    print("Training converged.")
                    break

            coef_dict = self.fourier()
            if len(coef_dict) > 0:
                fourier_coefs.append({**coef_dict, "step": i})

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
                print("Saving turbo encoder")
                torch.save(
                    self.encoder.state_dict(), safe_create_file(self.output_path)
                )
                if len(fourier_coefs) > 0:
                    print("Saving fourier coefs")
                    ####################
                    # DEBUG
                    # print("=====================")
                    # pprint(fourier_coefs)
                    # print("=====================")
                    ################################
                    torch.save(
                        fourier_coefs,
                        safe_create_file(str(self.output_path)) + ".fourier",
                    )
                if save_optimizer:
                    print("Saving optimizer")
                    torch.save(
                        optimizer.state_dict(),
                        safe_create_file(str(self.output_path) + ".opt"),
                    )

    def fourier(self):
        return {}

    def summary(self) -> None:
        print("Encoder:")
        print(self.encoder)
        print(self.encoder.settings())
        if all(p.numel() <= 256 for p in self.encoder.parameters()):
            print(list(self.encoder.parameters()))
        print("Channel:")
        print(self.channel.settings())
        print("Modulator:")
        print(self.modulator.settings())
        if self.validation_channel is not None:
            print("Validation Channel:")
            print(self.validation_channel.settings())


class BCJRTurboTrainer(DecoderBasedEncoderTrainer):
    def __init__(
        self,
        input_size: int,
        encoder: StreamedTurboEncoder,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        num_iter: int = 6,
        use_max: bool = False,
        output_path=None,
        batch_normalization=False,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ) -> None:
        super().__init__(
            input_size=input_size,
            encoder=encoder,
            channel=channel,
            validation_channel=validation_channel,
            output_path=output_path,
            device_manager=device_manager,
            batch_normalization=batch_normalization,
        )

        self.num_iter = num_iter
        self.use_max = use_max

    def forward(self, input_batch: torch.Tensor, validate=False):
        channel = self.validation_channel if validate else self.channel
        use_max = False if validate else self.use_max
        num_iter = 6 if validate else self.num_iter
        decoder = IteratedBCJRTurboDecoder(
            encoder=self.encoder,
            modulator=self.modulator,
            channel=channel,
            use_max=use_max,
            num_iter=num_iter,
            device_manager=self.device_manager,
        )
        modulated = self.modulator(self.encoder(input_batch))
        meandim = list(range(1, modulated.ndim)) if validate else None
        metrics = {
            "modulated_power": torch.mean(modulated**2, dim=meandim),
            "modulated_center": torch.mean(modulated, dim=meandim),
        }
        return decoder(channel(modulated)), metrics

    def reevaluate(self, input_batch: torch.Tensor, validate: bool = False):
        channel = self.validation_channel if validate else self.channel
        use_max = False if validate else self.use_max
        num_iter = 6 if validate else self.num_iter
        decoder = IteratedBCJRTurboDecoder(
            encoder=self.encoder,
            modulator=self.modulator,
            channel=channel,
            use_max=use_max,
            num_iter=num_iter,
            device_manager=self.device_manager,
        )
        modulated = self.modulator(self.encoder(input_batch))
        meandim = list(range(1, modulated.ndim)) if validate else None
        metrics = {
            "modulated_power": torch.mean(modulated**2, dim=meandim),
            "modulated_center": torch.mean(modulated, dim=meandim),
        }
        return decoder(channel.corrupt(modulated)), metrics


class CodebookEncoderTrainer(DecoderBasedEncoderTrainer):
    def __init__(
        self,
        input_size: int,
        encoder: Encoder,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        batch_normalization=False,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Settings to try out
        use_inputs_for_loss: bool = False,  # This will use XE instead of CE
    ) -> None:
        super().__init__(
            input_size=input_size,
            encoder=encoder,
            channel=channel,
            validation_channel=validation_channel,
            output_path=output_path,
            batch_normalization=batch_normalization,
            device_manager=device_manager,
        )

        self.use_inputs_for_loss = use_inputs_for_loss
        self._codebook = None

    def forward(self, input_batch: torch.Tensor, validate=False):
        modulated = self.modulator(self.encoder(input_batch))
        meandim = list(range(1, modulated.ndim)) if validate else None
        metrics = {
            "modulated_power": torch.mean(modulated**2, dim=meandim),
            "modulated_center": torch.mean(modulated, dim=meandim),
        }

        if (not validate) or (self.encoder.batch_dependent) or (self._codebook is None):
            # DEBUG
            if validate:
                print("Creating codebook during validation")
            self._codebook = self.encoder.to_codebook(dtype=torch.float)

        channel = self.validation_channel if validate else self.channel
        decoder = CodebookDecoder(
            encoder=self._codebook,
            modulator=self.modulator,
            channel=channel,
            device_manager=self.device_manager,
        )

        return decoder(channel(modulated)), metrics

    def reevaluate(self, input_batch: torch.Tensor, validate: bool = False):
        modulated = self.modulator(self.encoder(input_batch))
        meandim = list(range(1, modulated.ndim)) if validate else None
        metrics = {
            "modulated_power": torch.mean(modulated**2, dim=meandim),
            "modulated_center": torch.mean(modulated, dim=meandim),
        }

        if (not validate) or (self.encoder.batch_dependent) or (self._codebook is None):
            # DEBUG
            if validate:
                print("Creating codebook during validation")
            self._codebook = self.encoder.to_codebook(dtype=torch.float)

        channel = self.validation_channel if validate else self.channel
        decoder = CodebookDecoder(
            encoder=self._codebook,
            modulator=self.modulator,
            channel=channel,
            device_manager=self.device_manager,
        )

        return decoder(channel.corrupt(modulated)), metrics

    def metrics(self, inputs: torch.Tensor, logits: torch.FloatTensor, no_mean=False):
        mean_dim = -1 if no_mean else None
        ce = torch.mean(binary_entropy_with_logits(logits), dim=mean_dim)
        probs = torch.sigmoid(logits)
        true_ber = torch.mean(torch.minimum(probs, 1 - probs), dim=mean_dim)
        metrics = {
            **(super().metrics(inputs, logits, no_mean)[1]),
            "ce": ce,
            "true_ber": true_ber,
        }
        loss = metrics["xe"] if self.use_inputs_for_loss else metrics["ce"]
        return loss, metrics


def random_parity_table(window, streams, device_manager):
    generator, bias = gen_affine_convcode_generator(
        window, streams, device_manager=device_manager
    )
    return boolean_to_fourier(
        AffineConvolutionalEncoder._affine_encode(
            enumerate_binary_inputs(
                generator.shape[1], dtype=torch.float32, device=device_manager.device
            ),
            generator.float(),
            bias.float(),
        )[:, 0, :].float()
    )  # Inputs x Channels


class TurboFourierTrainerBase(CodingTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        init_method="fourier_normal",
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Other settings to try out
        constraint=None,
    ):

        CodingTrainer.__init__(self, device_manager=device_manager)

        init_callable = self.get_init_method(init_method)

        fourier_noninterleaved = nn.Parameter(
            init_callable(
                window=window,
                streams=num_noninterleaved_streams,
                device_manager=device_manager,
            ),
            requires_grad=True,
        )
        fourier_interleaved = nn.Parameter(
            init_callable(
                window=window,
                streams=num_interleaved_streams,
                device_manager=device_manager,
            ),
            requires_grad=True,
        )
        noninterleaved_encoder = FourierConvolutionalEncoder(
            num_steps=input_size,
            fourier_coefficients=fourier_noninterleaved,
            device_manager=device_manager,
            constraint=constraint,
        )
        interleaved_encoder = FourierConvolutionalEncoder(
            num_steps=input_size,
            fourier_coefficients=fourier_interleaved,
            device_manager=device_manager,
            constraint=constraint,
        )
        self.encoder = StreamedTurboEncoder[FourierConvolutionalEncoder](
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=device_manager,
        )

        self.init_method = init_method
        self.window = window

    def fourier(self):
        return {
            "noninterleaved": self.encoder.noninterleaved_encoder.fourier_coefficients.detach().clone(),
            "noninterleaved_window": self.encoder.noninterleaved_encoder.window,
            "interleaved": self.encoder.interleaved_encoder.fourier_coefficients.detach().clone(),
            "interleaved_window": self.encoder.interleaved_encoder.window,
        }

    @staticmethod
    def get_init_method(init_method):
        if init_method == "fourier_normal":
            return lambda window, streams, device_manager: torch.randn(
                size=(2**window, streams),
                device=device_manager.device,
                generator=device_manager.generator,
            )
        if init_method == "normal":
            return lambda window, streams, device_manager: table_to_fourier(
                torch.randn(
                    size=(2**window, streams),
                    device=device_manager.device,
                    generator=device_manager.generator,
                ),
                device_manager=device_manager,
            )
        elif init_method == "binary":
            return lambda window, streams, device_manager: table_to_fourier(
                boolean_to_fourier(
                    torch.randint(
                        low=0,
                        high=2,
                        size=(2**window, streams),
                        device=device_manager.device,
                        generator=device_manager.generator,
                        dtype=torch.float,
                    )
                ),
                device_manager=device_manager,
            )
        elif init_method == "parity":
            return lambda window, streams, device_manager: table_to_fourier(
                random_parity_table(
                    window=window, streams=streams, device_manager=device_manager
                ),
                device_manager=device_manager,
            )

        elif init_method == "bent":
            fc1 = [
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                -0.2500,
                0.0000,
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                -0.2500,
                0.0000,
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                0.2500,
                0.0000,
                -0.2500,
                0.0000,
                -0.2500,
                0.0000,
                -0.2500,
                0.0000,
                -0.2500,
                0.0000,
                0.2500,
                0.0000,
            ]
            fc2 = [
                0.2500,
                0.2500,
                0.2500,
                -0.2500,
                0.2500,
                0.2500,
                0.2500,
                -0.2500,
                0.2500,
                0.2500,
                0.2500,
                -0.2500,
                -0.2500,
                -0.2500,
                -0.2500,
                0.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
            fc3 = [
                0.2500,
                0.2500,
                0.2500,
                0.2500,
                0.2500,
                0.2500,
                -0.2500,
                -0.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.2500,
                -0.2500,
                0.2500,
                -0.2500,
                0.2500,
                -0.2500,
                -0.2500,
                0.2500,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]

            def make_bend(window, streams, device_manager):
                print("Using bent function as init")
                assert window == 5
                if streams == 2:
                    print("init noninterleaved")
                    coefs = [fc1, fc2]
                elif streams == 1:
                    print("init interleaved")
                    coefs = [fc3]
                else:
                    raise ValueError
                return torch.tensor(coefs, device=device_manager.device).transpose(1, 0)

            return make_bend

    @torch.no_grad()
    def apply_constraint(self):
        self.encoder.noninterleaved_encoder.apply_constraint_()
        self.encoder.interleaved_encoder.apply_constraint_()
        self.encoder.update()

    @staticmethod
    def fourier_std_mean(fourier_coefficients, reduce_dims=False):
        all_l2_norms_sq = (
            torch.norm(
                fourier_coefficients,
                p=2,
                dim=0,
                keepdim=False,
            )
            ** 2
        )
        all_means = fourier_coefficients[0]

        if not reduce_dims:
            all_vars = all_l2_norms_sq - (all_means**2)
            return torch.sqrt(all_vars), all_means
        else:
            l2_norm_sq = torch.mean(all_l2_norms_sq)
            mean = torch.mean(all_means)
            return torch.sqrt(l2_norm_sq - (mean**2)), mean

    def param_metrics(self):
        table_noni_std, table_noni_mean = torch.std_mean(
            self.encoder.noninterleaved_encoder.table.detach().clone()
        )
        table_i_std, table_i_mean = torch.std_mean(
            self.encoder.interleaved_encoder.table.detach().clone()
        )
        fourier_noni_std, fourier_noni_mean = self.fourier_std_mean(
            self.encoder.noninterleaved_encoder.fourier_coefficients.detach().clone(),
            reduce_dims=True,
        )
        fourier_i_std, fourier_i_mean = self.fourier_std_mean(
            self.encoder.interleaved_encoder.fourier_coefficients.detach().clone(),
            reduce_dims=True,
        )

        return {
            "table_noninterleaved.mean": table_noni_mean,
            "table_noninterleaved.std": table_noni_std,
            "table_interleaved.mean": table_i_mean,
            "table_interleaved.std": table_i_std,
            "fourier_noninterleaved.mean": fourier_noni_mean,
            "fourier_noninterleaved.std": fourier_noni_std,
            "fourier_interleaved.mean": fourier_i_mean,
            "fourier_interleaved.std": fourier_i_std,
        }


class TurboTableTrainerBase(CodingTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        init_method="normal",
        delay=0,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Other settings to try out
        constraint=None,
    ):

        CodingTrainer.__init__(self, device_manager=device_manager)

        self.encoder = self.make_turbo_encoder(
            input_size=input_size,
            window=window,
            interleaver=interleaver,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            init_method=init_method,
            delay=delay,
            device_manager=device_manager,
            constraint=constraint,
        )

        self.init_method = init_method
        self.window = window

    @staticmethod
    def make_turbo_encoder(
        input_size: int,
        window: int,
        interleaver: Interleaver,
        num_noninterleaved_streams: int,
        num_interleaved_streams: int,
        init_method: str,
        delay: int,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Other settings to try out
        constraint=None,
    ):
        init_callable = TurboTableTrainerBase.get_init_method(init_method)

        table_noninterleaved = nn.Parameter(
            init_callable(
                window=window,
                streams=num_noninterleaved_streams,
                device_manager=device_manager,
            ),
            requires_grad=True,
        )
        table_interleaved = nn.Parameter(
            init_callable(
                window=window,
                streams=num_interleaved_streams,
                device_manager=device_manager,
            ),
            requires_grad=True,
        )
        noninterleaved_encoder = GeneralizedConvolutionalEncoder(
            num_steps=input_size,
            table=table_noninterleaved,
            device_manager=device_manager,
            constraint=constraint,
            delay=delay,
        )
        interleaved_encoder = GeneralizedConvolutionalEncoder(
            num_steps=input_size,
            table=table_interleaved,
            device_manager=device_manager,
            constraint=constraint,
            delay=delay,
        )
        return StreamedTurboEncoder[GeneralizedConvolutionalEncoder](
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=device_manager,
        )

    def fourier(self):
        return {
            "noninterleaved": table_to_fourier(
                self.encoder.noninterleaved_encoder.table.detach().clone()
            ),
            "noninterleaved_window": self.encoder.noninterleaved_encoder.window,
            "interleaved": table_to_fourier(
                self.encoder.interleaved_encoder.table.detach().clone()
            ),
            "interleaved_window": self.encoder.interleaved_encoder.window,
        }

    @staticmethod
    def get_init_method(init_method):
        if init_method == "normal":
            return lambda window, streams, device_manager: torch.randn(
                size=(2**window, streams),
                device=device_manager.device,
                generator=device_manager.generator,
            )
        elif init_method == "binary":
            return lambda window, streams, device_manager: boolean_to_fourier(
                torch.randint(
                    low=0,
                    high=2,
                    size=(2**window, streams),
                    device=device_manager.device,
                    generator=device_manager.generator,
                    dtype=torch.float,
                )
            )
        elif init_method == "parity":
            return random_parity_table

    @torch.no_grad()
    def apply_constraint(self):
        self.encoder.noninterleaved_encoder.apply_constraint_()
        self.encoder.interleaved_encoder.apply_constraint_()
        self.encoder.update()

    def param_metrics(self):
        noni_std, noni_mean = torch.std_mean(
            self.encoder.noninterleaved_encoder.table.detach().clone()
        )
        i_std, i_mean = torch.std_mean(
            self.encoder.interleaved_encoder.table.detach().clone()
        )
        return {
            "noninterleaved.mean": noni_mean,
            "noninterleaved.std": noni_std,
            "interleaved.mean": i_mean,
            "interleaved.std": i_std,
        }


class TurboTableTrainer(TurboTableTrainerBase, CodebookEncoderTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Settings to try out
        constraint=None,
        use_inputs_for_loss: bool = False,  # This will use XE instead of CE
        init_method="normal",
    ) -> None:

        super(TurboTableTrainer, self).__init__(
            input_size=input_size,
            window=window,
            interleaver=interleaver,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            init_method=init_method,
            constraint=constraint,
            device_manager=device_manager,
        )
        super(TurboTableTrainerBase, self).__init__(
            input_size=input_size,
            encoder=self.encoder,
            channel=channel,
            validation_channel=validation_channel,
            output_path=output_path,
            batch_normalization=(constraint is None),
            device_manager=device_manager,
            use_inputs_for_loss=use_inputs_for_loss,
        )


class TurboTableTrainerBCJR(TurboTableTrainerBase, BCJRTurboTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        num_iter: int = 6,
        use_max: bool = False,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Settings to try out
        constraint=None,
        init_method="normal",
    ) -> None:

        super(TurboTableTrainerBCJR, self).__init__(
            input_size=input_size,
            window=window,
            interleaver=interleaver,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            init_method=init_method,
            constraint=constraint,
            device_manager=device_manager,
        )

        super(TurboTableTrainerBase, self).__init__(
            input_size=input_size,
            encoder=self.encoder,
            channel=channel,
            validation_channel=validation_channel,
            num_iter=num_iter,
            use_max=use_max,
            output_path=output_path,
            batch_normalization=(constraint is None),
            device_manager=device_manager,
        )


class TurboTableSwarmTrainerBCJR(DecoderBasedSwarmEncoderTrainer):
    def __init__(
        self,
        # Required Swarm settings
        swarm_size: int,
        # Required agent settings
        input_size: int,
        window: int,
        interleaver: Interleaver,
        channel: NoisyChannel,
        # Optional Swarm Settings
        kill_agents=False,
        tolm: float = 1e-4,
        merge_agents=True,
        tolmerge: float = 1e-3,
        communication_adj=2,
        step_adj=2,
        descent=0.2,
        shrinkage=0.9,
        failure=10,
        # Optional agent settings
        output_path=None,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        num_iter: int = 6,
        use_max: bool = False,
        constraint=None,
        init_method="normal",
        # Other
        device_manager=DEFAULT_DEVICE_MANAGER,
    ) -> None:
        self._input_size = input_size
        self.window = window
        self.interleaver = interleaver
        self.channel = channel
        self.num_noninterleaved_streams = num_noninterleaved_streams
        self.num_interleaved_streams = num_interleaved_streams
        self.num_iter = num_iter
        self.use_max = use_max
        self.device_manager = device_manager
        self.constraint = constraint
        self.init_method = init_method

        super().__init__(
            swarm_size=swarm_size,
            kill_agents=kill_agents,
            tolm=tolm,
            merge_agents=merge_agents,
            tolmerge=tolmerge,
            communication_adj=communication_adj,
            step_adj=step_adj,
            descent=descent,
            shrinkage=shrinkage,
            failure=failure,
            output_path=output_path,
        )

    def initialize_agent(self) -> TurboTableTrainerBCJR:
        return TurboTableTrainerBCJR(
            input_size=self.input_size,
            window=self.window,
            interleaver=self.interleaver,
            channel=self.channel,
            validation_channel=self.channel,
            output_path=None,
            num_noninterleaved_streams=self.num_noninterleaved_streams,
            num_interleaved_streams=self.num_interleaved_streams,
            num_iter=self.num_iter,
            use_max=self.use_max,
            device_manager=self.device_manager,
            constraint=self.constraint,
            init_method=self.init_method,
        )

    @property
    def input_size(self) -> int:
        return self._input_size


class TurboFourierTrainer(TurboFourierTrainerBase, CodebookEncoderTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Settings to try out
        constraint=None,
        use_inputs_for_loss: bool = False,  # This will use XE instead of CE
        init_method="normal",
    ) -> None:

        super(TurboFourierTrainer, self).__init__(
            input_size=input_size,
            window=window,
            interleaver=interleaver,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            init_method=init_method,
            constraint=constraint,
            device_manager=device_manager,
        )
        super(TurboFourierTrainerBase, self).__init__(
            input_size=input_size,
            encoder=self.encoder,
            channel=channel,
            validation_channel=validation_channel,
            output_path=output_path,
            batch_normalization=(constraint is None),
            device_manager=device_manager,
            use_inputs_for_loss=use_inputs_for_loss,
        )


class TurboFourierTrainerBCJR(TurboFourierTrainerBase, BCJRTurboTrainer):
    def __init__(
        self,
        input_size: int,
        window: int,
        interleaver: Interleaver,
        channel: NoisyChannel,
        validation_channel: NoisyChannel = None,
        output_path=None,
        num_noninterleaved_streams: int = 2,
        num_interleaved_streams: int = 1,
        num_iter: int = 6,
        use_max: bool = False,
        device_manager=DEFAULT_DEVICE_MANAGER,
        # Settings to try out
        constraint=None,
        init_method="normal",
    ) -> None:

        super(TurboFourierTrainerBCJR, self).__init__(
            input_size=input_size,
            window=window,
            interleaver=interleaver,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            init_method=init_method,
            constraint=constraint,
            device_manager=device_manager,
        )

        super(TurboFourierTrainerBase, self).__init__(
            input_size=input_size,
            encoder=self.encoder,
            channel=channel,
            validation_channel=validation_channel,
            num_iter=num_iter,
            use_max=use_max,
            output_path=output_path,
            batch_normalization=(constraint is None),
            device_manager=device_manager,
        )
