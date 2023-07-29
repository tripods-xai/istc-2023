from typing import Union
from pprint import pprint

from pathlib import Path
import torch
from torch.optim import Adam, SGD
import hashlib

from ..interleavers import FixedPermuteInterleaver, BatchRandomPermuteInterleaver
from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER
from ..engine import FileLogger
from ..training import (
    CodebookEncoderTrainer,
    TurboTableTrainer,
    TurboTableTrainerBCJR,
    TurboFourierTrainer,
    TurboFourierTrainerBCJR,
    TurboTableSwarmTrainerBCJR,
)
from ..channels import AWGN
from ..encoders import ENC_interCNN

from .experiment_utils import load_interleaver, load_optimizer_factory


def train_turbo_table(
    experiment_id: str,
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    use_inputs_for_loss: bool = False,  # This will use XE instead of CE
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    init_method="normal",
    constraint=None,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "use_inputs_for_loss": use_inputs_for_loss,
        "constraint": constraint,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device=manager.device).manual_seed(
                    interleaver_base_seed
                ),
                device=manager.device,
            )
            interleaver = FixedPermuteInterleaver(
                input_size=block_len, device_manager=manager, permutation=permutation
            )
        else:
            raise NotImplementedError(f"Interleaver base seed {interleaver_base_seed}")
    elif interleaver_type == "batch_random":
        interleaver = BatchRandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )
    else:
        raise NotImplementedError(f"Interleaver type {interleaver_type}")

    channel = AWGN(snr=snr, device_manager=manager)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    optimizer_factory = lambda p: Adam(p, lr=adam_lr)

    output_path = (
        Path(output_dir) / f"train_turbo_table_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = TurboTableTrainer(
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        num_noninterleaved_streams=num_noninterleaved_streams,
        num_interleaved_streams=num_interleaved_streams,
        device_manager=manager,
        use_inputs_for_loss=use_inputs_for_loss,
        init_method=init_method,
        constraint=constraint,
    )

    preamble = {
        "args": arguments,
        "encoder": trainer.encoder.long_settings(),
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def train_turboae_neural_encoder(
    experiment_id: str,
    block_len: int,
    enc_num_layer: int,
    enc_num_unit: int,
    enc_kernel_size: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    use_inputs_for_loss: bool = False,  # This will use XE instead of CE
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "enc_num_layer": enc_num_layer,
        "enc_num_unit": enc_num_unit,
        "enc_kernel_size": enc_kernel_size,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "use_inputs_for_loss": use_inputs_for_loss,
        "save_every": save_every,
        "output_dir": str(output_dir),
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device=manager.device).manual_seed(
                    interleaver_base_seed
                ),
                device=manager.device,
            )
            interleaver = FixedPermuteInterleaver(
                input_size=block_len, device_manager=manager, permutation=permutation
            )
        else:
            raise NotImplementedError(f"Interleaver base seed {interleaver_base_seed}")
    elif interleaver_type == "batch_random":
        interleaver = BatchRandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )
    else:
        raise NotImplementedError(f"Interleaver type {interleaver_type}")
    encoder = ENC_interCNN(
        enc_num_layer=enc_num_layer,
        enc_num_unit=enc_num_unit,
        enc_kernel_size=enc_kernel_size,
        interleaver=interleaver,
        device_manager=manager,
    )

    channel = AWGN(snr=snr, device_manager=manager)

    optimizer_factory = lambda p: Adam(p, lr=adam_lr)

    output_path = (
        Path(output_dir)
        / f"train_turboae_encoder_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = CodebookEncoderTrainer(
        input_size=block_len,
        encoder=encoder,
        channel=channel,
        validation_channel=channel,
        output_path=output_path,
        device_manager=manager,
        use_inputs_for_loss=use_inputs_for_loss,
    )

    preamble = {
        "args": arguments,
        "encoder": trainer.encoder.long_settings(),
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def train_turbo_fourier_bcjr(
    experiment_id: str,
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    num_iter: int = 6,
    use_max: bool = True,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    init_method="normal",
    constraint="unit_power",
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
        "num_iter": num_iter,
        "use_max": use_max,
        "constraint": constraint,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device=manager.device).manual_seed(
                    interleaver_base_seed
                ),
                device=manager.device,
            )
            interleaver = FixedPermuteInterleaver(
                input_size=block_len, device_manager=manager, permutation=permutation
            )
        else:
            raise NotImplementedError(f"Interleaver base seed {interleaver_base_seed}")
    elif interleaver_type == "batch_random":
        interleaver = BatchRandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )

    channel = AWGN(snr=snr, device_manager=manager)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    optimizer_factory = lambda p: Adam(p, lr=adam_lr)

    output_path = (
        Path(output_dir)
        / f"train_turbo_fourier_bcjr_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = TurboFourierTrainerBCJR(
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        num_noninterleaved_streams=num_noninterleaved_streams,
        num_interleaved_streams=num_interleaved_streams,
        device_manager=manager,
        num_iter=num_iter,
        use_max=use_max,
        init_method=init_method,
        constraint=constraint,
    )

    preamble = {
        "args": arguments,
        "encoder": trainer.encoder.long_settings(),
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def train_turbo_table_bcjr(
    experiment_id: str,
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    num_iter: int = 6,
    use_max: bool = False,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    init_method="normal",
    constraint=None,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
        "num_iter": num_iter,
        "use_max": use_max,
        "constraint": constraint,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device=manager.device).manual_seed(
                    interleaver_base_seed
                ),
                device=manager.device,
            )
            interleaver = FixedPermuteInterleaver(
                input_size=block_len, device_manager=manager, permutation=permutation
            )
        else:
            raise NotImplementedError(f"Interleaver base seed {interleaver_base_seed}")
    elif interleaver_type == "batch_random":
        interleaver = BatchRandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )

    channel = AWGN(snr=snr, device_manager=manager)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    optimizer_factory = lambda p: Adam(p, lr=adam_lr)

    output_path = (
        Path(output_dir)
        / f"train_turbo_table_bcjr_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = TurboTableTrainerBCJR(
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        num_noninterleaved_streams=num_noninterleaved_streams,
        num_interleaved_streams=num_interleaved_streams,
        device_manager=manager,
        num_iter=num_iter,
        use_max=use_max,
        init_method=init_method,
        constraint=constraint,
    )

    preamble = {
        "args": arguments,
        "encoder": trainer.encoder.long_settings(),
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def train_turbo_table_swarm_bcjr(
    experiment_id: str,
    swarm_size: int,
    # Agent specific
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    lr: float = 1e-1,
    delay=0,
    num_iter: int = 6,
    use_max: bool = False,
    save_every=10,
    init_method="normal",
    constraint=None,
    output_dir: Path = None,
    # Swarm Optional
    kill_agents=False,
    tolm: float = 1e-4,
    merge_agents=True,
    tolmerge: float = 1e-3,
    communication_adj=2,
    step_adj=1,
    descent=0.2,
    shrinkage=0.9,
    failure=10,
    # Other
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "swarm_size": swarm_size,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "lr": lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
        "num_iter": num_iter,
        "use_max": use_max,
        "constraint": constraint,
        "kill_agents": kill_agents,
        "tolm": tolm,
        "merge_agents": merge_agents,
        "tolmerge": tolmerge,
        "communication_adj": communication_adj,
        "step_adj": step_adj,
        "descent": descent,
        "shrinkage": shrinkage,
        "failure": failure,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )
    channel = AWGN(snr=snr, device_manager=manager)

    optimizer_factory = lambda p: SGD(p, lr=lr)

    output_path = (
        Path(output_dir)
        / f"train_turbo_table_swarm_bcjr_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = TurboTableSwarmTrainerBCJR(
        swarm_size=swarm_size,
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        channel=channel,
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
        num_noninterleaved_streams=num_noninterleaved_streams,
        num_interleaved_streams=num_interleaved_streams,
        device_manager=manager,
        num_iter=num_iter,
        use_max=use_max,
        init_method=init_method,
        constraint=constraint,
    )

    preamble = {
        "args": arguments,
        "swarm": trainer.swarm.long_settings(),
        "encoders_first_3": [
            a.encoder.long_settings() for a in trainer.swarm.agents[:3]
        ],
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            pprint(result)
            # print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            pprint(result)
            # print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def train_turbo_table_rerun_bcjr(
    experiment_id: str,
    num_rerun: int,
    # Agent specific
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type="fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    lr: float = 1e-1,
    optimizer: str = "sgd",
    delay=0,
    num_iter: int = 6,
    use_max: bool = False,
    save_every=10,
    init_method="normal",
    constraint=None,
    output_dir: Path = None,
    # backtracking Optional
    back_tracking=False,
    descent=0.2,
    shrinkage=0.9,
    failure=10,
    # Other
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "num_rerun": num_rerun,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "lr": lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
        "num_iter": num_iter,
        "use_max": use_max,
        "constraint": constraint,
        "back_tracking": back_tracking,
        "descent": descent,
        "shrinkage": shrinkage,
        "failure": failure,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )
    channel = AWGN(snr=snr, device_manager=manager)

    # optimizer_factory = lambda p: SGD(p, lr=lr)
    optimizer_factory = load_optimizer_factory(optimizer=optimizer, lr=lr)

    output_path = (
        Path(output_dir)
        / f"train_turbo_table_rerun_bcjr_block_len_{block_len}_{argument_hash}.pt"
    )
    for r in range(num_rerun):
        print(f"Run {r}")
        trainer = TurboTableTrainerBCJR(
            input_size=block_len,
            window=window,
            interleaver=interleaver,
            channel=channel,
            validation_channel=channel,
            output_path=output_path,
            num_noninterleaved_streams=num_noninterleaved_streams,
            num_interleaved_streams=num_interleaved_streams,
            device_manager=manager,
            num_iter=num_iter,
            use_max=use_max,
            init_method=init_method,
            constraint=constraint,
        )

        preamble = {
            "args": arguments,
            "run": r,
            "encoder": trainer.encoder.long_settings(),
            "channel": channel.long_settings(),
            "output_path": str(output_path),
            "argument_hash": argument_hash,
        }

        if file_logger is not None:
            file_logger.new_experiment(preamble=preamble)

        result_list = []
        for result in trainer.train(
            optimizer_factory=optimizer_factory,
            num_steps=num_steps,
            batch_size=batch_size,
            batches_per_update=batches_per_update,
            save_every=save_every,
            validate_every=save_every,
            num_validation_steps=num_validation_steps,
            back_tracking=back_tracking,
            descent=descent,
            shrinkage=shrinkage,
            failure=failure,
        ):
            if file_logger is not None:
                result_list.append(dict(result))
                file_logger.update(result_list)
            result_type = result.pop("type")
            if result_type == "training":
                step = result.pop("step")
                total = result.pop("total")
                print(f"Training step {step + 1}/{total}:")
                pprint(result)
                # print(get_result_str(result))
            if result_type == "validation":
                print("Validation:")
                pprint(result)
                # print(get_result_str(result))
        if file_logger is not None:
            file_logger.end_experiment()

    return file_logger


def train_turbo_table_fourier(
    experiment_id: str,
    block_len: int,
    window: int,
    interleaver_base_seed: int = None,
    interleaver_type: str = "fixed",
    num_noninterleaved_streams: int = 2,
    num_interleaved_streams: int = 1,
    use_inputs_for_loss: bool = False,  # This will use XE instead of CE
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    init_method="fourier_normal",
    constraint=None,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "batch_size": batch_size,
        "batches_per_update": batches_per_update,
        "snr": snr,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "window": window,
        "num_noninterleaved_streams": num_noninterleaved_streams,
        "num_interleaved_streams": num_interleaved_streams,
        "use_inputs_for_loss": use_inputs_for_loss,
        "constraint": constraint,
        "save_every": save_every,
        "output_dir": str(output_dir),
        "init_method": init_method,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device=manager.device).manual_seed(
                    interleaver_base_seed
                ),
                device=manager.device,
            )
            interleaver = FixedPermuteInterleaver(
                input_size=block_len, device_manager=manager, permutation=permutation
            )
        else:
            raise NotImplementedError(f"Interleaver base seed {interleaver_base_seed}")
    elif interleaver_type == "batch_random":
        interleaver = BatchRandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )

    channel = AWGN(snr=snr, device_manager=manager)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    optimizer_factory = lambda p: Adam(p, lr=adam_lr)

    output_path = (
        Path(output_dir)
        / f"train_turbo_fourier_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = TurboFourierTrainer(
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        num_noninterleaved_streams=num_noninterleaved_streams,
        num_interleaved_streams=num_interleaved_streams,
        device_manager=manager,
        use_inputs_for_loss=use_inputs_for_loss,
        init_method=init_method,
        constraint=constraint,
    )

    preamble = {
        "args": arguments,
        "encoder": trainer.encoder.long_settings(),
        "channel": channel.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in trainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
        fourier_every=1,
    ):
        if file_logger is not None:
            result_list.append(dict(result))
            file_logger.update(result_list)
        result_type = result.pop("type")
        if result_type == "training":
            step = result.pop("step")
            total = result.pop("total")
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result_type == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger
