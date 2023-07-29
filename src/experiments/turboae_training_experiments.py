from typing import Dict, Any, Tuple

import time
from pathlib import Path
from pprint import pprint
import hashlib
import torch
from torch.optim import Adam, SGD
from ..constants import MODELS_DIR

from ..decoders import TurboAEDecoder
from ..encoders import ENC_interCNN, get_info_from_checkpoint
from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    filter_state_dict,
    parse_timestamp,
)
from ..engine import FileLogger
from ..training import TurboAETrainer, TurboTableTrainerBase
from ..modulation import Normalization, IdentityModulation
from ..channels import VariableAWGN, AWGN
from ..graphs import nonrecursive_turbo_graph
from ..measurements import TurboAESampler

from .experiment_utils import (
    load_interleaver,
    InterleaverType,
    get_matching_checkpoints,
    run_measurement,
    serialize,
)


def get_optimizer_type(optimizer_type: str):
    if optimizer_type == "adam":
        return Adam
    elif optimizer_type == "sgd":

        def load_sgd(p, lr):
            return SGD(p, lr=lr, momentum=0.9, nesterov=True)

        return load_sgd
    else:
        raise NotImplementedError(f"Optimizer: {optimizer_type}")


def train_turboae(
    experiment_id: str,
    block_len: int,
    interleaver_type: str = InterleaverType,
    interleaver_base_seed: int = None,
    batch_size: int = 2000,
    batches_per_update: int = 1,
    enc_num_unit=100,
    enc_num_layer=2,
    enc_kernel_size=5,
    enc_snr: float = 2.0,
    enc_lr: float = 0.0001,
    enc_num_steps: int = 25,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    dec_snr_low: float = -1.5,
    dec_snr_high: float = 2.0,
    dec_lr: float = 0.0001,
    dec_num_steps: int = 125,
    num_epochs: int = 1000,
    validation_snr: float = 2.0,
    num_validation_steps: int = 2,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    write_checkpoints=False,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    reload_filename: Path = None,
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
        "enc_num_unit": enc_num_unit,
        "enc_num_layer": enc_num_layer,
        "enc_kernel_size": enc_kernel_size,
        "enc_snr": enc_snr,
        "enc_lr": enc_lr,
        "enc_num_steps": enc_num_steps,
        "front_pad": front_pad,
        "first_pad": first_pad,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "dec_snr_low": dec_snr_low,
        "dec_snr_high": dec_snr_high,
        "dec_lr": dec_lr,
        "dec_num_steps": dec_num_steps,
        "num_epochs": num_epochs,
        "validation_snr": validation_snr,
        "num_validation_steps": num_validation_steps,
        "save_every": save_every,
        "output_dir": str(output_dir) if output_dir is not None else output_dir,
        "model_dir": str(MODELS_DIR),
        "write_checkpoints": write_checkpoints,
        "reload_filename": reload_filename,
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
    # Does this work if I use delay=0
    encoder = ENC_interCNN(
        enc_num_layer=enc_num_layer,
        enc_num_unit=enc_num_unit,
        enc_kernel_size=enc_kernel_size,
        interleaver=interleaver,
        first_pad=first_pad,
        front_pad=front_pad,
        device_manager=manager,
    )
    interleaver = encoder.interleaver

    if write_checkpoints:
        print("Doing a check that the interleaver has the desired factor width.")
        # Just hardcoded seed because there is only one permutation I want to check.
        nt_graph = nonrecursive_turbo_graph(
            interleaver.permutation, window=encoder.window, delay=encoder.delay
        ).with_elimination_ordering(sample_thresh=3, tries=10, seed=67841)
        assert nt_graph.factor_width == 23

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        front_pad=front_pad,
        interleaver=interleaver,
        device_manager=manager,
    )

    if reload_filename is not None:
        print(f"Loading weight initialization from {reload_filename}")
        state_dict = torch.load(reload_filename, map_location=manager.device)
        encoder.load_state_dict(state_dict=filter_state_dict(state_dict, key="encoder"))
        decoder.load_state_dict(state_dict=filter_state_dict(state_dict, key="decoder"))

    encoder_channel = AWGN(snr=enc_snr, device_manager=manager)
    decoder_channel = VariableAWGN(snr_low=dec_snr_low, snr_high=dec_snr_high)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    def make_optimizer_factory(lr, reload_model_path=None, reload_key=None):
        def optimizer_factory(p):
            optimizer = Adam(p, lr=lr)
            if reload_model_path is not None:
                reload_optimizer_path = Path(str(reload_model_path) + ".opt")
                print(f"Loading key {reload_key} from {reload_optimizer_path}.")
                state_dict = torch.load(
                    reload_optimizer_path, map_location=manager.device
                )
                if reload_key is not None:
                    # I had saved these directly as dictionaries, not module_dicts
                    state_dict = state_dict[reload_key]
                optimizer.load_state_dict(state_dict=state_dict)
            return optimizer

        return optimizer_factory

    output_path = MODELS_DIR / f"{experiment_id}_{argument_hash[:5]}.pt"
    retrainer = TurboAETrainer(
        input_size=block_len,
        encoder=encoder,
        decoder=decoder,
        encoder_channel=encoder_channel,
        decoder_channel=decoder_channel,
        validation_channel=validation_channel,
        batch_normalization=(not first_pad),
        output_path=output_path,
        device_manager=manager,
    )

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "encoder_channel": encoder_channel.long_settings(),
        "decoder_channel": decoder_channel.long_settings(),
        "validation_channel": validation_channel.long_settings(),
        "decoder": decoder.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    result_list = []
    encoder_optimizer_factory = make_optimizer_factory(
        enc_lr, reload_model_path=reload_filename, reload_key="encoder_optimizer"
    )
    decoder_optimizer_factory = make_optimizer_factory(
        dec_lr, reload_model_path=reload_filename, reload_key="decoder_optimizer"
    )
    for result in retrainer.train(
        encoder_optimizer_factory=encoder_optimizer_factory,
        decoder_optimizer_factory=decoder_optimizer_factory,
        num_epochs=num_epochs,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        encoder_steps_per_epoch=enc_num_steps,
        decoder_steps_per_epoch=dec_num_steps,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
        save_optimizer=True,
        mode="alternating",
        write_checkpoints=write_checkpoints,
    ):
        if file_logger is not None:
            result_list.append(result)
            file_logger.update(result_list)
        if result["type"] == "training":
            step = result["step"]
            total = result["total"]
            print(f"Training step {step + 1}/{total}:")
            pprint(result)
        if result["type"] == "validation":
            print("Validation:")
            pprint(result)

    file_logger.end_experiment()

    return file_logger


def train_turboae_table(
    experiment_id: str,
    block_len: int,
    interleaver_type: str = InterleaverType,
    interleaver_base_seed: int = None,
    batch_size: int = 2000,
    batches_per_update: int = 1,
    window=5,
    enc_snr: float = 2.0,
    enc_lr: float = 1e-2,
    enc_num_steps: int = 25,
    front_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    dec_snr_low: float = -1.5,
    dec_snr_high: float = 2.0,
    dec_lr: float = 0.0001,
    dec_num_steps: int = 125,
    num_epochs: int = 1000,
    validation_snr: float = 2.0,
    num_validation_steps: int = 2,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    output_dir: Path = None,
    model_dir: Path = MODELS_DIR,
    write_checkpoints=False,
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
        "window": window,
        "enc_snr": enc_snr,
        "enc_lr": enc_lr,
        "enc_num_steps": enc_num_steps,
        "front_pad": front_pad,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "dec_snr_low": dec_snr_low,
        "dec_snr_high": dec_snr_high,
        "dec_lr": dec_lr,
        "dec_num_steps": dec_num_steps,
        "num_epochs": num_epochs,
        "validation_snr": validation_snr,
        "num_validation_steps": num_validation_steps,
        "save_every": save_every,
        "output_dir": str(output_dir) if output_dir is not None else output_dir,
        "model_dir": str(MODELS_DIR),
        "write_checkpoints": write_checkpoints,
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
    delay = 0 if front_pad else (window - 1) - ((window - 1) // 2)
    encoder = TurboTableTrainerBase.make_turbo_encoder(
        input_size=block_len,
        window=window,
        interleaver=interleaver,
        num_noninterleaved_streams=2,
        num_interleaved_streams=1,
        init_method="normal",
        delay=delay,
        device_manager=manager,
        constraint="opt_unit_power",
    )
    # Does this work if I use delay=0
    interleaver = encoder.interleaver

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        front_pad=False,
        interleaver=interleaver,
        device_manager=manager,
    )

    encoder_channel = AWGN(snr=enc_snr, device_manager=manager)
    decoder_channel = VariableAWGN(snr_low=dec_snr_low, snr_high=dec_snr_high)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    def make_optimizer_factory(lr):
        def optimizer_factory(p):
            optimizer = Adam(p, lr=lr)
            return optimizer

        return optimizer_factory

    output_path = model_dir / f"{experiment_id}_{argument_hash[:5]}.pt"
    retrainer = TurboAETrainer(
        input_size=block_len,
        encoder=encoder,
        decoder=decoder,
        encoder_channel=encoder_channel,
        decoder_channel=decoder_channel,
        validation_channel=validation_channel,
        batch_normalization=False,
        output_path=output_path,
        device_manager=manager,
    )

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "encoder_channel": encoder_channel.long_settings(),
        "decoder_channel": decoder_channel.long_settings(),
        "validation_channel": validation_channel.long_settings(),
        "decoder": decoder.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    result_list = []
    for result in retrainer.train(
        encoder_optimizer_factory=make_optimizer_factory(enc_lr),
        decoder_optimizer_factory=make_optimizer_factory(dec_lr),
        num_epochs=num_epochs,
        batch_size=batch_size,
        batches_per_update=batches_per_update,
        encoder_steps_per_epoch=enc_num_steps,
        decoder_steps_per_epoch=dec_num_steps,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
        save_optimizer=True,
        mode="alternating",
        write_checkpoints=write_checkpoints,
    ):
        if file_logger is not None:
            result_list.append(result)
            file_logger.update(result_list)
        if result["type"] == "training":
            step = result["step"]
            total = result["total"]
            print(f"Training step {step + 1}/{total}:")
            pprint(result)
        if result["type"] == "validation":
            print("Validation:")
            pprint(result)

    file_logger.end_experiment()

    return file_logger


def run_checkpoint_measurement(
    checkpoint_info: Dict[str, Any],
    sampler: TurboAESampler,
    num_samples: int,
    batch_size: int,
    stop_key: str = "xe",
    stop_tol=1e-1,
    patience=0,
):
    print(f"Measuring decomposition on checkpoint {checkpoint_info['filepath']}")
    state_dict = torch.load(
        checkpoint_info["filepath"], map_location=sampler.device_manager.device
    )
    sampler.load_state_dict(state_dict=state_dict)

    return run_measurement(
        sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key=stop_key,
        stop_tol=stop_tol,
        patience=patience,
    )


@torch.no_grad()
def xe_trajectory(
    experiment_id: str,
    checkpoint_basename: str,
    checkpoint_daterange: Tuple[str, str],
    num_samples: int,
    batch_size: int,
    snr: float,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    stop_tol=1e-1,
    resolution: int = 8,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
):

    orig_checkpoint_daterange = checkpoint_daterange
    checkpoint_daterange = tuple(parse_timestamp(dt) for dt in checkpoint_daterange)
    assert checkpoint_daterange[0] <= checkpoint_daterange[1]

    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "checkpoint_basename": checkpoint_basename,
        "checkpoint_daterange": orig_checkpoint_daterange,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "snr": snr,
        "front_pad": front_pad,
        "first_pad": first_pad,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "stop_tol": stop_tol,
        "resolution": resolution,
    }
    pprint(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    matching_checkpoint_info = get_matching_checkpoints(
        checkpoint_basename=checkpoint_basename,
        checkpoint_daterange=checkpoint_daterange,
    )
    model_info = get_info_from_checkpoint(
        matching_checkpoint_info[0]["filepath"], device_manager=manager
    )

    print(model_info)
    # Does this work if I use delay=0
    encoder = ENC_interCNN(
        enc_num_layer=model_info["enc_num_layer"],
        enc_num_unit=model_info["enc_num_unit"],
        enc_kernel_size=model_info["enc_kernel_size"],
        interleaver=model_info["interleaver"],
        first_pad=first_pad,
        front_pad=front_pad,
        device_manager=manager,
    )
    interleaver = encoder.interleaver

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        front_pad=front_pad,
        interleaver=interleaver,
        device_manager=manager,
    )
    modulator = IdentityModulation(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    sampler = TurboAESampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder=decoder,
        encoder_as_conv=first_pad,
        device_manager=manager,
    )

    avg_time = 0
    result_list = []
    to_measure_infos = sorted(matching_checkpoint_info, key=lambda d: d["datetime"])[
        ::resolution
    ]
    for i, checkpoint_info in enumerate(to_measure_infos):
        print(f"Measurement {i+1}/{len(to_measure_infos)}")
        start = time.time()
        result = run_checkpoint_measurement(
            checkpoint_info=checkpoint_info,
            sampler=sampler,
            num_samples=num_samples,
            batch_size=batch_size,
            stop_key="xe",
            stop_tol=stop_tol,
            patience=0,
        )
        result = {**checkpoint_info, "avg_time": avg_time, **result}

        if file_logger is not None:
            result_list.append(serialize(result))
            file_logger.update(result_list)

        pprint(result)

        step_time = time.time() - start
        avg_time = (i * avg_time + step_time) / (i + 1)
        est_time = (len(matching_checkpoint_info) - (i + 1)) * avg_time
        print(f"Estimated Remaining Time: {est_time}s")

    file_logger.end_experiment()

    return file_logger
