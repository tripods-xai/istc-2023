from typing import Union
from pprint import pprint

from pathlib import Path
import hashlib
import torch
from torch.optim import Adam, SGD

from ..decoders import TurboAEDecoder
from ..interleavers import (
    FixedPermuteInterleaver,
    Interleaver,
    BatchRandomPermuteInterleaver,
    RandomPermuteInterleaver,
)
from ..encoders import (
    turboae_binary_exact_nobd,
    GeneralizedConvolutionalEncoder,
    StreamedTurboEncoder,
    FourierConvolutionalEncoder,
)
from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER
from ..engine import FileLogger
from ..constants import TURBOAE_DECODER_BINARY_PATH, MODELS_DIR
from ..training import DecoderTrainer
from ..modulation import BPSK, Normalization, IdentityModulation
from ..channels import VariableAWGN, AWGN
from .experiment_utils import (
    load_interleaver,
    load_original_turboae_encoder_decoder,
    TurboAEType,
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


def retrain_turboae_for_new_block_len(
    experiment_id: str,
    block_len: int,
    interleaver_type: str = "fixed",
    interleaver_base_seed: int = None,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    batch_size: int = 2000,
    snr_low: float = -1.5,
    snr_high: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    delay=0,
    pre_init: bool = False,
    reload_optimizer: Union[str, Path] = None,
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
        "interleaver_base_seed": interleaver_base_seed,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "batch_size": batch_size,
        "snr_low": snr_low,
        "snr_high": snr_high,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "delay": delay,
        "pre_init": pre_init,
        "reload_optimizer": str(reload_optimizer),
        "save_every": save_every,
        "output_dir": str(output_dir),
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        interleaver_base_seed=interleaver_base_seed,
        manager=manager,
    )

    # Does this work if I use delay=0
    encoder = turboae_binary_exact_nobd(
        num_steps=block_len,
        device_manager=manager,
        delay=delay,
        interleaver=interleaver,
    )
    interleaver = encoder.interleaver

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        interleaver=interleaver,
        device_manager=manager,
    )
    if pre_init:
        state_dict = torch.load(
            TURBOAE_DECODER_BINARY_PATH, map_location=manager.device
        )
        decoder.pre_initialize(state_dict)

    modulator = BPSK(device_manager=manager)
    channel = VariableAWGN(snr_low=snr_low, snr_high=snr_high)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    def optimizer_factory(p):
        optimizer = Adam(p, lr=adam_lr)
        if reload_optimizer is not None:
            print(f"Initializing optimizer from path {reload_optimizer}")
            s_dict = torch.load(reload_optimizer, map_location=manager.device)
            optimizer.load_state_dict(s_dict)
        return optimizer

    output_path = Path(output_dir) / f"retrain_turboae_binary_block_len_{block_len}.pt"
    retrainer = DecoderTrainer(
        decoder=decoder,
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        device_manager=manager,
    )

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "validation_channel": validation_channel.long_settings(),
        "decoder": decoder.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    def get_result_str(results):
        return "\n".join([f"\t{k}: {v}" for k, v in results.items()])

    result_list = []
    for result in retrainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
    ):
        if file_logger is not None:
            result_list.append(result)
            file_logger.update(result_list)
        if result["type"] == "training":
            step = result["step"]
            total = result["total"]
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result["type"] == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger


def retrain_original_turboae_for_new_block_len(
    experiment_id: str,
    block_len: int,
    turboae_type: TurboAEType,
    interleaver_type: str = "fixed",
    interleaver_base_seed: int = None,
    batch_size: int = 2000,
    snr_low: float = -1.5,
    snr_high: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: int = 1000,
    num_validation_steps: int = 2,
    adam_lr: float = 1e-5,
    reload_optimizer: Union[str, Path] = None,
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
        "turboae_type": turboae_type,
        "interleaver_base_seed": interleaver_base_seed,
        "batch_size": batch_size,
        "snr_low": snr_low,
        "snr_high": snr_high,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "adam_lr": adam_lr,
        "reload_optimizer": str(reload_optimizer),
        "save_every": save_every,
        "output_dir": str(output_dir),
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        interleaver_base_seed=interleaver_base_seed,
        manager=manager,
    )

    encoder_decoder = load_original_turboae_encoder_decoder(
        interleaver, turboae_type=turboae_type, device_manager=manager
    )

    if turboae_type == "binary":
        modulator = IdentityModulation(device_manager=manager)
    elif turboae_type == "continuous":
        modulator = Normalization(device_manager=manager)
    else:
        raise ValueError

    channel = VariableAWGN(snr_low=snr_low, snr_high=snr_high)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    def optimizer_factory(p):
        optimizer = Adam(p, lr=adam_lr)
        if reload_optimizer is not None:
            print(f"Initializing optimizer from path {reload_optimizer}")
            s_dict = torch.load(reload_optimizer, map_location=manager.device)
            optimizer.load_state_dict(s_dict)
        return optimizer

    output_path = MODELS_DIR / f"{experiment_id}_{argument_hash[:5]}.pt"
    retrainer = DecoderTrainer(
        decoder=encoder_decoder["decoder"],
        encoder=encoder_decoder["encoder"],
        modulator=modulator,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        device_manager=manager,
    )

    preamble = {
        "args": arguments,
        "encoder": retrainer.encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "validation_channel": validation_channel.long_settings(),
        "decoder": retrainer.decoder.long_settings(),
        "output_path": str(output_path),
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    result_list = []
    for result in retrainer.train(
        optimizer_factory=optimizer_factory,
        num_steps=num_steps,
        batch_size=batch_size,
        save_every=save_every,
        validate_every=save_every,
        num_validation_steps=num_validation_steps,
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


def load_encoder_by_type(
    state_dict,
    type: str,
    encoder_constraint: str,
    input_size: int,
    interleaver: Interleaver,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    if type == "table":
        table_noninterleaved = state_dict["noninterleaved_encoder.table"].to(
            device_manager.device
        )
        table_interleaved = state_dict["interleaved_encoder.table"].to(
            device_manager.device
        )
        noninterleaved_encoder = GeneralizedConvolutionalEncoder(
            num_steps=input_size,
            table=table_noninterleaved,
            constraint=encoder_constraint,
            device_manager=device_manager,
        )
        interleaved_encoder = GeneralizedConvolutionalEncoder(
            num_steps=input_size,
            table=table_interleaved,
            constraint=encoder_constraint,
            device_manager=device_manager,
        )
        return StreamedTurboEncoder[GeneralizedConvolutionalEncoder](
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=device_manager,
        )
    if type == "fourier":
        fourier_noninterleaved = state_dict[
            "noninterleaved_encoder.fourier_coefficients"
        ].to(device_manager.device)
        fourier_interleaved = state_dict["interleaved_encoder.fourier_coefficients"].to(
            device_manager.device
        )
        noninterleaved_encoder = FourierConvolutionalEncoder(
            num_steps=input_size,
            fourier_coefficients=fourier_noninterleaved,
            constraint=encoder_constraint,
            device_manager=device_manager,
        )
        interleaved_encoder = FourierConvolutionalEncoder(
            num_steps=input_size,
            fourier_coefficients=fourier_interleaved,
            constraint=encoder_constraint,
            device_manager=device_manager,
        )
        return StreamedTurboEncoder[FourierConvolutionalEncoder](
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=device_manager,
        )
    else:
        raise NotImplementedError(f"type={type}")


def train_neural_decoder(
    experiment_id: str,
    block_len: int,
    encoder_path: Path,
    encoder_type: str,
    encoder_constraint: str,
    interleaver_base_seed: int = None,
    interleaver_type: str = "fixed",
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    batch_size: Union[int, list] = 256,
    batches_per_update: Union[int, list] = 1,
    snr_low: float = -1.5,
    snr_high: float = 2.0,
    validation_snr: float = 2.0,
    num_steps: Union[int, list] = 1000,
    num_validation_steps: int = 2,
    lr: float = 1e-5,
    optimizer_type: str = "adam",
    delay=0,
    pre_init=None,
    reload_optimizer: Union[str, Path] = None,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    save_every=10,
    constrain_batches=False,
    output_dir: Path = None,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "encoder_path": encoder_path,
        "encoder_type": encoder_type,
        "encoder_constraint": encoder_constraint,
        "interleaver_base_seed": interleaver_base_seed,
        "interleaver_type": interleaver_type,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "batch_size": batch_size,
        "snr_low": snr_low,
        "snr_high": snr_high,
        "validation_snr": validation_snr,
        "num_steps": num_steps,
        "num_validation_steps": num_validation_steps,
        "lr": lr,
        "optimizer_type": optimizer_type,
        "delay": delay,
        "pre_init": pre_init,
        "reload_optimizer": str(reload_optimizer),
        "save_every": save_every,
        "constrain_batches": constrain_batches,
        "batches_per_update": batches_per_update,
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
    elif interleaver_type == "sample_random":
        interleaver = RandomPermuteInterleaver(
            input_size=block_len, device_manager=manager
        )

    # Does this work if I use delay=0
    encoder_state_dict = torch.load(encoder_path, map_location=manager.device)
    encoder = load_encoder_by_type(
        state_dict=encoder_state_dict,
        type=encoder_type,
        encoder_constraint=encoder_constraint,
        input_size=block_len,
        interleaver=interleaver,
        device_manager=manager,
    )

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        interleaver=interleaver,
        device_manager=manager,
    )

    if pre_init is not None:
        print(f"Initializing decoder from path {pre_init}")
        s_dict = torch.load(pre_init, map_location=manager.device)
        decoder.pre_initialize(s_dict)

    modulator = (
        Normalization(device_manager=manager)
        if constrain_batches
        else IdentityModulation(device_manager=manager)
    )
    channel = VariableAWGN(snr_low=snr_low, snr_high=snr_high, device_manager=manager)
    validation_channel = AWGN(snr=validation_snr, device_manager=manager)

    def optimizer_factory(p):
        optimizer = get_optimizer_type(optimizer_type=optimizer_type)(p, lr=lr)
        if reload_optimizer is not None:
            print(f"Initializing optimizer from path {reload_optimizer}")
            s_dict = torch.load(reload_optimizer, map_location=manager.device)
            optimizer.load_state_dict(s_dict)
        print(f"Loaded optimizer {optimizer}.")
        return optimizer

    output_path = (
        Path(output_dir)
        / f"train_neural_decoder_block_len_{block_len}_{argument_hash}.pt"
    )
    trainer = DecoderTrainer(
        decoder=decoder,
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        validation_channel=validation_channel,
        output_path=output_path,
        device_manager=manager,
    )

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "validation_channel": validation_channel.long_settings(),
        "decoder": decoder.long_settings(),
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
            result_list.append(result)
            file_logger.update(result_list)
        if result["type"] == "training":
            step = result["step"]
            total = result["total"]
            print(f"Training step {step + 1}/{total}:")
            print(get_result_str(result))
        if result["type"] == "validation":
            print("Validation:")
            print(get_result_str(result))

    file_logger.end_experiment()

    return file_logger
