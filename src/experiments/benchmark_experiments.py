# TODO: Move stuff from conditional_entropy_experiments.py

import hashlib
import torch
from pprint import pprint


from ..utils import DeviceManager, DEFAULT_DEVICE_MANAGER, PRECISION_DTYPE_DICT
from ..encoders import (
    get_encoder,
    TrellisEncoder,
    TurboEncoder,
    StreamedTurboEncoder,
    ENC_interCNN,
)
from ..modulation import Modulator, get_modulator, IdentityModulation
from ..channels import AWGN, NoisyChannel
from ..measurements import (
    NeuralDecoderCrossEntropySampler,
    TurboAESampler,
    Sampler,
    JunctionTreeConditionalEntropySampler,
)
from ..engine import FileLogger, TqdmProgressBar, ResultsProcessor

from .experiment_utils import (
    load_interleaver,
    check_stop_criterion,
    load_turboae_checkpoint,
    load_turboae_encoder_checkpoint,
    TurboAEType,
    load_original_turboe_encoder,
    load_turboae_decoder,
    run_measurement,
)


@torch.no_grad()
def run_single_turbo_neural_benchmark(
    sampler: Sampler,
    file_logger: FileLogger = None,
    num_samples: int = 1000,
    batch_size: int = 1,
    stop_key: str = None,
    stop_tol: float = 1e-1,
    patience: int = 1,
    preamble: dict = None,
) -> FileLogger:

    if file_logger is not None:
        if preamble is None:
            preamble = {}
        file_logger.new_experiment(preamble=preamble)

    num_batches = num_samples // batch_size
    progress_bar = TqdmProgressBar()
    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    for i in range(0, num_samples, batch_size):
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cross_entropies = sampler.sample(cur_batch_size)
        results_processor.update(sampled_cross_entropies, num_samples=cur_batch_size)
        if stop_key is not None:
            if check_stop_criterion(
                i=i,
                results_processor=results_processor,
                key=stop_key,
                tol=stop_tol,
                patience=patience,
            ):
                print("Stop criterion met, stopping.")
                break

    results_processor.close()

    return file_logger


def benchmark_neural_turbo_codes(
    experiment_id: str,
    encoder_name: str,
    decoder_path: str,
    interleaver_type: str = "fixed",
    interleaver_base_seed: int = None,
    modulator_type: str = "identity",
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "decoder_path": decoder_path,
        "interleaver_type": interleaver_type,
        "interleaver_base_seed": interleaver_base_seed,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder = get_encoder(encoder_name)(num_steps=block_len, device_manager=manager)

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )

    if isinstance(encoder, (TurboEncoder, ENC_interCNN)):
        encoder.interleaver = interleaver
    elif isinstance(encoder, TrellisEncoder):
        # Assumes the encoder has 2 channels, will use the first 2
        # for noninterleaved and the last for the interleaved channel
        noninterleaved_encoder = encoder
        interleaved_encoder = encoder.get_encoder_channels([-1])
        encoder = StreamedTurboEncoder(
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=manager,
        )
    else:
        raise NotImplementedError(f"encoder type: {(encoder).__class__.__name__}")

    modulator = get_modulator(modulator_type=modulator_type)
    channel = AWGN(snr=snr, device_manager=manager)

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "neural",
    }

    cross_entropy_estimator = NeuralDecoderCrossEntropySampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder_path=decoder_path,
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        device_manager=manager,
    )

    file_logger = run_single_turbo_neural_benchmark(
        sampler=cross_entropy_estimator,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=0.1,
        patience=100000,
        preamble=preamble,
    )

    return file_logger


@torch.no_grad()
def benchmark_turboae_codes(
    experiment_id: str,
    encoder_decoder_path: str,
    modulator_type: str = "identity",
    batch_size: int = 1,
    num_samples: int = 1000,
    snr: float = 0.0,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_path": encoder_decoder_path,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "snr": snr,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "front_pad": front_pad,
        "first_pad": first_pad,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder_decoder = load_turboae_checkpoint(
        encoder_decoder_path=encoder_decoder_path,
        front_pad=front_pad,
        first_pad=first_pad,
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        device_manager=manager,
    )
    encoder = encoder_decoder["encoder"]
    decoder = encoder_decoder["decoder"]

    modulator = get_modulator(modulator_type=modulator_type)
    channel = AWGN(snr=snr, device_manager=manager)

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "decoder": decoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "neural",
        "encoder_type": "neural",
    }

    sampler = TurboAESampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder=decoder,
        device_manager=manager,
        encoder_as_conv=first_pad,
    )

    file_logger = run_single_turbo_neural_benchmark(
        sampler=sampler,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=0.1,
        patience=100000,
        preamble=preamble,
    )

    return file_logger


@torch.no_grad()
def benchmark_turboae_codes_jt(
    experiment_id: str,
    encoder_decoder_path: str,
    modulator_type: str = "identity",
    batch_size: int = 1,
    num_samples: int = 1000,
    snr: float = 0.0,
    front_pad: bool = False,
    first_pad: bool = False,
    elimination_seed: int = None,
    precision: str = "half",
    stop_tol: float = 1e-1,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_path": encoder_decoder_path,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "snr": snr,
        "front_pad": front_pad,
        "first_pad": first_pad,
        "stop_tol": stop_tol,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder = load_turboae_encoder_checkpoint(
        encoder_decoder_path=encoder_decoder_path,
        front_pad=front_pad,
        first_pad=first_pad,
        binarize=False,
        device_manager=manager,
    )

    modulator = get_modulator(modulator_type=modulator_type)
    channel = AWGN(snr=snr, device_manager=manager)

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "junction_tree",
        "encoder_type": "neural",
    }

    if file_logger is not None:
        if preamble is None:
            preamble = {}
        file_logger.new_experiment(preamble=preamble)

    if not first_pad:
        encoder.compute_mean_std_(batch_size=batch_size)

    sampler = JunctionTreeConditionalEntropySampler(
        encoder=encoder.to_conv_code() if first_pad else encoder,
        modulator=modulator,
        channel=channel,
        elimination_seed=elimination_seed,
        dtype=PRECISION_DTYPE_DICT[precision],
        device_manager=manager,
    )

    result = run_measurement(
        sampler=sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=stop_tol,
        patience=5 * batch_size,
    )

    pprint(result)
    if file_logger is not None:
        file_logger.update(result)

    file_logger.end_experiment()

    return file_logger


@torch.no_grad()
def benchmark_finetuned_turboae_codes_jt(
    experiment_id: str,
    block_len: int,
    interleaver_base_seed: int,
    turboae_type: TurboAEType,
    batch_size: int = 1,
    num_samples: int = 1000,
    snr: float = 0.0,
    elimination_seed: int = None,
    precision: str = "half",
    stop_tol: float = 1e-1,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "turboae_type": turboae_type,
        "experiment_id": experiment_id,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "snr": snr,
        "stop_tol": stop_tol,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type="fixed",
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )
    encoder = load_original_turboe_encoder(
        interleaver=interleaver, turboae_type=turboae_type, device_manager=manager
    )
    if turboae_type == "continuous":
        encoder.compute_mean_std_(batch_size=batch_size)

    modulator = IdentityModulation(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "junction_tree",
        "encoder_type": "neural",
    }

    if file_logger is not None:
        if preamble is None:
            preamble = {}
        file_logger.new_experiment(preamble=preamble)

    sampler = JunctionTreeConditionalEntropySampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        elimination_seed=elimination_seed,
        dtype=PRECISION_DTYPE_DICT[precision],
        device_manager=manager,
    )

    result = run_measurement(
        sampler=sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=stop_tol,
        patience=5 * batch_size,
    )

    pprint(result)
    if file_logger is not None:
        file_logger.update(result)

    file_logger.end_experiment()

    return file_logger


@torch.no_grad()
def benchmark_finetuned_original_turboae_codes(
    experiment_id: str,
    block_len: int,
    interleaver_base_seed: int,
    turboae_type: TurboAEType,
    decoder_path: str,
    batch_size: int = 1,
    num_samples: int = 1000,
    snr: float = 0.0,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "turboae_type": turboae_type,
        "decoder_path": decoder_path,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "snr": snr,
        "num_iteration": num_iteration,
        "num_iter_ft": num_iter_ft,
        "dec_num_layer": dec_num_layer,
        "dec_num_unit": dec_num_unit,
        "dec_kernel_size": dec_kernel_size,
        "front_pad": front_pad,
        "first_pad": first_pad,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    interleaver = load_interleaver(
        interleaver_type="fixed",
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )
    decoder = load_turboae_decoder(
        interleaver=interleaver,
        front_pad=front_pad,
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        device_manager=manager,
        decoder_path=decoder_path,
    )
    encoder = load_original_turboe_encoder(
        interleaver=interleaver, turboae_type=turboae_type, device_manager=manager
    )
    if turboae_type == "continuous":
        encoder.compute_mean_std_(batch_size=batch_size)

    modulator = IdentityModulation(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "decoder": decoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "neural",
        "encoder_type": "neural",
    }

    sampler = TurboAESampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder=decoder,
        device_manager=manager,
        encoder_as_conv=first_pad,
    )

    file_logger = run_single_turbo_neural_benchmark(
        sampler=sampler,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=0.1,
        patience=100000,
        preamble=preamble,
    )

    return file_logger
