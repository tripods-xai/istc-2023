from typing import Sequence
import hashlib

import torch

from ..utils import (
    DEFAULT_SEED,
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    gen_affine_convcode_generator,
)
from ..encoders import (
    get_encoder,
    UnrolledTrellis,
    TrellisEncoder,
    AffineConvolutionalEncoder,
    TurboEncoder,
    CodebookEncoder,
    StreamedTurboEncoder,
    is_systematic,
    Encoder,
)
from ..modulation import BPSK, Modulator, get_modulator
from ..interleavers import FixedPermuteInterleaver, RandomPermuteInterleaver
from ..channels import AWGN, NoisyChannel
from ..measurements import (
    TrellisConditionalEntropySampler,
    CodebookConditionalEntropySampler,
    IteratedTurboCrossEntropySampler,
    HazzysTurboCrossEntropySampler,
    JunctionTreeConditionalEntropySampler,
)
from ..engine import FileLogger, TqdmProgressBar, ResultsProcessor

from .experiment_utils import load_interleaver, check_stop_criterion


@torch.no_grad()
def run_single_trellis_encoder_conditional_entropy_experiment(
    encoder: TrellisEncoder,
    modulator: Modulator,
    channel: NoisyChannel,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    num_samples: int = 1000,
    batch_size: int = 1,
    stop_key: str = None,
    stop_tol: float = 1e-1,
    patience: int = 1,
    preamble: dict = None,
) -> FileLogger:
    cond_entropy_estimator = TrellisConditionalEntropySampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

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
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        results_processor.update(sampled_cond_entropies)
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


def benchmark_trellis_codes(
    experiment_id: str,
    encoder_name: str,
    modulator_type: str = "identity",
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    patience: int = None,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")
    if patience is None:
        patience = 5 * batch_size

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
        "patience": patience,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder = get_encoder(encoder_name)(num_steps=block_len, device_manager=manager)

    if not isinstance(encoder, TrellisEncoder):
        raise NotImplementedError(f"encoder type: {(encoder).__class__.__name__}")

    modulator = get_modulator(modulator_type=modulator_type)
    channel = AWGN(snr=snr, device_manager=manager)

    encoder_is_systematic = is_systematic(encoder_name)
    print(
        f"Encoder {encoder_name} is {'' if encoder_is_systematic else 'not '}systematic."
    )
    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "bcjr",
    }

    file_logger = run_single_trellis_encoder_conditional_entropy_experiment(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=0.1,
        patience=patience,
        preamble=preamble,
    )

    return file_logger


def trellis_encoder_conditional_entropy_experiment(
    experiment_id: str,
    encoder_name: str,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:

    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)

    encoder = get_encoder(encoder_name)(num_steps=block_len, device_manager=manager)
    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)
    cond_entropy_estimator = TrellisConditionalEntropySampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    if file_logger is not None:
        file_logger.new_experiment(
            preamble={
                "args": arguments,
                "encoder_name": encoder_name,
                "encoder": encoder.long_settings(),
                "modulator": modulator.long_settings(),
                "channel": channel.long_settings(),
            }
        )

    num_batches = num_samples // batch_size
    progress_bar = TqdmProgressBar()
    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    for i in range(0, num_samples, batch_size):
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        records = {
            "ce": sampled_cond_entropies,
        }
        results_processor.update(records)

    results_processor.close()

    return file_logger


def trellis_encoder_conditional_entropy_experiment_concat_encoders(
    experiment_id: str,
    encoder_name: str,
    additional_encoders: int = 3,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:

    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "additional_encoders": additional_encoders,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)

    encoder = get_encoder(encoder_name)(num_steps=block_len, device_manager=manager)
    encoder_list = [encoder]
    for i in range(1, additional_encoders):
        old_encoder = encoder_list[-1]
        new_output_tables = torch.randint(
            0,
            2,
            size=old_encoder.output_tables[..., 0:1].shape,
            device=manager.device,
            generator=manager.generator,
        )
        new_trellises = UnrolledTrellis(
            old_encoder.trellises.state_transitions,
            torch.concat(
                [old_encoder.trellises.output_tables, new_output_tables], dim=-1
            ),
        )
        new_encoder = TrellisEncoder(
            new_trellises,
            normalize_output_table=old_encoder.normalize_output_table,
            delay_state_transitions=old_encoder.delay_state_transitions,
            device_manager=manager,
        )
        encoder_list.append(new_encoder)

    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    for i, e in enumerate(encoder_list):
        cond_entropy_estimator = TrellisConditionalEntropySampler(
            encoder=e,
            modulator=modulator,
            channel=channel,
            device_manager=manager,
        )

        if file_logger is not None:
            file_logger.new_experiment(
                preamble={
                    "args": arguments,
                    "encoder_name": encoder_name,
                    "additional_concatenated_encoders": i,
                    "encoder": e.long_settings(),
                    "modulator": modulator.long_settings(),
                    "channel": channel.long_settings(),
                }
            )

        num_batches = num_samples // batch_size
        progress_bar = TqdmProgressBar()
        progress_bar.new_experiment(total=num_batches)

        listeners = [l for l in [progress_bar, file_logger] if l is not None]
        results_processor = ResultsProcessor(listeners=listeners)

        for i in range(0, num_samples, batch_size):
            cur_batch_size = min(num_samples - i, batch_size)
            sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
            records = {
                "ce": sampled_cond_entropies,
            }
            results_processor.update(records)

        results_processor.close()

    return file_logger


def trellis_encoder_conditional_entropy_experiment_random_encoder_variable_window_block_len(
    experiment_id: str,
    num_channels: int = 2,
    window: int = 3,
    trials: int = 5,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:

    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "num_channels": num_channels,
        "window": window,
        "trials": trials,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)

    nonsys_encoder_list = []
    rsc_encoder_list = []
    for i in range(trials):
        gen, bias = gen_affine_convcode_generator(
            window, num_channels, device_manager=manager
        )
        new_encoder = AffineConvolutionalEncoder(
            generator=gen, bias=bias, num_steps=block_len, device_manager=manager
        )
        nonsys_encoder_list.append(new_encoder)
        rsc_encoder_list.append(new_encoder.to_rsc())

    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    for i, e in enumerate(nonsys_encoder_list):
        print(f"Nonsys trial {i+1}/{trials}")
        preamble = {
            "args": arguments,
            "type": "nonsystematic",
            "trial": i,
            "encoder": e.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
        }
        file_logger = run_single_trellis_encoder_conditional_entropy_experiment(
            encoder=e,
            modulator=modulator,
            channel=channel,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
        )

    for i, e in enumerate(rsc_encoder_list):
        print(f"RSC trial {i+1}/{trials}")
        preamble = {
            "args": arguments,
            "type": "rsc",
            "encoder": e.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
        }
        file_logger = run_single_trellis_encoder_conditional_entropy_experiment(
            encoder=e,
            modulator=modulator,
            channel=channel,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
        )

    return file_logger


@torch.no_grad()
def run_single_codebook_encoder_conditional_entropy_experiment(
    encoder: Encoder,
    modulator: Modulator,
    channel: NoisyChannel,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
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

    num_batches = (num_samples + batch_size - 1) // batch_size
    progress_bar = TqdmProgressBar()

    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    for i in range(0, num_samples, batch_size):
        encoder(
            encoder.dummy_input(batch_size=1)
        )  # will setup any batch-specific stuff
        cond_entropy_estimator = CodebookConditionalEntropySampler(
            encoder=encoder.to_codebook(dtype=torch.half),
            modulator=modulator,
            channel=channel,
            device_manager=manager,
        )
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        # TODO write some logic so this is not hardcoded
        sampled_cond_entropies = {
            k: torch.mean(v) for k, v in sampled_cond_entropies.items()
        }
        results_processor.update(sampled_cond_entropies, num_samples=cur_batch_size)
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


@torch.no_grad()
def run_single_codebook_encoder_conditional_entropy_experiment_repro(
    encoder: Encoder,
    modulator: Modulator,
    channel: NoisyChannel,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
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

    num_batches = (num_samples + batch_size - 1) // batch_size
    progress_bar = TqdmProgressBar()

    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    encoder(encoder.dummy_input(batch_size=1))  # will setup any batch-specific stuff
    cond_entropy_estimator = CodebookConditionalEntropySampler(
        encoder=encoder.to_codebook(),
        modulator=modulator,
        channel=channel,
        device_manager=manager,
    )

    for i in range(0, num_samples, batch_size):
        encoder(
            encoder.dummy_input(batch_size=1)
        )  # will setup any batch-specific stuff

        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)

        results_processor.update(sampled_cond_entropies, num_samples=cur_batch_size)
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


@torch.no_grad()
def run_single_codebook_encoder_conditional_entropy_experiment_repro2(
    encoder: Encoder,
    modulator: Modulator,
    channel: NoisyChannel,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
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

    num_batches = (num_samples + batch_size - 1) // batch_size
    progress_bar = TqdmProgressBar()

    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    for i in range(0, num_samples, batch_size):
        encoder(
            encoder.dummy_input(batch_size=1)
        )  # will setup any batch-specific stuff
        cond_entropy_estimator = CodebookConditionalEntropySampler(
            encoder=encoder.to_codebook(dtype=torch.half),
            modulator=modulator,
            channel=channel,
            device_manager=manager,
        )
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        results_processor.update(sampled_cond_entropies, num_samples=cur_batch_size)
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


@torch.no_grad()
def run_single_junction_tree_conditional_entropy_experiment(
    encoder: Encoder,
    modulator: Modulator,
    channel: NoisyChannel,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    num_samples: int = 1000,
    batch_size: int = 1,
    elimination_tries: int = 50,
    elimination_seed: int = None,
    stop_key: str = None,
    stop_tol: float = 1e-1,
    patience: int = 1,
    preamble: dict = None,
) -> FileLogger:

    if file_logger is not None:
        if preamble is None:
            preamble = {}
        file_logger.new_experiment(preamble=preamble)

    num_batches = (num_samples + batch_size - 1) // batch_size
    progress_bar = TqdmProgressBar()

    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar, file_logger] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    # Make the cluster tree first, so we don't redo it every batch
    if elimination_seed is None:
        elimination_seed = manager.generate_seed()
    cluster_tree = (
        encoder.dependency_graph()
        .with_elimination_ordering(seed=elimination_seed, tries=elimination_tries)
        .as_cluster_tree()
    )

    for i in range(0, num_samples, batch_size):
        encoder(
            encoder.dummy_input(batch_size=1)
        )  # will setup any batch-specific stuff
        cond_entropy_estimator = JunctionTreeConditionalEntropySampler(
            encoder=encoder,
            modulator=modulator,
            channel=channel,
            device_manager=manager,
            cluster_tree=cluster_tree,
        )
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = cond_entropy_estimator.sample(cur_batch_size)
        # Since we are doing a fixed interleaver, all our samples are independent
        # sampled_cond_entropies = {
        #     k: torch.mean(v) for k, v in sampled_cond_entropies.items()
        # }
        results_processor.update(sampled_cond_entropies, num_samples=cur_batch_size)
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


@torch.no_grad()
def run_single_turbo_iterate_cross_entropy_experiment(
    encoder: StreamedTurboEncoder,
    modulator: Modulator,
    channel: NoisyChannel,
    num_iter: int = 6,
    systematic=False,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    num_samples: int = 1000,
    batch_size: int = 1,
    stop_key: str = None,
    stop_tol: float = 1e-1,
    patience: int = 1,
    preamble: dict = None,
) -> FileLogger:

    sampler_type = (
        HazzysTurboCrossEntropySampler
        # IteratedTurboCrossEntropySampler
        if systematic
        else IteratedTurboCrossEntropySampler
    )

    cross_entropy_estimator = sampler_type(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        device_manager=manager,
        num_iter=num_iter,
    )

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
        sampled_cross_entropies = cross_entropy_estimator.sample(cur_batch_size)
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


@torch.no_grad()
def benchmark_trellis_turbo_codes(
    experiment_id: str,
    encoder_name: str,
    interleaver_type: str = "fixed",
    interleaver_base_seed: int = None,
    modulator_type: str = "identity",
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    patience: int = None,
    num_iter: int = 6,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")
    if patience is None:
        patience = 5 * batch_size

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "interleaver_type": interleaver_type,
        "interleaver_base_seed": interleaver_base_seed,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
        "num_iter": num_iter,
        "patience": patience,
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
    encoder = get_encoder(encoder_name)(
        num_steps=block_len, interleaver=interleaver, device_manager=manager
    )

    if isinstance(encoder, TurboEncoder):
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

    encoder_is_systematic = is_systematic(encoder_name)
    print(
        f"Encoder {encoder_name} is {'' if encoder_is_systematic else 'not '}systematic."
    )
    preamble = {
        "args": arguments,
        "encoder": encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "hazzys" if encoder_is_systematic else "iterated_regular",
    }

    file_logger = run_single_turbo_iterate_cross_entropy_experiment(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        num_iter=num_iter,
        systematic=encoder_is_systematic,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ber",
        stop_tol=0.1,
        patience=patience,
        preamble=preamble,
    )

    return file_logger


def turbo_conditional_entropy_experiment(
    experiment_id: str,
    encoder_name: str,
    interleaver_type: str,
    interleaver_base_seed: int = None,
    delay: int = 0,
    modulator_type: str = "identity",
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "interleaver_type": interleaver_type,
        "interleaver_base_seed": interleaver_base_seed,
        "delay": delay,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder = get_encoder(encoder_name)(
        num_steps=block_len, device_manager=manager, delay=delay
    )

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )

    if isinstance(encoder, TurboEncoder):
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

    print(
        f"Testing Conditional Entropy for Turbo encoder naively at block len {block_len}."
    )
    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
        "encoder": encoder.long_settings(),
        "interleaver": interleaver.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "with_interleaver": True,
    }

    file_logger = run_single_codebook_encoder_conditional_entropy_experiment(
        encoder,
        modulator=modulator,
        channel=channel,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="ce",
        stop_tol=0.1,
        patience=10000,
        preamble=preamble,
    )

    return file_logger


def turbo_interleaver_conditional_entropy_same_encoder_experiment(
    experiment_id: str,
    encoder_name: str,
    interleaver_base_seed: int,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "interleaver_base_seed": interleaver_base_seed,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    base_encoder = get_encoder(encoder_name)(
        num_steps=block_len, device_manager=manager
    )
    # Assumes the encoder has 2 channels, will use the first 2
    # for noninterleaved and the last for the interleaved channel
    if isinstance(base_encoder, TurboEncoder):
        noninterleaved_encoder = base_encoder.noninterleaved_encoder
        interleaved_encoder = base_encoder.interleaved_encoder
    elif isinstance(base_encoder, TrellisEncoder):
        noninterleaved_encoder = base_encoder.get_encoder_channels(
            list(range(base_encoder.num_output_channels - 1))
        )
        interleaved_encoder = base_encoder.get_encoder_channels([-1])
    else:
        raise NotImplementedError(f"encoder type: {(base_encoder).__class__.__name__}")

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
        raise NotImplementedError(f"interleaver_base_seed={interleaver_base_seed}")

    nonturbo_encoder = noninterleaved_encoder.concat_outputs(interleaved_encoder)
    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    if block_len <= 20:
        print(
            f"Testing Conditional Entropy for Turbo encoder naively at block len {block_len}."
        )

        turbo_encoder = TurboEncoder(
            noninterleaved_encoder=noninterleaved_encoder,
            interleaved_encoder=interleaved_encoder,
            interleaver=interleaver,
            device_manager=manager,
        )
        codebook_turbo_encoder = turbo_encoder.to_codebook()

        preamble = {
            "args": arguments,
            "argument_hash": argument_hash,
            "encoder": codebook_turbo_encoder.long_settings(),
            "interleaver": interleaver.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
            "with_interleaver": True,
        }

        file_logger = run_single_codebook_encoder_conditional_entropy_experiment(
            codebook_turbo_encoder,
            modulator=modulator,
            channel=channel,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
        )

        turbo_encoder = None
        codebook_turbo_encoder = None

    preamble = {
        "args": arguments,
        "encoder": nonturbo_encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "with_interleaver": False,
    }

    file_logger = run_single_trellis_encoder_conditional_entropy_experiment(
        nonturbo_encoder,
        modulator=modulator,
        channel=channel,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        preamble=preamble,
    )

    return file_logger


def turbo_interleaver_conditional_entropy_same_encoder_experiment_repro(
    experiment_id: str,
    encoder_name: str,
    interleaver_base_seed: int,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "interleaver_base_seed": interleaver_base_seed,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    trellis_encoder = get_encoder(encoder_name)(
        num_steps=block_len, device_manager=manager
    )
    other_trellis_encoder = trellis_encoder.get_encoder_channels([-1])
    # Produces weird very good behavior at moderate block length
    # ANSWER: This is because once the codebook is generated, the original
    # interleaver is no longer used. When using the RandomPermuteInterleaver,
    # each entry in the codebook uses a different interleaver, and as a result
    # the code looks like a random code rather than a turbo code. This significantly
    # boosts BER and BLER performance.
    # interleaver = RandomPermuteInterleaver(input_size=block_len, device_manager=manager)
    interleaver = load_interleaver(
        "fixed",
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )
    print(interleaver)

    nonturbo_encoder = trellis_encoder.concat_outputs(other_trellis_encoder)
    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    # base_encoder = get_encoder(encoder_name)(
    #     num_steps=block_len, device_manager=manager
    # )
    # # Assumes the encoder has 2 channels, will use the first 2
    # # for noninterleaved and the last for the interleaved channel
    # if isinstance(base_encoder, TurboEncoder):
    #     noninterleaved_encoder = base_encoder.noninterleaved_encoder
    #     interleaved_encoder = base_encoder.interleaved_encoder
    # elif isinstance(base_encoder, TrellisEncoder):
    #     noninterleaved_encoder = base_encoder
    #     interleaved_encoder = base_encoder.get_encoder_channels([-1])
    # else:
    #     raise NotImplementedError(f"encoder type: {(base_encoder).__class__.__name__}")

    # if isinstance(interleaver_base_seed, int):
    #     permutation = torch.randperm(
    #         block_len,
    #         generator=torch.Generator(device=manager.device).manual_seed(
    #             interleaver_base_seed
    #         ),
    #         device=manager.device,
    #     )
    #     interleaver = FixedPermuteInterleaver(
    #         input_size=block_len, device_manager=manager, permutation=permutation
    #     )
    # else:
    #     raise NotImplementedError(f"interleaver_base_seed={interleaver_base_seed}")

    # nonturbo_encoder = noninterleaved_encoder.concat_outputs(interleaved_encoder)
    # modulator = BPSK(device_manager=manager)
    # channel = AWGN(snr=snr, device_manager=manager)

    if block_len <= 20:
        print(
            f"Testing Conditional Entropy for Turbo encoder naively at block len {block_len}."
        )

        turbo_encoder = TurboEncoder(
            noninterleaved_encoder=trellis_encoder,
            interleaved_encoder=other_trellis_encoder,
            interleaver=interleaver,
            device_manager=manager,
        )

        # turbo_encoder = TurboEncoder(
        #     noninterleaved_encoder=noninterleaved_encoder,
        #     interleaved_encoder=interleaved_encoder,
        #     interleaver=interleaver,
        #     device_manager=manager,
        # )
        # codebook_turbo_encoder = turbo_encoder.to_codebook()

        preamble = {
            "args": arguments,
            "argument_hash": argument_hash,
            "encoder": turbo_encoder.long_settings(),
            "interleaver": interleaver.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
            "with_interleaver": True,
        }

        file_logger = run_single_codebook_encoder_conditional_entropy_experiment_repro2(
            turbo_encoder,
            modulator=modulator,
            channel=channel,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
            stop_key="ce",
        )

        turbo_encoder = None
        codebook_turbo_encoder = None

    preamble = {
        "args": arguments,
        "encoder": nonturbo_encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "with_interleaver": False,
    }

    file_logger = run_single_trellis_encoder_conditional_entropy_experiment(
        nonturbo_encoder,
        modulator=modulator,
        channel=channel,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        preamble=preamble,
        stop_key="ce",
    )

    return file_logger


def turbo_iterated_vs_brute_force_experiment(
    experiment_id: str,
    encoder_name: str,
    interleaver_base_seed: int,
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    num_iter: int = 6,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
    }
    print(arguments)

    trellis_encoder = get_encoder(encoder_name)(
        num_steps=block_len, device_manager=manager
    )
    other_trellis_encoder = trellis_encoder.get_encoder_channels([-1])
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

    modulator = BPSK(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    turbo_encoder = StreamedTurboEncoder(
        noninterleaved_encoder=trellis_encoder,
        interleaved_encoder=other_trellis_encoder,
        interleaver=interleaver,
        device_manager=manager,
    )

    if block_len <= 16:
        print(
            f"Testing Conditional Entropy for Turbo encoder naively at block len {block_len}."
        )

        codebook_turbo_encoder = turbo_encoder.to_codebook()

        preamble = {
            "args": arguments,
            "encoder": codebook_turbo_encoder.long_settings(),
            "interleaver": interleaver.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
            "decoder_type": "brute_force",
        }

        file_logger = run_single_codebook_encoder_conditional_entropy_experiment(
            codebook_turbo_encoder,
            modulator=modulator,
            channel=channel,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
        )

        codebook_turbo_encoder = None

    print(
        f"Testing Cross Entropy for Turbo encoder with Iterated BCJR at block len {block_len}."
    )

    preamble = {
        "args": arguments,
        "encoder": turbo_encoder.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "decoder_type": "iterated_regular",
    }

    file_logger = run_single_turbo_iterate_cross_entropy_experiment(
        turbo_encoder,
        modulator=modulator,
        channel=channel,
        num_iter=num_iter,
        systematic=False,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        preamble=preamble,
    )

    if is_systematic(encoder_name):
        print(
            f"Testing Cross Entropy for Turbo encoder with Iterated Systematic BCJR at block len {block_len}."
        )

        preamble = {
            "args": arguments,
            "encoder": turbo_encoder.long_settings(),
            "modulator": modulator.long_settings(),
            "channel": channel.long_settings(),
            "decoder_type": "iterated_systematic",
        }

        file_logger = run_single_turbo_iterate_cross_entropy_experiment(
            turbo_encoder,
            modulator=modulator,
            channel=channel,
            num_iter=num_iter,
            systematic=True,
            manager=manager,
            file_logger=file_logger,
            num_samples=num_samples,
            batch_size=batch_size,
            preamble=preamble,
        )

    return file_logger


def turbo_conditional_entropy_experiment_jtree(
    experiment_id: str,
    encoder_name: str,
    interleaver_type: str,
    interleaver_base_seed: int = None,
    delay: int = 0,
    modulator_type: str = "identity",
    batch_size: int = 1,
    block_len: int = 10,
    num_samples: int = 1000,
    snr: float = 0.0,
    patience: int = None,
    elimination_tries: int = 50,
    elimination_seed: int = None,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
) -> FileLogger:
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")
    if patience is None:
        patience = 5 * batch_size

    arguments = {
        "experiment_id": experiment_id,
        "encoder_name": encoder_name,
        "interleaver_type": interleaver_type,
        "interleaver_base_seed": interleaver_base_seed,
        "delay": delay,
        "modulator_type": modulator_type,
        "batch_size": batch_size,
        "block_len": block_len,
        "num_samples": num_samples,
        "snr": snr,
        "elimination_tries": elimination_tries,
        "elimination_seed": elimination_seed,
        "patience": patience,
    }
    print(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    encoder = get_encoder(encoder_name)(
        num_steps=block_len, device_manager=manager, delay=delay
    )

    interleaver = load_interleaver(
        interleaver_type=interleaver_type,
        block_len=block_len,
        manager=manager,
        interleaver_base_seed=interleaver_base_seed,
    )

    if isinstance(encoder, TurboEncoder):
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
        # Temporary to make sure things are working
        assert encoder.is_nonrecursive
    else:
        raise NotImplementedError(f"encoder type: {(encoder).__class__.__name__}")

    modulator = get_modulator(modulator_type=modulator_type)
    channel = AWGN(snr=snr, device_manager=manager)

    print(
        f"Testing Conditional Entropy for Turbo encoder with Junction Tree at block len {block_len}."
    )
    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
        "encoder": encoder.long_settings(),
        "interleaver": interleaver.long_settings(),
        "modulator": modulator.long_settings(),
        "channel": channel.long_settings(),
        "with_interleaver": True,
    }

    file_logger = run_single_junction_tree_conditional_entropy_experiment(
        encoder,
        modulator=modulator,
        channel=channel,
        manager=manager,
        file_logger=file_logger,
        num_samples=num_samples,
        batch_size=batch_size,
        elimination_tries=elimination_tries,
        elimination_seed=elimination_seed,
        stop_key="ce",
        stop_tol=0.1,
        patience=patience,
        preamble=preamble,
    )

    return file_logger
