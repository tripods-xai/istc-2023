from typing import Tuple, Dict, Any
from glob import glob
import hashlib
from pathlib import Path
from pprint import pprint
import time

import torch

from ..engine import FileLogger, ResultsProcessor, TqdmProgressBar
from ..graphs import nonrecursive_turbo_graph
from ..utils import (
    DeviceManager,
    DEFAULT_DEVICE_MANAGER,
    parse_timestamp,
    TIME_FORMAT,
    Precision,
    PRECISION_DTYPE_DICT,
)
from ..constants import CHECKPOINTS_DIR
from ..measurements import TurboAEDecompositionSampler, BCJRDecompositionSampler
from ..encoders import ENC_interCNN, get_info_from_checkpoint
from ..decoders import TurboAEDecoder
from ..interleavers import FixedPermuteInterleaver
from ..modulation import IdentityModulation
from ..channels import AWGN

from .experiment_utils import (
    load_interleaver,
    serialize,
    run_measurement,
    get_matching_checkpoints,
    TurboAEType,
    load_original_turboe_encoder,
    load_turboae_decoder,
    load_turboae_encoder_checkpoint,
)


def cluster_tree_statistics(
    experiment_id: str,
    interleaver_base_seed: int,
    block_len: int,
    window: int,
    num_samples: int,
    delay: int = None,
    tries: int = 100,
    sample_thresh: int = 3,
    file_logger: FileLogger = None,
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    if delay is None:
        delay = (window - 1) - ((window - 1) // 2)

    arguments = {
        "experiment_id": experiment_id,
        "interleaver_base_seed": interleaver_base_seed,
        "block_len": block_len,
        "window": window,
        "delay": delay,
        "num_samples": num_samples,
    }
    pprint(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")
    interleaver_seed_generator = DeviceManager(no_cuda=True, seed=interleaver_base_seed)

    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
    }

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    results_processor = ResultsProcessor([])
    result_list = []

    for i in range(num_samples):
        print(f"Run {i+1}/{num_samples}")
        interleaver_seed = interleaver_seed_generator.generate_seed()
        elimination_seed = interleaver_seed_generator.generate_seed()
        interleaver = load_interleaver(
            interleaver_type="fixed",
            block_len=block_len,
            manager=manager,
            interleaver_base_seed=interleaver_seed,
        )
        nt_inference_graph = nonrecursive_turbo_graph(
            interleaver.permutation, window=window, delay=delay
        ).with_elimination_ordering(
            sample_thresh=sample_thresh, tries=tries, seed=elimination_seed
        )
        result = {
            "factor_width": nt_inference_graph.factor_width,
            "interleaver_seed": interleaver_seed,
            "elimination_seed": elimination_seed,
            "type": "trial",
        }
        results_processor.update(
            {"width": torch.tensor(nt_inference_graph.factor_width)}
        )

        if file_logger is not None:
            result_list.append(result)
            file_logger.update(result_list)

    summary_result = {**results_processor.results, "type": "summary"}
    pprint(summary_result)
    if file_logger is not None:
        result_list.append(result)
        file_logger.update(result_list)
        file_logger.end_experiment()

    return file_logger


def run_checkpoint_measurement(
    checkpoint_info: Dict[str, Any],
    sampler: TurboAEDecompositionSampler,
    num_samples: int,
    batch_size: int,
    stop_key: str = "kl",
    stop_tol=2e-1,
    patience=0,
):
    print(f"Measuring decomposition on checkpoint {checkpoint_info['filepath']}")
    state_dict = torch.load(
        checkpoint_info["filepath"], map_location=sampler.device_manager.device
    )
    sampler.load_state_dict(state_dict=state_dict, rebuild_jtree=False)

    return run_measurement(
        sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key=stop_key,
        stop_tol=stop_tol,
        patience=patience,
    )


@torch.no_grad()
def decomposition_trajectory(
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
    stop_tol=2e-1,
    resolution: int = 5,
    elimination_seed: int = None,
    precision: Precision = "half",
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
        "elimination_seed": elimination_seed,
        "precision": precision,
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

    sampler = TurboAEDecompositionSampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder=decoder,
        elimination_seed=elimination_seed,
        dtype=PRECISION_DTYPE_DICT[precision],
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
            stop_key="kl",
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
        est_time = (len(to_measure_infos) - (i + 1)) * avg_time
        print(f"Estimated Remaining Time: {est_time}s")

    file_logger.end_experiment()

    return file_logger


@torch.no_grad()
def decomposition_estimation_finetuned_tae(
    experiment_id: str,
    block_len: int,
    interleaver_base_seed: int,
    turboae_type: TurboAEType,
    decoder_path: str,
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
    stop_tol=2e-1,
    elimination_seed: int = None,
    precision: Precision = "half",
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "block_len": block_len,
        "interleaver_base_seed": interleaver_base_seed,
        "turboae_type": turboae_type,
        "decoder_path": decoder_path,
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
        "elimination_seed": elimination_seed,
        "precision": precision,
    }
    pprint(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
    }

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

    sampler = TurboAEDecompositionSampler(
        encoder=encoder,
        modulator=modulator,
        channel=channel,
        decoder=decoder,
        elimination_seed=elimination_seed,
        dtype=PRECISION_DTYPE_DICT[precision],
        device_manager=manager,
    )

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    result = run_measurement(
        sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="kl",
        stop_tol=stop_tol,
        patience=0,
    )

    pprint(result)
    if file_logger is not None:
        file_logger.update(result)

    file_logger.end_experiment()

    return file_logger


@torch.no_grad()
def decomposition_estimation_bcjr_trained_tae(
    experiment_id: str,
    encoder_decoder_path: str,
    num_samples: int,
    batch_size: int,
    snr: float,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iter: int = 6,
    stop_tol=1e-1,
    elimination_seed: int = None,
    precision: Precision = "half",
    manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    file_logger: FileLogger = None,
    **kwargs,
):
    if kwargs != {}:
        print(f"Unused arguments {kwargs}")

    arguments = {
        "experiment_id": experiment_id,
        "encoder_decoder_path": encoder_decoder_path,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "snr": snr,
        "front_pad": front_pad,
        "first_pad": first_pad,
        "num_iter": num_iter,
        "stop_tol": stop_tol,
        "elimination_seed": elimination_seed,
        "precision": precision,
    }
    pprint(arguments)
    argument_hash = hashlib.sha1(str(arguments).encode("utf-8")).hexdigest()
    print(f"Hash: {argument_hash}")

    preamble = {
        "args": arguments,
        "argument_hash": argument_hash,
    }

    encoder = load_turboae_encoder_checkpoint(
        encoder_decoder_path=encoder_decoder_path,
        front_pad=front_pad,
        first_pad=first_pad,
        binarize=False,
        device_manager=manager,
    )

    modulator = IdentityModulation(device_manager=manager)
    channel = AWGN(snr=snr, device_manager=manager)

    sampler = BCJRDecompositionSampler(
        encoder=encoder.to_conv_code(no_delay=True),
        modulator=modulator,
        channel=channel,
        num_iter=num_iter,
        elimination_seed=elimination_seed,
        dtype=PRECISION_DTYPE_DICT[precision],
        device_manager=manager,
    )

    if file_logger is not None:
        file_logger.new_experiment(preamble=preamble)

    result = run_measurement(
        sampler,
        num_samples=num_samples,
        batch_size=batch_size,
        stop_key="kl",
        stop_tol=stop_tol,
        patience=0,
    )

    pprint(result)
    if file_logger is not None:
        file_logger.update(result)

    file_logger.end_experiment()

    return file_logger
