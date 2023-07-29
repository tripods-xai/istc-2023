from typing import Literal, Dict, Tuple
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import SGD, Adam
from ..utils import (
    DeviceManager,
    TIME_FORMAT,
    DEFAULT_DEVICE_MANAGER,
    filter_state_dict,
    parse_timestamp,
)
from ..encoders import get_info_from_checkpoint, ENC_interCNN
from ..interleavers import (
    FixedPermuteInterleaver,
    RandomPermuteInterleaver,
    BatchRandomPermuteInterleaver,
    TurboAEInterleaver,
    Interleaver,
)
from ..decoders import TurboAEDecoder
from ..engine import ResultsProcessor, TqdmProgressBar
from ..measurements import Sampler
from ..constants import (
    TURBOAE_ENCODER_CONT_PATH,
    TURBOAE_DECODER_CONT_PATH,
    TURBOAE_DECODER_BINARY_PATH,
    TURBOAE_ENCODER_BINARY_PATH,
    CHECKPOINTS_DIR,
)

InterleaverType = Literal["fixed", "batch_random", "sample_random", "turboae"]


def check_stop_criterion(
    i: int, results_processor: ResultsProcessor, key: str, tol: float, patience: int
):
    if patience > i:
        return False
    results = results_processor.results
    err_diam = results[f"{key}__err"] * 2
    mean = results[f"{key}__mean"]
    return err_diam < (mean * tol)


def load_interleaver(
    interleaver_type: InterleaverType,
    block_len: int,
    manager: DeviceManager,
    interleaver_base_seed: int = None,
):
    if interleaver_type == "fixed":
        if isinstance(interleaver_base_seed, int):
            permutation = torch.randperm(
                block_len,
                generator=torch.Generator(device="cpu").manual_seed(
                    interleaver_base_seed
                ),
                device="cpu",
            ).to(manager.device)
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
    elif interleaver_type == "turboae":
        if block_len != 100:
            raise ValueError(
                f"Must use block length 100 with TurboAEInterleaver. Used {block_len}."
            )
        interleaver = TurboAEInterleaver(device_manager=manager)
    else:
        raise NotImplementedError(f"Interleaver type {interleaver_type}")

    return interleaver


def load_optimizer_factory(optimizer: str, lr: float):
    if optimizer == "sgd":
        return lambda p: SGD(p, lr=lr)
    elif optimizer == "adam":
        return lambda p: Adam(p, lr=lr)
    raise ValueError(f"Did not recognize optimizer {optimizer}.")


def serialize(d: dict):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            new_d[k] = v.strftime(TIME_FORMAT)
        elif isinstance(v, Path):
            new_d[k] = str(v)
        else:
            new_d[k] = v

    return new_d


def load_turboae_encoder(
    interleaver: Interleaver,
    enc_num_layer=2,
    enc_num_unit=100,
    enc_kernel_size=5,
    first_pad=False,
    front_pad=False,
    binarize=False,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    encoder_path: Path = None,
    state_dict=None,
):
    if encoder_path is None and state_dict is None:
        raise ValueError

    encoder = ENC_interCNN(
        enc_num_layer=enc_num_layer,
        enc_num_unit=enc_num_unit,
        enc_kernel_size=enc_kernel_size,
        interleaver=interleaver,
        first_pad=first_pad,
        front_pad=front_pad,
        binarize=binarize,
        device_manager=device_manager,
    )

    if encoder_path is not None:
        state_dict = torch.load(encoder_path, map_location=device_manager.device)

    encoder.load_state_dict(state_dict=state_dict)

    return encoder


def load_turboae_decoder(
    interleaver: Interleaver,
    front_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
    decoder_path: Path = None,
    state_dict=None,
):
    if decoder_path is None and state_dict is None:
        raise ValueError

    decoder = TurboAEDecoder(
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_num_unit=dec_num_unit,
        dec_kernel_size=dec_kernel_size,
        front_pad=front_pad,
        interleaver=interleaver,
        device_manager=device_manager,
    )

    if decoder_path is not None:
        state_dict = torch.load(decoder_path, map_location=device_manager.device)

    decoder.load_state_dict(state_dict=state_dict)

    return decoder


def load_turboae_encoder_checkpoint(
    encoder_decoder_path: str,
    front_pad: bool = False,
    first_pad: bool = False,
    binarize=False,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    print(f"Loading encoder weight initialization from {encoder_decoder_path}")
    state_dict = torch.load(encoder_decoder_path, map_location=device_manager.device)

    model_info = get_info_from_checkpoint(
        encoder_decoder_path, device_manager=device_manager
    )
    encoder = load_turboae_encoder(
        enc_num_layer=model_info["enc_num_layer"],
        enc_num_unit=model_info["enc_num_unit"],
        enc_kernel_size=model_info["enc_kernel_size"],
        interleaver=model_info["interleaver"],
        first_pad=first_pad,
        front_pad=front_pad,
        binarize=binarize,
        device_manager=device_manager,
        state_dict=filter_state_dict(state_dict, key="encoder"),
    )

    return encoder


def load_turboae_decoder_checkpoint(
    encoder_decoder_path: str,
    front_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    print(f"Loading decoder weight initialization from {encoder_decoder_path}")
    state_dict = torch.load(encoder_decoder_path, map_location=device_manager.device)

    model_info = get_info_from_checkpoint(
        encoder_decoder_path, device_manager=device_manager
    )
    decoder = load_turboae_decoder(
        interleaver=model_info["interleaver"],
        front_pad=front_pad,
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_kernel_size=dec_kernel_size,
        dec_num_unit=dec_num_unit,
        device_manager=device_manager,
        state_dict=filter_state_dict(state_dict, key="decoder"),
    )

    return decoder


def load_turboae_checkpoint(
    encoder_decoder_path: str,
    front_pad: bool = False,
    first_pad: bool = False,
    num_iteration: int = 6,
    num_iter_ft: int = 5,
    dec_num_layer: int = 5,
    dec_num_unit: int = 100,
    dec_kernel_size: int = 5,
    binarize=False,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    print(f"Loading weight initialization from {encoder_decoder_path}")
    state_dict = torch.load(encoder_decoder_path, map_location=device_manager.device)

    model_info = get_info_from_checkpoint(
        encoder_decoder_path, device_manager=device_manager
    )
    encoder = load_turboae_encoder(
        enc_num_layer=model_info["enc_num_layer"],
        enc_num_unit=model_info["enc_num_unit"],
        enc_kernel_size=model_info["enc_kernel_size"],
        interleaver=model_info["interleaver"],
        first_pad=first_pad,
        front_pad=front_pad,
        binarize=binarize,
        device_manager=device_manager,
        state_dict=filter_state_dict(state_dict, key="encoder"),
    )
    interleaver = encoder.interleaver

    decoder = load_turboae_decoder(
        interleaver=interleaver,
        front_pad=front_pad,
        num_iteration=num_iteration,
        num_iter_ft=num_iter_ft,
        dec_num_layer=dec_num_layer,
        dec_kernel_size=dec_kernel_size,
        dec_num_unit=dec_num_unit,
        device_manager=device_manager,
        state_dict=filter_state_dict(state_dict, key="decoder"),
    )

    return {"encoder": encoder, "decoder": decoder}


TurboAEType = Literal["continuous", "binary"]
ENCODER_PATH_DICT: Dict[TurboAEType, Path] = {
    "continuous": TURBOAE_ENCODER_CONT_PATH,
    "binary": TURBOAE_ENCODER_BINARY_PATH,
}
DECODER_PATH_DICT: Dict[TurboAEType, Path] = {
    "continuous": TURBOAE_DECODER_CONT_PATH,
    "binary": TURBOAE_DECODER_BINARY_PATH,
}


def load_original_turboe_encoder(
    interleaver: Interleaver,
    turboae_type: TurboAEType,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    encoder_path = ENCODER_PATH_DICT[turboae_type]
    state_dict = torch.load(encoder_path, map_location=device_manager.device)
    base_cnn = ENC_interCNN(
        enc_num_layer=state_dict["enc_num_layer"].item(),
        enc_num_unit=state_dict["enc_num_unit"].item(),
        enc_kernel_size=state_dict["enc_kernel_size"].item(),
        interleaver=interleaver,
        binarize=(turboae_type == "binary"),
        device_manager=device_manager,
    )
    base_cnn.pre_initialize(state_dict)

    return base_cnn


def load_original_turboae_decoder(
    interleaver: Interleaver,
    turboae_type: TurboAEType,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    decoder_path = DECODER_PATH_DICT[turboae_type]
    state_dict = torch.load(decoder_path, map_location=device_manager.device)
    decoder_cnn = TurboAEDecoder(
        num_iteration=6,
        num_iter_ft=5,
        dec_num_layer=5,
        dec_num_unit=100,
        dec_kernel_size=5,
        front_pad=False,
        interleaver=interleaver,
        device_manager=DEFAULT_DEVICE_MANAGER,
    )
    decoder_cnn.pre_initialize(state_dict)

    return decoder_cnn


def load_original_turboae_encoder_decoder(
    interleaver: Interleaver,
    turboae_type: TurboAEType,
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER,
):
    return {
        "encoder": load_original_turboe_encoder(
            interleaver=interleaver,
            turboae_type=turboae_type,
            device_manager=device_manager,
        ),
        "decoder": load_original_turboae_decoder(
            interleaver=interleaver,
            turboae_type=turboae_type,
            device_manager=device_manager,
        ),
    }


def parse_checkpoint_filepath(filepath: Path, basename: str):
    ep_and_datetime_parts = filepath.stem[len(f"{basename}_") :].split("_")
    ep_num = int(ep_and_datetime_parts[0][len("ep") :])
    datetime_str = "_".join(ep_and_datetime_parts[1:])
    datetime_obj = parse_timestamp(datetime_str)
    return {
        "basename": basename,
        "epoch": ep_num,
        "datetime": datetime_obj,
        "filepath": filepath,
    }


def get_matching_checkpoints(
    checkpoint_basename: str, checkpoint_daterange: Tuple[datetime, datetime]
):
    all_checkpoints = CHECKPOINTS_DIR.glob(f"{checkpoint_basename}_*.pt")
    matching_checkpoint_info = []
    for checkpoint_fpath in all_checkpoints:
        checkpoint_info = parse_checkpoint_filepath(
            checkpoint_fpath, basename=checkpoint_basename
        )
        if (
            checkpoint_daterange[0]
            <= checkpoint_info["datetime"]
            <= checkpoint_daterange[1]
        ):
            matching_checkpoint_info.append(checkpoint_info)

    return matching_checkpoint_info


def run_measurement(
    sampler: Sampler,
    num_samples: int,
    batch_size: int,
    stop_key: str = None,
    stop_tol=1e-1,
    patience=0,
):
    num_batches = (num_samples + batch_size - 1) // batch_size
    progress_bar = TqdmProgressBar(watch=stop_key)

    progress_bar.new_experiment(total=num_batches)

    listeners = [l for l in [progress_bar] if l is not None]
    results_processor = ResultsProcessor(listeners=listeners)

    for i in range(0, num_samples, batch_size):
        cur_batch_size = min(num_samples - i, batch_size)
        sampled_cond_entropies = sampler.sample(cur_batch_size)
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

    return results_processor.results
