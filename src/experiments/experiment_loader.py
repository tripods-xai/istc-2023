from typing import Dict, Callable
import traceback

from pathlib import Path
import json

from sklearn.model_selection import ParameterGrid

from ..utils import DeviceManager, get_timestamp
from ..engine import FileLogger

from .conditional_entropy_experiments import (
    trellis_encoder_conditional_entropy_experiment,
    trellis_encoder_conditional_entropy_experiment_concat_encoders,
    trellis_encoder_conditional_entropy_experiment_random_encoder_variable_window_block_len,
    turbo_interleaver_conditional_entropy_same_encoder_experiment,
    turbo_iterated_vs_brute_force_experiment,
    benchmark_trellis_turbo_codes,
    turbo_conditional_entropy_experiment,
    turbo_conditional_entropy_experiment_jtree,
    turbo_interleaver_conditional_entropy_same_encoder_experiment_repro,
    benchmark_trellis_codes,
)
from .benchmark_experiments import *
from .decoder_training_experiments import *
from .encoder_training_experiments import *
from .turboae_training_experiments import *
from .junction_tree_experiments import *


EXPERIMENT_REGISTRY: Dict[str, Callable[..., FileLogger]] = {
    trellis_encoder_conditional_entropy_experiment.__name__: trellis_encoder_conditional_entropy_experiment,
    trellis_encoder_conditional_entropy_experiment_concat_encoders.__name__: trellis_encoder_conditional_entropy_experiment_concat_encoders,
    trellis_encoder_conditional_entropy_experiment_random_encoder_variable_window_block_len.__name__: trellis_encoder_conditional_entropy_experiment_random_encoder_variable_window_block_len,
    turbo_interleaver_conditional_entropy_same_encoder_experiment.__name__: turbo_interleaver_conditional_entropy_same_encoder_experiment,
    turbo_iterated_vs_brute_force_experiment.__name__: turbo_iterated_vs_brute_force_experiment,
    retrain_turboae_for_new_block_len.__name__: retrain_turboae_for_new_block_len,
    train_turbo_table.__name__: train_turbo_table,
    train_turboae_neural_encoder.__name__: train_turboae_neural_encoder,
    train_turbo_table_bcjr.__name__: train_turbo_table_bcjr,
    train_turbo_table_fourier.__name__: train_turbo_table_fourier,
    train_neural_decoder.__name__: train_neural_decoder,
    benchmark_trellis_turbo_codes.__name__: benchmark_trellis_turbo_codes,
    turbo_conditional_entropy_experiment.__name__: turbo_conditional_entropy_experiment,
    benchmark_neural_turbo_codes.__name__: benchmark_neural_turbo_codes,
    train_turbo_fourier_bcjr.__name__: train_turbo_fourier_bcjr,
    turbo_conditional_entropy_experiment_jtree.__name__: turbo_conditional_entropy_experiment_jtree,
    turbo_interleaver_conditional_entropy_same_encoder_experiment_repro.__name__: turbo_interleaver_conditional_entropy_same_encoder_experiment_repro,
    benchmark_trellis_codes.__name__: benchmark_trellis_codes,
    train_turbo_table_swarm_bcjr.__name__: train_turbo_table_swarm_bcjr,
    train_turbo_table_rerun_bcjr.__name__: train_turbo_table_rerun_bcjr,
    train_turboae.__name__: train_turboae,
    train_turboae_table.__name__: train_turboae_table,
    cluster_tree_statistics.__name__: cluster_tree_statistics,
    decomposition_trajectory.__name__: decomposition_trajectory,
    decomposition_estimation_bcjr_trained_tae.__name__: decomposition_estimation_bcjr_trained_tae,
    benchmark_turboae_codes.__name__: benchmark_turboae_codes,
    retrain_original_turboae_for_new_block_len.__name__: retrain_original_turboae_for_new_block_len,
    benchmark_finetuned_original_turboae_codes.__name__: benchmark_finetuned_original_turboae_codes,
    xe_trajectory.__name__: xe_trajectory,
    decomposition_estimation_finetuned_tae.__name__: decomposition_estimation_finetuned_tae,
    benchmark_turboae_codes_jt.__name__: benchmark_turboae_codes_jt,
    benchmark_finetuned_turboae_codes_jt.__name__: benchmark_finetuned_turboae_codes_jt,
}


def run_all_experiments(
    experiment_settings_json: Path,
    no_cuda=False,
    output_dir: Path = None,
    log_every=200,
):
    print(f"Loading experiment settings from {experiment_settings_json}")
    with open(experiment_settings_json, mode="r") as e:
        experiment_dict = json.load(e)

    ids = experiment_dict.keys()
    print(f"Running all experiments {list(ids)}")
    for e_id in ids:
        run_experiments(
            experiment_id=e_id,
            experiment_settings_json=experiment_settings_json,
            no_cuda=no_cuda,
            output_dir=output_dir,
            log_every=log_every,
        )


def parse_dunder_params(run_params):
    new_run_params = {}
    for k, v in run_params.items():
        if "__" in k:
            new_keys = k.split("__")
            assert len(new_keys) == len(run_params[k])
            for nk, nv in zip(new_keys, run_params[k]):
                new_run_params[nk] = nv
        else:
            new_run_params[k] = v
    return new_run_params


def run_experiments(
    experiment_id: str,
    experiment_settings_json: Path,
    no_cuda=False,
    output_dir: Path = None,
    model_dir: Path = MODELS_DIR,
    log_every=200,
) -> FileLogger:

    timestamp = get_timestamp()

    print(f"Loading experiment settings from {experiment_settings_json}")
    with open(experiment_settings_json, mode="r") as e:
        experiment_dict = json.load(e)

    experiment_settings: dict = experiment_dict[experiment_id]
    experiment_runnable_name = experiment_settings["name"]
    seed = experiment_settings["seed"]
    experiment_runnable = EXPERIMENT_REGISTRY[experiment_runnable_name]
    compressed_run_settings = experiment_settings["runs"]
    run_settings = list(ParameterGrid(compressed_run_settings))

    manager = DeviceManager(no_cuda=no_cuda, seed=seed)

    if output_dir is not None:
        file_logger_fp = output_dir / f"{experiment_id}.json"
        print(f"Writing experiment outputs to {file_logger_fp}.")
        file_logger = FileLogger(
            preamble={
                "experiment_id": experiment_id,
                "experiment_settings_json": str(experiment_settings_json),
                "experiment_name": experiment_runnable_name,
                "timestamp": timestamp,
                "seed": seed,
            },
            log_every=log_every,
            fp=file_logger_fp,
        )

    else:
        file_logger = None

    updated_file_logger = file_logger
    for i, run in enumerate(run_settings):
        print(f"Running subexperiment {i+1}/{len(run_settings)}.")
        try:
            run = parse_dunder_params(run)
        except Exception:
            print(f"Unable to parse dunder params of run {run}.")
            print(traceback.format_exc())
            continue
        try:
            updated_file_logger = experiment_runnable(
                experiment_id=experiment_id,
                file_logger=updated_file_logger,
                manager=manager,
                **run,
                output_dir=output_dir,
                model_dir=model_dir,
            )
            updated_file_logger.flush()
        except Exception as e:
            print(f"Experiment failed due to error")
            print(traceback.format_exc())
        finally:
            updated_file_logger.flush()

    return updated_file_logger
