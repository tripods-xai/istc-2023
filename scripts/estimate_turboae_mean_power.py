from src.encoders import turboae_cont_exact_nn
import torch
import math
from src.engine import ResultsProcessor, TqdmProgressBar
from pprint import pprint
from src.utils import DeviceManager

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--compute", action="store_true")
parser.add_argument("--test", action="store_true")

with torch.no_grad():
    manager = DeviceManager(seed=4137)
    args = parser.parse_args()

    msg_length = 100

    num_samples = 10000000
    batch_size = 10000
    running_mean = 0
    running_sqmean = 0
    num_batches = num_samples // batch_size
    pbar = TqdmProgressBar()
    pbar.new_experiment(total=num_batches)
    results_processor = ResultsProcessor([pbar])
    if args.compute:
        encoder = turboae_cont_exact_nn(num_steps=msg_length, precomputed_stats=False)
        for i in range(0, num_samples, batch_size):
            # Don't worry about clipping batch_size if we're at the end.
            inputs = torch.randint(
                0,
                2,
                size=(batch_size, msg_length),
                dtype=torch.float,
                device=manager.device,
            )
            encoded = encoder(inputs)
            codeword_vars, codeword_means = torch.var_mean(
                encoded, dim=[1, 2], unbiased=False
            )
            assert codeword_means.ndim == 1
            assert codeword_vars.ndim == 1
            assert codeword_means.shape[0] == codeword_vars.shape[0] == batch_size
            results_processor.update({"mean": codeword_means, "var": codeword_vars})

        pprint(results_processor.results)

    if args.test:
        encoder = turboae_cont_exact_nn(num_steps=msg_length, precomputed_stats=True)
        for i in range(0, num_samples, batch_size):
            # Don't worry about clipping batch_size if we're at the end.
            inputs = torch.randint(
                0,
                2,
                size=(batch_size, msg_length),
                dtype=torch.float,
                device=manager.device,
            )
            encoded = encoder(inputs)
            codeword_vars, codeword_means = torch.var_mean(
                encoded, dim=[1, 2], unbiased=False
            )
            assert codeword_means.ndim == 1
            assert codeword_vars.ndim == 1
            assert codeword_means.shape[0] == codeword_vars.shape[0] == batch_size
            results_processor.update({"mean": codeword_means, "var": codeword_vars})

        pprint(results_processor.results)
