from typing import Any, Dict, Sequence
from numbers import Number
from pathlib import Path
from collections import defaultdict
import math
import json

from typing import Union
from tqdm import tqdm
import torch

from ..utils import safe_create_file


class ResultsProcessor:
    def __init__(self, listeners: Sequence["Listener"]) -> None:
        self.listeners = listeners
        self.running_means = defaultdict(lambda: 0)
        self.running_sqmeans = defaultdict(lambda: 0)
        self.running_counts = defaultdict(lambda: 0)
        self.num_samples = 0

    def update(self, records: Dict[str, torch.Tensor], num_samples=None):
        if num_samples is not None:
            self.num_samples += num_samples
        for name, record in records.items():
            mean = torch.mean(record.float())
            new_count = torch.numel(record)
            # print(f"{name} : {new_count}")
            self.running_means[name] = (
                self.running_means[name] * self.running_counts[name]
                + mean.item() * new_count
            ) / (self.running_counts[name] + new_count)

            sqmean = torch.mean(record.float() ** 2)
            self.running_sqmeans[name] = (
                self.running_sqmeans[name] * self.running_counts[name]
                + sqmean.item() * new_count
            ) / (self.running_counts[name] + new_count)

            self.running_counts[name] += new_count

        for listener in self.listeners:
            listener.update(self.results)

    def get_variance(self, name: str):
        counts = self.running_counts[name]
        if counts <= 1:
            res = 0
        else:
            res = (
                (self.running_sqmeans[name] - self.running_means[name] ** 2)
                * counts
                / (counts - 1)
            )
        if res < 0:
            print(f"variance became negative! {res}")
            res = 0
        return res

    @property
    def results(self):
        means = {f"{name}__mean": val for name, val in self.running_means.items()}
        stds = {
            f"{name}__std": math.sqrt(self.get_variance(name))
            for name in self.running_sqmeans.keys()
        }
        err = {
            f"{name}__err": 2
            * stds[f"{name}__std"]
            / math.sqrt(self.running_counts[name])
            for name in self.running_counts.keys()
        }
        return {
            **means,
            **stds,
            **err,
            **({"num_samples": self.num_samples} if self.num_samples > 0 else {}),
        }

    def close(self):
        for listener in self.listeners:
            listener.end_experiment()


class TqdmProgressBar:
    def __init__(self, log_every=1, watch=None) -> None:
        self.log_every = log_every
        self.log_counter = 0
        self.watch = watch

    def new_experiment(self, total: int = None):
        self.progress_bar = tqdm(total=total)

    def update(self, records: Dict[str, Number]):
        self.log_counter = (self.log_counter + 1) % self.log_every

        if self.log_counter == 0:
            self.progress_bar.update(n=self.log_every)
            if self.watch is not None:
                watch_keys = {k for k in records.keys() if k.startswith(self.watch)}
                self.progress_bar.set_postfix(
                    **{k: v for k, v in records.items() if k in watch_keys}
                )
            else:
                self.progress_bar.set_postfix(**records)

        return records

    def end_experiment(self):
        self.progress_bar.close()


class FileLogger:
    def __init__(self, preamble: Any, fp: Path, log_every=1) -> None:
        self.fp = fp
        self.log_every = log_every
        self.preamble = preamble
        self.data = []

        self.num_experiments = 0
        self.log_counter = 0

    def new_experiment(self, preamble):
        self.log_counter = 0
        self.num_experiments += 1
        self.data.append({"preamble": preamble, "experiment_num": self.num_experiments})
        self.flush()

    def flush(self):
        print("Flushing experiment results")
        with open(safe_create_file(self.fp), "w", encoding="utf-8") as fd:
            json.dump(
                {"preamble": self.preamble, "data": self.data},
                fd,
                ensure_ascii=False,
                indent=4,
            )

    def update(self, records: Dict[str, Number]):
        self.log_counter = (self.log_counter + 1) % self.log_every
        self.data[-1] = {**self.data[-1], "results": records}

        if self.log_counter == 0:
            self.flush()

    def end_experiment(self):
        self.flush()


Listener = TqdmProgressBar
