from pprint import pprint
from typing import Tuple, Dict, Generic, TypeVar, List, Any, Union
from dataclasses import dataclass, replace
import itertools as it
import time
import abc
from collections.abc import Iterator, Iterable
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer

from ..utils import (
    DEFAULT_DEVICE_MANAGER,
    EPSILON,
    DeviceManager,
    WithSettings,
    GeneratorWithLength,
    RepeatableGenerator,
    data_gen,
)
from ..engine import ResultsProcessor


def canonicalize_schedules(*args: Union[int, list]):
    assert len(args) > 0
    if any(isinstance(arg, list) for arg in args):
        assert all(isinstance(arg, list) for arg in args)
        assert all(len(arg) == len(args[0]) for arg in args)
        list_args = args
    else:
        list_args = [[arg] for arg in args]
    return list_args


class CodingTrainer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ) -> None:
        self.device_manager = device_manager

    @property
    def device_manager(self) -> DeviceManager:
        return self._device_manager

    @device_manager.setter
    def device_manager(self, value):
        self._device_manager = value

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        pass

    def summary(self) -> None:
        pass

    def data_gen(
        self,
        num_steps: int,
        batch_size: int,
        num_batches: int = 1,
    ):
        return data_gen(
            input_size=self.input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            num_batches=num_batches,
            device_manager=self.device_manager,
        )

    @torch.no_grad()
    def back_tracking(
        self,
        input_batches: GeneratorWithLength[torch.Tensor],
        prev_params: Dict[str, torch.Tensor],
        prev_loss: torch.Tensor,
        descent: float,
        shrinkage: float,
        failure: int,
    ) -> Dict[str, Any]:
        named_params_dict = {k: v for k, v in self.named_parameters()}
        grad_norm = (
            torch.norm(
                torch.cat(
                    [
                        param.grad.detach().reshape(-1)
                        for k, param in named_params_dict.items()
                    ]
                ),
                p=2,
            )
            ** 2
        )
        diffs = {
            name: (named_params_dict[name].detach() - prev_param)
            for name, prev_param in prev_params.items()
        }
        success = False
        for i in range(failure):
            if i != 0:
                for name, param in named_params_dict.items():
                    param.data = prev_params[name] + diffs[name] * (shrinkage**i)
            self.apply_constraint()

            loss = torch.tensor(0.0, device=self.device_manager.device)
            for input_batch in input_batches:
                logits, _ = self.reevaluate(input_batch.float(), validate=False)
                loss += self.metrics(input_batch.float(), logits)[0] / len(
                    input_batches
                )
            if loss <= prev_loss - descent * grad_norm:
                success = True
                break
        if not success:
            print("Backtracking failure...")

        return {
            "back_steps": i,
            "shrinkage": shrinkage**i,
            "descent": float(descent),
            "back_success": success,
        }

    def train_step(
        self,
        input_batches: GeneratorWithLength[torch.Tensor],
        optimizer: Optimizer,
        prev_params=None,
        prev_grads=None,
        param_filter=None,
        # Backtracking
        back_tracking=False,
        descent=0.2,
        shrinkage=0.9,
        failure=10,
        # Other details for forward
        **forward_kwargs,
    ) -> dict:
        # zero the parameter gradients
        optimizer.zero_grad()

        #########################
        metrics_accum = defaultdict(
            lambda: torch.tensor(0.0, device=self.device_manager.device)
        )
        num_samples = 0
        total_loss = torch.tensor(0.0, device=self.device_manager.device)
        for input_batch in input_batches:
            logits, forward_metrics = self.forward(
                input_batch.float(), validate=False, **forward_kwargs
            )

            loss, metrics = self.metrics(input_batch.float(), logits)
            (loss / len(input_batches)).backward()
            total_loss += loss.detach() / len(input_batches)
            for k, v in metrics.items():
                metrics_accum[k] += v.detach() / len(input_batches)
            for k, v in forward_metrics.items():
                metrics_accum[k] += v.detach() / len(input_batches)
            num_samples += input_batch.shape[0]

        optimizer.step()
        back_tracking_res = {}
        if back_tracking:
            assert prev_params is not None
            back_tracking_res = self.back_tracking(
                input_batches,
                prev_params=prev_params,
                prev_loss=total_loss,
                descent=descent,
                shrinkage=shrinkage,
                failure=failure,
            )

        self.apply_constraint()
        grad_metrics = self.grad_metrics(
            named_params=self.named_parameters(),
            prev_params=prev_params,
            prev_grads=prev_grads,
            param_filter=param_filter,
            long=False,
        )
        self.post_step()

        return {
            **{k: v.item() for k, v in metrics_accum.items()},
            **{k: v.item() for k, v in grad_metrics.items()},
            **{k: v for k, v in back_tracking_res.items()},
            "type": "training",
            "num_samples": num_samples,
        }
        ###################################

    @torch.no_grad()
    def validate(
        self,
        batch_size,
        num_validation_steps=20,
        # Other details for forward
        **forward_kwargs,
    ):
        results_processor = ResultsProcessor([])

        for i, input_batches in enumerate(
            self.data_gen(
                num_steps=num_validation_steps, batch_size=batch_size, num_batches=1
            )
        ):
            input_batch = list(input_batches)[0]
            print(f"Validation Step {i+1}/{num_validation_steps}")
            start = time.time()
            logits, forward_metrics = self.forward(
                input_batch.float(), validate=True, **forward_kwargs
            )

            _, metrics = self.metrics(input_batch, logits, no_mean=True)
            results_processor.update({**metrics, **forward_metrics})
            print(f"Took {time.time() - start}s")

        return results_processor.results

    @abc.abstractmethod
    def forward(
        self, inputs: torch.Tensor, validate: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def reevaluate(
        self, inputs: torch.Tensor, validate: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Rerun the last call of forward. If forward call
        has stochasticity, this should be overridden.
        """
        return self.forward(inputs, validate=validate)

    @property
    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        pass

    def parameters(self):
        return self.trainable_module.parameters()

    def named_parameters(self):
        return self.trainable_module.named_parameters()

    def post_step(self):
        pass

    def apply_constraint(self):
        pass

    def param_metrics(self):
        return {}

    def grad_metrics(
        self,
        named_params,
        prev_params=None,
        prev_grads=None,
        param_filter=None,
        long=True,
    ):
        if prev_grads is None:
            prev_grads = {}
        if prev_params is None:
            prev_params = {}

        def check_param_filter(k):
            if param_filter is not None:
                return k in param_filter
            else:
                return True

        named_params_dict = {k: v for k, v in named_params if check_param_filter(k)}
        grad_dict = {
            k: param.grad.detach().reshape(-1) for k, param in named_params_dict.items()
        }
        prev_grads = {
            name: prev_grad.reshape(-1) for name, prev_grad in prev_grads.items()
        }
        diffs = {
            name: (named_params_dict[name].detach() - prev_param).reshape(-1)
            for name, prev_param in prev_params.items()
        }

        long_grad_metrics = {}
        if long:
            long_grad_metrics = {
                **{
                    f"{name}_grad_l2": torch.norm(grad, p=2)
                    for name, grad in grad_dict.items()
                },
                **{
                    f"{name}_grad_linf": torch.norm(grad, p=torch.inf)
                    for name, grad in grad_dict.items()
                },
                **{
                    f"{name}_update_l2": torch.norm(diff, p=2)
                    for name, diff in diffs.items()
                },
                **{
                    f"{name}_update_linf": torch.norm(diff, p=torch.inf)
                    for name, diff in diffs.items()
                },
                **{
                    f"{name}_update_grad_ip": F.cosine_similarity(
                        grad_dict[name],
                        diff,
                        dim=0,
                        eps=EPSILON,
                    )
                    for name, diff in diffs.items()
                },
                **{
                    f"{name}_grad_ip": F.cosine_similarity(
                        grad_dict[name],
                        prev_grad,
                        dim=0,
                        eps=EPSILON,
                    )
                    for name, prev_grad in prev_grads.items()
                },
            }

        short_grad_metrics = {
            **(
                {"grad_avg_l2": torch.norm(torch.cat(list(grad_dict.values())), p=2)}
                if len(grad_dict) > 0
                else {}
            ),
            **(
                {
                    "grad_avg_linf": torch.norm(
                        torch.cat(list(grad_dict.values())), p=torch.inf
                    )
                }
                if len(grad_dict) > 0
                else {}
            ),
            **(
                {"update_avg_l2": torch.norm(torch.cat(list(diffs.values())), p=2)}
                if len(diffs) > 0
                else {}
            ),
            **(
                {
                    "update_avg_linf": torch.norm(
                        torch.cat(list(diffs.values())), p=torch.inf
                    )
                }
                if len(diffs) > 0
                else {}
            ),
            **(
                {
                    "update_grad_avg_ip": F.cosine_similarity(
                        torch.cat(list(grad_dict[name] for name in diffs.keys())),
                        torch.cat(list(diffs.values())),
                        dim=0,
                        eps=EPSILON,
                    )
                }
                if len(diffs) > 0
                else {}
            ),
            **(
                {
                    "grad_avg_ip": F.cosine_similarity(
                        torch.cat(list(grad_dict[name] for name in prev_grads.keys())),
                        torch.cat(list(prev_grads.values())),
                        dim=0,
                        eps=EPSILON,
                    )
                }
                if len(prev_grads) > 0
                else {}
            ),
        }

        return {
            "converged": short_grad_metrics["update_avg_l2"] < 1e-7,
            **(long_grad_metrics if long else {}),
            **short_grad_metrics,
        }

    def metrics(
        self, inputs: torch.Tensor, logits: torch.FloatTensor, no_mean=False
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        mean_dim = -1 if no_mean else None
        xe = torch.mean(
            F.binary_cross_entropy_with_logits(
                logits, inputs.float(), reduction="none"
            ),
            dim=mean_dim,
        )
        with torch.no_grad():
            hard_decision = logits > 0
            mismatch = hard_decision != inputs.bool()
        metrics = {
            "xe": xe,
            "ber": torch.mean(mismatch.float(), dim=mean_dim),
            "bler": torch.mean(
                torch.any(mismatch, dim=-1, keepdim=True).float(), dim=mean_dim
            ),
        }
        return metrics["xe"], metrics

    def sync_generator(other_device_manager: DeviceManager):
        raise NotImplementedError


R = TypeVar("R", bound=CodingTrainer)


@dataclass
class Swarm(Generic[R], WithSettings):
    agents: List[R]
    masses: torch.Tensor
    agent_names: List[str] = None
    converged_agents: List[bool] = None
    kill_agents: bool = False
    tolm: float = 1e-4
    merge_agents: bool = True
    tolmerge: float = 1e-3
    communication_adj: int = 2
    step_adj: int = 1
    descent: float = 0.2
    shrinkage: float = 0.9
    failure: int = 10
    device_manager: DeviceManager = DEFAULT_DEVICE_MANAGER

    def __post_init__(self):
        self.coordination_device_manager = DeviceManager(
            self.device_manager.device, seed=self.device_manager.generate_seed()
        )
        if self.agent_names is None:
            self.agent_names = [str(i) for i in range(len(self))]
        if self.converged_agents is None:
            self.converged_agents = [False] * len(self)

        assert self.masses.ndim == 1
        assert (
            len(self.agent_names)
            == len(self.agents)
            == self.masses.shape[0]
            == len(self.converged_agents)
        )

    def __len__(self) -> int:
        return len(self.agents)

    def heaviest_agent_i(self) -> int:
        return torch.argmax(self.masses).item()

    # Using NotImplementedError instead of ABC for dataclasses
    @torch.no_grad()
    def agent_distance(self, agent, other_agent) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def merge(self, clone=True) -> "Swarm[R]":
        if not self.merge_agents:
            print("self.merge_agents=False, but still merging.")

        new_masses = self.masses.clone() if clone else self.masses
        new_agent_names = list(self.agent_names)
        merged = set()
        for i, j in it.combinations(range(len(self)), r=2):
            if (i in merged) or (j in merged):
                continue
            distance = self.agent_distance(self.agents[i], self.agents[j])
            # Merge into the smaller index so we don't need to double merge.
            if distance < self.tolmerge:
                new_masses[i] += new_masses[j]
                new_agent_names[i] = "_".join(new_agent_names[i], new_agent_names[j])
                merged.add(j)
        new_agent_mask = [i not in merged for i in range(len(self))]
        new_agents = [self.agents[i] for i in range(len(self)) if new_agent_mask[i]]
        new_agent_names = [
            self.agent_names[i] for i in range(len(self)) if new_agent_mask[i]
        ]
        new_converged_agents = [
            self.converged_agents[i] for i in range(len(self)) if new_agent_mask[i]
        ]
        new_masses = new_masses[new_agent_mask]
        return replace(
            self,
            agents=new_agents,
            agent_names=new_agent_names,
            converged_agents=new_converged_agents,
            masses=new_masses,
        )

    @torch.no_grad()
    def kill(self, clone=True) -> "Swarm[R]":
        if not self.kill_agents:
            print("self.kill_agents=False, but still merging.")

        survival_inds = self.masses > (self.tolm * (1 / len(self)))
        new_masses = (
            self.masses[survival_inds].clone() if clone else self.masses[survival_inds]
        )
        new_agents = [self.agents[i] for i in range(len(self)) if survival_inds[i]]
        new_agent_names = [
            self.agent_names[i] for i in range(len(self)) if survival_inds[i]
        ]
        new_converged_agents = [
            self.converged_agents[i] for i in range(len(self)) if survival_inds[i]
        ]
        new_swarm = replace(
            self,
            agents=new_agents,
            agent_names=new_agent_names,
            converged_agents=new_converged_agents,
            masses=new_masses,
        )
        assert len(new_swarm) != 0
        return new_swarm

    def clone(self) -> "Swarm":
        return replace(
            self,
            masses=self.masses.clone(),
            agent_names=list(self.agent_names),
            converged_agents=list(self.converged_agents),
        )

    @torch.no_grad()
    def mass_transition(self, losses: torch.Tensor, clone=True) -> "Swarm[R]":
        best_agent_i = torch.argmin(losses)
        best_loss = losses[best_agent_i]
        worst_loss = torch.max(losses)
        communication_weights = (losses - best_loss) / (worst_loss - best_loss)
        total_mass = torch.sum(self.masses)
        new_masses = self.masses.clone() if clone else self.masses
        mass_updates = -(communication_weights**self.communication_adj) * new_masses
        mass_updates[best_agent_i] = 0.0
        new_masses += mass_updates
        # Cutoff
        new_masses = new_masses.clamp_(min=(self.tolm * (1 / len(self))))

        new_masses[best_agent_i] += total_mass - torch.sum(
            new_masses
        )  # Mass is conserved

        return replace(
            self,
            masses=new_masses,
        )

    def update(
        self,
        input_batches: GeneratorWithLength[torch.Tensor],
        optimizer_swarm: List[Optimizer],
    ) -> Tuple["Swarm[R]", List[Dict[str, Any]]]:
        new_swarm = self.clone()
        if self.merge_agents:
            new_swarm = new_swarm.merge(clone=False)
        if self.kill_agents:
            new_swarm = new_swarm.kill(clone=False)
        # Reset the best agent to be based on the provided input_batches
        print("Evaluating Swarm:")
        losses = new_swarm.evaluate_swarm(input_batches=input_batches)
        num_agent_preview = min(len(new_swarm), 3)
        print(f"Top {num_agent_preview} agents")
        for i in torch.topk(losses, k=num_agent_preview, largest=False).indices:
            print(
                f"Agent {self.agent_names[i]} : Loss = {losses[i].item()} : Mass = {self.masses[i].item()} : Converged = {self.converged_agents[i]}"
            )
        new_swarm = new_swarm.mass_transition(losses=losses, clone=False)

        # Run the gradient step
        print("Train Stepping Swarm:")
        training_result = new_swarm.train_step(
            input_batches=input_batches, optimizer_swarm=optimizer_swarm
        )
        new_swarm.converged_agents = [r["converged"] for r in training_result]

        return new_swarm, training_result

    def sync_agent_generators(self):
        assert len(self) > 0
        for agent in self.agents:
            agent.sync_generator(self.coordination_device_manager)

    def train_step(
        self,
        input_batches: GeneratorWithLength[torch.Tensor],
        optimizer_swarm: List[Optimizer],
    ) -> List[Dict[str, Any]]:
        heaviest_mass = torch.max(self.masses)
        relative_masses = self.masses / heaviest_mass
        # Run the update for each agent
        agent_descent = self.descent * relative_masses**self.step_adj
        swarm_train_results = []
        # Sync the random number generators
        self.sync_agent_generators()
        for i, agent in enumerate(tqdm(self.agents)):
            # Skip all converged agents to save time.
            if self.converged_agents[i]:
                swarm_train_results.append({"converged": True})
                continue
            prev_params = {k: v.detach().clone() for k, v in agent.named_parameters()}
            prev_grads = {
                k: v.grad.detach().clone()
                for k, v in agent.named_parameters()
                if v.grad is not None
            }
            agent_results = agent.train_step(
                input_batches=input_batches,
                optimizer=optimizer_swarm[i],
                prev_params=prev_params,
                prev_grads=prev_grads,
                back_tracking=True,
                descent=agent_descent[i],
                shrinkage=self.shrinkage,
                failure=self.failure,
            )
            swarm_train_results.append(agent_results)
        assert len(swarm_train_results) == len(self)
        return swarm_train_results

    @torch.no_grad()
    def evaluate_swarm(
        self, input_batches: GeneratorWithLength[torch.Tensor]
    ) -> torch.Tensor:
        losses = [
            torch.tensor(0.0, device=self.device_manager.device)
            for _ in range(len(self))
        ]
        for sample in input_batches:
            self.sync_agent_generators()
            for i, agent in enumerate(tqdm(self.agents)):
                logits, _ = agent.forward(input_batch=sample, validate=False)
                loss, _ = agent.metrics(inputs=sample, logits=logits, no_mean=False)
                losses[i] += loss / len(input_batches)
        losses = torch.stack(losses)

        return losses

    def apply_constraint(self):
        for agent in self.agents:
            agent.apply_constraint()

    @property
    def trainable_modules(self) -> List[nn.Module]:
        return [agent.trainable_module for agent in self.agents]

    def settings(self) -> Dict[str, Any]:
        return {
            "masses": self.masses.cpu().tolist() if len(self) < 100 else "...",
            "kill_agents": self.kill_agents,
            "tolm": self.tolm,
            "merge_agents": self.merge_agents,
            "tolmerge": self.tolmerge,
            "communication_adj": self.communication_adj,
            "step_adj": self.step_adj,
            "descent": self.descent,
            "shrinkage": self.shrinkage,
            "failure": self.failure,
        }


@dataclass
class L2DistanceSwarm(Swarm[R]):
    @torch.no_grad()
    def agent_distance(self, agent: R, other_agent: R) -> torch.Tensor:
        agent_params = torch.cat([p.reshape(-1) for p in agent.parameters()])
        other_agent_params = torch.cat(
            [p.reshape(-1) for p in other_agent.parameters()]
        )
        return torch.norm(agent_params - other_agent_params, p=2)


class SwarmTrainer(Generic[R], metaclass=abc.ABCMeta):
    def __init__(
        self,
        swarm_size: int,
        kill_agents=False,
        tolm: float = 1e-4,
        merge_agents=True,
        tolmerge: float = 1e-3,
        communication_adj=2,
        step_adj=2,
        descent=0.2,
        shrinkage=0.9,
        failure=10,
        output_path=None,
        device_manager=DEFAULT_DEVICE_MANAGER,
    ):
        self.device_manager = device_manager
        self.swarm = L2DistanceSwarm(
            [self.initialize_agent() for _ in range(swarm_size)],
            masses=torch.full(
                (swarm_size,),
                fill_value=1.0 / swarm_size,
                device=self.device_manager.device,
            ),
            kill_agents=kill_agents,
            tolm=tolm,
            merge_agents=merge_agents,
            tolmerge=tolmerge,
            communication_adj=communication_adj,
            step_adj=step_adj,
            descent=descent,
            shrinkage=shrinkage,
            failure=failure,
            device_manager=self.device_manager,
        )

        self.output_path = output_path

    @abc.abstractmethod
    def initialize_agent(self) -> R:
        pass

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        pass

    def summary(self) -> None:
        pass

    def data_gen(
        self,
        num_steps: int,
        batch_size: int,
        num_batches: int = 1,
    ):
        return data_gen(
            input_size=self.input_size,
            num_steps=num_steps,
            batch_size=batch_size,
            num_batches=num_batches,
            device_manager=self.device_manager,
        )
