from typing import Dict, Sequence, Set, List
import abc
from collections import defaultdict, deque
import gc

import numpy as np
import torch
import networkx as nx
from tqdm import trange

from ..utils import NamedTensor, peek

from . import graph_utils as gu


class InferenceGraph(metaclass=abc.ABCMeta):
    def __init__(
        self,
        graph: nx.Graph,
        factor_groups: Dict[str, Set[str]],
        elimination_ordering=None,
        factor_width=None,
        do_copy=True,
    ) -> None:
        self.graph = graph.copy() if do_copy else graph
        self.factor_groups = factor_groups
        self.elimination_ordering = elimination_ordering
        self.factor_width = factor_width

        self.validate()

    @staticmethod
    def from_factor_groups(
        factor_groups: Dict[str, Set[str]], state_sizes: Dict[str, int] = None
    ):
        if state_sizes is None:
            print("No state sizes provided, using 1 by default.")
            state_sizes = defaultdict(lambda: 1)

        graph = gu.factor_groups_to_graph(factor_groups, state_sizes)
        return InferenceGraph(graph, factor_groups, do_copy=False)

    def validate(self):
        gu.check_node_attribute(self.graph, "weight")

    def is_chordal(self) -> bool:
        return nx.is_chordal(self.graph)

    def with_elimination_ordering(
        self, sample_thresh=3, tries=50, seed=None
    ) -> "InferenceGraph":
        if self.is_chordal():
            # Doesn't really matter which  metric we use
            elimination_ordering = gu.greedy_ordering(self.graph, metric=gu.min_weight)
            _, fw = gu.compute_elimination_factors(self.graph, elimination_ordering)
            print(
                f"Graph was already chordally complete, max factor weight is {max(fw)}."
            )
            return InferenceGraph(
                self.graph,
                self.factor_groups,
                elimination_ordering=elimination_ordering,
                factor_width=max(fw),
            )

        rng = np.random.default_rng(seed=seed)
        print(seed)
        metrics = [gu.min_neighbors, gu.min_weight, gu.min_fill, gu.weighted_min_fill]
        best = float("inf")
        best_elimination_ordering = None
        pbar = trange(tries)
        for _ in pbar:
            for m in metrics:
                # print(f"====== {m.__name__} ========")
                elimination_ordering = gu.stochastic_greedy_ordering(
                    self.graph, m, sample_thresh=sample_thresh, rng=rng
                )
                _, fw = gu.compute_elimination_factors(self.graph, elimination_ordering)
                # print(f)
                # print(max(fw))
                res = max(fw)
                if res < best:
                    best = res
                    best_elimination_ordering = elimination_ordering
                    pbar.set_postfix({"best": best})

        return InferenceGraph(
            self.graph,
            self.factor_groups,
            elimination_ordering=best_elimination_ordering,
            factor_width=best,
        )

    def as_cluster_tree(self):
        assert self.elimination_ordering is not None, "Need an elimination ordering."
        elimination_factors, _ = gu.compute_elimination_factors(
            self.graph, self.elimination_ordering
        )
        cluster_tree = gu.build_cluster_tree(elimination_factors)
        cluster_tree = gu.prune_cluster_tree(cluster_tree)

        return ClusterTree(cluster_tree, self.factor_groups, do_copy=False)


# def check_ready(dir_tree: nx.DiGraph):
#     ready = []
#     for node in dir_tree.nodes():
#         out_edge_result = {
#             succ: (dir_tree.edges[(node, succ)]["msg"] is not None)
#             for succ in dir_tree.neighbors(node)
#         }
#         in_edge_result = {
#             pred: (dir_tree.edges[(pred, node)]["msg"] is not None)
#             for pred in dir_tree.predecessors(node)
#         }
#         for


class ClusterTree:
    def __init__(
        self,
        graph: nx.Graph,
        factor_groups: Dict[str, Set[str]],
        factor_group_assigments=None,
        variable_assignments=None,
        do_copy=True,
    ) -> None:
        self.graph = graph.copy() if do_copy else graph
        self.factor_groups = factor_groups
        if factor_group_assigments is None:
            self.factor_group_assignments = self.assign_factor_groups(
                graph, factor_groups
            )
        if variable_assignments is None:
            self.variable_assignments = self.assign_variables(graph)
        self.variable_loads = {}
        for variable, assignment in self.variable_assignments.items():
            self.variable_loads[variable] = len(graph.nodes[assignment]["variables"])

        self.annotate_graph()
        self.validate()

    def annotate_graph(self):
        for node in self.graph.nodes():
            self.graph.nodes[node]["factors"] = []
            self.graph.nodes[node]["variable_assignments"] = []
        for factor_name, node in self.factor_group_assignments.items():
            self.graph.nodes[node]["factors"].append(factor_name)
        for variable, node in self.variable_assignments.items():
            self.graph.nodes[node]["variable_assignments"].append(variable)

    @staticmethod
    def assign_variables(graph: nx.Graph):
        # TODO Like below but for individual variables. This will be used to figure out which
        # node in the tree to marginalize over to get the appropriate posterior.
        variable_assignments = {}
        all_variables = set(
            var for node in graph.nodes for var in graph.nodes[node]["variables"]
        )
        for variable in all_variables:
            valid_nodes = [
                node
                for node in graph.nodes
                if variable in graph.nodes[node]["variables"]
            ]
            assert len(valid_nodes) > 0
            variable_assignments[variable] = min(
                valid_nodes, key=lambda node: len(graph.nodes[node]["variables"])
            )

        return variable_assignments

    @staticmethod
    def assign_factor_groups(
        graph: nx.Graph, factor_groups: Dict[str, Set[str]]
    ) -> Dict[str, str]:
        """Map factor groups to the cluster tree node containing all variables for that factor

        Parameters
        ----------
        graph : nx.Graph
            The underlying graph object of the cluster tree.
        factor_groups : Dict[str, Set[str]]
            A mapping of factor group names to a set of names of variables in the factor group.
        [name] ([shape]) : [type]
            [desc]

        Returns
        -------
        Dict[str, str]
            A mapping of factor group names to node names in the underlying graph of the
            cluster tree. The factor group is mapped to the node containing the factor
            with the least number of variables.
        [type]
            [desc]

        """
        factor_group_assignments = {}
        node_variables = {node: graph.nodes[node]["variables"] for node in graph.nodes}
        for name, factor in factor_groups.items():
            valid_nodes = [
                node
                for node, variables in node_variables.items()
                if factor.issubset(variables)
            ]
            assert len(valid_nodes) > 0, f"Factor {factor} failed to have valid node."
            factor_group_assignments[name] = min(
                valid_nodes, key=lambda node: len(node_variables[node])
            )
        return factor_group_assignments

    def validate(self):
        gu.check_cluster_tree(self.graph)
        assert gu.is_pruned(self.graph)
        gu.check_node_attribute(self.graph, "factors")

    def propogate_evidence(self, evidence: Dict[str, NamedTensor]) -> nx.DiGraph:
        """Complete a round of message passing for belief propogation

        Parameters
        ----------
        evidence (Batch x ...(named for relevant variable)) : Dict[str, NamedTensor]
            A mapping that maps each factor name to a tensor
            of evidence values. The dimensions after the batch dim are aranged
            in an undefined order with each dimension named after a variable
            in the factor.
        [name] ([shape]) : [type]
            [desc]

        Returns
        -------
        [type]
            [desc]

        """
        dir_graph = self.graph.to_directed()
        for edge in dir_graph.edges():
            dir_graph.edges[edge]["msg_data"] = None  # Actual NamedTensor

        init_nodes = [
            node for node in dir_graph.nodes if dir_graph.out_degree(node) == 1
        ]
        init_msg_edges = [
            (node, next(dir_graph.neighbors(node))) for node in init_nodes
        ]
        # appendleft and pop
        ready_q = deque(init_msg_edges)

        while len(ready_q) != 0:
            source, target = ready_q.pop()
            # print(
            #     f"Popped edge {(source, target)} with intersection set {dir_graph.edges[(source, target)]['intersection']}"
            # )
            if dir_graph.edges[(source, target)]["msg_data"] is not None:
                # print(f"Already sent data along {(source, target)}, skipping.")
                continue
            self._sp_msg(dir_graph, source, target, evidence)
            target_received = self.received(dir_graph, target)
            target_to_send = self.to_send(dir_graph, target)
            discrepancy = target_to_send - target_received
            if len(discrepancy) == 1:
                ready_q.appendleft((target, peek(discrepancy)))
            elif len(discrepancy) == 0:
                for next_target in target_to_send:
                    ready_q.appendleft((target, next_target))

        for edge in dir_graph.edges():
            # At end we should have sent all messages.
            assert dir_graph.edges[edge]["msg_data"] is not None

        return dir_graph

    def compute_posteriors(
        self,
        msg_digraph: nx.DiGraph,
        variables: Sequence[str],
        evidence: Dict[str, NamedTensor],
        delete_data: bool = False,
    ) -> Dict[str, NamedTensor]:
        log_clique_potential = None
        variable_set = set(variables)
        variable_logits: Dict[str, NamedTensor] = {}
        for variable in sorted(
            variable_set, key=lambda v: self.variable_loads[v], reverse=False
        ):
            if variable in variable_logits:
                # We've already computed the posterior
                continue
            node = self.variable_assignments[variable]
            node_factors = msg_digraph.nodes[node]["factors"]
            node_factor_data = [evidence[factor] for factor in node_factors]
            msg_data: List[NamedTensor] = [
                msg_digraph.edges[(pred, node)]["msg_data"]
                for pred in msg_digraph.predecessors(node)
            ]
            # In log space, so sum is prod
            log_clique_potential: NamedTensor = NamedTensor.sum(
                msg_data + node_factor_data
            )
            if delete_data:
                for factor in node_factors:
                    del evidence[factor]
                for pred in msg_digraph.predecessors(node):
                    del msg_digraph.edges[(pred, node)]["msg_data"]
                del msg_data
                del node_factor_data
                gc.collect()

            for keep_variable in variable_set & msg_digraph.nodes[node]["variables"]:
                marginalize_variables = log_clique_potential.dim_names[
                    log_clique_potential.dim_names != keep_variable
                ]
                variable_logits[keep_variable] = log_clique_potential.logsumexp(
                    dim=marginalize_variables
                )
            del log_clique_potential
            gc.collect()
        return variable_logits

    @staticmethod
    def add_dummy_dims(
        tensor: torch.Tensor, dim_labels: List[str], all_labels: List[str], batch_dims=1
    ):
        assert set(dim_labels).issubset(set(all_labels))
        assert tensor.ndim - batch_dims == len(dim_labels)
        new_shape = np.ones(len(all_labels), dtype=int)
        new_shape[np.in1d(all_labels, dim_labels)] = np.array(tensor.shape[batch_dims:])
        new_shape = np.concatenate([np.array(tensor.shape[:batch_dims]), new_shape])
        return tensor.reshape(new_shape)

    @staticmethod
    def received(dir_graph: nx.DiGraph, node: str):
        return {
            pred
            for pred in dir_graph.predecessors(node)
            if dir_graph.edges[(pred, node)]["msg_data"] is not None
        }

    @staticmethod
    def to_send(dir_graph: nx.DiGraph, node: str):
        return {
            succ
            for succ in dir_graph.successors(node)
            if dir_graph.edges[(node, succ)]["msg_data"] is None
        }

    @staticmethod
    def _sp_msg(
        dir_graph: nx.DiGraph,
        source,
        target,
        evidence: Dict[str, NamedTensor],
    ):
        assert dir_graph.edges[(source, target)]["msg_data"] is None

        # Collect all evidence needed for sum-product
        node_factor_names = dir_graph.nodes[source]["factors"]
        node_factor_data = [evidence[name] for name in node_factor_names]
        incoming_edges = [
            (other, source)
            for other in dir_graph.predecessors(source)
            if other != target
        ]
        incoming_msg_data = [dir_graph.edges[e]["msg_data"] for e in incoming_edges]
        all_data = node_factor_data + incoming_msg_data

        # In log space we take sums instead of products
        res: NamedTensor = NamedTensor.sum(
            all_data
        )  # Should broadcast because axes names will be the same
        transmit_variable_set: set = dir_graph.edges[(source, target)]["intersection"]
        res = res.logsumexp(dim=list(res._dim_name_set - transmit_variable_set))
        # assert set(res.active_dims()) == transmit_variable_set  I think we only need the weaker condition below
        # TODO: Prove this
        assert transmit_variable_set.issuperset(res._dim_name_set)

        dir_graph.edges[(source, target)]["msg_data"] = res
