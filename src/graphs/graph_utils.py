from typing import List, Callable, Tuple, Set, Dict
from numbers import Number
import itertools as it

import numpy as np
import networkx as nx


def check_node_attribute(G: nx.Graph, key: str, attr_type: type=None):
    for n in G.nodes:
        assert key in G.nodes[n], f"Failed to find attr {key} on node {n}"
        if attr_type is not None:
            assert isinstance(G.nodes[n][key], attr_type)


def check_edge_attribute(G: nx.Graph, key: str, attr_type: type=None):
    for e in G.edges:
        assert key in G.edges[e], f"Failed to find attr {key} on edge {e}"
        if attr_type is not None:
            assert isinstance(G.edges[e][key], attr_type)


def connect_nodes(G: nx.Graph, nodes: List[str], inplace=False):
    if not inplace:
        G = G.copy()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                # print(f"Adding edge from {nodes[i]} to {nodes[j]}")
                G.add_edge(nodes[i], nodes[j])
    return G

def factor_groups_to_graph(factor_groups: Dict[str, Set[str]], state_sizes: Dict[str, int]) -> nx.Graph:
    G = nx.Graph()
    for name, factor in factor_groups.items():
        for node in factor:
            G.add_node(node, weight=2 ** state_sizes[node], from_factor=name)
        G = connect_nodes(G, list(factor), inplace=True)
    return G


def greedy_ordering(
    G: nx.Graph, metric: Callable[[nx.Graph, str], Number]
) -> List[str]:
    ordering = []
    G = G.copy()
    num_nodes = len(G.nodes)
    for _ in range(num_nodes):
        minimizer = min([node for node in G.nodes()], key=lambda node: metric(G, node))
        ordering.append(minimizer)
        neighbors = list(nx.neighbors(G, minimizer))
        G = connect_nodes(G, nodes=neighbors, inplace=True)
        G.remove_node(minimizer)

    return ordering


def stochastic_greedy_ordering(
    G: nx.Graph,
    metric: Callable[[nx.Graph, str], Number],
    sample_thresh=5,
    rng= None,
) -> List[str]:
    if rng is None:
        rng = np.random.default_rng()
    ordering = []
    G = G.copy()
    num_nodes = len(G.nodes)
    for _ in range(num_nodes):
        sorted_nodes = sorted(
            [node for node in G.nodes()], key=lambda node: metric(G, node)
        )
        minimizer = rng.choice(sorted_nodes[:sample_thresh])
        ordering.append(minimizer)
        neighbors = list(nx.neighbors(G, minimizer))
        G = connect_nodes(G, nodes=neighbors, inplace=True)
        G.remove_node(minimizer)

    return ordering


def min_neighbors(G: nx.Graph, node: str) -> Number:
    return len(list(nx.neighbors(G, node)))


def min_weight(G: nx.Graph, node: str) -> Number:
    return int(np.prod([G.nodes[neighb]["weight"] for neighb in nx.neighbors(G, node)]))


def min_fill(G: nx.Graph, node: str) -> Number:
    neighbors = list(nx.neighbors(G, node))
    new_edges = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if not G.has_edge(neighbors[i], neighbors[j]):
                new_edges += 1
    return new_edges


def weighted_min_fill(G: nx.Graph, node: str) -> Number:
    neighbors = list(nx.neighbors(G, node))
    edge_cost = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            if not G.has_edge(neighbors[i], neighbors[j]):
                edge_cost += (
                    G.nodes[neighbors[i]]["weight"] * G.nodes[neighbors[j]]["weight"]
                )
    return edge_cost


def compute_elimination_factors(G: nx.Graph, ordering: List[str]) -> Tuple[List[Set[str]], List[int]]:
    G = G.copy()
    factors: List[Set[str]] = []
    factor_log_weights = []
    for i, v in enumerate(ordering):
        neighbors = list(nx.neighbors(G, v))        
        variables = {v, *neighbors}
        
        factors.append(variables)
        factor_log_weights.append(
            sum([int(np.log2(G.nodes[n]["weight"])) for n in variables])
        )
        G = connect_nodes(G, nodes=neighbors, inplace=True)
        G.remove_node(v)

    return factors, factor_log_weights


def build_cluster_tree(factors: List[Set[str]]):
    cluster_tree = nx.Graph()
    # Build the clique network
    for i, factor in enumerate(factors):
       cluster_tree.add_node(i, variables=factor)
    for n1, n2 in it.combinations(cluster_tree.nodes, 2):
        n1_vars: Set[str] = cluster_tree.nodes[n1]["variables"]
        n2_vars: Set[str] = cluster_tree.nodes[n2]["variables"]
        if not n1_vars.isdisjoint(n2_vars):
            intersection = n1_vars & n2_vars
            # print(intersection)
            cluster_tree.add_edge(n1, n2, intersection=intersection, weight=len(intersection))
    
    # Reduce to a max-spanning-tree
    cluster_tree = nx.maximum_spanning_tree(cluster_tree)
    return cluster_tree

def check_cluster_tree(G: nx.Graph):
    assert nx.is_tree(G), "G is not a tree."
    check_edge_attribute(G, "intersection", attr_type=set)
    check_node_attribute(G, "variables", attr_type=set)
    for n1, n2 in it.combinations(G.nodes, 2):
        # Only one simple path since this is a tree
        path = next(nx.all_simple_edge_paths(G, n1, n2))
        n1_vars: set = G.nodes[n1]["variables"]
        n2_vars: set = G.nodes[n2]["variables"]
        shared_vars = n1_vars & n2_vars
        for e in path:
            intersection: set = G.edges[e]["intersection"]
            assert shared_vars.issubset(
                intersection
            ), f"Failed to validate RIP of shared vars {shared_vars} : path - {n1_vars} -> {n2_vars} : failed edge - {G.nodes[e[0]]['variables']}-{G.nodes[e[1]]['variables']} w/ intersection {intersection}."


def is_pruned(G: nx.Graph):
    for n1, n2 in G.edges:
        n1_vars: set = G.nodes[n1]["variables"]
        n2_vars: set = G.nodes[n2]["variables"]
        if n2_vars.issubset(n1_vars) or n1_vars.issubset(n2_vars):
            return False
    return True


def prune_cluster_tree(G: nx.Graph):
    check_cluster_tree(G)
    G = G.copy()

    found_edge = True
    while found_edge:
        found_edge = False
        for n1, n2 in list(G.edges):
            n1_vars: set = G.nodes[n1]["variables"]
            n2_vars: set = G.nodes[n2]["variables"]
            if n2_vars.issubset(n1_vars):
                # Swap n1 and n2 so we always have n2_vars subset of n1_vars
                n1, n2 = n2, n1
                n1_vars: set = G.nodes[n1]["variables"]
                n2_vars: set = G.nodes[n2]["variables"]
                assert n1_vars.issubset(n2_vars)
            if n1_vars.issubset(n2_vars):
                n1_neighbors = list(G.neighbors(n1))
                for neighb in n1_neighbors:
                    if neighb != n2:
                        G.add_edge(
                            neighb, n2, intersection=G.nodes[neighb]["variables"] & n2_vars
                        )
                G.remove_node(n1)
                found_edge = True
                break
    
    assert is_pruned(G)
    return G
    
    