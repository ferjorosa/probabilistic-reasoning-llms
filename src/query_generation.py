from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

@dataclass
class QuerySpec:
    # One or two query nodes with chosen states (state labels)
    targets: List[Tuple[str, str]]
    # Evidence assignments as mapping node -> state label
    evidence: Dict[str, str]
    # Metadata about difficulty dimensions
    meta: Dict[str, Any]

def _to_nx_dag(model: DiscreteBayesianNetwork) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    return G


def _all_state_labels(model: DiscreteBayesianNetwork) -> Dict[str, List[str]]:
    labels: Dict[str, List[str]] = {}
    for cpd in model.get_cpds():
        var = cpd.variable
        labels[var] = list(cpd.state_names[var])
    return labels


def _shortest_undirected_distance(G: nx.DiGraph, a: str, b: str) -> int:
    UG = G.to_undirected()
    try:
        return nx.shortest_path_length(UG, a, b)
    except nx.NetworkXNoPath:
        return 10**9


def _pick_nodes_by_distance(
    G: nx.DiGraph,
    rng: np.random.Generator,
    desired_min: int,
    desired_max: int,
) -> Tuple[str, str, int]:
    nodes = list(G.nodes())
    for _ in range(500):
        a, b = rng.choice(nodes, 2, replace=False)
        d = _shortest_undirected_distance(G, a, b)
        if desired_min <= d <= desired_max:
            return a, b, d
    # Fallback: just return a closest pair found
    a, b = rng.choice(nodes, 2, replace=False)
    d = _shortest_undirected_distance(G, a, b)
    return a, b, d


def _choose_states(rng: np.random.Generator, state_labels: Dict[str, List[str]], nodes: Sequence[str]) -> Dict[str, str]:
    return {n: rng.choice(state_labels[n]).item() for n in nodes}


def generate_queries(
    model: DiscreteBayesianNetwork,
    *,
    num_queries: int = 20,
    query_node_counts: Sequence[int] = (1, 2),
    evidence_counts: Sequence[int] = (0, 1, 2, 3),
    # Distance buckets in undirected graph between target and evidence nodes
    distance_buckets: Sequence[Tuple[int, int]] = ((0, 1), (2, 3), (4, 99)),
    prefer_zero_determinism: bool = True,
    seed: Optional[int] = None,
) -> List[QuerySpec]:
    """Generate query/evidence sets spanning difficulty dimensions from ideas.md.

    Each QuerySpec defines:
      - targets: list of (node, state)
      - evidence: mapping node->state
    Metadata includes counts and approximate min distance between targets and evidence.
    """
    rng = np.random.default_rng(seed)
    G = _to_nx_dag(model)
    state_labels = _all_state_labels(model)
    nodes = list(G.nodes())

    results: List[QuerySpec] = []
    bucket_cycle = list(distance_buckets)
    if not bucket_cycle:
        bucket_cycle = [(0, 99)]

    for i in range(num_queries):
        qk = int(rng.choice(query_node_counts))
        ek = int(rng.choice(evidence_counts))
        dmin, dmax = bucket_cycle[i % len(bucket_cycle)]

        # Choose target nodes (distinct)
        q_nodes = list(rng.choice(nodes, size=min(qk, len(nodes)), replace=False))

        # Evidence selection: choose nodes at a desired distance from at least one target
        e_nodes: List[str] = []
        if ek > 0:
            pool = set(nodes) - set(q_nodes)
            for _ in range(5 * ek):  # limited attempts
                if not pool:
                    break
                t = rng.choice(q_nodes)
                e = rng.choice(list(pool))
                d = _shortest_undirected_distance(G, t, e)
                if dmin <= d <= dmax:
                    e_nodes.append(e)
                    pool.remove(e)
                    if len(e_nodes) >= ek:
                        break
            # Fallback if we couldn't satisfy distance: pick random remaining
            while len(e_nodes) < ek and pool:
                e = rng.choice(list(pool))
                e_nodes.append(e)
                pool.remove(e)

        # Assign random states
        q_states = _choose_states(rng, state_labels, q_nodes)
        e_states = _choose_states(rng, state_labels, e_nodes)

        # Compute an aggregate distance metric: min distance from any target to any evidence
        if e_nodes:
            min_dist = min(_shortest_undirected_distance(G, t, e) for t in q_nodes for e in e_nodes)
        else:
            min_dist = 0

        targets = [(n, q_states[n]) for n in q_nodes]
        evidence = {n: e_states[n] for n in e_nodes}

        meta = {
            "num_query_nodes": len(q_nodes),
            "num_evidence_nodes": len(e_nodes),
            "distance_bucket": (dmin, dmax),
            "min_target_evidence_distance": int(min_dist),
        }
        results.append(QuerySpec(targets=targets, evidence=evidence, meta=meta))

    return results


__all__ = ["QuerySpec", "generate_queries"]



