"""
Discrete BN generation from DAGs with controllable CPT properties.

This module builds pgmpy DiscreteBayesianNetwork instances from NetworkX DAGs,
sampling Conditional Probability Tables (CPTs) according to configurable
parameters:

- Variable arity strategy (fixed or ranged)
- CPT skewness via Dirichlet(alpha)
- Determinism fraction: proportion of CPT columns set to 0/1

Example (API):
    >>> import networkx as nx
    >>> from graph_generation import generate_dag_with_treewidth
    >>> dag, _, _ = generate_dag_with_treewidth(5, 2, seed=42)
    >>> bn, meta = generate_discrete_bn_from_dag(
    ...     dag,
    ...     arity_strategy={"type": "range", "min": 2, "max": 3},
    ...     dirichlet_alpha=0.5,
    ...     determinism_fraction=0.0,
    ...     seed=123,
    ... )

CLI:
    uv run python -m bn_generation --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Local import: support both `python src/bn_generation.py` and `python -m src.bn_generation`
try:  # direct import when src/ is on sys.path
    from graph_generation import generate_dag_with_treewidth  # type: ignore
    from llm_calling import create_probability_prompt, run_llm_call, extract_numeric_answer  # type: ignore
except ModuleNotFoundError:  # relative import when running as a module
    from .graph_generation import generate_dag_with_treewidth  # type: ignore
    from .llm_calling import create_probability_prompt, run_llm_call, extract_numeric_answer  # type: ignore


# ------------------------------
# Types and configuration models
# ------------------------------

@dataclass
class ArityStrategy:
    type: str  # "fixed" | "range"
    fixed: Optional[int] = None
    min: Optional[int] = None
    max: Optional[int] = None

    def draw_cardinalities(self, nodes: Sequence[Any], rng: np.random.Generator) -> Dict[Any, int]:
        if self.type == "fixed":
            if not self.fixed or self.fixed < 2:
                raise ValueError("fixed arity must be >= 2")
            return {n: int(self.fixed) for n in nodes}
        elif self.type == "range":
            if not self.min or not self.max or self.min < 2 or self.max < self.min:
                raise ValueError("range arity requires 2 <= min <= max")
            return {n: int(rng.integers(self.min, self.max + 1)) for n in nodes}
        else:
            raise ValueError("Unsupported arity strategy; use 'fixed' or 'range'")


# ------------------------------
# Core generation functions
# ------------------------------

def _enumerate_parent_assignments(parents: Sequence[Any], parent_cards: Dict[Any, int]) -> List[Tuple[int, ...]]:
    """Enumerate parent assignments as tuples of state indices in cartesian order.

    For parents [P1, P2] with cards [c1, c2], produces:
        (0,0), (0,1), ..., (0,c2-1), (1,0), ..., (c1-1, c2-1)
    """
    if not parents:
        return []
    ranges = [list(range(parent_cards[p])) for p in parents]
    # Product in natural order: parents[0] is slowest changing
    # We achieve that by nested loops or using numpy indexing order.
    # Simple recursive style using python product (fast enough for our scope).
    from itertools import product

    return list(product(*ranges))


def _sample_cpt_for_node(
    node: Any,
    parents: Sequence[Any],
    var_card: int,
    parent_cards: Dict[Any, int],
    dirichlet_alpha: float,
    determinism_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a CPT matrix for 'node' given parents.

    Returns an array of shape (var_card, product(parent_cards)) for conditional nodes,
    or shape (var_card,) for root nodes.
    """
    if not parents:
        # Root prior
        if determinism_fraction > 0.0 and rng.random() < determinism_fraction:
            one_hot = np.zeros(var_card, dtype=float)
            one_hot[int(rng.integers(0, var_card))] = 1.0
            return one_hot
        probs = rng.dirichlet([dirichlet_alpha] * var_card)
        return probs

    parent_assignments = _enumerate_parent_assignments(parents, parent_cards)
    num_cols = len(parent_assignments)
    values = np.zeros((var_card, num_cols), dtype=float)

    deterministic_cols = set()
    if determinism_fraction > 0.0:
        num_deterministic = int(round(determinism_fraction * num_cols))
        if num_deterministic > 0:
            deterministic_cols = set(rng.choice(num_cols, size=num_deterministic, replace=False))

    for col in range(num_cols):
        if col in deterministic_cols:
            idx = int(rng.integers(0, var_card))
            values[idx, col] = 1.0
        else:
            values[:, col] = rng.dirichlet([dirichlet_alpha] * var_card)

    return values


def generate_discrete_bn_from_dag(
    dag: nx.DiGraph,
    arity_strategy: Dict[str, Any] | ArityStrategy,
    dirichlet_alpha: float = 1.0,
    determinism_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[DiscreteBayesianNetwork, Dict[str, Any]]:
    """Generate a DiscreteBayesianNetwork with sampled CPTs for the given DAG.

    Args:
        dag: A NetworkX DiGraph (assumed acyclic)
        arity_strategy: dict or ArityStrategy specifying node cardinalities
        dirichlet_alpha: Dirichlet concentration for CPT columns (<=1 skewed, 1 uniform, >1 flat)
        determinism_fraction: fraction of CPT columns set to deterministic 0/1 (0.0 recommended by default)
        seed: RNG seed

    Returns:
        (model, metadata) where model is a pgmpy DiscreteBayesianNetwork and metadata includes
        chosen arities and generation parameters.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Input graph must be a DAG")

    rng = np.random.default_rng(seed)
    if isinstance(arity_strategy, dict):
        strat = ArityStrategy(**arity_strategy)  # type: ignore[arg-type]
    else:
        strat = arity_strategy

    nodes = list(dag.nodes())
    node_cards: Dict[Any, int] = strat.draw_cardinalities(nodes, rng)

    model = DiscreteBayesianNetwork(list(dag.edges()))

    # Create state names per node
    state_names: Dict[Any, List[str]] = {n: [f"s{i}" for i in range(node_cards[n])] for n in nodes}

    # Build CPDs in topological order
    cpds: List[TabularCPD] = []
    for node in nx.topological_sort(dag):
        parents = list(dag.predecessors(node))
        var_card = node_cards[node]
        parent_cards = {p: node_cards[p] for p in parents}

        values = _sample_cpt_for_node(
            node=node,
            parents=parents,
            var_card=var_card,
            parent_cards=parent_cards,
            dirichlet_alpha=dirichlet_alpha,
            determinism_fraction=determinism_fraction,
            rng=rng,
        )

        if parents:
            evidence = parents
            evidence_card = [node_cards[p] for p in parents]
            cpd = TabularCPD(
                variable=node,
                variable_card=var_card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
                state_names={**{node: state_names[node]}, **{p: state_names[p] for p in parents}},
            )
        else:
            cpd = TabularCPD(
                variable=node,
                variable_card=var_card,
                values=values.reshape(var_card, 1),
                state_names={node: state_names[node]},
            )

        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()

    metadata: Dict[str, Any] = {
        "node_cardinalities": node_cards,
        "dirichlet_alpha": dirichlet_alpha,
        "determinism_fraction": determinism_fraction,
        "seed": seed,
    }
    return model, metadata


def generate_variants_for_dag(
    dag: nx.DiGraph,
    variants: List[Dict[str, Any]],
    base_seed: int = 0,
) -> List[Tuple[DiscreteBayesianNetwork, Dict[str, Any]]]:
    """Generate multiple BN variants for a single DAG.

    'variants' is a list of dicts with keys accepted by generate_discrete_bn_from_dag.
    Each variant gets a deterministically offset seed (base_seed + idx).
    """
    results: List[Tuple[DiscreteBayesianNetwork, Dict[str, Any]]] = []
    for idx, cfg in enumerate(variants):
        cfg = dict(cfg)  # shallow copy
        if "seed" not in cfg:
            cfg["seed"] = int(base_seed + idx * 9973)  # prime step
        bn, meta = generate_discrete_bn_from_dag(dag, **cfg)
        meta["variant_index"] = idx
        results.append((bn, meta))
    return results


# ------------------------------
# CLI
# ------------------------------

def _parse_arity(arg: str) -> ArityStrategy:
    """Parse arity string like 'fixed:2' or 'range:2-4'."""
    if arg.startswith("fixed:"):
        val = int(arg.split(":", 1)[1])
        return ArityStrategy(type="fixed", fixed=val)
    if arg.startswith("range:"):
        span = arg.split(":", 1)[1]
        lo, hi = span.split("-", 1)
        return ArityStrategy(type="range", min=int(lo), max=int(hi))
    raise argparse.ArgumentTypeError("Arity must be 'fixed:<k>' or 'range:<min>-<max>'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate discrete BN variants from a generated DAG")
    parser.add_argument("--n-nodes", type=int, default=8, help="Number of nodes for generated DAG")
    parser.add_argument("--target-treewidth", type=int, default=2, help="Target treewidth for DAG generation")
    parser.add_argument("--dag-method", type=str, default="random", choices=["random", "topological", "bfs", "dfs"], help="DAG orientation method")
    parser.add_argument("--variants", type=int, default=3, help="Number of BN variants to generate")
    parser.add_argument("--arity", type=_parse_arity, default=ArityStrategy(type="range", min=2, max=3), help="Variable arity strategy: fixed:k or range:min-max")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet alpha for CPT sampling (<=1 skewed, 1 uniform, >1 flat)")
    parser.add_argument("--determinism", type=float, default=0.0, help="Deterministic fraction for CPT columns (0.0 recommended)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")

    args = parser.parse_args()

    dag, achieved_tw, meta = generate_dag_with_treewidth(
        n_nodes=args.n_nodes,
        target_treewidth=args.target_treewidth,
        dag_method=args.dag_method,
        seed=args.seed,
    )

    variant_cfgs: List[Dict[str, Any]] = []
    for i in range(args.variants):
        # Vary alpha a bit across variants while keeping determinism mostly zero
        alpha = max(1e-3, args.alpha * (0.5 + 0.5 * (i + 1) / args.variants))
        cfg: Dict[str, Any] = {
            "arity_strategy": args.arity,
            "dirichlet_alpha": alpha,
            "determinism_fraction": args.determinism,
        }
        variant_cfgs.append(cfg)

    bns = generate_variants_for_dag(dag, variant_cfgs, base_seed=args.seed)

    print(f"Generated DAG with achieved treewidth = {achieved_tw}")
    for idx, (bn, meta) in enumerate(bns):
        cards = meta["node_cardinalities"]
        card_str = ", ".join(f"{n}:{c}" for n, c in list(cards.items())[:6])
        if len(cards) > 6:
            card_str += ", ..."
        print(f"Variant {idx}: alpha={meta['dirichlet_alpha']:.3f}, det={meta['determinism_fraction']:.2f}, seed={meta['seed']} | cards: {card_str}")


if __name__ == "__main__":
    main()


def _parse_field(val: Any) -> Any:
    import ast, re
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return None
    s = re.sub(r"np\.str_\(\'([^']*)\'\)", r"'\1'", s)
    s = re.sub(r'np\.str_\("([^"]*)"\)', r"'\1'", s)
    try:
        return ast.literal_eval(s)
    except Exception:
        return val


def inspect_row_and_call_llm(
    *,
    full_df: pd.DataFrame,
    all_bayesian_networks: List[Dict[str, Any]],
    row_index: int,
    openai_client: Any,
    model: str,
    system_prompt: str = "You are a probability calculator. Provide exact numerical answers.",
    draw_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if draw_kwargs is None:
        draw_kwargs = {}

    row = full_df.iloc[row_index]
    if "bn_index" not in row:
        raise ValueError("full_df is missing 'bn_index' column")
    bn_idx = int(row["bn_index"])  # type: ignore[arg-type]
    bn = all_bayesian_networks[bn_idx]["bn"]

    q_vars = _parse_field(row.get("query_vars"))
    q_states = _parse_field(row.get("query_states"))
    evidence = _parse_field(row.get("evidence"))
    if not isinstance(q_vars, list) or not isinstance(q_states, list):
        raise ValueError("Row has invalid 'query_vars' or 'query_states'")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("Row has invalid 'evidence' (expected dict or None)")

    # Draw BN
    G = nx.DiGraph()
    G.add_nodes_from(bn.nodes())
    G.add_edges_from(bn.edges())
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=draw_kwargs.pop("figsize", (6, 4)))
    nx.draw(G, pos, with_labels=True, node_color="#A7C7E7", arrows=True, **draw_kwargs)
    plt.title("Bayesian Network Structure")
    plt.show()

    # Build query text
    if len(q_vars) == 1:
        if evidence:
            ev_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
            q_text = f"P({q_vars[0]}={q_states[0]} | {ev_str})"
        else:
            q_text = f"P({q_vars[0]}={q_states[0]})"
    else:
        parts = [f"{v}={s}" for v, s in zip(q_vars, q_states)]
        if evidence:
            ev_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
            q_text = f"P({', '.join(parts)} | {ev_str})"
        else:
            q_text = f"P({', '.join(parts)})"
    print("Query:", q_text)

    exact_prob = row.get("probability", None)
    if exact_prob is not None:
        exact_prob = float(exact_prob)
    print("Exact probability:", exact_prob)

    prompt_str = create_probability_prompt(bn, q_vars, q_states, evidence)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_str},
    ]
    try:
        response, _ = run_llm_call(openai_client=openai_client, model=model, messages=messages)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        response = None
    llm_prob = extract_numeric_answer(response) if response else None
    print("LLM probability:", llm_prob)

    delta = None
    if llm_prob is not None and exact_prob is not None:
        delta = float(abs(llm_prob - exact_prob))
        print("Absolute error:", delta)

    return {
        "query": q_text,
        "exact_probability": exact_prob,
        "llm_probability": llm_prob,
        "delta": delta,
        "llm_response": response,
    }
