from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pgmpy.inference import VariableElimination

# Local imports (support running as module or with src on sys.path)
try:
    from graph_generation import generate_dag_with_treewidth  # type: ignore
    from bn_generation import generate_variants_for_dag  # type: ignore
    from query_generation import generate_queries, QuerySpec  # type: ignore
    from cpd_utils import cpd_to_ascii_table  # type: ignore
    from llm_calling import (
        run_llm_call,
        create_probability_prompt,
        extract_numeric_answer,
    )  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from .graph_generation import generate_dag_with_treewidth  # type: ignore
    from .bn_generation import generate_variants_for_dag  # type: ignore
    from .query_generation import generate_queries, QuerySpec  # type: ignore
    from .cpd_utils import cpd_to_ascii_table  # type: ignore
    from .llm_calling import (  # type: ignore
        run_llm_call,
        create_probability_prompt,
        extract_numeric_answer,
    )


def _arity_to_str(spec: Dict[str, Any]) -> str:
    if spec.get("type") == "fixed":
        return f"fixed:{spec['fixed']}"
    if spec.get("type") == "range":
        return f"range:{spec['min']}-{spec['max']}"
    return str(spec)


def generate_bn_sweep(
    *,
    ns: Sequence[int],
    treewidths: Sequence[int],
    naming_strategies: Sequence[str],
    arity_specs: Sequence[Dict[str, Any]],
    dirichlet_alphas: Sequence[float],
    determinism_fracs: Sequence[float],
    variants_per_combo: int = 2,
    base_seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate a sweep of DAGs and BN variants without any LLM calls.

    Returns (df, all_bayesian_networks), where:
      - df has one row per BN variant with metadata only (no queries yet)
      - all_bayesian_networks is a list of dicts with keys: {"bn", "meta"}
    """
    rows: List[Dict[str, Any]] = []
    all_bayesian_networks: List[Dict[str, Any]] = []

    sample_counter = 0
    for n in ns:
        for tw in treewidths:
            for naming in naming_strategies:
                dag, achieved_tw, _ = generate_dag_with_treewidth(
                    n_nodes=n, target_treewidth=tw, node_naming=naming, seed=int(base_seed + sample_counter)
                )
                for arity in arity_specs:
                    for alpha in dirichlet_alphas:
                        for det in determinism_fracs:
                            cfgs: List[Dict[str, Any]] = []
                            for i in range(variants_per_combo):
                                cfgs.append(
                                    {
                                        "arity_strategy": arity,
                                        "dirichlet_alpha": float(alpha),
                                        "determinism_fraction": float(det),
                                    }
                                )
                            variants = generate_variants_for_dag(
                                dag, cfgs, base_seed=int(base_seed + sample_counter)
                            )
                            for idx, (bn, meta) in enumerate(variants):
                                all_bayesian_networks.append(
                                    {
                                        "bn": bn,
                                        "meta": {
                                            "n": n,
                                            "target_tw": tw,
                                            "achieved_tw": achieved_tw,
                                            "naming": naming,
                                            "arity": _arity_to_str(arity),
                                            "alpha": meta["dirichlet_alpha"],
                                            "determinism": meta["determinism_fraction"],
                                            "seed": meta["seed"],
                                            "variant_index": idx,
                                            "num_edges": bn.number_of_edges(),
                                            "num_nodes": bn.number_of_nodes(),
                                        },
                                    }
                                )
                                rows.append(
                                    {
                                        "n": n,
                                        "target_tw": tw,
                                        "achieved_tw": achieved_tw,
                                        "naming": naming,
                                        "arity": _arity_to_str(arity),
                                        "alpha": meta["dirichlet_alpha"],
                                        "determinism": meta["determinism_fraction"],
                                        "seed": meta["seed"],
                                        "variant_index": idx,
                                        "num_edges": bn.number_of_edges(),
                                        "num_nodes": bn.number_of_nodes(),
                                    }
                                )
                            sample_counter += 1

    df = pd.DataFrame(rows)
    return df, all_bayesian_networks


def generate_queries_for_bn_list(
    *,
    all_bayesian_networks: List[Dict[str, Any]],
    bn_df: pd.DataFrame,
    num_queries_per_bn: int = 5,
    query_node_counts: Sequence[int] = (1, 2),
    evidence_counts: Sequence[int] = (0, 1, 2),
    distance_buckets: Sequence[Tuple[int, int]] = ((1, 1), (2, 3), (1, 3)),
    base_query_seed: int = 1000,
) -> Tuple[pd.DataFrame, List[List[QuerySpec]]]:
    """Generate queries and exact probabilities for each BN, without LLM calls.

    Returns (full_df, all_bn_queries):
      - full_df: one row per (BN, query) with metadata and exact probability
      - all_bn_queries: list aligned with all_bayesian_networks; each entry is the list of QuerySpec
    """
    query_rows: List[Dict[str, Any]] = []
    all_bn_queries: List[List[QuerySpec]] = []

    for idx, bn_dict in enumerate(all_bayesian_networks):
        bn = bn_dict["bn"]
        query_seed = int(base_query_seed + idx)
        queries = generate_queries(
            bn,
            num_queries=int(num_queries_per_bn),
            query_node_counts=tuple(query_node_counts),
            evidence_counts=tuple(evidence_counts),
            distance_buckets=tuple(distance_buckets),
            seed=query_seed,
        )
        all_bn_queries.append(queries)

        # BN metadata row (assume bn_df is aligned with all_bayesian_networks by index)
        bn_row = bn_df.iloc[idx].to_dict()

        infer = VariableElimination(bn)
        for qidx, query in enumerate(queries):
            query_vars = [v for v, _ in query.targets]
            query_states = [s for _, s in query.targets]
            evidence = query.evidence if query.evidence else None

            # Compute exact probability for 1- or 2-variable targets
            prob: Optional[float]
            try:
                result = infer.query(variables=query_vars, evidence=evidence, show_progress=False)
                assignment = dict(zip(query_vars, query_states))
                prob = float(result.get_value(**assignment))
            except Exception:
                prob = None

            row = dict(bn_row)
            row.update(
                {
                    "bn_index": idx,
                    "query_index": qidx,
                    # Store structured values (not stringified) for downstream use
                    "query_vars": list(query_vars),
                    "query_states": list(query_states),
                    "evidence": dict(query.evidence),
                    "distance": int(query.meta.get("min_target_evidence_distance", 0)),
                    "num_evidence": int(query.meta.get("num_evidence_nodes", 0)),
                    "probability": prob,
                }
            )
            query_rows.append(row)

    full_df = pd.DataFrame(query_rows)
    return full_df, all_bn_queries


def call_llms_for_rows(
    *,
    full_df: pd.DataFrame,
    row_indices: Sequence[int],
    all_bayesian_networks: List[Dict[str, Any]],
    all_bn_queries: List[List[QuerySpec]],
    openai_client: Any,
    model: str,
    system_prompt: str = "You are a probability calculator. Provide exact numerical answers.",
) -> pd.DataFrame:
    """Run LLM calls only for selected rows, augmenting full_df with LLM outputs.

    Adds/updates columns: 'llm_probability', 'llm_response'.
    Does not modify input full_df; returns a copy.
    """
    import ast
    import re

    def _maybe_parse(val):
        # If already a structure, return as-is
        if isinstance(val, (list, dict)):
            return val
        if not isinstance(val, str):
            return val
        s = val.strip()
        if not s:
            return None
        # Normalize occurrences like np.str_('X') -> 'X'
        s = re.sub(r"np\.str_\(\'([^']*)\'\)", r"'\1'", s)
        # Also the double-quoted variant
        s = re.sub(r'np\.str_\("([^"]*)"\)', r"'\1'", s)
        try:
            return ast.literal_eval(s)
        except Exception:
            # Fallback: return original string
            return val

    out = full_df.copy()
    if "llm_probability" not in out.columns:
        out["llm_probability"] = pd.Series([None] * len(out), dtype=object)
    if "llm_response" not in out.columns:
        out["llm_response"] = pd.Series([None] * len(out), dtype=object)

    for ridx in row_indices:
        row = out.iloc[ridx]
        bn_index = int(row["bn_index"]) if "bn_index" in row else None
        query_index = int(row["query_index"]) if "query_index" in row else None
        if bn_index is None or query_index is None:
            continue

        bn = all_bayesian_networks[bn_index]["bn"]
        # Pull query info from DataFrame (structured lists/dicts)
        query_vars_val = _maybe_parse(row.get("query_vars"))
        query_states_val = _maybe_parse(row.get("query_states"))
        evidence_val = _maybe_parse(row.get("evidence"))

        # Ensure proper types
        query_vars: List[str] = list(query_vars_val) if isinstance(query_vars_val, list) else []
        query_states: List[str] = list(query_states_val) if isinstance(query_states_val, list) else []
        evidence: Optional[Dict[str, str]] = (
            dict(evidence_val) if isinstance(evidence_val, dict) else None
        )

        prompt_str = create_probability_prompt(bn, query_vars, query_states, evidence)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_str},
        ]
        try:
            response, _ = run_llm_call(openai_client=openai_client, model=model, messages=messages)
        except Exception as e:  # pragma: no cover
            response = None

        llm_prob: Optional[float] = None
        if response:
            llm_prob = extract_numeric_answer(response)

        out.at[out.index[ridx], "llm_probability"] = llm_prob
        out.at[out.index[ridx], "llm_response"] = response

    return out


__all__ = [
    "generate_bn_sweep",
    "generate_queries_for_bn_list",
    "call_llms_for_rows",
]
