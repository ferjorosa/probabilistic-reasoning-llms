from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Local imports (support running as module or with src on sys.path)
try:
    from cpd_utils import cpd_to_ascii_table  # type: ignore
    from llm_calling import (
        run_llm_call,
        create_probability_prompt,
        create_system_and_user_prompts,
        extract_numeric_answer,
    )  # type: ignore
    from yaml_utils import load_yaml  # type: ignore
    from graph_generation import generate_dag_with_treewidth  # type: ignore
    from bn_generation import generate_variants_for_dag  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from .cpd_utils import cpd_to_ascii_table  # type: ignore
    from .llm_calling import (  # type: ignore
        run_llm_call,
        create_probability_prompt,
        create_system_and_user_prompts,
        extract_numeric_answer,
    )
    from .yaml_utils import load_yaml  # type: ignore
    from .graph_generation import generate_dag_with_treewidth  # type: ignore
    from .bn_generation import generate_variants_for_dag  # type: ignore




def _arity_to_str(spec: Dict[str, Any]) -> str:
    """Convert arity specification to string representation."""
    if spec["type"] == "fixed":
        return f"fixed:{spec['fixed']}"
    return f"range:{spec['min']}-{spec['max']}"


def generate_bayesian_networks_and_metadata(
    ns: List[int],
    treewidths: List[int], 
    arity_specs: List[Dict[str, Any]],
    dirichlet_alphas: List[float],
    determinism_fracs: List[float],
    naming_strategies: List[str],
    variants_per_combo: int = 4,
    base_seed: int = 42,
    max_preview_samples: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any]]:
    """
    Generate Bayesian networks and populate metadata rows based on parameter sweeps.
    
    This function sweeps over DAG/BN generation parameters and materializes multiple 
    discrete BN variants per DAG, similar to the logic in BN_generation_sweep.ipynb.
    
    Parameters:
    - ns: List of numbers of variables
    - treewidths: List of target treewidths
    - arity_specs: List of arity specifications (fixed or range)
    - dirichlet_alphas: List of Dirichlet alpha values for CPT skewness
    - determinism_fracs: List of determinism fractions (mostly 0%)
    - naming_strategies: List of naming strategies ('simple', 'confusing', 'semantic')
    - variants_per_combo: Number of variants per parameter combination
    - base_seed: Base seed for reproducibility
    - max_preview_samples: Maximum number of preview samples to collect
    
    Returns:
    - Tuple of (all_bayesian_networks, rows, preview_samples)
        - all_bayesian_networks: List of BN dictionaries with 'bn' and 'meta' keys
        - rows: List of metadata dictionaries for DataFrame creation
        - preview_samples: List of BN objects for preview (first few samples)
    """
    rows = []
    preview_samples = []
    sample_counter = 0
    all_bayesian_networks = []  # Store all BNs and their metadata

    for n in ns:
        for tw in treewidths:
            for naming in naming_strategies:
                dag, achieved_tw, _ = generate_dag_with_treewidth(
                    n, tw, node_naming=naming, seed=base_seed + sample_counter
                )
                for arity in arity_specs:
                    for alpha in dirichlet_alphas:
                        for det in determinism_fracs:
                            cfgs = []
                            for i in range(variants_per_combo):
                                cfgs.append({
                                    "arity_strategy": arity,
                                    "dirichlet_alpha": alpha,
                                    "determinism_fraction": det,
                                })
                            variants = generate_variants_for_dag(
                                dag, cfgs, base_seed=base_seed + sample_counter
                            )
                            for idx, (bn, meta) in enumerate(variants):
                                # Store BN and its metadata for later access
                                all_bayesian_networks.append({
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
                                    }
                                })
                                rows.append({
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
                                })
                                if sample_counter < max_preview_samples:  # collect a few previews
                                    preview_samples.append(bn)
                            sample_counter += 1

    return all_bayesian_networks, rows, preview_samples


def _parse_field(val: Any) -> Any:
    """Parse field values that might be stringified lists/dicts from DataFrame storage."""
    import ast
    import re

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


def inspect_row_and_call_llm(
    *,
    full_df: pd.DataFrame,
    all_bayesian_networks: List[Dict[str, Any]],
    row_index: int,
    openai_client: Any,
    model: str,
    prompts_path: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    draw_kwargs: Optional[Dict[str, Any]] = None,
    print_prompts: bool = True,
) -> Dict[str, Any]:
    """Inspect a specific row from full_df: retrieve BN and query, draw BN, print query, call LLM, and compare results.
    
    Args:
        full_df: DataFrame containing BN and query information
        all_bayesian_networks: List of BN dictionaries with 'bn' and 'meta' keys
        row_index: Index of the row in full_df to inspect
        openai_client: Initialized OpenAI client
        model: Model name to use
        prompts_path: Path to prompts.yaml file (optional)
        system_prompt: Custom system prompt (optional, overrides prompts_path)
        draw_kwargs: Additional arguments for BN drawing (optional)
        print_prompts: Whether to print system and user prompts to console (default: True)
    
    Returns:
        Dictionary with query, exact probability, LLM probability, error, and LLM response
    """
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

    # Print prompts to console if requested (before calling LLM)
    if print_prompts:
        # Create prompts just for display purposes
        system_prompt, prompt_str = create_system_and_user_prompts(
            bn=bn,
            query_vars=q_vars,
            query_states=q_states,
            evidence=evidence,
            prompts_path=prompts_path,
            system_prompt=system_prompt,
        )
        print("\n" + "="*50)
        print("SYSTEM PROMPT:")
        print("="*50)
        print(system_prompt)
        print("\n" + "="*50)
        print("USER PROMPT:")
        print("="*50)
        print(prompt_str)
        print("="*50 + "\n")

    # Call LLM using the shared function
    llm_prob, response = call_llm_for_query(
        bn=bn,
        query_vars=q_vars,
        query_states=q_states,
        evidence=evidence,
        openai_client=openai_client,
        model=model,
        prompts_path=prompts_path,
    )
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


def call_llm_for_query(
    bn: Any,
    query_vars: List[str],
    query_states: List[str],
    evidence: Optional[Dict[str, str]],
    openai_client: Any = None,
    model: str = None,
    prompts_path: Optional[Path] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Call LLM to get probability for a specific query.
    
    Args:
        bn: Bayesian network object
        query_vars: List of query variable names
        query_states: List of query variable states
        evidence: Optional evidence dictionary
        openai_client: OpenAI client (optional, can be set globally)
        model: Model name (optional, can be set globally)
        prompts_path: Path to prompts.yaml file (optional, can be set globally)
    
    Returns:
        Tuple of (llm_probability, llm_response)
    """
    # Import here to avoid circular imports
    import os
    from openai import OpenAI
    
    # Use provided parameters or try to get from global scope
    if openai_client is None:
        # Try to get from global scope (notebook context)
        try:
            import sys
            frame = sys._getframe(1)
            openai_client = frame.f_locals.get('client')
            if openai_client is None:
                # Fallback: create new client
                openai_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY")
                )
        except:
            raise ValueError("openai_client must be provided or available in global scope as 'client'")
    
    if model is None:
        try:
            import sys
            frame = sys._getframe(1)
            model = frame.f_locals.get('MODEL')
            if model is None:
                model = "deepseek/deepseek-chat-v3.1:free"  # fallback
        except:
            model = "deepseek/deepseek-chat-v3.1:free"  # fallback
    
    if prompts_path is None:
        try:
            import sys
            frame = sys._getframe(1)
            prompts_path = frame.f_locals.get('prompt_path')
            if prompts_path is None:
                # Fallback: construct path
                from pathlib import Path
                prompts_path = Path("notebooks") / "discrete" / "prompts.yaml"
        except:
            from pathlib import Path
            prompts_path = Path("notebooks") / "discrete" / "prompts.yaml"
    
    try:
        # Create system and user prompts using YAML template
        system_prompt, prompt_str = create_system_and_user_prompts(
            bn=bn,
            query_vars=query_vars,
            query_states=query_states,
            evidence=evidence,
            prompts_path=prompts_path,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_str}
        ]
        
        response, _ = run_llm_call(
            openai_client=openai_client,
            model=model,
            messages=messages
        )
        
        if response:
            numeric_answer = extract_numeric_answer(response)
            return numeric_answer, response
        else:
            return None, None
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None, None


def compute_query_complexity(full_df, all_bayesian_networks, row_index, verbose=True):
    """
    Compute the complexity metrics for a specific query from a row in full_df.
    Now properly takes the query into account by only eliminating non-query variables.
    
    Parameters:
    - full_df: DataFrame containing query information
    - all_bayesian_networks: List of BN dictionaries with 'bn' and 'meta' keys
    - row_index: Index of the row in full_df to analyze
    - verbose: If True, print detailed progress information
    
    Returns:
    - dict: Complexity metrics including induced width, total cost, max factor size, etc.
    """
    from pgmpy.inference.EliminationOrder import WeightedMinFill
    from pgmpy.inference import VariableElimination
    import numpy as np
    
    # Get the row data
    row = full_df.iloc[row_index]
    bn_index = int(row['bn_index'])
    
    # Get the Bayesian network
    bn = all_bayesian_networks[bn_index]['bn']
    
    # Parse query information
    query_vars = _parse_field(row['query_vars']) or []
    query_states = _parse_field(row['query_states']) or []
    evidence = _parse_field(row['evidence']) or {}
    
    if verbose:
        print(f"Computing complexity for query: P({query_vars}={query_states} | {evidence})")
        print(f"BN: {bn_index}, Query: {int(row['query_index'])}")
    
    # Ensure the model is valid
    bn.check_model()
    
    # Get cardinalities
    card = bn.get_cardinality()
    if verbose:
        print(f"Variable cardinalities: {card}")
    
    # QUERY-SPECIFIC COMPLEXITY COMPUTATION
    # Identify which variables need to be eliminated vs kept for the query
    all_vars = set(bn.nodes())
    query_vars_set = set(query_vars)
    evidence_vars_set = set(evidence.keys()) if evidence else set()
    
    # Variables that must be kept until the end (query variables)
    keep_vars = query_vars_set
    
    # Variables that can be eliminated (all others)
    eliminate_vars = all_vars - keep_vars
    
    if verbose:
        print(f"Variables to keep (query): {sorted(keep_vars)}")
        print(f"Variables to eliminate: {sorted(eliminate_vars)}")
        print(f"Evidence variables: {sorted(evidence_vars_set)}")
    
    # Handle evidence by reducing cardinalities
    # Evidence variables are instantiated, so they don't contribute to factor sizes
    effective_card = card.copy()
    for evar in evidence_vars_set:
        effective_card[evar] = 1  # Evidence variables are fixed, so cardinality = 1
    
    if verbose:
        print(f"Effective cardinalities (after evidence): {dict(effective_card)}")
    
    # Create elimination orderer and get optimal order for variables to eliminate
    if eliminate_vars:
        orderer = WeightedMinFill(bn)
        elim_order = orderer.get_elimination_order(nodes=list(eliminate_vars))
    else:
        elim_order = []  # No variables to eliminate
    
    if verbose:
        print(f"Elimination order (variables to eliminate): {elim_order}")
        if elim_order:
            complete_elim_order = elim_order + list(keep_vars)
            print(f"Complete elimination order: {complete_elim_order}")
    
    # Calculate induced width for the elimination order
    if elim_order:
        # For induced width calculation, we need to create a complete elimination order
        # that includes all variables, with query variables at the end
        complete_elim_order = elim_order + list(keep_vars)
        ve = VariableElimination(bn)
        induced_width = ve.induced_width(complete_elim_order)
    else:
        complete_elim_order = list(keep_vars)  # Only query variables
        induced_width = 0  # No elimination needed
    
    if verbose:
        print(f"Induced width: {induced_width}")
    
    # Simulate variable elimination to compute cost metrics
    cost = 0
    max_factor_size = 0
    moral = bn.to_markov_model()  # moralized undirected graph
    
    # Track factor sizes for each elimination step
    factor_sizes = []
    
    for step, x in enumerate(elim_order):
        nbrs = list(moral.neighbors(x))
        
        # Size of the intermediate factor created when eliminating x
        # Use effective cardinalities (evidence variables have cardinality 1)
        size = 1
        for v in nbrs + [x]:
            size *= effective_card[v]
        
        cost += size
        max_factor_size = max(max_factor_size, size)
        factor_sizes.append(size)
        
        if verbose:
            print(f"Step {step+1}: Eliminating {x}, neighbors: {nbrs}, factor size: {size}")
        
        # Connect neighbors (fill-in) and remove x
        for i in range(len(nbrs)):
            for j in range(i+1, len(nbrs)):
                moral.add_edge(nbrs[i], nbrs[j])
        moral.remove_node(x)
    
    # Calculate final factor size (the remaining query variables)
    if keep_vars:
        # The final factor contains all remaining query variables
        final_factor_size = 1
        for v in keep_vars:
            final_factor_size *= effective_card[v]
        cost += final_factor_size
        max_factor_size = max(max_factor_size, final_factor_size)
        if verbose:
            print(f"Final factor (query variables): {sorted(keep_vars)}, size: {final_factor_size}")
    
    # Calculate additional metrics
    num_vars = len(bn.nodes())
    num_edges = bn.number_of_edges()
    
    # Query-specific metrics
    num_query_vars = len(query_vars)
    num_evidence_vars = len(evidence) if evidence else 0
    num_eliminated_vars = len(elim_order)
    
    # Complexity metrics
    complexity_metrics = {
        'row_index': row_index,
        'bn_index': bn_index,
        'query_index': int(row['query_index']),
        'query_vars': query_vars,
        'query_states': query_states,
        'evidence': evidence,
        'num_vars': num_vars,
        'num_edges': num_edges,
        'num_query_vars': num_query_vars,
        'num_evidence_vars': num_evidence_vars,
        'num_eliminated_vars': num_eliminated_vars,
        'elimination_order': elim_order,
        'complete_elimination_order': complete_elim_order,
        'induced_width': induced_width,
        'total_cost': cost,
        'max_factor_size': max_factor_size,
        'avg_factor_size': cost / len(elim_order) if elim_order else 0,
        'factor_sizes': factor_sizes,
        'log_total_cost': np.log2(cost) if cost > 0 else 0,
        'log_max_factor_size': np.log2(max_factor_size) if max_factor_size > 0 else 0,
        'keep_vars': sorted(keep_vars),
        'eliminate_vars': sorted(eliminate_vars),
    }
    
    if verbose:
        print(f"\nQuery-Specific Complexity Summary:")
        print(f"  Variables eliminated: {num_eliminated_vars}/{num_vars}")
        print(f"  Query variables kept: {sorted(keep_vars)}")
        print(f"  Induced width: {induced_width}")
        print(f"  Total factor work: {cost:,}")
        print(f"  Max intermediate factor size: {max_factor_size:,}")
        print(f"  Average factor size: {cost / len(elim_order) if elim_order else 0:.1f}")
        print(f"  Log2(total cost): {np.log2(cost):.2f}")
        print(f"  Log2(max factor size): {np.log2(max_factor_size):.2f}")
    
    return complexity_metrics


def compute_all_query_complexities(full_df, all_bayesian_networks, verbose=False):
    """
    Compute complexity metrics for all queries in full_df.
    
    Parameters:
    - full_df: DataFrame containing query information
    - all_bayesian_networks: List of BN dictionaries with 'bn' and 'meta' keys
    - verbose: If True, print progress information
    
    Returns:
    - pd.DataFrame: DataFrame with complexity metrics for each query
    """
    complexity_results = []
    
    for idx in range(len(full_df)):
        if verbose:
            print(f"Processing query {idx+1}/{len(full_df)}...")
        
        try:
            result = compute_query_complexity(full_df, all_bayesian_networks, idx, verbose=False)
            complexity_results.append(result)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Add a row with error information
            complexity_results.append({
                'row_index': idx,
                'error': str(e),
                'induced_width': None,
                'total_cost': None,
                'max_factor_size': None,
            })
    
    # Convert to DataFrame
    complexity_df = pd.DataFrame(complexity_results)
    
    if verbose:
        print(f"\nComputed complexity for {len(complexity_results)} queries")
        if 'error' in complexity_df.columns:
            successful = len(complexity_df[complexity_df['error'].isna()])
            failed = len(complexity_df[complexity_df['error'].notna()])
        else:
            successful = len(complexity_df)
            failed = 0
        print(f"Successful computations: {successful}")
        print(f"Failed computations: {failed}")
    
    return complexity_df


__all__ = [
    "inspect_row_and_call_llm",
    "call_llm_for_query",
    "compute_query_complexity",
    "compute_all_query_complexities",
    "generate_bayesian_networks_and_metadata",
]
