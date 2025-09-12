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
except ModuleNotFoundError:  # pragma: no cover
    from .cpd_utils import cpd_to_ascii_table  # type: ignore
    from .llm_calling import (  # type: ignore
        run_llm_call,
        create_probability_prompt,
        create_system_and_user_prompts,
        extract_numeric_answer,
    )
    from .yaml_utils import load_yaml  # type: ignore




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


__all__ = [
    "inspect_row_and_call_llm",
    "call_llm_for_query",
]
