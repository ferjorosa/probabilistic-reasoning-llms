"""
Create a simple 3-node Bayesian Network, run exact inference with pgmpy,
optionally query an LLM using the provided prompt templates, and compare results.

Usage examples:

  - Run exact inference only (skip LLM):
      python scripts/test_llm_inference_discrete.py --no-llm

  - Run with OpenAI (requires OPENAI_API_KEY):
      python scripts/test_llm_inference_discrete.py --model gpt-4o-mini

Environment variables for LLM:
  - OPENAI_API_KEY: API key for OpenAI
  - (Optional) OPENAI_BASE_URL: Alternate base URL

Notes:
  - This script loads prompt templates from `notebooks/discrete/prompts.yaml`.
  - It uses `src/llm_calling.run_llm_call` for the chat completion call.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from inference_discrete import (
    format_probability_query,
    query_probability,
)
from llm_calling import run_llm_call
from yaml_utils import load_yaml


# ------------------------------
# BN construction
# ------------------------------

def build_three_node_bn() -> Tuple[DiscreteBayesianNetwork, VariableElimination]:
    """Create a simple 3-node Discrete BN: A -> B -> C with binary states yes/no.

    CPTs chosen to be intuitive and produce non-trivial posteriors.
    """
    model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

    # Root prior P(A)
    cpd_a = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.3], [0.7]],  # [P(A=yes), P(A=no)]
        state_names={"A": ["yes", "no"]},
    )

    # P(B | A)
    # Columns are A=yes, A=no
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[
            [0.8, 0.2],  # P(B=yes | A=yes), P(B=yes | A=no)
            [0.2, 0.8],  # P(B=no  | A=yes), P(B=no  | A=no)
        ],
        evidence=["A"],
        evidence_card=[2],
        state_names={
            "B": ["yes", "no"],
            "A": ["yes", "no"],
        },
    )

    # P(C | B)
    # Columns are B=yes, B=no
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.7, 0.1],  # P(C=yes | B=yes), P(C=yes | B=no)
            [0.3, 0.9],  # P(C=no  | B=yes), P(C=no  | B=no)
        ],
        evidence=["B"],
        evidence_card=[2],
        state_names={
            "C": ["yes", "no"],
            "B": ["yes", "no"],
        },
    )

    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()

    return model, VariableElimination(model)


# ------------------------------
# CPT formatting for LLM prompt
# ------------------------------

def _format_table(rows: List[List[str]]) -> str:
    """Render a simple ASCII table given rows of cells (all strings)."""
    # Compute column widths
    widths: List[int] = []
    for row in rows:
        for i, cell in enumerate(row):
            if i >= len(widths):
                widths.append(len(cell))
            else:
                widths[i] = max(widths[i], len(cell))

    def horiz() -> str:
        parts = ["+" + "-" * (w + 2) for w in widths]
        return "".join(parts) + "+"

    def fmt_row(row: List[str]) -> str:
        cells = [f" {cell.ljust(w)} " for cell, w in zip(row, widths)]
        return "|" + "|".join(cells) + "|"

    out: List[str] = []
    out.append(horiz())
    for r in rows:
        out.append(fmt_row(r))
        out.append(horiz())
    return "\n".join(out)


def _parent_assignments(
    parents: Sequence[str], state_names: Dict[str, Sequence[str]]
) -> List[Tuple[str, ...]]:
    """Enumerate parent assignments in the natural cartesian order of parents.

    For parents [P1, P2], this yields:
        (P1=s1, P2=t1), (P1=s1, P2=t2), ..., (P1=s2, P2=t1), ...
    """
    from itertools import product

    domains = [state_names[p] for p in parents]
    return list(product(*domains)) if parents else []


def cpd_to_ascii_table(cpd: TabularCPD) -> str:
    """Convert a pgmpy TabularCPD to an ASCII table matching the prompt's guidance."""
    var = cpd.variable
    var_states = list(cpd.state_names[cpd.variable])
    parents = list(cpd.variables[1:]) if hasattr(cpd, "variables") else list(cpd.evidence or [])

    rows: List[List[str]] = []

    if not parents:
        # Root CPT: rows are var(value) and Probability
        rows.append(["Node(Value)", "Probability"])  # header
        # values has shape (card,) for root nodes
        for s_idx, s in enumerate(var_states):
            prob = float(cpd.values[s_idx])
            rows.append([f"{var}({s})", f"{prob:.4f}"])
        return _format_table(rows)

    # Conditional CPT: parent headers + child rows
    parent_assigns = _parent_assignments(parents, cpd.state_names)

    # For each parent, create a row listing the parent name then assignment values per column
    for p in parents:
        header = [p]
        # For each column (assignment), append the value of this parent under that assignment
        for assign in parent_assigns:
            # assign is a tuple aligned with parents
            val = assign[parents.index(p)]
            header.append(f"{p}({val})")
        rows.append(header)

    # Now child rows: one per child state
    # cpd.values shape: (var_card, num_cols)
    for s_idx, s in enumerate(var_states):
        row = [f"{var}({s})"]
        for col in range(len(parent_assigns)):
            prob = float(cpd.values[s_idx][col])
            row.append(f"{prob:.4f}")
        rows.append(row)

    return _format_table(rows)


def bn_to_cpt_text(model: DiscreteBayesianNetwork) -> str:
    """Create a combined CPT text block for all nodes in the BN."""
    cpds: List[TabularCPD] = model.get_cpds()
    # Keep a stable order: parents before children if possible
    order = list(model.nodes())
    cpd_map = {cpd.variable: cpd for cpd in cpds}
    out_blocks: List[str] = []
    for node in order:
        cpd = cpd_map[node]
        out_blocks.append(cpd_to_ascii_table(cpd))
        out_blocks.append("")
    return "\n".join(out_blocks).strip()


# ------------------------------
# LLM helpers
# ------------------------------

def extract_probability_from_response(text: str) -> Optional[float]:
    """Extract the numeric probability from the LLM's final answer line.

    Looks for a pattern like:
        Final Answer: P(...) = 0.1234
    Falls back to searching for a float in [0,1].
    """
    # Prefer explicit "Final Answer" lines
    m = re.search(r"Final Answer\s*:\s*P\([^=]+\)\s*=\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Fallback: find last float in [0,1]
    candidates = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if candidates:
        try:
            return float(candidates[-1])
        except ValueError:
            return None
    return None


def build_messages(system_prompt: str, prompt_template: str, cpt_text: str, query_text: str):
    user_prompt = prompt_template.format(cpts=cpt_text, query=query_text)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def maybe_make_openai_client() -> Optional[object]:
    """Create an OpenAI client if API key is present; otherwise return None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ------------------------------
# Main
# ------------------------------


def run_three_node_tests(
    no_llm: bool = True,
    model: str = "gpt-4o-mini",
    prompts: Path | None = None,
    verbose: bool = True,
):
    """Run a set of queries on the 3-node BN, optionally with LLM.

    Returns a list of dicts: [{"query": str, "exact": float, "llm": Optional[float], "delta": Optional[float]}]
    """
    if prompts is None:
        # Resolve prompts.yaml relative to project root (src/ is one level below root)
        project_root = Path(__file__).resolve().parents[1]
        prompts = project_root / "notebooks" / "discrete" / "prompts.yaml"

    parser = argparse.ArgumentParser(description="Test LLM vs exact inference on a 3-node BN")
    # Build BN + exact inference engine
    model, engine = build_three_node_bn()
    cpt_text = bn_to_cpt_text(model)

    prompts_yaml = load_yaml(prompts)
    system_prompt: str = prompts_yaml["system_prompt"]
    prompt_template: str = prompts_yaml["prompt_base"]

    # Define a set of queries to test
    queries: List[Tuple[str, str, Optional[Dict[str, str]]]] = [
        ("C", "yes", None),
        ("C", "yes", {"A": "yes"}),
        ("B", "no", {"C": "yes"}),
        ("A", "yes", {"C": "yes"}),
    ]

    # Set up LLM client if desired
    openai_client = None
    if not no_llm:
        openai_client = maybe_make_openai_client()
        if openai_client is None:
            if verbose:
                print("[warn] No OPENAI_API_KEY found or OpenAI client unavailable; skipping LLM calls.")

    # Run queries
    if verbose:
        print("BN CPTs presented to LLM:\n")
        print(cpt_text)
        print("\n---\n")

    results = []
    for (var, val, evidence) in queries:
        q_text = format_probability_query(var, val, evidence)
        exact = query_probability(engine, var, val, evidence)

        if verbose:
            print(f"Query: {q_text}")
            print(f"Exact: {exact:.6f}")

        llm_prob: Optional[float] = None
        if openai_client is not None:
            messages = build_messages(system_prompt, prompt_template, cpt_text, q_text)
            try:
                content, _ = run_llm_call(openai_client, model, messages)
                if content is None:
                    if verbose:
                        print("LLM: [no content returned]")
                else:
                    llm_prob = extract_probability_from_response(content)
                    if llm_prob is None:
                        if verbose:
                            print("LLM: could not parse probability from response.")
                            print("Response snippet:\n" + content)
                    else:
                        if verbose:
                            print(f"LLM:   {llm_prob:.6f}")
                            print(f"Delta: {abs(exact - llm_prob):.6f}")
            except Exception as e:
                if verbose:
                    print(f"LLM call failed: {e}")
        if verbose:
            print("---")

        results.append({
            "query": q_text,
            "exact": float(exact),
            "llm": llm_prob,
            "delta": (abs(float(exact) - llm_prob) if llm_prob is not None else None),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test LLM vs exact inference on a 3-node BN")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip calling the LLM; only run exact inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for OpenAI client (e.g., gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Path to prompts YAML (defaults to project_root/notebooks/discrete/prompts.yaml)",
    )
    args = parser.parse_args()

    run_three_node_tests(
        no_llm=args.no_llm,
        model=args.model,
        prompts=args.prompts,
        verbose=True,
    )


if __name__ == "__main__":
    main()

__all__ = [
    "cpd_to_ascii_table",
]
__all__ = [
    # Public helpers that other scripts can import
    "cpd_to_ascii_table",
]
