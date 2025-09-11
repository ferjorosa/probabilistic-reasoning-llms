from typing import Any, List, Dict, Optional, Tuple

# Optional imports for prompt helpers
try:  # type: ignore
    from cpd_utils import cpd_to_ascii_table  # type: ignore
except Exception:  # pragma: no cover
    cpd_to_ascii_table = None  # type: ignore

def run_llm_call(
    openai_client: Any,
    model: str,
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Sends a basic chat message to a language model (no tool calling).

    Args:
        openai_client (Any): Initialized OpenAI client (e.g., from OpenAI or OpenRouter).
        model (str): Model name to use (e.g., 'openai/gpt-3.5-turbo').
        messages (List[Dict[str, str]]): List of chat messages in OpenAI format.

    Returns:
        Tuple[Optional[str], List[Dict[str, str]]]: A tuple of (final_response_content, full_message_list), 
        where final_response_content is the assistant's response content, or None if no response is returned,
        and full_message_list is the list of messages including the assistant's response.
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )

    message = response.choices[0].message

    messages.append(message.model_dump())

    return message.content, messages


def extract_numeric_answer(response_text: str) -> Optional[float]:
    """Extract a numeric probability from free-form LLM response text.

    Looks for patterns like:
      - "Final Answer: P(...) = 0.1234"
      - "= 0.1234"
      - "probability: 0.1234"
      - trailing number at end of text
    Returns the last matched number as float, or None.
    """
    import re

    patterns = [
        r"Final Answer:.*?=\s*([0-9]*\.?[0-9]+)",
        r"=\s*([0-9]*\.?[0-9]+)\s*$",
        r"probability[:\s]*([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response_text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


def create_probability_prompt(
    bn: Any,
    query_vars: List[str],
    query_states: List[str],
    evidence: Optional[Dict[str, str]],
) -> str:
    """Create a machine-readable prompt for BN probability queries.

    Includes CPTs as ASCII tables and a query string. Works for 1 or 2 targets.
    """
    if cpd_to_ascii_table is None:  # pragma: no cover
        raise RuntimeError("cpd_utils.cpd_to_ascii_table not available")

    # Format CPTs
    cpd_strings: List[str] = []
    for cpd in bn.get_cpds():
        cpd_strings.append(cpd_to_ascii_table(cpd))
    cpds_as_string = "\n\n".join(cpd_strings)

    # Format query string
    if len(query_vars) == 1:
        if evidence:
            ev_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
            query_str = f"P({query_vars[0]}={query_states[0]} | {ev_str})"
        else:
            query_str = f"P({query_vars[0]}={query_states[0]})"
    else:
        parts = [f"{v}={s}" for v, s in zip(query_vars, query_states)]
        if evidence:
            ev_str = ", ".join([f"{k}={v}" for k, v in evidence.items()])
            query_str = f"P({', '.join(parts)} | {ev_str})"
        else:
            query_str = f"P({', '.join(parts)})"

    prompt_str = (
        "You are a probability calculator. Compute the exact probability for the given query.\n\n"
        "Conditional Probability Tables:\n"
        f"{cpds_as_string}\n\n"
        f"Query: {query_str}\n\n"
        "Instructions:\n"
        "1. Use exact inference methods (variable elimination, etc.)\n"
        "2. Show your work step by step\n"
        "3. Provide the final answer in this exact format: Final Answer: "
        f"{query_str} = [NUMBER]\n\n"
        "Where [NUMBER] is the exact probability as a decimal (e.g., 0.1234)."
    )
    return prompt_str
