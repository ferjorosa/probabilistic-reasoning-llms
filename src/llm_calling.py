from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path

# Optional imports for prompt helpers
try:  # type: ignore
    from cpd_utils import cpd_to_ascii_table  # type: ignore
    from yaml_utils import load_yaml  # type: ignore
except Exception:  # pragma: no cover
    cpd_to_ascii_table = None  # type: ignore
    load_yaml = None  # type: ignore

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
    """Extract a numeric probability from LLM response text.

    Looks for patterns like:
      - "Final Answer: P(...) = 0.1234"
      - "= 0.1234"
      - "probability: 0.1234"
      - trailing number at end of text
    Returns the last matched number as float, or None.
    """
    import re

    if not response_text or not response_text.strip():
        return None

    # Clean up the response text
    response_text = response_text.strip()
    
    # Patterns in order of preference (most specific first)
    patterns = [
        # Most specific: "Final Answer: P(...) = 0.1234"
        r"Final Answer:\s*P\([^)]+\)\s*=\s*([0-9]*\.?[0-9]+)",
        # General: "Final Answer: ... = 0.1234"
        r"Final Answer:.*?=\s*([0-9]*\.?[0-9]+)",
        # "= 0.1234" at end of line
        r"=\s*([0-9]*\.?[0-9]+)\s*$",
        # "probability: 0.1234"
        r"probability[:\s]*([0-9]*\.?[0-9]+)",
        # Any number at end of text
        r"([0-9]*\.?[0-9]+)\s*$",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                value = float(matches[-1])
                # Validate that it's a probability (0-1)
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                continue
    
    # If no valid probability found, print the response for debugging
    print(f"Could not extract numeric answer from response: {response_text}")
    return None


def create_probability_prompt(
    bn: Any,
    query_vars: List[str],
    query_states: List[str],
    evidence: Optional[Dict[str, str]],
    prompts_path: Optional[Path] = None,
) -> str:
    """Create a machine-readable prompt for BN probability queries.

    Includes CPTs as ASCII tables and a query string. Works for 1 or 2 targets.
    Uses prompts.yaml template if prompts_path is provided, otherwise uses default format.
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

    # Use YAML template - REQUIRED
    if prompts_path is None or load_yaml is None:
        raise ValueError("prompts_path must be provided and yaml_utils must be available")
    
    try:
        prompts = load_yaml(prompts_path)
        prompt_template = prompts.get("prompt_base", "")
        
        if not prompt_template:
            raise ValueError("prompt_base not found in YAML file")
            
        return prompt_template.format(cpts=cpds_as_string, query=query_str)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts from YAML: {e}")


def create_system_and_user_prompts(
    bn: Any,
    query_vars: List[str],
    query_states: List[str],
    evidence: Optional[Dict[str, str]],
    prompts_path: Optional[Path] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    """Create system and user prompts for BN probability queries using YAML templates.

    Args:
        bn: Bayesian network object
        query_vars: List of query variable names
        query_states: List of query variable states
        evidence: Optional evidence dictionary
        prompts_path: Path to prompts.yaml file (optional)
        system_prompt: Custom system prompt (optional, overrides prompts_path)

    Returns:
        Tuple of (system_prompt, user_prompt)
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

    # Load prompts from YAML - REQUIRED
    if prompts_path is None or load_yaml is None:
        raise ValueError("prompts_path must be provided and yaml_utils must be available")
    
    try:
        prompts = load_yaml(prompts_path)
        system_prompt = prompts.get("system_prompt", "You are a probability calculator. Provide exact numerical answers.")
        prompt_template = prompts.get("prompt_base", "")
        
        if not prompt_template:
            raise ValueError("prompt_base not found in YAML file")
            
        user_prompt = prompt_template.format(cpts=cpds_as_string, query=query_str)
        return system_prompt, user_prompt
        
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts from YAML: {e}")


__all__ = [
    "run_llm_call",
    "extract_numeric_answer", 
    "create_probability_prompt",
    "create_system_and_user_prompts",
]
