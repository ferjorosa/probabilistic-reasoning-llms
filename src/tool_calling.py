import json
from typing import Callable, Dict, List, Tuple, Any, Optional

def run_tool_call_loop(
    openai_client: Any,
    model: str,
    tools: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_mapping: Dict[str, Callable[..., Any]],
    max_iterations: int = 5,
    verbose: bool = True
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Runs a loop to interact with an LLM that may invoke tools during its response generation.

    The function sends messages to the language model, checks for tool calls in the response, 
    executes the appropriate tools, appends the tool outputs to the message history, and 
    continues the loop until a final response is returned or the max iteration limit is reached.

    Args:
        openai_client (Any): The initialized OpenAI client (e.g., via OpenRouter or OpenAI SDK).
        model (str): The model to use (e.g., 'openai/gpt-4', 'google/gemini-2.0-pro').
        tools (List[Dict[str, Any]]): List of tool/function specifications (OpenAI-compatible format).
        messages (List[Dict[str, Any]]): Message history including system/user/assistant/tool messages.
        tool_mapping (Dict[str, Callable[..., Any]]): Mapping of tool names to corresponding Python callables.
        max_iterations (int, optional): Maximum number of LLM-tool call interaction cycles.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        Tuple[Optional[str], List[Dict[str, Any]]]: A tuple of (final_response_content, full_message_list), 
        or (None, messages) if no final response is produced within the max_iterations limit.

    Raises:
        ValueError: If the model requests a tool that isn't defined in `tool_mapping`.
    """
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        response = openai_client.chat.completions.create(
            model=model,
            tools=tools,
            messages=messages
        )
        
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if not getattr(msg, "tool_calls", None):
            if verbose:
                print("🔚 Final model output:\n", msg.content[:500])
            return msg.content, messages

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_func = tool_mapping.get(tool_name)

            if not tool_func:
                raise ValueError(f"No tool function mapped for '{tool_name}'")

            if verbose:
                print(f"🛠️  Model requested tool: {tool_name} with args: {tool_args}")
            tool_result = tool_func(**tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            })

    if verbose:
        print("⚠️ Max iterations reached without final response.")
    return None, messages
