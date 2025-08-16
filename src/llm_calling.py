from typing import Any, List, Dict, Optional, Tuple

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