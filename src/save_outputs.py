import json
from pathlib import Path
from typing import List, Union

def save_output(
    response: Union[str, None],
    messages: List[dict],
    base_path: Path,
    timestamp: str,
    dir_name: str,
    print_output: bool = True
) -> None:
    """
    Save the final response and message history from an llm / tool calling loop.

    Args:
        response (str | None): The final text response to save.
        messages (List[dict]): The full message history to save as JSON.
        base_path (Path): Base directory where the results will be stored.
        timestamp (str): Timestamp string to organize result folders.
        dir_name (str): Name of the subdirectory to store results in.
    """
    # Create timestamped directory
    results_dir = base_path / timestamp / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    txt_path = results_dir / "final_response.txt"
    json_path = results_dir / "message_history.json"

    # Save final response (if it exists)
    if response:
        txt_path.write_text(response, encoding="utf-8")
    else:
        print("❌ No response to save")

    # Save full message history as JSON
    if messages:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    else:
        print("❌ No message history to save")

    # Optional: print where the files were saved
    if print_output:
        print(f"✅ Final response saved to: {txt_path}")
        print(f"✅ Message history saved to: {json_path}")