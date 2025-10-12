"""Python code execution tool with sandboxed environment.

This module provides a safe way to execute Python code snippets in a restricted environment.
It uses RestrictedPython to limit available operations and multiprocessing to enforce timeouts.
The tool is designed to be used with OpenAI's function calling interface for mathematical
and logical computations.

Example:
    >>> result = execute_python("a = 10; b = 20; sum_ab = a + b")
    >>> print(result)
    {'a': 10, 'b': 20, 'sum_ab': 30}
"""

import multiprocessing
from RestrictedPython import compile_restricted
from RestrictedPython import safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
)

# Tool schema compatible with OpenAI Function Calling
PYTHON_MATH_EXECUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Executes a safe subset of Python code for mathematical computations and returns all user-defined variables. "
            "Use this tool for multi-step math, probability, or logic operations.\n\n"
            "Examples:\n"
            "```python\n"
            "a = 10\n"
            "b = 20\n"
            "sum_ab = a + b\n"
            "product_ab = a * b\n"
            "```\n"
            "```python\n"
            "P_A = 0.01\n"
            "P_B_given_A = 0.9\n"
            "P_B = 0.05\n"
            "P_A_given_B = (P_B_given_A * P_A) / P_B\n"
            "```"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code snippet to evaluate. Use standard math operations and variables."
                }
            },
            "required": ["code"]
        }
    }
}

def _run_sandbox(code: str, variables: dict, return_dict: dict) -> None:
    """Execute code in a restricted Python environment.
    
    Args:
        code: Python code to execute
        variables: Dictionary of variables to make available in the sandbox
        return_dict: Dictionary to store execution results or errors
    """
    restricted_globals = safe_globals.copy()
    restricted_globals.update({
        "__builtins__": safe_builtins,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "math": __import__("math"),
    })

    restricted_locals = dict(variables)

    try:
        byte_code = compile_restricted(code, filename="<string>", mode="exec")
        exec(byte_code, restricted_globals, restricted_locals)

        result = {
            k: v for k, v in restricted_locals.items()
            if not k.startswith("_") and k != "__builtins__"
        }
        return_dict["result"] = result

    except Exception as e:
        return_dict["error"] = str(e)

def execute_python(code: str, timeout: int = 2) -> dict:
    """Execute Python code in a sandboxed environment with timeout.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary containing either:
        - Execution results as key-value pairs of variables
        - Error message if execution fails or times out
        
    Example:
        >>> result = execute_python("x = 5; y = x * 2")
        >>> print(result)
        {'x': 5, 'y': 10}
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    process = multiprocessing.Process(
        target=_run_sandbox, args=(code, {}, return_dict)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        return {"error": "Execution timed out."}

    return dict(return_dict)
