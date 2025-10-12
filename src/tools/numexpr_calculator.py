"""Mathematical expression evaluation tool using numexpr.

This module provides a safe way to evaluate mathematical expressions using the numexpr library.
It supports standard arithmetic operations and common mathematical constants like pi and e.
The tool is designed to be used with OpenAI's function calling interface for mathematical
computations.

Example:
    >>> result = numexpr_calculator("37593 * 67")
    >>> print(result)
    '2518731'
"""

import math
import numexpr

NUMEXPR_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "numexpr_calculator",
        "description": "Evaluate a mathematical expression using the numexpr library.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A single-line mathematical expression to evaluate. "
                        "Supports standard arithmetic operations and constants like pi and e. "
                        "Examples: '37593 * 67', '37593**(1/5)', 'sin(pi / 2)'"
                    )
                }
            },
            "required": ["expression"]
        }
    }
}


def numexpr_calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Args:
        expression: A single-line mathematical expression to evaluate.
            Supports standard arithmetic operations and constants like pi and e.

    Returns:
        str: The evaluated result as a string.

    Examples:
        >>> numexpr_calculator("37593 * 67")
        '2518731'
        >>> numexpr_calculator("37593**(1/5)")
        '8.234567890123456'
        >>> numexpr_calculator("sin(pi / 2)")
        '1.0'
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )