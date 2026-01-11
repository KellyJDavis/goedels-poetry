"""Safe nested value extraction helpers."""

from typing import Any


def _extract_nested_value(node: dict, path: list[int | str], default: Any = None) -> Any:
    """
    Safely extract value from nested structure using a path.

    Based on Lean's Syntax AST structure from AstExport.lean:
    - Syntax nodes have structure: {"kind": kind, "args": args, "info": info}
    - Atoms/Idents have structure: {"val": val, "info": info}
    - Path can mix string keys and integer indices

    Parameters
    ----------
    node: dict
        Starting node (must be a dict)
    path: list[Union[int, str]]
        List of keys/indices to traverse, e.g., ["args", 1, "args", 0, "val"]
    default: Any
        Default value to return if path doesn't exist

    Returns
    -------
    Any
        The value at the path, or default if path doesn't exist

    Examples
    --------
    >>> node = {"args": [{"val": "test"}]}
    >>> _extract_nested_value(node, ["args", 0, "val"])
    'test'
    >>> _extract_nested_value(node, ["args", 1, "val"], "default")
    'default'
    """
    if not isinstance(node, dict):
        return default

    current = node
    for step in path:
        if isinstance(step, int):
            # Integer index - access list or dict by position
            if isinstance(current, list):
                if step < 0 or step >= len(current):
                    return default
                current = current[step]
            elif isinstance(current, dict):
                # For dicts, convert to list of values (order may vary)
                values = list(current.values())
                if step < 0 or step >= len(values):
                    return default
                current = values[step]
            else:
                return default
        else:
            # String key - access dict
            if not isinstance(current, dict) or step not in current:
                return default
            current = current[step]

    return current
