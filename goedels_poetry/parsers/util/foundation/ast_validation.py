"""AST structure validation."""

from .constants import Node


def _validate_ast_structure(ast: Node, raise_on_error: bool = False) -> bool:  # noqa: C901
    """
    Validate that the AST has a basic valid structure.

    The AST from kimina-lean-server can be:
    - A top-level dict with "header" and "commands" fields (from AstExport.lean)
    - A dict representing a single node with "kind" field
    - A list of nodes
    - Any nested combination of the above

    Parameters
    ----------
    ast: Node
        The AST node to validate
    raise_on_error: bool
        If True, raise ValueError on invalid structure. If False, return False.

    Returns
    -------
    bool
        True if the AST structure appears valid, False otherwise.

    Raises
    ------
    ValueError
        If raise_on_error is True and the AST structure is invalid.
    """
    if ast is None:
        if raise_on_error:
            raise ValueError("AST cannot be None")  # noqa: TRY003
        return False

    # AST must be a dict or list
    if not isinstance(ast, dict | list):
        if raise_on_error:
            raise TypeError(f"AST must be a dict or list, got {type(ast).__name__}")  # noqa: TRY003
        return False

    # If it's a dict, check for expected structure
    if isinstance(ast, dict):
        # Top-level AST from AstExport has "header" and/or "commands"
        if "header" in ast or "commands" in ast:
            # Validate commands if present
            if "commands" in ast:
                commands = ast["commands"]
                if not isinstance(commands, list):
                    if raise_on_error:
                        raise TypeError("AST 'commands' field must be a list")  # noqa: TRY003
                    return False
            return True

        # Node-level AST should have "kind" field (though some nodes might not)
        # We're lenient here - if it's a dict, we consider it potentially valid
        # The actual structure will be validated during traversal
        return True

    # If it's a list, validate that all elements are valid nodes
    if isinstance(ast, list):
        for item in ast:
            if not isinstance(item, dict | list):
                if raise_on_error:
                    raise TypeError(f"AST list contains invalid item type: {type(item).__name__}")  # noqa: TRY003
                return False
            # Recursively validate nested structures (with depth limit to avoid infinite recursion)
            if isinstance(item, dict) and ("header" in item or "commands" in item or "kind" in item):
                # This looks like a valid node, continue
                pass
            elif isinstance(item, dict | list):
                # Nested structure - validate recursively but limit depth
                # For now, we'll be lenient and just check it's a dict/list
                pass

    return True
