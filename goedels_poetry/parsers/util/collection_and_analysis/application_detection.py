"""Detect function applications using AST.

Finds where a subgoal is used in the parent sketch and checks if it's an application.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goedels_poetry.parsers.ast import AST


def find_subgoal_usage_in_ast(ast: "AST", hole_name: str) -> list[dict]:  # noqa: C901
    """
    Find all usages of a subgoal in the AST.

    Uses AST structure to find references to the subgoal name.
    Returns both the identifier nodes and their parent app nodes (if any).

    Parameters
    ----------
    ast: AST
        The parent sketch AST
    hole_name: str
        The subgoal name to find

    Returns
    -------
    list[dict]
        List of AST nodes where the subgoal is used (includes app nodes if usage is in application)
    """
    ast_node = ast.get_ast()
    usages = []

    def find_usages(node: Any, parent: dict | None = None, grandparent: dict | None = None) -> None:  # noqa: C901
        """Recursively find usages of hole_name."""
        if isinstance(node, dict):
            # Check if this node is an app node that might contain the subgoal
            if is_app_node(node):
                # Check if any args contain the hole_name
                args = node.get("args", [])
                for arg in args:
                    if isinstance(arg, dict):
                        arg_val = arg.get("val")
                        if isinstance(arg_val, str) and arg_val == hole_name:
                            # Found app node containing the subgoal
                            usages.append({"node": node, "parent": parent, "is_app": True})
                            break
                    elif isinstance(arg, list):
                        # Recursively check list args
                        for item in arg:
                            if isinstance(item, dict):
                                item_val = item.get("val")
                                if isinstance(item_val, str) and item_val == hole_name:
                                    usages.append({"node": node, "parent": parent, "is_app": True})
                                    break

            # Check if this node references the subgoal
            val = node.get("val")
            if isinstance(val, str) and val == hole_name:
                # Check if this is an identifier node (not a declaration)
                # Use exact AST node kinds (not string matching)
                kind = node.get("kind", "")
                IDENTIFIER_KINDS = {
                    "Lean.Parser.Term.ident",
                    "Lean.Parser.Tactic.ident",
                    "Lean.binderIdent",
                    "Lean.Parser.Term.binderIdent",
                    # Add other identifier kinds as discovered from actual AST structures
                }
                if kind in IDENTIFIER_KINDS or (not kind and val == hole_name):
                    # Check if parent is an app node
                    if parent and is_app_node(parent):
                        # Store the app node (parent) instead of just the identifier
                        usages.append({"node": parent, "parent": grandparent, "is_app": True})
                    else:
                        # Store the identifier node
                        usages.append({"node": node, "parent": parent, "is_app": False})

            # Recursively search children
            for _key, value in node.items():
                if isinstance(value, dict | list):
                    find_usages(value, parent=node, grandparent=parent)
        elif isinstance(node, list):
            for item in node:
                find_usages(item, parent=parent, grandparent=grandparent)

    find_usages(ast_node)
    # Filter out any None nodes and return unique nodes
    result: list[dict] = []
    seen_nodes: set[int] = set()  # Use id() to track seen nodes
    for usage in usages:
        node = usage.get("node")
        if isinstance(node, dict):
            node_id = id(node)
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                result.append(node)
    return result


def is_app_node(node: dict) -> bool:
    """
    Determine if an AST node is a function application.

    Uses exact AST node kinds from Lean parser, not string matching.

    Parameters
    ----------
    node: dict
        AST node to check

    Returns
    -------
    bool
        True if application node, False otherwise
    """
    if not isinstance(node, dict):
        return False

    kind = node.get("kind", "")

    # Use exact AST node kinds from Lean parser (not string matching)
    # These should be determined by examining actual AST structures
    APPLICATION_KINDS = {
        "Lean.Parser.Term.app",  # Function application: f x
        # Note: "Lean.Parser.Term.paren" is NOT always an application
        # Only include if verified to be application-related
        # Add other application-related kinds as discovered
        # Should be verified against actual Lean AST structures
    }

    if kind in APPLICATION_KINDS:
        return True

    # Additional check: if node has structure of application
    # App nodes typically have structure: [function, argument1, argument2, ...]
    # This is a structural check using exact node kinds
    args = node.get("args", [])
    if len(args) >= 2:
        # Might be an application - check if first arg is a function-like node
        first_arg = args[0] if args else None
        if isinstance(first_arg, dict):
            first_kind = first_arg.get("kind", "")
            # Use exact node kinds for function-like nodes (not string matching)
            FUNCTION_LIKE_KINDS = {
                "Lean.Parser.Term.ident",
                "Lean.Parser.Term.const",
                # Add other function-like kinds as discovered from actual AST structures
            }
            if first_kind in FUNCTION_LIKE_KINDS:
                return True

    return False
