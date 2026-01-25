"""Analyze types using AST structure.

Detects function types (Pi, Forall) using AST, not string matching.
"""


def is_function_type(type_ast: dict | None) -> bool:
    """
    Determine if a type AST represents a function type.

    Uses AST structure analysis, not string matching.
    Checks for Pi types (∀) and function types (→).

    Parameters
    ----------
    type_ast: dict | None
        The type AST node

    Returns
    -------
    bool
        True if function type, False otherwise
    """
    if not isinstance(type_ast, dict):
        return False

    kind = type_ast.get("kind", "")

    # Use exact AST node kinds from Lean parser (not string matching)
    # Based on codebase: type_extraction.py uses these exact kinds
    FUNCTION_TYPE_KINDS = {
        "Lean.Parser.Term.forall",  # ∀ (Pi type)
        "Lean.Parser.Term.arrow",  # → (arrow type)
        # Add other function type kinds as discovered from actual AST structures
    }

    if kind in FUNCTION_TYPE_KINDS:
        return True

    # Recursively check nested types
    args = type_ast.get("args", [])
    return any(isinstance(arg, dict) and is_function_type(arg) for arg in args)


def is_pi_or_forall_type(type_ast: dict | None) -> bool:
    """
    Specifically check for Pi/Forall types (universal quantification).

    Parameters
    ----------
    type_ast: dict | None
        The type AST node

    Returns
    -------
    bool
        True if Pi/Forall type, False otherwise
    """
    if not isinstance(type_ast, dict):
        return False

    kind = type_ast.get("kind", "")

    # Use exact AST node kinds from Lean parser (not string matching)
    PI_OR_FORALL_KINDS = {
        "Lean.Parser.Term.forall",
        # Add other Pi/Forall kinds as discovered from actual AST structures
    }

    if kind in PI_OR_FORALL_KINDS:
        return True

    # Recursively check nested types
    args = type_ast.get("args", [])
    return any(isinstance(arg, dict) and is_pi_or_forall_type(arg) for arg in args)
