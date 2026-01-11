"""Construct binder AST nodes."""

import logging

from .type_extraction import __strip_leading_colon


def __make_binder(name: str, type_ast: dict | None) -> dict:
    if type_ast is None:
        type_ast = {"val": "Prop", "info": {"leading": " ", "trailing": " "}}
    inner_type = __strip_leading_colon(type_ast)
    binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": name, "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            inner_type,
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    return binder


def __make_equality_binder(hypothesis_name: str, var_name: str, value_ast: dict) -> dict:
    """
    Create a binder for an equality hypothesis like (hs : s = value).

    Parameters
    ----------
    hypothesis_name: str
        The name of the hypothesis (e.g., "hs")
    var_name: str
        The name of the variable being defined (e.g., "s")
    value_ast: dict
        The AST of the value expression (should be a __value_container or similar)
    """
    # Create the equality expression: var_name = value
    # We'll create a simple structure that serializes as "var_name = value"
    # The value_ast might be a __value_container, so we extract its args
    value_args = value_ast.get("args", []) if value_ast.get("kind") == "__value_container" else [value_ast]

    # Create nodes for the equality: var_name, "=", and the value
    var_node = {"val": var_name, "info": {"leading": "", "trailing": " "}}
    eq_node = {"val": "=", "info": {"leading": " ", "trailing": " "}}

    # Create a container that will serialize as "var_name = value"
    # We use a simple structure that _ast_to_code will handle correctly
    equality_expr = {
        "kind": "__equality_expr",
        "args": [var_node, eq_node, *value_args],
    }

    binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": hypothesis_name, "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            equality_expr,
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    return binder


def __make_binder_from_type_string(name: str, type_str: str) -> dict:
    """
    Create a binder AST node from a name and type string.
    """
    # Clean common artifacts like leading ':' or extra whitespace to avoid malformed binders
    cleaned_type_str = type_str.lstrip(":").strip() if isinstance(type_str, str) else ""
    if cleaned_type_str:
        cleaned_type_str = " ".join(cleaned_type_str.split())  # normalize whitespace
    if not cleaned_type_str:
        cleaned_type_str = "Prop"
    # Create a simple type AST node from the string
    type_ast = {"val": cleaned_type_str, "info": {"leading": "", "trailing": ""}}
    return __make_binder(name, type_ast)


def __generate_equality_hypothesis_name(var_name: str, existing_names: set[str]) -> str:
    """
    Generate a hypothesis name for an equality from a variable name, avoiding conflicts.
    Examples: s -> hs, sOdd -> hsOdd, sEven -> hsEven
    If the base name conflicts, tries h2{var_name}, h3{var_name}, etc.

    Parameters
    ----------
    var_name: str
        The variable name (e.g., "s")
    existing_names: set[str]
        Set of names that already exist (binders, hypotheses, etc.)

    Returns
    -------
    str
        A unique hypothesis name (e.g., "hs", "h2s", "h3s", etc.)
    """
    base_name = f"h{var_name}"
    if base_name not in existing_names:
        return base_name

    # Try numbered variants: h2s, h3s, h4s, etc.
    counter = 2
    while True:
        candidate = f"h{counter}{var_name}"
        if candidate not in existing_names:
            return candidate
        counter += 1
        # Safety limit to avoid infinite loops
        if counter > 1000:
            logging.warning(f"Could not generate unique hypothesis name for '{var_name}' after 1000 attempts")
            return f"h{counter}{var_name}"
