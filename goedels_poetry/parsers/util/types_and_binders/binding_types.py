"""Determine types for bindings and handle set/let as equality."""

import logging
from typing import Any

from ..foundation.ast_walkers import __find_first
from ..foundation.goal_context import __parse_goal_context
from ..names_and_bindings.binding_name_extraction import (
    __extract_set_name,
    __extract_set_with_hypothesis_name,
)
from ..names_and_bindings.binding_value_extraction import (
    __extract_let_value,
    __extract_set_value,
)
from .binder_construction import (
    __generate_equality_hypothesis_name,
    __make_binder,
    __make_binder_from_type_string,
    __make_equality_binder,
)
from .type_extraction import __extract_type_ast


def __construct_set_with_hypothesis_type(set_node: dict, hypothesis_name: str) -> dict | None:  # noqa: C901
    """
    Construct the type AST for a set_with_hypothesis binding from the set statement.

    For `set S := Finset.range 10000 with hS`, the hypothesis `hS` has type `S = Finset.range 10000`.
    This function constructs that equality type from the set statement AST.

    Parameters
    ----------
    set_node: dict
        The set statement AST node
    hypothesis_name: str
        The hypothesis name from the `with` clause (e.g., "hS")

    Returns
    -------
    Optional[dict]
        The equality type AST (e.g., representing "S = Finset.range 10000"), or None if construction fails.

    Notes
    -----
    The constructed type AST uses the `__equality_expr` kind, which serializes as "var_name = value".
    This can be used directly as a type_ast in `__make_binder`.

    This function handles two AST structures:
    1. Mathlib.Tactic.setTactic with Mathlib.Tactic.setArgsRest (extracts directly from setArgsRest)
    2. Lean.Parser.Tactic.tacticSet_ with Lean.Parser.Term.setDecl (uses __extract_set_name/__extract_set_value)
    """
    if not isinstance(set_node, dict):
        logging.debug("__construct_set_with_hypothesis_type: set_node is not a dict")
        return None

    # Verify the set node has a 'with' clause matching the hypothesis name
    extracted_hypothesis_name = __extract_set_with_hypothesis_name(set_node)
    if not extracted_hypothesis_name or extracted_hypothesis_name != hypothesis_name:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Hypothesis name mismatch or no 'with' clause. "
            f"Expected: {hypothesis_name}, Got: {extracted_hypothesis_name}"
        )
        return None

    var_name: str | None = None
    value_args: list = []

    # Check if this is a Mathlib.Tactic.setTactic structure (setArgsRest format)
    set_args_rest = __find_first(
        set_node,
        lambda n: n.get("kind")
        in {
            "Mathlib.Tactic.setArgsRest",
            "Lean.Parser.Tactic.setArgsRest",
            "Lean.Parser.Term.setArgsRest",
        },
    )

    if set_args_rest and isinstance(set_args_rest, dict):
        # Handle Mathlib.Tactic.setTactic structure: setArgsRest.args = [var_name, [], ":=", value, ["with", [], h]]
        sar_args = set_args_rest.get("args", [])
        if len(sar_args) >= 4:
            # Extract variable name (first arg)
            var_name_node = sar_args[0]
            if isinstance(var_name_node, dict) and var_name_node.get("val"):
                var_name = var_name_node.get("val")
            elif isinstance(var_name_node, str):
                var_name = var_name_node

            # Find ":=" token and extract value after it
            assign_idx = None
            for i, arg in enumerate(sar_args):
                if (isinstance(arg, dict) and arg.get("val") == ":=") or (isinstance(arg, str) and arg == ":="):
                    assign_idx = i
                    break

            if assign_idx is not None and assign_idx + 1 < len(sar_args):
                # Extract value tokens, stopping at "with" clause
                value_tokens = []
                for i in range(assign_idx + 1, len(sar_args)):
                    arg = sar_args[i]
                    # Stop at "with" clause (list starting with "with")
                    if (
                        isinstance(arg, list)
                        and len(arg) > 0
                        and (
                            (isinstance(arg[0], dict) and arg[0].get("val") == "with")
                            or (isinstance(arg[0], str) and arg[0] == "with")
                        )
                    ):
                        break
                    value_tokens.append(arg)
                if value_tokens:
                    value_args = value_tokens

    # If we didn't extract from setArgsRest, try using existing extraction functions
    if not var_name or not value_args:
        var_name = __extract_set_name(set_node)
        value_ast = __extract_set_value(set_node)
        if value_ast:
            value_args = value_ast.get("args", []) if value_ast.get("kind") == "__value_container" else [value_ast]

    if not var_name:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Could not extract variable name from set statement "
            f"for hypothesis '{hypothesis_name}'"
        )
        return None

    if not value_args:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Could not extract value from set statement "
            f"for hypothesis '{hypothesis_name}' (variable: {var_name})"
        )
        return None

    # Construct the equality type AST: var_name = value
    # This uses the same structure as __equality_expr for consistency
    var_node = {"val": var_name, "info": {"leading": "", "trailing": " "}}
    eq_node = {"val": "=", "info": {"leading": " ", "trailing": " "}}

    equality_type_ast = {
        "kind": "__equality_expr",
        "args": [var_node, eq_node, *value_args],
    }

    return equality_type_ast


def __determine_general_binding_type(
    binding_name: str,
    binding_type: str,
    binding_node: dict,
    goal_var_types: dict[str, str],
) -> dict:
    """
    Determine the type for a general binding (have, obtain, choose, generalize, match, suffices).

    Uses a fallback chain appropriate for each binding type:
    - have/suffices: goal context → AST extraction → Prop
    - obtain/choose/generalize/match: goal context → Prop (types not in AST)

    Parameters
    ----------
    binding_name: str
        The name of the binding
    binding_type: str
        The type of binding ("have", "obtain", "choose", "generalize", "match", "suffices")
    binding_node: dict
        The AST node for the binding
    goal_var_types: dict[str, str]
        Dictionary mapping variable names to their types from goal context

    Returns
    -------
    dict
        A binder AST node with the determined type, or Prop if all methods fail
    """
    # Binding types that have types in AST (can extract from AST)
    ast_extractable_types = {"have", "suffices"}

    # Binding types that rely solely on goal context (types inferred, not in AST)
    goal_context_only_types = {"obtain", "choose", "generalize", "match"}

    # Try goal context first (most accurate for all binding types)
    if binding_name in goal_var_types:
        logging.debug(
            f"__determine_general_binding_type: Found type for {binding_type} '{binding_name}' in goal context"
        )
        return __make_binder_from_type_string(binding_name, goal_var_types[binding_name])

    # For have and suffices, try AST extraction as fallback
    if binding_type in ast_extractable_types:
        logging.debug(
            f"__determine_general_binding_type: Goal context unavailable for {binding_type} '{binding_name}', "
            "trying AST extraction"
        )
        binding_type_ast = __extract_type_ast(binding_node, binding_name=binding_name)
        if binding_type_ast is not None:
            logging.debug(
                f"__determine_general_binding_type: Successfully extracted type from AST for {binding_type} '{binding_name}'"
            )
            return __make_binder(binding_name, binding_type_ast)
        else:
            logging.warning(
                f"Could not determine type for {binding_type} binding '{binding_name}': "
                "goal context unavailable and AST extraction failed, using Prop"
            )
            return __make_binder(binding_name, None)

    # For obtain, choose, generalize, match: types must come from goal context
    if binding_type in goal_context_only_types:
        logging.warning(
            f"Could not determine type for {binding_type} binding '{binding_name}': "
            "types are inferred and not in AST, goal context unavailable, using Prop"
        )
        return __make_binder(binding_name, None)

    # Unknown binding type (shouldn't happen, but handle gracefully)
    logging.warning(
        f"Could not determine type for binding '{binding_name}' (unknown type '{binding_type}'): using Prop as fallback"
    )
    return __make_binder(binding_name, None)


def __get_binding_type_from_node(node: dict | None) -> str | None:
    """
    Determine if a node represents a set or let binding.
    Returns "set", "let", or None.
    """
    if not isinstance(node, dict):
        return None
    kind = node.get("kind", "")
    if kind == "Lean.Parser.Tactic.tacticSet_":
        return "set"
    if kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
        return "let"
    return None


def __handle_set_let_binding_as_equality(
    var_name: str,
    binding_type: str,
    binding_node: dict,
    existing_names: set[str],
    variables_in_equality_hypotheses: set[str],
    goal_var_types: dict[str, str] | None = None,
    sorries: list[dict[str, Any]] | None = None,
) -> tuple[dict | None, bool]:
    """
    Handle a set or let binding by creating an equality hypothesis.

    Parameters
    ----------
    var_name: str
        The variable name from the binding (e.g., "l", "s")
    binding_type: str
        Either "set" or "let"
    binding_node: dict
        The AST node for the binding
    existing_names: set[str]
        Set of names that already exist (for conflict resolution)
    variables_in_equality_hypotheses: set[str]
        Set to track variables already handled as equality hypotheses
    goal_var_types: Optional[dict[str, str]]
        Optional dictionary mapping variable names to their types from goal context.
        Used as fallback when value extraction fails.
    sorries: Optional[list[dict[str, Any]]]
        Optional list of sorries for defensive goal context parsing when goal_var_types is empty.

    Returns
    -------
    tuple[Optional[dict], bool]
        A tuple of (binder, was_handled):
        - binder: The equality hypothesis binder if successful, None if all fallbacks failed
        - was_handled: True if an equality hypothesis was created, False if all attempts failed
    """
    # Extract the value expression from the binding
    # Pass binding_name to ensure we extract from the correct binding if multiple exist
    value_ast = None
    if binding_type == "let":
        value_ast = __extract_let_value(binding_node, binding_name=var_name)
    elif binding_type == "set":
        value_ast = __extract_set_value(binding_node, binding_name=var_name)

    if value_ast is not None:
        # Generate hypothesis name (e.g., "hl" for "l"), avoiding conflicts
        hypothesis_name = __generate_equality_hypothesis_name(var_name, existing_names)
        # Add the generated hypothesis name to existing_names to avoid future conflicts
        existing_names.add(hypothesis_name)
        # Create equality binder: (hl : l = value)
        binder = __make_equality_binder(hypothesis_name, var_name, value_ast)
        # Track that this variable is included as an equality hypothesis
        variables_in_equality_hypotheses.add(var_name)
        return (binder, True)

    # Value extraction failed - try fallback strategies
    # Fallback 1: Try to extract type from AST and create a type annotation instead
    # This is better than nothing - at least we have the type information
    binding_type_ast = __extract_type_ast(binding_node, binding_name=var_name)
    if binding_type_ast is not None:
        # We have type information, create a type annotation binder
        # This is not ideal (we wanted an equality), but better than skipping
        logging.debug(
            f"__handle_set_let_binding_as_equality: Value extraction failed for {binding_type} '{var_name}', "
            "but type extraction succeeded, using type annotation as fallback"
        )
        binder = __make_binder(var_name, binding_type_ast)
        variables_in_equality_hypotheses.add(var_name)  # Still track it
        return (binder, True)

    # Fallback 2: Try to use goal context types if available
    if goal_var_types and var_name in goal_var_types:
        logging.debug(
            f"__handle_set_let_binding_as_equality: Value and type extraction failed for {binding_type} '{var_name}', "
            "but found type in goal context, using as fallback"
        )
        binder = __make_binder_from_type_string(var_name, goal_var_types[var_name])
        variables_in_equality_hypotheses.add(var_name)  # Still track it
        return (binder, True)

    # Fallback 3: Defensive goal context parsing - if goal_var_types is empty but sorries exist, try parsing directly
    if (not goal_var_types or var_name not in goal_var_types) and sorries:
        logging.debug(
            f"__handle_set_let_binding_as_equality: goal_var_types is empty or missing '{var_name}', "
            "trying defensive goal context parsing from sorries"
        )
        for sorry in sorries:
            goal = sorry.get("goal", "")
            if goal and var_name in goal:
                parsed_types = __parse_goal_context(goal)
                if var_name in parsed_types:
                    logging.debug(
                        f"__handle_set_let_binding_as_equality: Found '{var_name}' in defensive goal context parsing, "
                        f"type: {parsed_types[var_name]}"
                    )
                    binder = __make_binder_from_type_string(var_name, parsed_types[var_name])
                    variables_in_equality_hypotheses.add(var_name)
                    return (binder, True)

    # All fallbacks exhausted - return failure
    logging.debug(
        f"__handle_set_let_binding_as_equality: All extraction attempts failed for {binding_type} '{var_name}' "
        "(value extraction, type extraction, goal context, and defensive parsing all failed)"
    )
    return (None, False)
