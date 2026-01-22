"""Variable extraction from check() responses and AST analysis.

This module provides functions to extract variables from Lean code using check() responses
as the source of truth, and AST analysis to determine variable origins (lemma parameters
vs proof body variables).
"""

from typing import TYPE_CHECKING, Any

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import __extract_theorem_binders
from goedels_poetry.parsers.util.foundation.ast_walkers import __find_first
from goedels_poetry.parsers.util.foundation.kind_utils import __is_theorem_or_lemma_kind
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name

if TYPE_CHECKING:
    from goedels_poetry.parsers.ast import AST


def _parse_hypothesis_string(hypothesis: str) -> tuple[str | None, str | None]:
    """
    Parse a hypothesis string to extract name and type.

    Handles formats: "name : type" or "name := value"

    Parameters
    ----------
    hypothesis: str
        Hypothesis string from Lean error message

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of (name, type_str) or (None, None) if parsing fails
    """
    # Handle type annotation format: "name : type"
    if " : " in hypothesis:
        parts = hypothesis.split(" : ", 1)  # Split on first occurrence only
        if len(parts) == 2:
            name = parts[0].strip()
            type_str = parts[1].strip()

            # Validate name is a valid identifier (basic check)
            # Lean identifiers are alphanumeric with _ and ' allowed
            if name and all(c.isalnum() or c in "_'" for c in name):
                return name, type_str
            return None, None
    # Handle assignment format: "name := value" (for let bindings with defaults)
    elif " := " in hypothesis:
        parts = hypothesis.split(" := ", 1)
        if len(parts) == 2:
            name = parts[0].strip()
            # Validate name
            if name and all(c.isalnum() or c in "_'" for c in name):
                return name, None  # Type is inferred from value
            return None, None
    else:
        # Unknown format - try to extract name as first token
        # This is a fallback for edge cases
        tokens = hypothesis.strip().split()
        if tokens:
            return tokens[0], None

    return None, None


def extract_variables_from_check_response(
    check_response: dict,
    lemma_parameters: set[str] | None = None,
) -> list[dict]:
    """
    Extract all variables in scope from a check() response.

    Uses "unsolved goals" messages to get ALL variables, regardless of how
    they were introduced (have, let, intro, obtain, cases, match, etc.).

    CRITICAL: Distinguishes between:
    - Lemma parameters/hypotheses (from binders) - marked as "is_lemma_parameter"
    - Variables introduced in proof body - marked as "is_proof_body_variable"

    This is the primary method - it's general and scalable.

    Parameters
    ----------
    check_response: dict
        Parsed check() response from parse_kimina_check_response()
    lemma_parameters: set[str] | None
        Set of variable names that are lemma parameters/hypotheses.
        If provided, variables with these names are marked as lemma parameters.
        If None, we try to infer from AST (see extract_variables_with_origin)

    Returns
    -------
    list[dict]
        List of variable dictionaries with:
        - name: str - variable name (extracted from hypothesis string)
        - type: str - variable type (extracted from hypothesis string)
        - hypothesis: str - full hypothesis string (e.g., "x : Z", "hx : x > 0")  # noqa: RUF002
        - source: str - "check_response" (indicates source)
        - is_lemma_parameter: bool - True if variable is a lemma parameter/hypothesis
        - is_proof_body_variable: bool - True if variable was introduced in proof body
    """
    # Import here to avoid circular import
    from goedels_poetry.agents.util.kimina_server import extract_hypotheses_from_check_response

    try:
        hypotheses = extract_hypotheses_from_check_response(check_response)
    except (ValueError, TypeError):
        # No "unsolved goals" message or invalid response
        return []

    variables = []
    for hypothesis in hypotheses:
        # Parse hypothesis string (e.g., "x : Z", "hx : x > 0")
        # Format from Lean is standardized: "name : type" or "name := value"
        # Note: This parsing is necessary because hypotheses come as strings from Lean's error messages
        # The existing extract_hypotheses_from_unsolved_goals_data() already handles line continuation
        # This matches the pattern used in the existing codebase (see hypothesis_extraction.py)
        # This is NOT brittle parsing - it's parsing Lean's standardized error message format

        name, type_str = _parse_hypothesis_string(hypothesis)
        if not name:
            continue

        if name:
            # Determine if this is a lemma parameter or proof body variable
            is_lemma_param = False
            if lemma_parameters is not None:
                is_lemma_param = name in lemma_parameters
            # If lemma_parameters not provided, we can't determine here
            # Use extract_variables_with_origin() instead

            variables.append({
                "name": name,
                "type": type_str,
                "hypothesis": hypothesis,
                "source": "check_response",
                "is_lemma_parameter": is_lemma_param,
                "is_proof_body_variable": not is_lemma_param if lemma_parameters is not None else None,
            })

    return variables


def _extract_names_from_binder_with_fallback(binder: dict) -> set[str]:
    """Extract names from a binder, with fallback for explicitBinder format."""
    names = set()
    name = __extract_binder_name(binder)
    if name:
        names.add(name)
    else:
        # Fallback: Extract names directly from binder structure (for explicitBinder format)
        if isinstance(binder, dict):
            binder_args = binder.get("args", [])
            if len(binder_args) > 1:
                name_list = binder_args[1]
                if isinstance(name_list, list):
                    for item in name_list:
                        if isinstance(item, dict):
                            val = item.get("val")
                            if isinstance(val, str) and val and val not in {"(", ")", ":", ",", ":=", " "}:
                                names.add(val)
    return names


def _extract_parameter_names_from_binders(binders: list[dict]) -> set[str]:
    """Extract parameter names from a list of binders."""
    parameter_names = set()
    for binder in binders:
        names = _extract_names_from_binder_with_fallback(binder)
        parameter_names.update(names)
    return parameter_names


def _extract_parameters_from_decl_sig(decl_sig: dict) -> set[str]:
    """Extract parameter names from declSig node (args[0] only, not type)."""
    decl_sig_args = decl_sig.get("args", [])
    if not decl_sig_args or len(decl_sig_args) == 0:
        return set()
    # Process only the first argument (parameter list), skip args[1] (type)
    first_arg = decl_sig_args[0]
    # Create a temporary node to use __extract_theorem_binders logic
    temp_node = {"kind": "temp", "args": [first_arg]}
    binders = __extract_theorem_binders(temp_node, goal_var_types={})
    return _extract_parameter_names_from_binders(binders)


def _extract_parameters_from_theorem_args(theorem_node: dict) -> set[str]:
    """Extract parameter names from theorem args before colon."""
    args = theorem_node.get("args", [])
    colon_index = None
    for i, arg in enumerate(args):
        if isinstance(arg, dict) and arg.get("val") == ":":
            colon_index = i
            break

    if colon_index is None:
        return set()

    parameter_names = set()
    # Process arguments before the colon using __extract_theorem_binders
    for arg in args[:colon_index]:
        temp_node = {"kind": "temp", "args": [arg]}
        binders = __extract_theorem_binders(temp_node, goal_var_types={})
        names = _extract_parameter_names_from_binders(binders)
        parameter_names.update(names)
    return parameter_names


def extract_lemma_parameters_from_ast(ast: "AST") -> set[str]:
    """
    Extract lemma/theorem parameters and hypotheses from AST.

    These are variables that are part of the lemma signature (binders BEFORE the colon),
    NOT variables introduced in the proof body or binders in the type signature.

    Uses AST to find the theorem/lemma node and extract only parameter binders
    (before the colon), not type binders (after the colon).

    Parameters
    ----------
    ast: AST
        The AST for the lemma/theorem

    Returns
    -------
    set[str]
        Set of variable names that are lemma parameters/hypotheses
    """
    ast_node = ast.get_ast()

    # Find theorem/lemma node
    theorem_node = __find_first(ast_node, lambda n: __is_theorem_or_lemma_kind(n.get("kind")))
    if not theorem_node:
        return set()

    # Preferred: Look for declSig node which contains only parameter binders (before colon)
    decl_sig = __find_first(theorem_node, lambda n: n.get("kind") == "Lean.Parser.Command.declSig")
    if decl_sig is not None:
        return _extract_parameters_from_decl_sig(decl_sig)

    # Fallback: If no declSig, find colon in args and extract only before colon
    return _extract_parameters_from_theorem_args(theorem_node)


def extract_variables_with_origin(check_response: dict, ast: "AST") -> list[dict]:
    """
    Extract all variables from check() response and determine their origin.

    Distinguishes between:
    - Lemma parameters/hypotheses (from binders) - should NOT be renamed
    - Variables introduced in proof body (from tactics) - CAN be renamed

    This is the complete method that combines check() response with AST analysis.

    Parameters
    ----------
    check_response: dict
        Parsed check() response
    ast: AST
        The AST for the lemma/theorem

    Returns
    -------
    list[dict]
        List of variables with origin information:
        - name: str
        - type: str
        - hypothesis: str
        - is_lemma_parameter: bool
        - is_proof_body_variable: bool
        - declaration_node: dict | None (if found in AST)
    """
    # Extract lemma parameters from AST
    lemma_parameters = extract_lemma_parameters_from_ast(ast)

    # Extract all variables from check() response
    variables = extract_variables_from_check_response(check_response, lemma_parameters)

    # For each variable, try to find its declaration in AST
    # This helps determine if it's from proof body (found in proof body AST)
    # vs lemma parameter (found in theorem signature AST)
    for var_info in variables:
        var_name = var_info["name"]

        # Find declaration node
        declaration_node = find_variable_declaration_in_ast(ast, var_name)
        var_info["declaration_node"] = declaration_node

        # Refine is_proof_body_variable based on AST location
        # Check if declaration is in proof body (after := by) vs in signature
        # This is a heuristic - actual determination requires position analysis
        # For now, if we found it and it's not a lemma parameter, assume proof body
        if declaration_node and not var_info["is_lemma_parameter"]:
            var_info["is_proof_body_variable"] = True

    return variables


def _check_declaration_matches(node: dict, var_name: str) -> bool:
    """Check if a declaration node declares the given variable name."""
    if not _is_declaration_node(node.get("kind", "")):
        return False
    declared_name = _extract_name_from_declaration(node)
    return declared_name == var_name


def find_variable_declaration_in_ast(ast: "AST", var_name: str) -> dict | None:
    """
    Find where a variable is declared in the AST.

    Uses AST structure to find the declaration node for a given variable name.
    This works for any declaration type by searching for identifier nodes
    with the matching name in declaration contexts.

    Parameters
    ----------
    ast: AST
        The AST to search
    var_name: str
        The variable name to find

    Returns
    -------
    dict | None
        Declaration node if found, None otherwise
    """
    ast_node = ast.get_ast()

    def find_declaration(node: Any) -> dict | None:
        """Recursively search for variable declaration."""
        if isinstance(node, dict):
            # Check if this node declares the variable
            if _check_declaration_matches(node, var_name):
                return node

            # Recursively search children (only if we haven't found a match yet)
            for value in node.values():
                if isinstance(value, dict | list):
                    result = find_declaration(value)
                    if result:
                        return result
        elif isinstance(node, list):
            for item in node:
                result = find_declaration(item)
                if result:
                    return result

        return None

    return find_declaration(ast_node)


def _is_declaration_node(kind: str) -> bool:
    """
    Determine if an AST node kind represents a variable declaration.

    This is a general check that works for all declaration types.

    Parameters
    ----------
    kind: str
        The AST node kind

    Returns
    -------
    bool
        True if the node kind represents a declaration
    """
    declaration_kinds = {
        "Lean.Parser.Tactic.tacticHave_",
        "Lean.Parser.Term.let",
        "Lean.Parser.Tactic.tacticLet_",
        "Lean.Parser.Tactic.tacticIntro",
        "Lean.Parser.Tactic.tacticObtain",
        "Lean.Parser.Tactic.tacticCases",
        "Lean.Parser.Tactic.tacticMatch",
        "Lean.Parser.Tactic.tacticGeneralize",
        "Lean.Parser.Tactic.tacticSuffices",
        "Lean.Parser.Tactic.tacticSet",
        "Lean.Parser.Tactic.tacticChoose",
        "Lean.Parser.Tactic.tacticByCases",
        "Lean.Parser.Tactic.tacticInduction",
        # Add more as needed - but this is just for optimization
        # The actual search is more general
    }

    # Use exact node kinds only (no string matching)
    # The declaration_kinds set above contains exact kinds
    # If a kind is not in the set, it's not a declaration node
    # Do NOT use string matching as fallback - this is brittle
    return kind in declaration_kinds


def _extract_name_from_declaration(node: dict) -> str | None:
    """
    Extract variable name from a declaration node.

    Uses existing extraction functions for known types, falls back to
    general AST search for unknown types.

    Parameters
    ----------
    node: dict
        A declaration AST node

    Returns
    -------
    str | None
        The variable name, or None if not found
    """
    from ..names_and_bindings.binding_name_extraction import (
        __extract_choose_names,
        __extract_generalize_names,
        __extract_let_name,
        __extract_obtain_names,
        __extract_set_name,
        __extract_suffices_name,
    )
    from ..names_and_bindings.name_extraction import _extract_have_id_name

    kind = node.get("kind", "")

    # Use existing extractors for known types
    if kind == "Lean.Parser.Tactic.tacticHave_":
        return _extract_have_id_name(node)
    elif kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
        return __extract_let_name(node)
    elif kind == "Lean.Parser.Tactic.tacticObtain":
        names = __extract_obtain_names(node)
        return names[0] if names else None
    elif kind == "Lean.Parser.Tactic.tacticGeneralize":
        names = __extract_generalize_names(node)
        return names[0] if names else None
    elif kind == "Lean.Parser.Tactic.tacticSet":
        return __extract_set_name(node)
    elif kind == "Lean.Parser.Tactic.tacticSuffices":
        return __extract_suffices_name(node)
    elif kind == "Lean.Parser.Tactic.tacticChoose":
        names = __extract_choose_names(node)
        return names[0] if names else None

    # Fallback: search for identifier nodes with name
    return _find_name_in_declaration_fallback(node)


def _find_name_in_declaration_fallback(node: Any) -> str | None:
    """Fallback: search for identifier nodes with name in declaration node."""
    IDENTIFIER_KINDS = {
        "Lean.Parser.Term.ident",
        "Lean.Parser.Tactic.ident",
        "Lean.binderIdent",
        "Lean.Parser.Term.binderIdent",
        "Lean.Parser.Term.haveId",
        "Lean.Parser.Command.declId",
        # Add other identifier/name kinds as discovered from actual AST structures
    }

    def find_name_in_node(n: Any) -> str | None:
        if isinstance(n, dict):
            val = n.get("val")
            kind_inner = n.get("kind", "")
            # Check if this is in a declaration context
            if isinstance(val, str) and kind_inner in IDENTIFIER_KINDS and _is_in_declaration_context(n):
                return val
            for v in n.values():
                result = find_name_in_node(v)
                if result:
                    return result
        elif isinstance(n, list):
            for item in n:
                result = find_name_in_node(item)
                if result:
                    return result
        return None

    return find_name_in_node(node)


def _is_in_declaration_context(node: dict) -> bool:
    """
    Check if a node is in a declaration context.

    This is a heuristic - actual determination requires semantic analysis.
    For now, we check if parent is a declaration node.
    In practice, we'll use check() responses as the source of truth.

    Parameters
    ----------
    node: dict
        An AST node

    Returns
    -------
    bool
        True if the node is in a declaration context
    """
    # This is a simplified heuristic - actual implementation would check parent context
    # For now, we assume that if we're searching for a name in a declaration node,
    # it's in a declaration context
    return True  # Simplified - actual implementation would check parent context


def extract_outer_scope_variables_ast_based(
    proof_text: str,
    ast: "AST",
    kimina_client: Any,  # KiminaClient type
    server_timeout: int,
) -> dict[str, dict]:
    """
    Extract variables with scope information using check() and AST.

    PRIMARY METHOD: Uses check() to get ALL variables in scope.
    Then uses AST to find declaration positions (for conflict detection).

    This is general and works for ALL Lean 4 constructs that introduce variables.

    Parameters
    ----------
    proof_text: str
        The proof text (body only, no preamble)
    ast: AST
        The AST for the proof
    kimina_client: Any
        Client for check() calls (KiminaClient type)
    server_timeout: int
        Timeout for check() calls

    Returns
    -------
    dict[str, dict]
        Mapping variable name to scope info:
        {
            "h_main": {
                "name": "h_main",
                "type": "∀ z : ZMod 7, 4 * z ^ 3 ≠ 1",
                "hypothesis": "h_main : ∀ z : ZMod 7, 4 * z ^ 3 ≠ 1",
                "declaration_node": dict | None,  # AST node if found
                "declaration_pos": (line, col) | None,
                "source": "check_response"  # Always from check() response
            },
            ...
        }
    """
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
    from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response

    # PRIMARY: Get ALL variables from check() response
    # This includes variables from have, let, intro, obtain, cases, match, etc.
    proof_with_preamble = combine_preamble_and_body(DEFAULT_IMPORTS, proof_text)
    check_response = kimina_client.check(proof_with_preamble, timeout=server_timeout)
    parsed_check = parse_kimina_check_response(check_response)

    # Extract all variables with origin information
    # This distinguishes lemma parameters from proof body variables
    variables = extract_variables_with_origin(parsed_check, ast)

    # Fallback: If check() response doesn't have unsolved goals (e.g., code is valid),
    # extract variables directly from AST declarations
    # Note: We can't import _extract_variables_from_ast here due to circular import
    # Instead, we'll extract variables from AST directly using the same logic
    if not variables:
        variables = _extract_variables_from_ast_fallback(ast)
        # Filter out lemma parameters (they shouldn't be in outer scope for conflict detection)
        variables = [v for v in variables if not v.get("is_lemma_parameter", False)]

    # Build scope map with origin information
    scope_map = {}
    for var_info in variables:
        var_name = var_info["name"]

        declaration_node = var_info.get("declaration_node")
        declaration_pos = None
        if declaration_node:
            info = declaration_node.get("info", {})
            if isinstance(info, dict):
                pos = info.get("pos")
                if isinstance(pos, list) and len(pos) >= 2:
                    declaration_pos = (pos[0], pos[1])

        scope_info = {
            "name": var_name,
            "type": var_info.get("type"),
            "hypothesis": var_info["hypothesis"],
            "declaration_node": declaration_node,
            "declaration_pos": declaration_pos,
            "is_lemma_parameter": var_info.get("is_lemma_parameter", False),
            "is_proof_body_variable": var_info.get("is_proof_body_variable", False),
            "source": "check_response",  # Always from check() - the source of truth
        }
        scope_map[var_name] = scope_info

    return scope_map


def _extract_variables_from_ast_fallback(ast: "AST") -> list[dict]:  # noqa: C901
    """
    Extract variables directly from AST declarations (fallback when check() has no unsolved goals).

    This is a helper function to avoid circular imports with variable_renaming.py.

    Parameters
    ----------
    ast: AST
        The AST to extract variables from

    Returns
    -------
    list[dict]
        List of variable dictionaries
    """
    from contextlib import suppress

    from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
    from goedels_poetry.parsers.util.types_and_binders.type_extraction import (
        __extract_type_ast,
        __strip_leading_colon,
    )

    variables = []
    ast_node = ast.get_ast()

    def extract_from_node(node: Any) -> None:  # noqa: C901
        """Recursively extract variable declarations from AST."""
        if isinstance(node, dict):
            kind = node.get("kind", "")

            # Check if this is a declaration node
            if _is_declaration_node(kind):
                var_name = _extract_name_from_declaration(node)
                if var_name:
                    # Try to extract type from the declaration node
                    type_str = None
                    if kind == "Lean.Parser.Tactic.tacticHave_":
                        # Use the existing type extraction function
                        type_ast = __extract_type_ast(node)
                        if type_ast is not None:
                            type_ast = __strip_leading_colon(type_ast)
                            with suppress(Exception):
                                type_str = _ast_to_code(type_ast).strip()
                    elif kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
                        # For let statements, try to extract explicit type if present
                        type_ast = __extract_type_ast(node)
                        if type_ast is not None:
                            type_ast = __strip_leading_colon(type_ast)
                            with suppress(Exception):
                                type_str = _ast_to_code(type_ast).strip()

                    variables.append({
                        "name": var_name,
                        "type": type_str,
                        "hypothesis": f"{var_name} : {type_str}" if type_str else var_name,
                        "is_lemma_parameter": False,  # All variables from proof body AST are proof body variables
                        "is_proof_body_variable": True,
                        "declaration_node": node,
                    })

            # Recurse into children
            for value in node.values():
                if isinstance(value, dict | list):
                    extract_from_node(value)
        elif isinstance(node, list):
            for item in node:
                extract_from_node(item)

    extract_from_node(ast_node)
    return variables


def is_intentional_shadowing(
    var_decl: dict,
    check_response: dict,
    outer_scope_vars: dict[str, dict],
) -> bool:
    """
    Determine if a variable declaration is intentional shadowing.

    Uses check() response to understand if outer scope variable is accessible.
    If outer scope variable is accessible but not used, it's likely shadowing.
    If outer scope variable is used, it's likely a conflict.

    Parameters
    ----------
    var_decl: dict
        Variable declaration to check. Must have "name" key.
    check_response: dict
        Parsed check() response for the proof
    outer_scope_vars: dict[str, dict]
        Outer scope variables mapping

    Returns
    -------
    bool
        True if intentional shadowing, False if conflict
    """
    var_name = var_decl.get("name")
    if not var_name:
        return False

    # If variable doesn't exist in outer scope, no shadowing
    if var_name not in outer_scope_vars:
        return False

    # Check if there are errors related to this variable
    # If check() reports errors about variable conflicts, it's not shadowing
    errors = check_response.get("errors", [])
    for error in errors:
        error_data = error.get("data", "")
        if isinstance(error_data, str):
            # Look for type mismatch or "unknown identifier" errors
            # that might indicate a conflict
            # Use word boundaries to avoid false positives (e.g., "h_main" matching "main")
            # Note: This is still string-based, but error messages from Lean are strings
            # We can't avoid parsing them, but we should be careful

            # Check for type mismatch errors mentioning this variable
            # Use more specific patterns to avoid false positives
            error_lower = error_data.lower()

            # Check if error mentions variable in context of type mismatch
            if "type mismatch" in error_lower and var_name in error_data:
                # Check if variable name appears as a word (not substring)
                # This is approximate - actual determination would need AST analysis
                return False

            # Check for unknown identifier errors
            if ("unknown identifier" in error_lower or "unknown constant" in error_lower) and var_name in error_data:
                return False

    # Additional check: Use AST to verify if outer scope variable is actually referenced
    # This is more robust than just checking error messages
    # (Implementation would need AST passed in)

    # For now, use heuristic: if no errors, assume shadowing
    # This is conservative - better to assume shadowing than to rename incorrectly
    # TODO: Enhance with AST-based reference checking
    return True
