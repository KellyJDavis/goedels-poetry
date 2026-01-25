"""Rename conflicting variables using AST and check() calls.

Uses AST for structure, check() for scoping, preserves qualified names.
"""

import copy
import uuid
from typing import TYPE_CHECKING, Any

from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
    extract_variables_with_origin,
    is_intentional_shadowing,
)
from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code

if TYPE_CHECKING:
    from kimina_client import KiminaClient

    from goedels_poetry.parsers.ast import AST


def _identify_conflicts(
    proof_variables: list[dict],
    outer_scope_vars: dict[str, dict],
    parsed_check: dict,
) -> list[dict]:
    """
    Identify conflicts (variables that exist in outer scope AND are not shadowed).

    CRITICAL: Only consider proof body variables for renaming, NOT lemma parameters.

    Strategy: Compare types - if outer scope variable has different type than proof body
    variable, it's likely a conflict (not shadowing). Also check for errors in check() response.

    Parameters
    ----------
    proof_variables: list[dict]
        Variables extracted from check() response with origin information
    outer_scope_vars: dict[str, dict]
        Outer scope variables mapping
    parsed_check: dict
        Parsed check() response

    Returns
    -------
    list[dict]
        List of conflicting variable declarations
    """
    conflicts = []
    outer_scope_names = set(outer_scope_vars.keys())

    for var_info in proof_variables:
        var_name = var_info["name"]

        # Skip lemma parameters - these are part of the signature and should NOT be renamed
        if var_info.get("is_lemma_parameter", False):
            continue

        # Only consider proof body variables for conflict detection
        # If variable is in outer scope and is NOT a lemma parameter, it's a potential conflict
        # We consider it a proof body variable if it's not a lemma parameter
        is_lemma_param = var_info.get("is_lemma_parameter", False)

        # Check if variable is in outer scope - if so, it's a potential conflict
        # We don't require is_proof_body_variable to be True, because if it's not a lemma parameter
        # and it's in the proof body's check() response, it's a proof body variable
        if var_name in outer_scope_names and not is_lemma_param:
            # Check if types are different - if so, it's likely a conflict
            outer_var = outer_scope_vars[var_name]
            outer_type = outer_var.get("type") or ""
            proof_type = var_info.get("type") or ""

            # If types are different and both are non-empty, it's a conflict (not shadowing)
            # Shadowing typically means same name, same or compatible type
            # Empty types might indicate inferred types, so we can't compare them
            is_type_conflict = bool(outer_type and proof_type and outer_type != proof_type)

            var_decl = {
                "name": var_name,
                "node": var_info.get("declaration_node"),
                "hypothesis": var_info["hypothesis"],
                "is_lemma_parameter": False,
                "is_proof_body_variable": True,
            }

            # Check if this is intentional shadowing
            # If types are different, it's a conflict regardless of error messages
            if is_type_conflict or not is_intentional_shadowing(var_decl, parsed_check, outer_scope_vars):
                conflicts.append(var_decl)

    return conflicts


def _create_renaming_map(conflicts: list[dict], hole_name: str) -> dict[str, str]:
    """
    Create renaming map from conflicts.

    Parameters
    ----------
    conflicts: list[dict]
        List of conflicting variable declarations
    hole_name: str
        Name of the hole (for generating unique names)

    Returns
    -------
    dict[str, str]
        Mapping from old name to new name
    """
    rename_map = {}
    for var_decl in conflicts:
        old_name = var_decl["name"]
        new_name = f"{old_name}_{hole_name}_{uuid.uuid4().hex[:8]}"
        rename_map[old_name] = new_name
    return rename_map


def _extract_renamed_body(renamed_ast: "AST") -> str:
    """
    Extract body from renamed AST.

    Parameters
    ----------
    renamed_ast: AST
        AST with renamed variables

    Returns
    -------
    str
        Extracted body text
    """
    # Import here to avoid circular import
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS

    # Convert back to code
    renamed_code = _ast_to_code(renamed_ast.get_ast())

    # Extract body using AST body_start (not string manipulation)
    body_start = renamed_ast.get_body_start()
    if body_start is not None:
        source_text = renamed_ast.get_source_text()
        if source_text and body_start < len(source_text):
            renamed_code = source_text[body_start:].strip()
        else:
            # Fallback: use converted code as-is
            renamed_code = renamed_code.strip()
    else:
        # Fallback: try to remove preamble if present
        # This is a last resort - should not happen if AST is correct
        if renamed_code.startswith(DEFAULT_IMPORTS):
            renamed_code = renamed_code[len(DEFAULT_IMPORTS) :].lstrip()
        else:
            renamed_code = renamed_code.strip()

    return renamed_code


def rename_conflicting_variables_ast_based(  # noqa: C901
    proof_body: str,
    outer_scope_vars: dict[str, dict],
    kimina_client: "KiminaClient",
    server_timeout: int,
    hole_name: str,
) -> str:
    """
    Rename conflicting variables using AST and check() calls.

    Strategy:
    1. Parse proof_body to get AST
    2. Use check() to get scope information
    3. Identify actual conflicts (not shadowing)
    4. Rename only conflicting declarations
    5. Update references using AST (preserve qualified names)

    Parameters
    ----------
    proof_body: str
        The proof body to rename variables in
    outer_scope_vars: dict[str, dict]
        Outer scope variables mapping
    kimina_client: KiminaClient
        Client for AST and check() calls
    server_timeout: int
        Timeout for calls
    hole_name: str
        Name of the hole (for generating unique names)

    Returns
    -------
    str
        Proof body with renamed variables
    """
    # Import here to avoid circular import
    from goedels_poetry.agents.util.common import (
        DEFAULT_IMPORTS,
        combine_preamble_and_body,
        remove_default_imports_from_ast,
    )
    from goedels_poetry.agents.util.kimina_server import (
        parse_kimina_ast_code_response,
        parse_kimina_check_response,
    )
    from goedels_poetry.parsers.ast import AST

    # Parse proof body
    # CRITICAL: Proof body might not be valid standalone Lean code (e.g., "have x : ℕ := 1\n  sorry")  # noqa: RUF003
    # We need to wrap it in a temporary lemma for parsing and variable extraction
    # This ensures the AST contains the declarations we need
    # IMPORTANT: Indent all non-empty lines of the proof body so they're part of the `by` block
    # Preserve existing relative indentation by finding the minimum indentation and normalizing
    lines = proof_body.split("\n")
    if lines:
        # Normalize: remove existing indentation, then add 2 spaces for the `by` block
        indented_lines = []
        for line in lines:
            if line.strip():
                # Remove existing indentation, add 2 spaces
                indented_lines.append("  " + line.lstrip())
            else:
                indented_lines.append(line)
        indented_body = "\n".join(indented_lines)
    else:
        indented_body = proof_body
    temp_lemma = f"lemma _temp_ : True := by\n{indented_body}"
    temp_full = combine_preamble_and_body(DEFAULT_IMPORTS, temp_lemma)
    ast_response = kimina_client.ast_code(temp_full, timeout=server_timeout)
    parsed_ast = parse_kimina_ast_code_response(ast_response)

    if parsed_ast.get("error") or not parsed_ast.get("ast"):
        # Fallback: return as-is if parsing fails
        return proof_body

    # Create AST from the temporary lemma
    temp_ast_without = remove_default_imports_from_ast(parsed_ast["ast"], preamble=DEFAULT_IMPORTS)
    temp_ast = AST(
        temp_ast_without,
        sorries=parsed_ast.get("sorries"),
        source_text=temp_full,
        body_start=len(DEFAULT_IMPORTS.strip()) + 2,
    )

    # Get check response for the temporary lemma
    check_response = kimina_client.check(temp_full, timeout=server_timeout)
    parsed_check = parse_kimina_check_response(check_response)

    # Extract ALL variables from check() response with origin information
    # This distinguishes lemma parameters from proof body variables
    proof_variables = extract_variables_with_origin(parsed_check, temp_ast)

    # Fallback: If check() response doesn't have unsolved goals, extract variables directly from AST
    if not proof_variables:
        proof_variables = _extract_variables_from_ast(temp_ast)

    # Filter out the temporary lemma parameter (if any)
    # We only want variables from the proof body
    proof_variables = [v for v in proof_variables if v.get("name") != "_temp_"]

    # Identify conflicts
    # Note: We check conflicts by comparing types - if outer scope variable has different type
    # than proof body variable, it's likely a conflict (not shadowing)
    conflicts = _identify_conflicts(proof_variables, outer_scope_vars, parsed_check)

    # Debug: Log conflicts for troubleshooting
    if conflicts:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Found {len(conflicts)} conflicts: {[c['name'] for c in conflicts]}")

    # If no conflicts, return as-is
    if not conflicts:
        return proof_body

    # Create renaming map
    rename_map = _create_renaming_map(conflicts, hole_name)

    # Apply renaming using AST (preserves qualified names, doesn't break shadowing)
    # Use the temporary lemma AST for renaming
    renamed_ast = _rename_variables_in_ast(temp_ast, rename_map)

    # Extract the proof body from the renamed lemma
    # Use _extract_renamed_body to get the full lemma code, then extract just the body part
    renamed_code = _extract_renamed_body(renamed_ast)

    # Extract body part (after "by")
    # The renamed_code will be the full lemma, we need to extract the body
    if renamed_code and "by" in renamed_code:
        # Find the "by" and extract everything after it
        by_index = renamed_code.find("by")
        if by_index != -1:
            # Skip "by" and any whitespace after it
            body_start = by_index + 2
            while body_start < len(renamed_code) and renamed_code[body_start] in " \n":
                body_start += 1
            renamed_code = renamed_code[body_start:].strip()

    return renamed_code if renamed_code else proof_body


def _is_qualified_name(node: dict) -> bool:
    """Check if node represents a qualified name."""
    kind = node.get("kind", "")
    val = node.get("val", "")
    return "." in str(val) or kind in {"Lean.Parser.Term.field", "Lean.Parser.Term.proj"}


def _should_rename_identifier(
    node: dict, rename_map: dict[str, str], parent_is_qualified: bool, in_declaration: bool
) -> bool:
    """Determine if an identifier node should be renamed."""
    if parent_is_qualified:
        return False

    val = node.get("val")
    if not isinstance(val, str) or val not in rename_map:
        return False

    kind = node.get("kind", "")
    IDENTIFIER_KINDS = {
        "Lean.Parser.Term.ident",
        "Lean.Parser.Tactic.ident",
        "Lean.binderIdent",
        "Lean.Parser.Term.binderIdent",
    }

    # If node has a val that matches rename_map and is not qualified,
    # it's likely an identifier node even if kind is empty or not in IDENTIFIER_KINDS
    # (some identifier nodes don't have explicit kinds in the AST)
    # We check kind first for safety, but allow nodes with val if they're in declaration or reference context
    if kind and kind not in IDENTIFIER_KINDS:
        # If kind is explicitly set and not an identifier kind, don't rename
        # (e.g., keywords, operators, etc.)
        return False

    # If kind is empty or is an identifier kind, check context
    return in_declaration or _is_reference_context(node, kind)


def _is_declaration_kind(kind: str) -> bool:
    """Check if node kind is a declaration."""
    DECLARATION_KINDS = {
        "Lean.Parser.Tactic.tacticHave_",
        "Lean.Parser.Term.let",
        "Lean.Parser.Tactic.tacticLet_",
        "Lean.Parser.Tactic.tacticIntro",
        "Lean.Parser.Tactic.tacticObtain",
    }
    return kind in DECLARATION_KINDS


def _rename_in_node_recursive(
    node: Any, rename_map: dict[str, str], in_declaration: bool = False, parent_is_qualified: bool = False
) -> None:
    """Recursively rename variables in AST node."""
    if isinstance(node, dict):
        kind = node.get("kind", "")

        # Check for qualified names
        is_qualified = _is_qualified_name(node)
        if is_qualified:
            parent_is_qualified = True

        # Check if this is a variable reference (identifier node) that should be renamed
        if _should_rename_identifier(node, rename_map, parent_is_qualified, in_declaration):
            val = node.get("val")
            if isinstance(val, str):
                node["val"] = rename_map[val]

        # Process children based on whether this is a declaration
        is_decl = _is_declaration_kind(kind)
        for value in node.values():
            if isinstance(value, dict | list):
                _rename_in_node_recursive(value, rename_map, in_declaration=is_decl, parent_is_qualified=is_qualified)
    elif isinstance(node, list):
        for item in node:
            _rename_in_node_recursive(item, rename_map, in_declaration, parent_is_qualified)


def _rename_variables_in_ast(ast: "AST", rename_map: dict[str, str]) -> "AST":
    """
    Rename variables in AST according to rename_map.

    Preserves qualified names and doesn't break shadowing.
    Only renames the specific declarations and their references.

    Uses AST structure to identify:
    - Declaration nodes (where variable is declared)
    - Reference nodes (where variable is used)
    - Qualified names (should not be renamed)
    """
    ast_node = ast.get_ast()
    renamed_node = copy.deepcopy(ast_node)

    _rename_in_node_recursive(renamed_node, rename_map)

    # Create new AST with renamed node
    # Note: source_text and body_start might need updating after renaming
    # For now, we'll regenerate source_text by converting AST back to code
    # This ensures consistency
    # Ensure renamed_node is a dict (not a list) for AST constructor
    # Import here to avoid circular import
    from goedels_poetry.parsers.ast import AST

    if isinstance(renamed_node, dict):
        return AST(
            renamed_node,
            sorries=ast.get_sorries(),
            source_text=None,  # Will be regenerated from AST
            body_start=ast.get_body_start(),  # Should remain the same
        )
    else:
        # Fallback: if renamed_node is a list, wrap it or return original AST
        # This should not happen in practice, but handle it defensively
        return ast


def _extract_variables_from_ast(ast: "AST") -> list[dict]:  # noqa: C901
    """
    Extract variables directly from AST declarations when check() response doesn't provide them.

    This is a fallback when check() response has no "unsolved goals" (e.g., code is valid).

    Parameters
    ----------
    ast: AST
        The AST to extract variables from

    Returns
    -------
    list[dict]
        List of variable dictionaries with:
        - name: str
        - type: str | None (extracted from AST)
        - hypothesis: str (constructed from name and type)
        - is_lemma_parameter: bool (False for proof body variables)
        - is_proof_body_variable: bool (True for all extracted variables)
        - declaration_node: dict (the declaration node)
    """
    from contextlib import suppress

    from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
        _extract_name_from_declaration,
        _is_declaration_node,
    )
    from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code

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
                    # Use existing type extraction function for have statements
                    type_str = None
                    if kind == "Lean.Parser.Tactic.tacticHave_":
                        # Use the existing type extraction function
                        from goedels_poetry.parsers.util.high_level.subgoal_extraction_v2 import __strip_leading_colon
                        from goedels_poetry.parsers.util.types_and_binders.type_extraction import __extract_type_ast

                        type_ast = __extract_type_ast(node)
                        if type_ast is not None:
                            type_ast = __strip_leading_colon(type_ast)
                            with suppress(Exception):
                                type_str = _ast_to_code(type_ast).strip()
                    elif kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
                        # For let statements, type is optional (can be inferred)
                        # Try to extract explicit type if present
                        from goedels_poetry.parsers.util.high_level.subgoal_extraction_v2 import __strip_leading_colon
                        from goedels_poetry.parsers.util.types_and_binders.type_extraction import __extract_type_ast

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


def _is_reference_context(node: dict, node_kind: str) -> bool:
    """
    Determine if node is in a reference context (not declaration).

    Uses AST structure, not heuristics.
    """
    # Declaration nodes are identified by their kind
    # If we're not in a declaration node, we're in a reference context
    DECLARATION_KINDS = {
        "Lean.Parser.Tactic.tacticHave_",
        "Lean.Parser.Term.let",
        "Lean.Parser.Tactic.tacticLet_",
        # Add other declaration kinds
    }

    return node_kind not in DECLARATION_KINDS
