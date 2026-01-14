"""New subgoal extraction using check responses from theorem portions."""

from goedels_poetry.agents.util.kimina_server import extract_hypotheses_from_check_response
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
from goedels_poetry.parsers.util.foundation.ast_walkers import __find_first
from goedels_poetry.parsers.util.foundation.kind_utils import __is_theorem_or_lemma_kind
from goedels_poetry.parsers.util.hypothesis_extraction import parse_hypothesis_strings_to_binders
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import (
    _extract_decl_id_name,
    _extract_have_id_name,
)
from goedels_poetry.parsers.util.types_and_binders.type_extraction import (
    __extract_type_ast,
    __strip_leading_colon,
)


def extract_subgoal_with_check_responses(
    ast: AST,
    check_responses: dict[str, dict],
    target_subgoal_identifier: str,
    target_subgoal_name: str,
) -> str:
    """
    Extract a standalone lemma from a subgoal using check responses from theorem portions.

    Parameters
    ----------
    ast: AST
        The AST of the decomposed theorem
    check_responses: dict[str, dict]
        Dictionary mapping subgoal identifiers to their parsed check responses
    target_subgoal_identifier: str
        Identifier for the target subgoal (includes ##sorry## suffix for multiple sorries)
        Used to look up the check response
    target_subgoal_name: str
        Base name of the target subgoal (without ##sorry## suffix)
        Used to look up the subgoal in the AST

    Returns
    -------
    str
        Standalone lemma code string (e.g., "lemma hn_nonneg (n : Z) (hn : n > 1) : 0 <= n := by sorry")

    Raises
    ------
    ValueError
        If the subgoal is not found in the AST or check response is missing.
    """
    MAIN_BODY_NAME = "<main body>"

    # Special handling for <main body> subgoal
    if target_subgoal_name == MAIN_BODY_NAME:
        # Find enclosing theorem/lemma (not a have statement)
        ast_node = ast.get_ast()
        theorem_node = __find_first(ast_node, lambda n: __is_theorem_or_lemma_kind(n.get("kind")))
        if theorem_node is None:
            raise ValueError("main body target found but no enclosing theorem/lemma in AST")  # noqa: TRY003

        # Extract theorem name for lemma name
        decl_name = _extract_decl_id_name(theorem_node) or "unknown_decl"
        name = f"gp_main_body__{decl_name}"

        # Extract type from theorem node
        type_ast = __extract_type_ast(theorem_node)
        if type_ast is not None:
            type_ast = __strip_leading_colon(type_ast)
            type_str = _ast_to_code(type_ast)
        else:
            # Type extraction fallback: __extract_type_ast() returns None for:
            # - match bindings: Types are inferred from pattern matching, not in AST
            # - choose/obtain/generalize bindings: Types come from goal context, not AST
            # - Malformed AST nodes or nodes where type cannot be determined
            # Using "Prop" as fallback is acceptable as a safe default
            type_str = "Prop"
    else:
        # Find the have statement node in AST
        have_node = ast.get_named_subgoal_ast(target_subgoal_name)
        if have_node is None:
            raise ValueError(f"Subgoal '{target_subgoal_name}' not found in AST")  # noqa: TRY003

        # Extract the have statement's type from the AST node
        type_ast = __extract_type_ast(have_node)
        if type_ast is not None:
            type_ast = __strip_leading_colon(type_ast)
            type_str = _ast_to_code(type_ast)
        else:
            # Type extraction fallback: __extract_type_ast() returns None for:
            # - match bindings: Types are inferred from pattern matching, not in AST
            # - choose/obtain/generalize bindings: Types come from goal context, not AST
            # - Malformed AST nodes or nodes where type cannot be determined
            # Using "Prop" as fallback is acceptable as a safe default
            type_str = "Prop"

        # Extract have statement name
        name = _extract_have_id_name(have_node) or target_subgoal_name
        # For anonymous haves with synthetic names like gp_anon_have__<decl>__<idx>,
        # use the target_subgoal_name as-is (required by reconstruct_complete_proof())

    # Find the corresponding check response
    check_response = check_responses.get(target_subgoal_identifier)
    if check_response is None:
        raise ValueError(f"Check response not found for subgoal identifier '{target_subgoal_identifier}'")  # noqa: TRY003

    # Extract hypotheses from the check response
    hypotheses = extract_hypotheses_from_check_response(check_response)

    # Convert hypothesis strings to binder strings
    binder_strings = parse_hypothesis_strings_to_binders(hypotheses)

    # Format binders string: join with single space
    binders_str = " ".join(binder_strings)

    # Construct standalone lemma code
    lemma_code = f"lemma {name} {binders_str} : {type_str} := by sorry"

    return lemma_code
