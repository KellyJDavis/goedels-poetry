"""New subgoal extraction using check responses from theorem portions."""

from typing import Any

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


def _find_proof_recursive(node: Any) -> dict | None:
    """
    Recursively search for byTactic or tacticSeq node in AST structure.

    Parameters
    ----------
    node: Any
        AST node to search (dict, list, or other)

    Returns
    -------
    dict | None
        The proof node (byTactic or tacticSeq), or None if not found
    """
    if isinstance(node, dict):
        kind = node.get("kind", "")
        if kind in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
            return node
        for v in node.values():
            result = _find_proof_recursive(v)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = _find_proof_recursive(item)
            if result:
                return result
    return None


def _find_proof_in_have_decl(have_decl: dict) -> dict | None:
    """
    Find proof node in haveDecl structure by looking for ":=" followed by proof node.

    Parameters
    ----------
    have_decl: dict
        A Lean.Parser.Term.haveDecl node

    Returns
    -------
    dict | None
        The proof node (byTactic or tacticSeq), or None if not found
    """
    decl_args = have_decl.get("args", [])
    seen_assign = False

    for arg in decl_args:
        # Check if this is the ":=" token
        if isinstance(arg, dict) and arg.get("val") == ":=":
            seen_assign = True
            continue

        # After ":=", look for proof node
        if seen_assign and isinstance(arg, dict):
            kind = arg.get("kind", "")
            if kind in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
                return arg

    return None


def _extract_proof_node_from_have(have_node: dict) -> dict | None:
    """
    Extract the proof node (byTactic or tacticSeq) from a have statement AST node.

    Uses AST structure analysis, not regex.

    Structure: tacticHave_ -> args[1] = haveDecl -> args contains [haveIdDecl, type, ":=", byTactic/tacticSeq]

    Parameters
    ----------
    have_node: dict
        A Lean.Parser.Tactic.tacticHave_ node

    Returns
    -------
    dict | None
        The proof node (byTactic or tacticSeq), or None if not found
    """
    if not isinstance(have_node, dict):
        return None

    if have_node.get("kind") != "Lean.Parser.Tactic.tacticHave_":
        return None

    # Structure: tacticHave_ -> args[1] = haveDecl
    have_args = have_node.get("args", [])
    if len(have_args) < 2:
        return None

    # Get haveDecl (second argument)
    have_decl = have_args[1]
    if isinstance(have_decl, dict) and have_decl.get("kind") == "Lean.Parser.Term.haveDecl":
        # Look in haveDecl.args for ":=" followed by byTactic/tacticSeq
        proof_node = _find_proof_in_have_decl(have_decl)
        if proof_node:
            return proof_node

    # Fallback: recursive search in all args if structure is different
    for arg in have_args:
        proof_node = _find_proof_recursive(arg)
        if proof_node:
            return proof_node

    return None


def _extract_tactics_from_proof_node(proof_node: dict, ast: AST) -> str:
    """
    Extract tactics string from a proof node (byTactic or tacticSeq).

    Uses AST-to-code conversion, not regex.

    Parameters
    ----------
    proof_node: dict
        A byTactic or tacticSeq node
    ast: AST
        The AST containing source text for position tracking

    Returns
    -------
    str
        The tactics as a string (without "by" keyword)
    """
    if not isinstance(proof_node, dict):
        return "sorry"

    kind = proof_node.get("kind", "")
    if kind not in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
        return "sorry"

    # Extract tactics based on AST structure (not string manipulation)
    # byTactic structure: [{"val": "by"}, tactics_node]
    # tacticSeq structure: [tactic_nodes...]
    if kind == "Lean.Parser.Term.byTactic":
        # Extract tactics node (second argument of byTactic)
        args = proof_node.get("args", [])
        if len(args) >= 2:
            # Second arg is the tactics (first is "by" keyword)
            tactics_node = args[1]
            tactics = _ast_to_code(tactics_node)
        else:
            tactics = "sorry"
    elif kind == "Lean.Parser.Tactic.tacticSeq":
        # Already tactics, no "by" to remove
        tactics = _ast_to_code(proof_node)
    else:
        tactics = "sorry"

    if not tactics:
        return "sorry"

    tactics = tactics.strip()

    # If tactics is empty or only whitespace, return "sorry"
    if not tactics:
        return "sorry"

    return tactics


def _extract_main_body_info(ast: AST) -> tuple[str, str]:
    """
    Extract name and type for main body subgoal.

    Parameters
    ----------
    ast: AST
        The AST of the decomposed theorem

    Returns
    -------
    tuple[str, str]
        Tuple of (name, type_str)

    Raises
    ------
    ValueError
        If no enclosing theorem/lemma is found in AST.
    """
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
        # _ast_to_code preserves token trailing whitespace; strip to avoid types ending in " ... x ".
        type_str = _ast_to_code(type_ast).strip()
    else:
        # Type extraction fallback: __extract_type_ast() returns None for:
        # - match bindings: Types are inferred from pattern matching, not in AST
        # - choose/obtain/generalize bindings: Types come from goal context, not AST
        # - Malformed AST nodes or nodes where type cannot be determined
        # Using "Prop" as fallback is acceptable as a safe default
        type_str = "Prop"

    return name, type_str


def _extract_have_statement_info(ast: AST, target_subgoal_name: str) -> tuple[str, str, dict | None]:
    """
    Extract name, type, and have_node for have statement subgoal.

    Parameters
    ----------
    ast: AST
        The AST of the decomposed theorem
    target_subgoal_name: str
        Base name of the target subgoal

    Returns
    -------
    tuple[str, str, dict | None]
        Tuple of (name, type_str, have_node)

    Raises
    ------
    ValueError
        If subgoal is not found in AST.
    """
    # Find the have statement node in AST
    have_node = ast.get_named_subgoal_ast(target_subgoal_name)
    if have_node is None:
        raise ValueError(f"Subgoal '{target_subgoal_name}' not found in AST")  # noqa: TRY003

    # Extract the have statement's type from the AST node
    type_ast = __extract_type_ast(have_node)
    if type_ast is not None:
        type_ast = __strip_leading_colon(type_ast)
        # _ast_to_code preserves token trailing whitespace; strip to avoid types ending in " ... x ".
        # These trailing spaces can later produce binders like `(hx : 1 < x )` when wrapped.
        type_str = _ast_to_code(type_ast).strip()
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

    return name, type_str, have_node


def _extract_proof_body_from_have(
    have_node: dict | None, target_subgoal_name: str, main_body_name: str, ast: AST
) -> str:
    """
    Extract proof body from have statement if applicable.

    Parameters
    ----------
    have_node: dict | None
        The have statement node, or None for main body
    target_subgoal_name: str
        Base name of the target subgoal
    main_body_name: str
        The main body name constant
    ast: AST
        The AST for position tracking

    Returns
    -------
    str
        The tactics string, or "sorry" if not applicable
    """
    if (
        target_subgoal_name == main_body_name
        or not have_node
        or have_node.get("kind") != "Lean.Parser.Tactic.tacticHave_"
    ):
        return "sorry"

    # Extract proof node
    proof_node = _extract_proof_node_from_have(have_node)
    if not proof_node:
        return "sorry"

    # Extract tactics from proof node
    tactics = _extract_tactics_from_proof_node(proof_node, ast)
    # If tactics doesn't contain "sorry", add it (maintains hole for prover)
    if "sorry" not in tactics:
        tactics = tactics + "\n  sorry"

    return tactics


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

    # Extract name and type based on subgoal type
    if target_subgoal_name == MAIN_BODY_NAME:
        name, type_str = _extract_main_body_info(ast)
        have_node = None
    else:
        name, type_str, have_node = _extract_have_statement_info(ast, target_subgoal_name)

    # Find the corresponding check response
    check_response = check_responses.get(target_subgoal_identifier)
    if check_response is None:
        raise ValueError(f"Check response not found for subgoal identifier '{target_subgoal_identifier}'")  # noqa: TRY003

    # Extract hypotheses from the check response
    hypotheses = extract_hypotheses_from_check_response(check_response)

    # Prefer AST-extracted types for hypotheses that correspond to named `have` statements.
    #
    # The "unsolved goals" pretty printer often omits type ascriptions/coercions (e.g. `(30 : ℝ)`),  # noqa: RUF003
    # which can make a hypothesis like `hprod : 30 * (13 / 2) = 195` ambiguous when re-parsed as
    # a lemma binder (Lean may default it to `Nat`). By mapping such hypotheses back to their
    # original `have` types from the sketch AST, we preserve the intended types and make the
    # standalone lemma binders round-trip safely.
    rewritten: list[str] = []
    for hyp in hypotheses:
        if ":=" in hyp:
            rewritten.append(hyp)
            continue
        if ":" not in hyp:
            rewritten.append(hyp)
            continue

        lhs, _rhs = hyp.split(":", 1)
        hyp_name = lhs.strip()
        # Skip multi-name binders like "b h v : ℝ".  # noqa: RUF003
        if not hyp_name or " " in hyp_name:
            rewritten.append(hyp)
            continue

        try:
            _n, have_type_str, _have_node = _extract_have_statement_info(ast, hyp_name)
        except Exception:
            rewritten.append(hyp)
            continue

        rewritten.append(f"{hyp_name} : {have_type_str}")

    hypotheses = rewritten

    # Convert hypothesis strings to binder strings
    binder_strings = parse_hypothesis_strings_to_binders(hypotheses)

    # Format binders string: join with single space
    binders_str = " ".join(binder_strings)

    # Extract proof body from have statement
    tactics = _extract_proof_body_from_have(have_node, target_subgoal_name, MAIN_BODY_NAME, ast)

    # Construct standalone lemma with proof body
    # Format: "lemma name (binders) : type := by\n  tactics"
    lemma_code = f"lemma {name} {binders_str} : {type_str} := by\n  {tactics}"

    return lemma_code
