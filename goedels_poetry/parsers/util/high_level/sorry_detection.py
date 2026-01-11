"""Sorry hole detection and tracking."""

from typing import Any, cast

from ..foundation.ast_validation import _validate_ast_structure
from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node
from ..names_and_bindings.anonymous_haves import __collect_anonymous_haves
from ..names_and_bindings.name_extraction import _context_after_decl, _context_after_have, _extract_have_id_name


def _record_sorry(node: dict[str, Any], context: dict[str, str | None], results: dict[str | None, list[str]]) -> None:
    if node.get("kind") == "Lean.Parser.Tactic.tacticSorry":
        theorem = context.get("theorem")
        have = context.get("have")
        results.setdefault(theorem, []).append(have or "<main body>")


def _get_unproven_subgoal_names(
    node: Node,
    context: dict[str, str | None],
    results: dict[str | None, list[str]],
    anon_have_by_id: dict[int, str] | None = None,
) -> None:
    # Initialize anonymous-have mapping once at the root call and thread it through recursion.
    if anon_have_by_id is None:
        anon_have_by_id, _anon_by_name = __collect_anonymous_haves(node)

    if isinstance(node, dict):
        context = _context_after_decl(node, context)
        # Update context for have statements; if the have is anonymous, attach its synthetic name.
        if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
            have_name = _extract_have_id_name(node)
            if have_name:
                context = {**context, "have": have_name}
            else:
                synthetic = anon_have_by_id.get(id(node)) if anon_have_by_id is not None else None
                if synthetic:
                    context = {**context, "have": synthetic}
        else:
            context = _context_after_have(node, context)

        _record_sorry(node, context, results)
        for _key, val in node.items():
            _get_unproven_subgoal_names(val, dict(context), results, anon_have_by_id)
    elif isinstance(node, list):
        for item in node:
            _get_unproven_subgoal_names(item, dict(context), results, anon_have_by_id)


def _get_sorry_holes_by_name(ast: Node) -> dict[str, list[tuple[int, int]]]:  # noqa: C901
    """
    Return a mapping of subgoal-name -> (start, end) character offsets for each `sorry` token
    occurring inside a theorem/lemma proof.

    Names match the decomposition pipeline:
    - Named `have` statements use their have-id name (e.g. `hv'`)
    - Anonymous `have : ...` statements use synthetic names (`gp_anon_have__<decl>__<idx>`)
    - Standalone `sorry` in the main body uses the special name `"<main body>"`

    Notes
    -----
    The returned offsets are for the *full* source text that Kimina parsed (i.e. including any
    preamble that was passed to the server). Callers that want body-relative offsets should
    translate these offsets accordingly.
    """
    if not _validate_ast_structure(ast, raise_on_error=False):
        raise ValueError("Invalid AST structure: AST must be a dict or list")  # noqa: TRY003

    anon_have_by_id, _anon_by_name = __collect_anonymous_haves(ast)

    holes: dict[str, list[tuple[int, int]]] = {}

    def record(name: str, span: tuple[int, int]) -> None:
        holes.setdefault(name, []).append(span)

    def find_sorry_span(node: dict[str, Any]) -> tuple[int, int] | None:
        tok = __find_first(
            node,
            lambda n: n.get("val") == "sorry"
            and isinstance(n.get("info"), dict)
            and isinstance((n.get("info") or {}).get("pos"), list)
            and len(cast(list[Any], (n.get("info") or {}).get("pos"))) == 2,
        )
        if not tok:
            return None
        pos = (tok.get("info") or {}).get("pos")
        if not isinstance(pos, list) or len(pos) != 2:
            return None
        try:
            return int(pos[0]), int(pos[1])
        except Exception:
            return None

    def rec(node: Node, context: dict[str, str | None]) -> None:
        if isinstance(node, dict):
            # Update theorem context
            context = _context_after_decl(node, context)

            # Update have context (including synthetic anonymous have names)
            if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(node)
                if have_name:
                    context = {**context, "have": have_name}
                else:
                    synthetic = anon_have_by_id.get(id(node))
                    if synthetic:
                        context = {**context, "have": synthetic}
            else:
                context = _context_after_have(node, context)

            # Record sorry holes
            if node.get("kind") == "Lean.Parser.Tactic.tacticSorry":
                have = context.get("have")
                hole_name = have or "<main body>"
                span = find_sorry_span(node)
                if span is not None:
                    record(hole_name, span)

            for val in node.values():
                rec(val, dict(context))
        elif isinstance(node, list):
            for item in node:
                rec(item, dict(context))

    rec(ast, {"theorem": None, "have": None})
    return holes
