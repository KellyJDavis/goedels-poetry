"""Named subgoal AST operations."""

import logging
from typing import Any

from ..foundation.constants import __ANON_HAVE_NAME_PREFIX, Node
from ..foundation.kind_utils import __is_theorem_or_lemma_kind, __normalize_kind
from ..names_and_bindings.anonymous_haves import __collect_anonymous_haves
from ..names_and_bindings.name_extraction import _extract_decl_id_name, _extract_have_id_name


def _get_named_subgoal_ast(node: Node, target_name: str) -> dict[str, Any] | None:  # noqa: C901
    """
    Find the sub-AST for a given theorem/lemma/have name.
    Returns the entire subtree rooted at that declaration.

    Parameters
    ----------
    node: Node
        The AST node to search
    target_name: str
        The name of the subgoal to find

    Returns
    -------
    Optional[dict[str, Any]]
        The AST of the named subgoal, or None if not found.
    """
    # Validate target_name
    if not isinstance(target_name, str) or not target_name:
        logging.warning(f"Invalid target_name: expected non-empty string, got {type(target_name).__name__}")
        return None

    if isinstance(node, dict):
        # Synthetic anonymous-have name support
        if isinstance(target_name, str) and target_name.startswith(__ANON_HAVE_NAME_PREFIX):
            _anon_by_id, anon_by_name = __collect_anonymous_haves(node)
            found = anon_by_name.get(target_name)
            if found is not None:
                return found

        kind = __normalize_kind(node.get("kind"))

        # Theorem or lemma
        # Structure documented in _extract_decl_id_name()
        if __is_theorem_or_lemma_kind(kind):
            name = _extract_decl_id_name(node)
            if name == target_name:
                return node

        # Have subgoal
        # Structure documented in _extract_have_id_name()
        if kind == "Lean.Parser.Tactic.tacticHave_":
            have_name = _extract_have_id_name(node)
            if have_name == target_name:
                return node

        # Recurse into children
        for val in node.values():
            result = _get_named_subgoal_ast(val, target_name)
            if result is not None:
                return result

    elif isinstance(node, list):
        for item in node:
            result = _get_named_subgoal_ast(item, target_name)
            if result is not None:
                return result

    return None
