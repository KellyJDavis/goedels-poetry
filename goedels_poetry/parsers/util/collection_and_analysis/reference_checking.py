"""Check references and containment in AST."""

from ..foundation.constants import Node
from ..foundation.kind_utils import __is_theorem_or_lemma_kind, __normalize_kind
from ..names_and_bindings.name_extraction import _extract_decl_id_name, _extract_have_id_name


def __is_referenced_in(subtree: Node, name: str) -> bool:
    """
    Check if a variable name is referenced in the given subtree.
    """
    if isinstance(subtree, dict):
        # Check if this node has a val that matches the name
        if subtree.get("val") == name:
            # Make sure it's not a binding occurrence
            kind = subtree.get("kind", "")
            if kind not in {
                "Lean.Parser.Term.haveId",
                "Lean.Parser.Command.declId",
                "Lean.binderIdent",
                "Lean.Parser.Term.binderIdent",
            }:
                return True
        # Recurse into children
        for v in subtree.values():
            if __is_referenced_in(v, name):
                return True
    elif isinstance(subtree, list):
        for item in subtree:
            if __is_referenced_in(item, name):
                return True
    return False


def __contains_target_name(node: Node, target_name: str, name_map: dict[str, dict]) -> bool:
    """
    Check if the given node contains the target by name.
    Uses name_map to check if target is defined within this node.
    """
    if isinstance(node, dict):
        # Check various node types that might contain the target
        kind = node.get("kind", "")
        if kind == "Lean.Parser.Tactic.tacticHave_":
            # Structure documented in _extract_have_id_name()
            have_name = _extract_have_id_name(node)
            if have_name == target_name:
                return True
        # Recurse into children
        for v in node.values():
            if __contains_target_name(v, target_name, name_map):
                return True
    elif isinstance(node, list):
        for item in node:
            if __contains_target_name(item, target_name, name_map):
                return True
    return False


def __find_enclosing_theorem(  # noqa: C901
    ast: Node, target_name: str, anon_have_by_id: dict[int, str] | None = None
) -> dict | None:
    """
    Find the theorem/lemma that encloses the given target (typically a have statement).
    Returns the theorem/lemma node if found, None otherwise.
    """

    def contains_target(node: Node) -> bool:  # noqa: C901
        """Check if the given node contains the target by name."""
        if isinstance(node, dict):
            # Check for theorem/lemma names
            kind = __normalize_kind(node.get("kind", ""))
            if __is_theorem_or_lemma_kind(kind):
                # Structure documented in _extract_decl_id_name()
                name = _extract_decl_id_name(node)
                if name == target_name:
                    return True
            # Check for have statement names
            # Structure documented in _extract_have_id_name()
            if kind == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(node)
                if have_name == target_name:
                    return True
                if (not have_name) and anon_have_by_id is not None:
                    synthetic = anon_have_by_id.get(id(node))
                    if synthetic == target_name:
                        return True
            # Recurse into children
            for v in node.values():
                if contains_target(v):
                    return True
        elif isinstance(node, list):
            for item in node:
                if contains_target(item):
                    return True
        return False

    if isinstance(ast, dict):
        kind = __normalize_kind(ast.get("kind", ""))
        # If this is a theorem/lemma and it contains the target, return it
        if __is_theorem_or_lemma_kind(kind) and contains_target(ast):
            return ast
        # Otherwise, recurse into children
        for v in ast.values():
            result = __find_enclosing_theorem(v, target_name, anon_have_by_id)
            if result is not None:
                return result
    elif isinstance(ast, list):
        for item in ast:
            result = __find_enclosing_theorem(item, target_name, anon_have_by_id)
            if result is not None:
                return result
    return None
