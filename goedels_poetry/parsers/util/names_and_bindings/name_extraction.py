"""Extract names from AST nodes (declarations, have statements, binders)."""

from typing import Any

from ..foundation.ast_walkers import __find_first
from ..foundation.kind_utils import __is_theorem_or_lemma_kind, __normalize_kind
from ..foundation.nested_extraction import _extract_nested_value


def _extract_decl_id_name(node: dict[str, Any]) -> str | None:
    """
    Extract the name from a Lean.Parser.Command.declId node.

    Structure (based on Lean parser grammar):
    - theorem/lemma node: {"kind": "Lean.Parser.Command.theorem", "args": [..., declId, ...]}
    - declId: {"kind": "Lean.Parser.Command.declId", "args": [name_node, ...]}
    - name_node: {"val": "theorem_name", "info": {...}} (from Syntax.ident or Syntax.atom)

    Origin: Based on Lean's parser grammar where declId is the first argument after
    the theorem/lemma keyword, and the name is the first argument of declId.

    Parameters
    ----------
    node: dict[str, Any]
        A theorem or lemma node

    Returns
    -------
    Optional[str]
        The theorem/lemma name, or None if not found
    """
    # Kimina AST can represent declarations in multiple shapes.
    # Robust approach:
    # - locate the first declId node anywhere in this subtree
    # - extract the first non-empty `val` string within that declId
    decl_id = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Command.declId")
    if not decl_id:
        return None
    val_node = __find_first(decl_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if val_node is None:
        return None
    val = val_node.get("val")
    return str(val) if val is not None else None


def _extract_have_id_name(node: dict[str, Any]) -> str | None:
    """
    Extract the name from a Lean.Parser.Tactic.tacticHave_ node.

    Structure (based on Lean parser grammar):
    - tacticHave_: {"kind": "Lean.Parser.Tactic.tacticHave_", "args": [..., haveDecl, ...]}
    - haveDecl: {"kind": "Lean.Parser.Term.haveDecl", "args": [haveIdDecl, ...]}
    - haveIdDecl: {"kind": "Lean.Parser.Term.haveIdDecl", "args": [haveId, ...]}
    - haveId: {"kind": "Lean.Parser.Term.haveId", "args": [name_node, ...]}
    - name_node: {"val": "have_name", "info": {...}} (from Syntax.ident)

    Origin: Based on Lean's parser grammar where:
    - haveDecl is the second argument of tacticHave_ (args[1])
    - haveIdDecl is the first argument of haveDecl (args[0])
    - haveId is the first argument of haveIdDecl (args[0])
    - name is the first argument of haveId (args[0])

    This creates the path: args[1] -> args[0] -> args[0] -> args[0] -> val

    Parameters
    ----------
    node: dict[str, Any]
        A tacticHave_ node

    Returns
    -------
    Optional[str]
        The have statement name, or None if not found
    """
    # Structure: node["args"][1] = haveDecl
    #           haveDecl["args"][0] = haveIdDecl
    #           haveIdDecl["args"][0] = haveId
    #           haveId["args"][0] = name_node
    #           name_node["val"] = name
    # Based on Lean parser: have haveIdDecl : type := proof
    name_node = _extract_nested_value(node, ["args", 1, "args", 0, "args", 0, "args", 0])
    if isinstance(name_node, dict):
        val = name_node.get("val")
        if not isinstance(val, str):
            return None
        # Kimina sometimes represents anonymous haves as a placeholder identifier "[anonymous]".
        # Treat these as truly anonymous so we can assign stable synthetic names.
        #
        # We also treat `have _ : ...` as anonymous (non-referable) for decomposition purposes.
        val = val.strip()
        if not val or val in {"[anonymous]", "_"}:
            return None
        return val
    return None


def _context_after_decl(node: dict[str, Any], context: dict[str, str | None]) -> dict[str, str | None]:
    """
    Update context after encountering a theorem or lemma declaration.

    Structure documented in _extract_decl_id_name().
    """
    kind = __normalize_kind(node.get("kind"))
    if __is_theorem_or_lemma_kind(kind):
        name = _extract_decl_id_name(node)
        if name:
            return {"theorem": name, "have": None}
    return context


def _context_after_have(node: dict[str, Any], context: dict[str, str | None]) -> dict[str, str | None]:
    """
    Update context after encountering a have statement.

    Structure documented in _extract_have_id_name().
    """
    if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
        have_name = _extract_have_id_name(node)
        if have_name:
            return {**context, "have": have_name}
    return context


def __extract_binder_name(binder: dict) -> str | None:
    """
    Extract the variable name from a binder AST node.

    Returns None if the name cannot be extracted, with debug logging for failures.
    """
    import logging

    if not isinstance(binder, dict):
        logging.debug("__extract_binder_name: binder is not a dict")
        return None

    # Look for binderIdent node
    binder_ident = __find_first(binder, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"})
    if not binder_ident:
        logging.debug("__extract_binder_name: Could not find binderIdent in binder")
        return None

    name_node = __find_first(binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if not name_node:
        logging.debug("__extract_binder_name: Could not find val node with non-empty string in binderIdent")
        return None

    val = name_node.get("val")
    if val is None:
        logging.debug("__extract_binder_name: val node exists but val is None")
        return None

    return str(val)
