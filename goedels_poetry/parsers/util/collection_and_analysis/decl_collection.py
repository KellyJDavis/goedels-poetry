"""Collect named declarations and analyze dependencies."""

from typing import Any

from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node
from ..foundation.kind_utils import __is_decl_command_kind, __normalize_kind
from ..names_and_bindings.binding_name_extraction import (
    __extract_choose_names,
    __extract_generalize_names,
    __extract_let_name,
    __extract_obtain_names,
    __extract_set_name,
    __extract_suffices_name,
)
from ..names_and_bindings.name_extraction import _extract_have_id_name


def __collect_named_decls(ast: Node) -> dict[str, dict]:  # noqa: C901
    name_map: dict[str, dict] = {}

    def rec(n: Any) -> None:  # noqa: C901
        if isinstance(n, dict):
            k = __normalize_kind(n.get("kind", ""))
            # Collect theorems, lemmas, and definitions
            if __is_decl_command_kind(k):
                decl_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Command.declId")
                if decl_id:
                    val_node = __find_first(decl_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
            # Collect have statements (only if referable by a real name).
            # Note: Kimina may emit placeholder "[anonymous]" for anonymous have statements; those
            # are handled separately via synthetic `gp_anon_have__...` names.
            if k == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(n)
                if have_name:
                    name_map[have_name] = n
            # Collect let bindings
            if k in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
                let_name = __extract_let_name(n)
                if let_name:
                    name_map[let_name] = n
            # Collect obtain statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticObtain_":
                obtained_names = __extract_obtain_names(n)
                for name in obtained_names:
                    if name:
                        name_map[name] = n
            # Collect set statements
            if k == "Lean.Parser.Tactic.tacticSet_":
                set_name = __extract_set_name(n)
                if set_name:
                    name_map[set_name] = n
            # Collect suffices statements
            if k == "Lean.Parser.Tactic.tacticSuffices_":
                suffices_name = __extract_suffices_name(n)
                if suffices_name:
                    name_map[suffices_name] = n
            # Collect choose statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticChoose_":
                chosen_names = __extract_choose_names(n)
                for name in chosen_names:
                    if name:
                        name_map[name] = n
            # Collect generalize statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticGeneralize_":
                generalized_names = __extract_generalize_names(n)
                for name in generalized_names:
                    if name:
                        name_map[name] = n
            for v in n.values():
                rec(v)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(ast)
    return name_map


def __collect_defined_names(subtree: Node) -> set[str]:  # noqa: C901
    names: set[str] = set()

    def rec(n: Any) -> None:  # noqa: C901
        if isinstance(n, dict):
            k = n.get("kind", "")
            if k == "Lean.Parser.Term.haveId":
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            if k == "Lean.Parser.Command.declId":
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            if k in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            for v in n.values():
                rec(v)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(subtree)
    return names


def __find_dependencies(subtree: Node, name_map: dict[str, dict]) -> set[str]:
    defined = __collect_defined_names(subtree)
    deps: set[str] = set()

    def rec(n: Any) -> None:
        if isinstance(n, dict):
            v = n.get("val")
            if isinstance(v, str) and v in name_map and v not in defined:  # noqa: SIM102
                if n.get("kind") not in {
                    "Lean.Parser.Term.haveId",
                    "Lean.Parser.Command.declId",
                    "Lean.binderIdent",
                    "Lean.Parser.Term.binderIdent",
                }:
                    deps.add(v)
            for val in n.values():
                rec(val)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(subtree)
    return deps
