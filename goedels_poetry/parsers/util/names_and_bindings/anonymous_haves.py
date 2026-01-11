"""Anonymous have statement collection and naming."""

from typing import Any

from ..foundation.constants import __ANON_HAVE_NAME_PREFIX, Node
from ..foundation.kind_utils import (
    __is_decl_command_kind,
    __normalize_kind,
    __sanitize_lean_ident_fragment,
)
from .name_extraction import _extract_decl_id_name, _extract_have_id_name


def __collect_anonymous_haves(ast: Node) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    """
    Collect anonymous `have : ... := ...` tactic nodes and assign stable synthetic names.

    Naming scheme (1-based index per enclosing theorem/lemma/def):
      gp_anon_have__<theorem_name>__<idx>

    Returns
    -------
    (by_id, by_name):
      - by_id: maps `id(node)` to synthetic name
      - by_name: maps synthetic name to the original dict node
    """
    by_id: dict[int, str] = {}
    by_name: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}

    def rec(n: Any, current_decl: str | None) -> None:
        if isinstance(n, dict):
            k = __normalize_kind(n.get("kind", ""))
            # Track enclosing decl name for stable per-declaration numbering
            if __is_decl_command_kind(k):
                decl = _extract_decl_id_name(n) or "unknown_decl"
                current_decl = __sanitize_lean_ident_fragment(decl)
                counters.setdefault(current_decl, 0)

            if k == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(n)
                if not have_name:
                    decl_key = current_decl or "unknown_decl"
                    counters.setdefault(decl_key, 0)
                    counters[decl_key] += 1
                    synthetic = f"{__ANON_HAVE_NAME_PREFIX}{decl_key}__{counters[decl_key]}"
                    by_id[id(n)] = synthetic
                    # Only record the first occurrence if somehow duplicated (defensive)
                    by_name.setdefault(synthetic, n)

            for v in n.values():
                rec(v, current_decl)
        elif isinstance(n, list):
            for it in n:
                rec(it, current_decl)

    rec(ast, None)
    return by_id, by_name
