"""Kind normalization and checking utilities."""

from .constants import __DECL_KIND_ALIASES


def __normalize_kind(kind: str | None) -> str:
    if not isinstance(kind, str):
        return ""
    return __DECL_KIND_ALIASES.get(kind, kind)


def __is_decl_command_kind(kind: str | None) -> bool:
    k = __normalize_kind(kind)
    return k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}


def __is_theorem_or_lemma_kind(kind: str | None) -> bool:
    k = __normalize_kind(kind)
    return k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}


def __sanitize_lean_ident_fragment(s: str) -> str:
    """
    Best-effort sanitizer to keep synthetic identifiers readable and safe.

    Lean identifiers can be unicode-rich, but for synthetic names we keep to a conservative
    subset to avoid surprising tokenization issues in downstream text processing.
    """
    if not isinstance(s, str) or not s:
        return "unknown"
    # Keep ASCII letters/digits/underscore; replace everything else with '_'
    return "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in s)
