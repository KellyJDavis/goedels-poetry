"""Constants and type definitions for parser utilities."""

from typing import Any

Node = dict[str, Any] | list[Any]

__ANON_HAVE_NAME_PREFIX = "gp_anon_have__"

__DECL_KIND_ALIASES: dict[str, str] = {
    # Kimina can emit either fully-qualified kinds or unqualified ones for top-level commands.
    # Normalize both forms so downstream code can match reliably.
    "theorem": "Lean.Parser.Command.theorem",
    "lemma": "Lean.Parser.Command.lemma",
    "def": "Lean.Parser.Command.def",
}
