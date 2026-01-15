from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from goedels_poetry.parsers.util import (
    _ast_to_code,
    _get_named_subgoal_ast,
    _get_named_subgoal_rewritten_ast,
    _get_sorry_holes_by_name,
    _get_unproven_subgoal_names,
)


class AST:
    """
    Class representing Lean code's abstract syntax tree (AST).
    """

    def __init__(
        self,
        ast: dict[str, Any],
        sorries: list[dict[str, Any]] | None = None,
        *,
        source_text: str | None = None,
        body_start: int = 0,
    ):
        """
        Constructs an AST using the AST dict[str, Any] representation provided by the Kimin server.

        Parameters
        ----------
        ast: dict[str, Any]
            The AST representation provided by the Kimin server.
        sorries: Optional[list[dict[str, Any]]]
            Optional list of sorry entries from check response containing goal context with type information.

        Raises
        ------
        ValueError
            If the AST structure is invalid.
        """
        from goedels_poetry.parsers.util import _validate_ast_structure

        # Validate AST structure (will raise ValueError if invalid)
        _validate_ast_structure(ast, raise_on_error=True)

        self._ast: dict[str, Any] = ast
        self._sorries: list[dict[str, Any]] = sorries or []
        self._source_text: str | None = source_text
        self._body_start: int = max(0, int(body_start))
        self._byte_prefix: list[int] | None = None

    @dataclass(frozen=True)
    class Token:
        start: int
        end: int
        val: str
        synthetic: bool

    def get_ast(self) -> dict[str, Any] | list[Any]:
        """
        Returns the AST representation.

        Returns
        -------
        dict
            Representation of the AST.
        """
        return self._ast

    def get_source_text(self) -> str | None:
        """
        Returns the exact Lean source text that produced this AST (as parsed by Kimina), if known.
        """
        return self._source_text

    def _ensure_byte_prefix(self) -> list[int]:
        source = self._source_text
        if source is None:
            return []
        if self._byte_prefix is None:
            prefix: list[int] = [0]
            for ch in source:
                prefix.append(prefix[-1] + len(ch.encode("utf-8")))
            self._byte_prefix = prefix
        return self._byte_prefix

    def _byte_to_char_index(self, byte_off: int) -> int:
        source = self._source_text
        if source is None:
            return byte_off
        prefix = self._ensure_byte_prefix()
        if not prefix:
            return byte_off
        i = bisect.bisect_right(prefix, byte_off) - 1
        if i < 0:
            return 0
        if i > len(source):
            return len(source)
        return i

    def byte_to_char_index(self, byte_off: int) -> int:
        return self._byte_to_char_index(byte_off)

    def _translate_span_to_body(self, start_b: int, end_b: int) -> tuple[int, int]:
        start_c = self._byte_to_char_index(start_b)
        end_c = self._byte_to_char_index(end_b)
        return start_c - self._body_start, end_c - self._body_start

    def translate_span_to_body(self, start_b: int, end_b: int) -> tuple[int, int]:
        return self._translate_span_to_body(start_b, end_b)

    @staticmethod
    def _is_synthetic_info(info: dict[str, Any] | None) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(info.get("synthetic") or info.get("canonical"))

    def _iter_tokens(self, node: dict[str, Any] | list[Any]) -> Iterable[Token]:
        if isinstance(node, dict):
            info = node.get("info")
            if "val" in node and isinstance(info, dict):
                pos = info.get("pos")
                if isinstance(pos, list) and len(pos) == 2:
                    try:
                        start_b = int(pos[0])
                        end_b = int(pos[1])
                    except Exception:
                        start_b = None
                        end_b = None
                    if start_b is not None and end_b is not None:
                        val = node.get("val")
                        if isinstance(val, str):
                            yield self.Token(
                                start=start_b,
                                end=end_b,
                                val=val,
                                synthetic=self._is_synthetic_info(info),
                            )
            for val in node.values():
                yield from self._iter_tokens(val)
        elif isinstance(node, list):
            for item in node:
                yield from self._iter_tokens(item)

    def get_tokens(self, node: dict[str, Any] | list[Any]) -> list[Token]:
        return list(self._iter_tokens(node))

    def get_token_span(
        self, node: dict[str, Any] | list[Any], *, include_synthetic: bool = False
    ) -> tuple[int, int] | None:
        tokens = self.get_tokens(node)
        if not tokens:
            return None
        if not include_synthetic:
            tokens = [tok for tok in tokens if not tok.synthetic]
            if not tokens:
                return None
        start = min(tok.start for tok in tokens)
        end = max(tok.end for tok in tokens)
        return start, end

    def get_body_start(self) -> int:
        """
        Returns the character offset within `source_text` that corresponds to the start of the "body"
        region (i.e., after the preamble), if provided by the caller.

        For sketches/proofs we often parse `preamble + "\\n\\n" + body` for Kimina, but we want to
        do reconstruction in the body-only coordinate system. `body_start` lets callers translate.
        """
        return self._body_start

    def get_body_text(self) -> str | None:
        """
        Return the exact body text slice from the original source text, if available.

        This preserves the exact character layout used by Kimina parsing, which is required
        for stable offset-based reconstruction.
        """
        if self._source_text is None:
            return None
        start = max(0, int(self._body_start))
        return self._source_text[start:]

    def set_source_text(self, source_text: str | None, *, body_start: int | None = None) -> None:
        """
        Set/replace the source text metadata for this AST.
        """
        self._source_text = source_text
        if body_start is not None:
            self._body_start = max(0, int(body_start))

    def get_sorries(self) -> list[dict[str, Any]]:
        """
        Returns any Kimina `sorries` metadata that was attached to this AST.
        """
        return list(self._sorries)

    def get_unproven_subgoal_names(self) -> list[str]:
        """
        Returns a list of all unproven subgoals, i.e. soory proved subgoals.

        Returns
        -------
        list[str]
            List of unproven subgoals.
        """
        results: dict[str | None, list[str]] = {}
        _get_unproven_subgoal_names(self._ast, {}, results)
        return [name for names in list(results.values()) for name in names]

    def get_named_subgoal_ast(self, subgoal_name: str) -> dict | None:
        """
        Gets the AST of the named subgoal.

        Parameters
        ----------
        subgoal_name: str
            The name of the subgoal to retrive the AST for.

        Returns
        -------
        dict
            The AST of the named subgoal.
        """
        return _get_named_subgoal_ast(self._ast, subgoal_name)

    def get_named_subgoal_code(self, subgoal_name: str) -> str:
        """
        Gets the Lean code of the named subgoal.


        Parameters
        ----------
        subgoal_name: str
            The name of the subgoal to retrive the lean code for.

        Returns
        -------
        str
            The Lean code of the named subgoal.
        """
        rewritten_subgoal_ast = _get_named_subgoal_rewritten_ast(self._ast, subgoal_name, self._sorries)
        return str(_ast_to_code(rewritten_subgoal_ast))

    def get_sorry_holes_by_name(self) -> dict[str, list[tuple[int, int]]]:
        """
        Returns a mapping from subgoal-name -> (start, end) span (body-relative) for each `sorry`
        placeholder found in this AST.

        Subgoal names include:
        - Named `have` holes (e.g. `hv'`)
        - Synthetic anonymous-have names (e.g. `gp_anon_have__<decl>__1`)
        - The special marker `"<main body>"` for a standalone main-body sorry.
        """
        holes = _get_sorry_holes_by_name(self._ast)

        # Kimina/ast_export positions are byte offsets in UTF-8, while Python string slicing is
        # codepoint-based. For sketches containing unicode identifiers (e.g. `h₁`), byte offsets
        # differ from character indices. Convert byte offsets -> character indices using the
        # stored source text.
        translated: dict[str, list[tuple[int, int]]] = {}

        if self._source_text is None:
            # Fallback: assume offsets are already character indices.
            for name, spans in holes.items():
                translated[name] = [(start - self._body_start, end - self._body_start) for start, end in spans]
            return translated

        for name, spans in holes.items():
            out_spans: list[tuple[int, int]] = []
            for start_b, end_b in spans:
                out_spans.append(self._translate_span_to_body(start_b, end_b))
            translated[name] = out_spans

        return translated
