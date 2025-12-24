from __future__ import annotations

import bisect
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

    def get_body_start(self) -> int:
        """
        Returns the character offset within `source_text` that corresponds to the start of the "body"
        region (i.e., after the preamble), if provided by the caller.

        For sketches/proofs we often parse `preamble + "\\n\\n" + body` for Kimina, but we want to
        do reconstruction in the body-only coordinate system. `body_start` lets callers translate.
        """
        return self._body_start

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
        source = self._source_text
        translated: dict[str, list[tuple[int, int]]] = {}

        if source is None:
            # Fallback: assume offsets are already character indices.
            for name, spans in holes.items():
                translated[name] = [(start - self._body_start, end - self._body_start) for start, end in spans]
            return translated

        byte_prefix: list[int] = [0]
        for ch in source:
            byte_prefix.append(byte_prefix[-1] + len(ch.encode("utf-8")))

        def byte_to_char_index(byte_off: int) -> int:
            # Find i such that byte_prefix[i] <= byte_off < byte_prefix[i+1]
            # Using bisect_right gives the insertion point after any exact match.
            i = bisect.bisect_right(byte_prefix, byte_off) - 1
            if i < 0:
                return 0
            if i > len(source):
                return len(source)
            return i

        for name, spans in holes.items():
            out_spans: list[tuple[int, int]] = []
            for start_b, end_b in spans:
                start_c = byte_to_char_index(start_b)
                end_c = byte_to_char_index(end_b)
                out_spans.append((start_c - self._body_start, end_c - self._body_start))
            translated[name] = out_spans

        return translated
