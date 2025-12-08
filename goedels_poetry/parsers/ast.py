from __future__ import annotations

from typing import Any

from goedels_poetry.parsers.util import (
    _ast_to_code,
    _get_named_subgoal_ast,
    _get_named_subgoal_rewritten_ast,
    _get_unproven_subgoal_names,
)


class AST:
    """
    Class representing Lean code's abstract syntax tree (AST).
    """

    def __init__(self, ast: dict[str, Any], sorries: list[dict[str, Any]] | None = None):
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

    def get_ast(self) -> dict[str, Any] | list[Any]:
        """
        Returns the AST representation.

        Returns
        -------
        dict
            Representation of the AST.
        """
        return self._ast

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
