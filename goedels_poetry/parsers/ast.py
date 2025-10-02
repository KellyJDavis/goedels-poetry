import json

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

    def __init__(self, ast: str):
        """
        Constructs an AST using the AST string representation provided by the Kimin server.

        Parameters
        ----------
        ast: str
            The AST string representation provided by the Kimin server.
        """
        self._ast = json.loads(ast, strict=False)

    def get_ast(self) -> dict:
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
        results = {}
        _get_unproven_subgoal_names(self._ast, {}, results)
        return [name for names in list(results.values()) for name in names]

    def get_named_subgoal_ast(self, subgoal_name: str) -> dict:
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
        rewritten_subgoal_ast = _get_named_subgoal_rewritten_ast(self._ast, subgoal_name)
        return _ast_to_code(rewritten_subgoal_ast)
