"""
High-level operation modules.
"""

from .named_subgoals import _get_named_subgoal_ast
from .sorry_detection import (
    _get_sorry_holes_by_name,
    _get_unproven_subgoal_names,
    _record_sorry,
)
from .subgoal_rewriting import _get_named_subgoal_rewritten_ast

__all__ = [
    # Public API
    "_get_named_subgoal_ast",
    "_get_named_subgoal_rewritten_ast",
    "_get_sorry_holes_by_name",
    "_get_unproven_subgoal_names",
    "_record_sorry",
]
