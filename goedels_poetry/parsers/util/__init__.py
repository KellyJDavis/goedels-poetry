"""
Parser utilities module.

This module provides utilities for working with Lean AST (Abstract Syntax Tree) structures.
It is organized into subdirectories by functional area:
- foundation: Low-level utilities and constants
- names_and_bindings: Name and binding extraction
- types_and_binders: Type extraction and binder construction
- collection_and_analysis: Declaration collection and analysis
- high_level: High-level operations

All public API functions are re-exported here for backward compatibility.
"""

# Re-export all public API functions (single underscore prefix)
from .foundation import Node, _ast_to_code, _extract_nested_value, _validate_ast_structure
from .foundation.goal_context import __parse_goal_context
from .high_level import (
    _get_named_subgoal_ast,
    _get_named_subgoal_rewritten_ast,
    _get_sorry_holes_by_name,
    _get_unproven_subgoal_names,
    _record_sorry,
)
from .names_and_bindings import (
    _context_after_decl,
    _context_after_have,
    _extract_decl_id_name,
    _extract_have_id_name,
)
from .names_and_bindings.binding_name_extraction import __extract_set_with_hypothesis_name
from .names_and_bindings.binding_value_extraction import __extract_let_value, __extract_set_value
from .types_and_binders.binding_types import (
    __construct_set_with_hypothesis_type,
    __determine_general_binding_type,
    __handle_set_let_binding_as_equality,
)
from .types_and_binders.type_extraction import __extract_type_ast

__all__ = [
    # Type alias
    "Node",
    "__construct_set_with_hypothesis_type",
    "__determine_general_binding_type",
    "__extract_let_value",
    "__extract_set_value",
    "__extract_set_with_hypothesis_name",
    "__extract_type_ast",
    "__handle_set_let_binding_as_equality",
    # Private functions (double underscore prefix) - re-exported for test compatibility
    "__parse_goal_context",
    "_ast_to_code",
    "_context_after_decl",
    "_context_after_have",
    "_extract_decl_id_name",
    "_extract_have_id_name",
    "_extract_nested_value",
    "_get_named_subgoal_ast",
    "_get_named_subgoal_rewritten_ast",
    "_get_sorry_holes_by_name",
    "_get_unproven_subgoal_names",
    "_record_sorry",
    # Public API functions (single underscore prefix)
    "_validate_ast_structure",
]
