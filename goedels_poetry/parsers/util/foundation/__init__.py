"""
Foundation modules - low-level utilities, constants, and basic AST operations.
"""

from .ast_to_code import _ast_to_code
from .ast_validation import _validate_ast_structure
from .ast_walkers import __find_all, __find_first
from .constants import (
    __ANON_HAVE_NAME_PREFIX,
    __DECL_KIND_ALIASES,
    Node,
)
from .goal_context import __parse_goal_context, __parse_goal_context_line
from .kind_utils import (
    __is_decl_command_kind,
    __is_theorem_or_lemma_kind,
    __normalize_kind,
    __sanitize_lean_ident_fragment,
)
from .nested_extraction import _extract_nested_value

__all__ = [
    # Constants (for internal use, but exported for convenience)
    "__ANON_HAVE_NAME_PREFIX",
    "__DECL_KIND_ALIASES",
    # Types
    "Node",
    "__find_all",
    # Internal functions (exported for use within util package)
    "__find_first",
    "__is_decl_command_kind",
    "__is_theorem_or_lemma_kind",
    "__normalize_kind",
    "__parse_goal_context",
    "__parse_goal_context_line",
    "__sanitize_lean_ident_fragment",
    # Public API
    "_ast_to_code",
    "_extract_nested_value",
    "_validate_ast_structure",
]
