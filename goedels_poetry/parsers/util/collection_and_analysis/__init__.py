"""
Declaration collection and analysis modules.
"""

from .application_detection import find_subgoal_usage_in_ast, is_app_node
from .decl_collection import (
    __collect_defined_names,
    __collect_named_decls,
    __find_dependencies,
)
from .reference_checking import (
    __contains_target_name,
    __find_enclosing_theorem,
    __is_referenced_in,
)
from .theorem_binders import (
    __extract_theorem_binders,
    __find_earlier_bindings,
    __parse_pi_binders_from_type,
)
from .variable_extraction import (
    extract_lemma_parameters_from_ast,
    extract_outer_scope_variables_ast_based,
    extract_variables_from_check_response,
    extract_variables_with_origin,
    find_variable_declaration_in_ast,
    is_intentional_shadowing,
)
from .variable_renaming import rename_conflicting_variables_ast_based

__all__ = [
    # Internal functions (exported for use within util package)
    "__collect_defined_names",
    "__collect_named_decls",
    "__contains_target_name",
    "__extract_theorem_binders",
    "__find_dependencies",
    "__find_earlier_bindings",
    "__find_enclosing_theorem",
    "__is_referenced_in",
    "__parse_pi_binders_from_type",
    # Phase 2: Variable extraction functions
    "extract_lemma_parameters_from_ast",
    "extract_outer_scope_variables_ast_based",
    "extract_variables_from_check_response",
    "extract_variables_with_origin",
    "find_subgoal_usage_in_ast",  # Phase 4
    "find_variable_declaration_in_ast",
    "is_app_node",  # Phase 4
    "is_intentional_shadowing",
    "rename_conflicting_variables_ast_based",
]
