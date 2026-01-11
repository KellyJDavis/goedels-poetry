"""
Name and binding extraction modules.
"""

from .anonymous_haves import __collect_anonymous_haves
from .binding_name_extraction import (
    __extract_choose_names,
    __extract_generalize_names,
    __extract_let_name,
    __extract_match_names,
    __extract_match_pattern_names,
    __extract_obtain_names,
    __extract_set_name,
    __extract_set_with_hypothesis_name,
    __extract_suffices_name,
)
from .binding_value_extraction import __extract_let_value, __extract_set_value
from .name_extraction import (
    __extract_binder_name,
    _context_after_decl,
    _context_after_have,
    _extract_decl_id_name,
    _extract_have_id_name,
)

__all__ = [
    # Internal functions (exported for use within util package)
    "__collect_anonymous_haves",
    "__extract_binder_name",
    "__extract_choose_names",
    "__extract_generalize_names",
    "__extract_let_name",
    "__extract_let_value",
    "__extract_match_names",
    "__extract_match_pattern_names",
    "__extract_obtain_names",
    "__extract_set_name",
    "__extract_set_value",
    "__extract_set_with_hypothesis_name",
    "__extract_suffices_name",
    "_context_after_decl",
    "_context_after_have",
    # Public API
    "_extract_decl_id_name",
    "_extract_have_id_name",
]
