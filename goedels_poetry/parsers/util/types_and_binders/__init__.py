"""
Type extraction and binder construction modules.
"""

from .binder_construction import (
    __generate_equality_hypothesis_name,
    __make_binder,
    __make_binder_from_type_string,
    __make_equality_binder,
)
from .binding_types import (
    __construct_set_with_hypothesis_type,
    __determine_general_binding_type,
    __get_binding_type_from_node,
    __handle_set_let_binding_as_equality,
)
from .type_extraction import (
    __extract_all_exists_witness_binders,
    __extract_exists_witness_binder,
    __extract_type_ast,
    __strip_leading_colon,
)

__all__ = [
    "__construct_set_with_hypothesis_type",
    "__determine_general_binding_type",
    "__extract_all_exists_witness_binders",
    "__extract_exists_witness_binder",
    # Internal functions (exported for use within util package)
    "__extract_type_ast",
    "__generate_equality_hypothesis_name",
    "__get_binding_type_from_node",
    "__handle_set_let_binding_as_equality",
    "__make_binder",
    "__make_binder_from_type_string",
    "__make_equality_binder",
    "__strip_leading_colon",
]
