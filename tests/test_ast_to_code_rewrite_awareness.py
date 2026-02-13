from __future__ import annotations

from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code


def test_ast_to_code_suppresses_kimina_anonymous_sentinel() -> None:
    # Kimina can emit `val == "[anonymous]"` (with rawVal == "") for anonymous binders.
    # We must NOT serialize this sentinel into Lean code.
    node = {"val": "[anonymous]", "rawVal": "", "info": {"leading": "", "trailing": ""}}
    assert _ast_to_code(node) == ""


def test_ast_to_code_prefers_rawval_when_val_unchanged() -> None:
    # When AST node has a stable token, rawVal is the closest to original surface text.
    node = {"val": "x", "rawVal": "x", "info": {"leading": "", "trailing": ""}}
    assert _ast_to_code(node) == "x"


def test_ast_to_code_is_rewrite_aware_when_val_differs_from_rawval() -> None:
    # After programmatic AST rewrites (e.g. variable renaming), val can differ from rawVal.
    # In that case we should serialize val, otherwise the rewrite is lost.
    node = {"val": "x_hole1_deadbeef", "rawVal": "x", "info": {"leading": "", "trailing": ""}}
    assert _ast_to_code(node) == "x_hole1_deadbeef"
