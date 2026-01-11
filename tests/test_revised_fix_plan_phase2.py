"""Tests for Phase 2: Fix Double Colon Syntax Error

Phase 2 fix: Enhanced __strip_leading_colon() to handle val fields with leading colons.
This prevents double colon syntax errors like (N : : ℕ ) -> (N : ℕ ).
"""

# ruff: noqa: RUF001, RUF002

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.types_and_binders.binder_construction import __make_binder
from goedels_poetry.parsers.util.types_and_binders.type_extraction import __strip_leading_colon


def test_strip_leading_colon_val_field_with_leading_colon() -> None:
    """
    Test that __strip_leading_colon strips leading colons from val fields.

    Input: {"val": ": ℕ", "info": {}}
    Expected: {"val": "ℕ", "info": {}}
    """
    type_ast = {"val": ": ℕ", "info": {"leading": "", "trailing": ""}}

    result = __strip_leading_colon(type_ast)

    assert isinstance(result, dict), "Result should be a dict"
    assert result.get("val") == "ℕ", f"val should be 'ℕ', got {result.get('val')}"
    assert result.get("info") == {"leading": "", "trailing": ""}, "info should be preserved"

    print("\nPhase 2 Test 1 Result:")
    print(f"  Input val: {type_ast['val']!r}")
    print(f"  Output val: {result.get('val')!r}")
    print("  ✓ Leading colon stripped from val field")


def test_strip_leading_colon_val_field_no_colon() -> None:
    """
    Test that __strip_leading_colon doesn't modify val fields without leading colons.

    Input: {"val": "ℕ", "info": {}}
    Expected: {"val": "ℕ", "info": {}}
    """
    type_ast = {"val": "ℕ", "info": {"leading": "", "trailing": ""}}

    result = __strip_leading_colon(type_ast)

    assert isinstance(result, dict), "Result should be a dict"
    assert result.get("val") == "ℕ", f"val should be 'ℕ', got {result.get('val')}"

    print("\nPhase 2 Test 2 Result:")
    print(f"  Input val: {type_ast['val']!r}")
    print(f"  Output val: {result.get('val')!r}")
    print("  ✓ Val field without colon unchanged")


def test_make_binder_with_leading_colon_in_val() -> None:
    """
    Test that __make_binder correctly handles type_ast with leading colon in val field.

    Input: type_ast = {"val": ": ℕ", "info": {}}
    Expected: Binder serializes as (N : ℕ) not (N : : ℕ )
    """
    type_ast = {"val": ": ℕ", "info": {"leading": "", "trailing": ""}}

    binder = __make_binder("N", type_ast)
    binder_code = _ast_to_code(binder)

    # Should NOT have double colon
    has_double_colon = "::" in binder_code or ": :" in binder_code
    # Should have single colon
    has_single_colon = "(N :" in binder_code or "(N:" in binder_code
    # Should have the type
    has_type = "ℕ" in binder_code

    assert not has_double_colon, f"Should NOT have double colon. Got: {binder_code}"
    assert has_single_colon, f"Should have single colon. Got: {binder_code}"
    assert has_type, f"Should have type ℕ. Got: {binder_code}"

    print("\nPhase 2 Test 3 Result:")
    print(f"  Binder code: {binder_code}")
    print(f"  Has double colon: {has_double_colon} (should be False)")
    print(f"  Has single colon: {has_single_colon} (should be True)")
    print("  ✓ No double colon syntax error")


def test_strip_leading_colon_with_whitespace() -> None:
    """
    Test that __strip_leading_colon handles whitespace correctly.

    Input: {"val": "  :  ℕ  ", "info": {}}
    Expected: {"val": "ℕ", "info": {}}
    """
    type_ast = {"val": "  :  ℕ  ", "info": {"leading": "", "trailing": ""}}

    result = __strip_leading_colon(type_ast)

    assert isinstance(result, dict), "Result should be a dict"
    # Should strip leading colons and whitespace
    assert result.get("val") == "ℕ", f"val should be 'ℕ', got {result.get('val')!r}"

    print("\nPhase 2 Test 4 Result:")
    print(f"  Input val: {type_ast['val']!r}")
    print(f"  Output val: {result.get('val')!r}")
    print("  ✓ Leading colon and whitespace stripped")
