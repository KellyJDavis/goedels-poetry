"""Tests for Phase 1: Arrow Domain Hypothesis Name Matching.

This test validates that arrow domain hypotheses use correct names from goal context
instead of auto-generated names.
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __parse_pi_binders_from_type,
)
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name


def test_arrow_domain_name_matching_with_goal_var_types() -> None:
    """
    Test that arrow domain hypotheses use names from goal_var_types when provided.

    Expected: When goal_var_types contains "hn" with type "n > 1", the arrow domain
    hypothesis should use name "hn" instead of auto-generated "h".
    """
    # Create type AST representing ∀ n : ℤ, n > 1 → ...
    type_ast = {
        "kind": "Lean.Parser.Term.forall",
        "args": [
            # args[0]: binder (n : ℤ)
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "ℤ", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            # args[1]: body (n > 1 → ...)
            {
                "kind": "Lean.Parser.Term.arrow",
                "args": [
                    {
                        "kind": "__type_container",
                        "args": [
                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                            {"val": "1", "info": {"leading": " ", "trailing": ""}},
                        ],
                    },
                    {"val": "...", "info": {"leading": " ", "trailing": ""}},
                ],
            },
        ],
    }

    # goal_var_types with correct hypothesis name
    goal_var_types = {"n": "ℤ", "hn": "n > 1"}

    # Parse without goal_var_types (should use auto-generated name)
    binders_without = __parse_pi_binders_from_type(type_ast, goal_var_types=None)
    binder_names_without = [__extract_binder_name(b) for b in binders_without if __extract_binder_name(b)]

    # Parse with goal_var_types (should use correct name)
    binders_with = __parse_pi_binders_from_type(type_ast, goal_var_types=goal_var_types)
    binder_names_with = [__extract_binder_name(b) for b in binders_with if __extract_binder_name(b)]

    print("\n" + "=" * 70)
    print("Test: Arrow Domain Name Matching with goal_var_types")
    print("=" * 70)
    print("Type AST: ∀ n : ℤ, n > 1 → ...")
    print(f"goal_var_types: {goal_var_types}")
    print()
    print(f"Without goal_var_types: {binder_names_without}")
    print(f"With goal_var_types: {binder_names_with}")
    print()

    # VALIDATION
    # Without goal_var_types, should have auto-generated name (h, h1, etc.)
    has_auto_name_without = any(name and name.startswith("h") and name != "hn" for name in binder_names_without)
    assert "n" in binder_names_without, "Should have n from forall binder"
    assert has_auto_name_without, "Should have auto-generated name without goal_var_types"

    # With goal_var_types, should have correct name (hn)
    assert "n" in binder_names_with, "Should have n from forall binder"
    assert "hn" in binder_names_with, "Should have hn from goal_var_types instead of auto-generated name"

    print("✓ Test passed: Arrow domain hypothesis uses correct name (hn) from goal_var_types")
    print(f"  Changed from: {[n for n in binder_names_without if n != 'n']}")
    print(f"  Changed to: {[n for n in binder_names_with if n != 'n']}")


def test_arrow_domain_name_matching_fallback() -> None:
    """
    Test that arrow domain hypotheses fall back to auto-generated names if no match found.

    Expected: When goal_var_types doesn't contain a matching type, should use auto-generated name.
    """
    # Create type AST with arrow domain
    type_ast = {
        "kind": "Lean.Parser.Term.arrow",
        "args": [
            {"val": "P", "info": {"leading": "", "trailing": ""}},
            {"val": "Q", "info": {"leading": "", "trailing": ""}},
        ],
    }

    # goal_var_types without matching type
    goal_var_types = {"n": "ℤ", "hn": "n > 1"}

    binders = __parse_pi_binders_from_type(type_ast, goal_var_types=goal_var_types)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\n" + "=" * 70)
    print("Test: Arrow Domain Name Matching Fallback")
    print("=" * 70)
    print("Type AST: P → Q")
    print(f"goal_var_types: {goal_var_types} (no match for 'P')")
    print(f"Binder names: {binder_names}")
    print()

    # Should fall back to auto-generated name since "P" doesn't match any type in goal_var_types
    has_auto_name = any(name and name.startswith("h") for name in binder_names)
    assert has_auto_name, "Should fall back to auto-generated name when no match found"

    print("✓ Test passed: Falls back to auto-generated name when no match in goal_var_types")


if __name__ == "__main__":
    test_arrow_domain_name_matching_with_goal_var_types()
    test_arrow_domain_name_matching_fallback()
    print("\n" + "=" * 70)
    print("Phase 1 Tests Complete")
    print("=" * 70)
