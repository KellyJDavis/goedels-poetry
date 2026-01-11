"""
Investigation tests for Phase 3: Missing Type Annotations

These tests are designed to validate root causes through focused Python tests,
not to fix the bugs. The goal is to understand WHERE variables are lost in the
extraction pipeline.

Test Strategy:
1. Test if variables N, m appear in goal_var_types dictionary
2. Test if dependency tracking finds these variables (__find_dependencies)
3. Test if binder creation code path is reached for N, m
4. Test how goal_var_types is used in binder creation
"""

# ruff: noqa: RUF001, RUF002

from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context


def test_goal_context_parsing_for_let_binding() -> None:
    """
    Test: Does goal context parsing extract N from "N : ℕ := Int.toNat n"?

    This validates that goal context parsing works (already validated, but double-check).
    """
    goal_line = "N : ℕ := Int.toNat n"
    parsed = __parse_goal_context(goal_line)

    print("Test 1 - Goal context parsing:")
    print(f"  Input: '{goal_line}'")
    print(f"  Parsed: {parsed}")
    print(f"  'N' in parsed: {'N' in parsed}")
    if "N" in parsed:
        print(f"  Type of 'N': {parsed['N']}")
    print()

    assert "N" in parsed, "Goal context parsing should extract N"
    assert parsed["N"] == "ℕ", f"Type of N should be 'ℕ', got '{parsed['N']}'"


def test_goal_context_multiple_variables() -> None:
    """
    Test: Does goal context parsing extract multiple variables like N and m?
    """
    goal_text = """n : ℤ
N : ℕ := Int.toNat n
m : ℕ := some_expression
x : Type := other_value"""

    parsed = __parse_goal_context(goal_text)

    print("Test 2 - Goal context multiple variables:")
    print(f"  Parsed variables: {list(parsed.keys())}")
    print(f"  Parsed types: {parsed}")
    print(f"  'N' in parsed: {'N' in parsed}")
    print(f"  'm' in parsed: {'m' in parsed}")
    print(f"  'n' in parsed: {'n' in parsed}")
    print()

    # Check that N and m are extracted
    assert "N" in parsed, "Should extract N"
    assert "m" in parsed, "Should extract m"
    assert "n" in parsed, "Should extract n"


if __name__ == "__main__":
    print("=" * 60)
    print("INVESTIGATION TESTS: Missing Type Annotations (Phase 3)")
    print("=" * 60)
    print()

    test_goal_context_parsing_for_let_binding()
    test_goal_context_multiple_variables()

    print("=" * 60)
    print("SUMMARY:")
    print("  Goal context parsing works: To be validated")
    print("=" * 60)
