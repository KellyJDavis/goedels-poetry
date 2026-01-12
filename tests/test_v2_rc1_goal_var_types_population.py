"""Test if goal_var_types contains theorem signature variables when let bindings are present."""

# ruff: noqa: RUF001

from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context

# Simulate goal context from a have statement that depends on let binding
# This is what would be in the sorry goal for "have hn0 : 0 ≤ n"
goal_context_str = "n : ℤ\nhn : n > 1\nN : ℕ := Int.toNat n\n⊢ 0 ≤ n"

print("=" * 70)
print("VALIDATION TEST: goal_var_types population with let bindings")
print("=" * 70)
print(f"Goal context string: {goal_context_str!r}")
print()

parsed_types = __parse_goal_context(goal_context_str)

print(f"Parsed types: {parsed_types}")
print()

# VALIDATION
has_n = "n" in parsed_types
has_hn = "hn" in parsed_types
has_N = "N" in parsed_types

if has_n and has_hn:
    print("✓ ASSUMPTION IS FALSE: goal_var_types DOES contain theorem signature variables (n, hn)")
    print("  Root Cause 1 assumption is PARTIALLY INVALIDATED - goal_var_types is populated")
    print(f"  n : {parsed_types.get('n')}")
    print(f"  hn : {parsed_types.get('hn')}")
elif has_n and not has_hn:
    print("✗ ASSUMPTION IS PARTIALLY TRUE: goal_var_types contains n but NOT hn")
    print("  Root Cause 1 is PARTIALLY VALIDATED - hn is missing from goal_var_types")
elif not has_n:
    print("✗ ASSUMPTION IS TRUE: goal_var_types does NOT contain n (or hn)")
    print("  Root Cause 1 is VALIDATED - goal_var_types population fails")
else:
    print("? UNKNOWN: Could not determine goal_var_types content")

print()
print("=" * 70)
