"""Test what fallback_source contains when let bindings are present."""

# ruff: noqa: RUF001

from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context

# Simulate goal context from sorry for "have hn0 : 0 ≤ n"
# This is what would be in the sorry goal
goal_context_str = "n : ℤ\nhn : n > 1\nN : ℕ := Int.toNat n\n⊢ 0 ≤ n"

print("=" * 70)
print("VALIDATION TEST: Fallback source content")
print("=" * 70)
print(f"Goal context string: {goal_context_str!r}")
print()

# Parse goal context (this is what goal_var_types would contain)
goal_var_types = __parse_goal_context(goal_context_str)

print(f"goal_var_types: {goal_var_types}")
print()

# Simulate target_goal_var_types (from target sorry)
# In the actual code, this would be parsed from the target sorry's goal
target_goal_var_types = __parse_goal_context(goal_context_str)

print(f"target_goal_var_types: {target_goal_var_types}")
print()

# Check what fallback_source would be (line 179)
fallback_source = target_goal_var_types or goal_var_types

print(f"fallback_source (line 179): {fallback_source}")
print()

# Check if it contains theorem signature variables
has_n = "n" in fallback_source
has_hn = "hn" in fallback_source

print(f"  Contains 'n': {has_n}")
print(f"  Contains 'hn': {has_hn}")
print()

if has_n and has_hn:
    print("✓ VALIDATED: fallback_source DOES contain theorem signature variables (n, hn)")
    print("  Root Cause 1: If fallback is triggered, it should have access to n and hn")
else:
    print("✗ VALIDATED: fallback_source does NOT contain theorem signature variables")
    print("  Root Cause 1: fallback_source is missing n or hn")

print("=" * 70)
