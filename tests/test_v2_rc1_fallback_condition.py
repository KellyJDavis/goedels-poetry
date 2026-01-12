"""Test if fallback condition is met when let bindings are present."""

# ruff: noqa: RUF001

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
)
from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context

# Create theorem AST where type is string (not parsed)
theorem_ast = {
    "kind": "Lean.Parser.Command.theorem",
    "args": [
        {"val": "theorem", "info": {"leading": "", "trailing": " "}},
        {
            "kind": "Lean.Parser.Command.declId",
            "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
        },
        {"val": ":", "info": {"leading": " ", "trailing": " "}},
        {"val": "∀ n : ℤ, n > 1 → Prop", "info": {"leading": "", "trailing": " "}},  # Type as string
        {"val": ":=", "info": {"leading": " ", "trailing": " "}},
        {"kind": "Lean.Parser.Term.byTactic", "args": []},
    ],
}

# Goal context from sorry
goal_context_str = "n : ℤ\nhn : n > 1\n⊢ 0 ≤ n"
goal_var_types = __parse_goal_context(goal_context_str)

print("=" * 70)
print("VALIDATION TEST: Fallback condition check")
print("=" * 70)
print("Condition at line 178: if not theorem_binders and goal_var_types:")
print()

# Check theorem_binders
theorem_binders = __extract_theorem_binders(theorem_ast, goal_var_types)
print(f"theorem_binders: {len(theorem_binders)} binders")
print(f"  Empty? {len(theorem_binders) == 0}")
print()

print(f"goal_var_types: {goal_var_types}")
print(f"  Non-empty? {bool(goal_var_types)}")
print()

# Check condition
condition_met = not theorem_binders and goal_var_types
print(f"Fallback condition (line 178): {condition_met}")
print(f"  not theorem_binders: {not theorem_binders}")
print(f"  goal_var_types: {bool(goal_var_types)}")
print()

if condition_met:
    print("✓ VALIDATED: Fallback condition IS met")
    print("  This means fallback logic SHOULD be triggered")
    print("  If it's not working, the problem is INSIDE the fallback logic")
else:
    print("✗ VALIDATED: Fallback condition is NOT met")
    print("  This means fallback logic will NOT be triggered")

print("=" * 70)
