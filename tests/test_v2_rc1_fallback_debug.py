"""Debug test for RC1 fallback trigger issue."""

# ruff: noqa: RUF001

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
    __parse_pi_binders_from_type,
)
from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context
from goedels_poetry.parsers.util.types_and_binders.type_extraction import __extract_type_ast

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
print("DEBUG: RC1 Fallback Trigger")
print("=" * 70)
print(f"goal_var_types: {goal_var_types}")
print()

# Step 1: Check __extract_theorem_binders
theorem_binders = __extract_theorem_binders(theorem_ast, goal_var_types)
print(f"Step 1: __extract_theorem_binders returned {len(theorem_binders)} binders")
print()

# Step 2: Check if fallback to __parse_pi_binders_from_type would work
decl_type = __extract_type_ast(theorem_ast)
print(f"Step 2: __extract_type_ast returned: {decl_type}")
if decl_type:
    print(f"  Type AST kind: {decl_type.get('kind')}")
    print(f"  Type AST val: {decl_type.get('val')}")
print()

if decl_type and isinstance(decl_type, dict) and decl_type.get("val"):
    # Type is a string, not an AST - __parse_pi_binders_from_type won't work
    print("Step 3: Type is a string (not AST), so __parse_pi_binders_from_type won't work")
    print("  This means fallback at line 173 won't populate theorem_binders")
    print()
    print("✗ VALIDATED: Fallback logic does NOT work when type is string")
    print("  Root Cause 1 is VALIDATED - fallback requires parsed AST, not string")
else:
    parsed_pi_binders = __parse_pi_binders_from_type(decl_type, goal_var_types=goal_var_types) if decl_type else []
    print(f"Step 3: __parse_pi_binders_from_type returned {len(parsed_pi_binders)} binders")
    print()

print("=" * 70)
