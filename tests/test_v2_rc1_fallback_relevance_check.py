"""Test if relevance check in fallback logic works correctly."""

# ruff: noqa: RUF001

from goedels_poetry.parsers.util.collection_and_analysis.decl_collection import __find_dependencies
from goedels_poetry.parsers.util.collection_and_analysis.reference_checking import __is_referenced_in

# Simulate target have statement
target_ast = {
    "kind": "Lean.Parser.Tactic.tacticHave_",
    "args": [
        {"val": "have", "info": {"leading": "", "trailing": " "}},
        {
            "kind": "Lean.Parser.Term.haveDecl",
            "args": [
                {
                    "kind": "Lean.Parser.Term.haveIdDecl",
                    "args": [
                        {
                            "kind": "Lean.Parser.Term.haveId",
                            "args": [{"val": "hn0", "info": {"leading": "", "trailing": " "}}],
                        }
                    ],
                },
                {"val": ":", "info": {"leading": " ", "trailing": " "}},
                {"val": "0", "info": {"leading": "", "trailing": " "}},
                {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                {"val": "n", "info": {"leading": " ", "trailing": " "}},
            ],
        },
    ],
}

# Simulate name_map (empty for this test)
name_map = {}

# Check dependencies
deps = __find_dependencies(target_ast, name_map)

print("=" * 70)
print("VALIDATION TEST: Relevance check in fallback logic")
print("=" * 70)
print("Target: have hn0 : 0 ≤ n")
print()

# Check relevance for n and hn
n_in_target = __is_referenced_in(target_ast, "n")
n_in_deps = "n" in deps
hn_in_target = __is_referenced_in(target_ast, "hn")
hn_in_deps = "hn" in deps

print(f"Dependencies: {deps}")
print()
print(f"  n is referenced in target: {n_in_target}")
print(f"  n is in deps: {n_in_deps}")
print(f"  n passes relevance check (line 190): {n_in_target or n_in_deps}")
print()
print(f"  hn is referenced in target: {hn_in_target}")
print(f"  hn is in deps: {hn_in_deps}")
print(f"  hn passes relevance check (line 190): {hn_in_target or hn_in_deps}")
print()

# Check type reference
goal_var_types = {"n": "ℤ", "hn": "n > 1"}
n_type = goal_var_types.get("n", "")
hn_type = goal_var_types.get("hn", "")

print("Type reference check (line 206):")
print(f"  hn type '{hn_type}' references 'n': {'n' in hn_type}")
print("  If n is kept, hn should be added via type reference check")
print()

# VALIDATION
if n_in_target or n_in_deps:
    print("✓ VALIDATED: n SHOULD pass relevance check and be added in first pass")
    if "n" in hn_type:
        print("✓ VALIDATED: hn type references n, so hn SHOULD be added in second pass")
        print("  Root Cause 1: If fallback is triggered, both n and hn should be added")
    else:
        print("? UNEXPECTED: hn type does not reference n")
else:
    print("✗ VALIDATED: n does NOT pass relevance check")
    print("  Root Cause 1: n won't be added, so hn won't be added either")

print("=" * 70)
