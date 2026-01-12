"""Test if relevance check works correctly for n and hn when let bindings are present."""

from goedels_poetry.parsers.util.collection_and_analysis.reference_checking import __is_referenced_in

# Simulate target have statement: "have hn0 : 0 ≤ n"
# The target AST would reference "n" but not directly reference "hn"
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

print("=" * 70)
print("VALIDATION TEST: Relevance check for n and hn")
print("=" * 70)
print("Target: have hn0 : 0 ≤ n")
print()

# Check if n is referenced
n_referenced = __is_referenced_in(target_ast, "n")
hn_referenced = __is_referenced_in(target_ast, "hn")

print(f"  'n' is referenced in target: {n_referenced}")
print(f"  'hn' is referenced in target: {hn_referenced}")
print()

# VALIDATION
if n_referenced and not hn_referenced:
    print("✓ VALIDATED: n is referenced, hn is NOT directly referenced")
    print("  This means:")
    print("  - n should pass relevance check (line 190)")
    print("  - hn should NOT pass relevance check (line 190)")
    print("  - hn should be added via type reference check (line 206) if n is kept")
    print()
    print("  Root Cause 1: If fallback is triggered, n should be added, then hn via type reference")
elif n_referenced and hn_referenced:
    print("? UNEXPECTED: Both n and hn are referenced")
else:
    print("? UNEXPECTED: n is not referenced")

print("=" * 70)
