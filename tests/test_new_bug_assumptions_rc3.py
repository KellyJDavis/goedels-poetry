"""Validation test for Root Cause 3: Goal context fallback.

This test validates whether the goal context fallback includes all theorem signature variables.
"""

# ruff: noqa: RUF001, RUF002

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast


def test_assumption_rc3_fallback_with_empty_theorem_binders() -> None:
    """
    VALIDATION TEST: Does goal context fallback include all theorem signature variables?

    Assumption: Goal context fallback may not include all theorem signature variables,
    especially when relevance check fails and only first name is kept.

    Test: Create a theorem AST with empty theorem_binders (simulating quantifier-only signature
    that __extract_theorem_binders can't handle), set goal_var_types with n and hn,
    extract a have statement that references n, and check if both n and hn are added.

    This simulates the scenario where:
    - __extract_theorem_binders returns empty list (theorem_binders = [])
    - goal_var_types contains {"n": "ℤ", "hn": "n > 1"}
    - We extract a have statement that references n
    - Fallback code should add both n and hn
    """
    # Create a minimal theorem AST where __extract_theorem_binders would return empty
    # (e.g., quantifier-only signature without declSig or explicit binders)
    # For testing, we'll create a structure that results in empty theorem_binders
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            # No declSig - this would result in empty theorem_binders
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "∀ n : ℤ, n > 1 → Prop", "info": {"leading": "", "trailing": ""}},  # Type as string
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {
                                        "kind": "Lean.Parser.Tactic.haveId",
                                        "args": [
                                            {"val": "hn_nonneg", "info": {"leading": "", "trailing": ""}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0 ≤ n", "info": {"leading": "", "trailing": ""}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }

    # Create sorries with goal context containing theorem signature variables
    sorries = [
        {
            "name": "hn_nonneg",
            "goal": "n : ℤ\nhn : n > 1\n⊢ 0 ≤ n",
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
        }
    ]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 3 - Goal context fallback")
    print("=" * 70)
    print("Theorem AST: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  have hn_nonneg : 0 ≤ n := by sorry")
    print()
    print("Goal context: 'n : ℤ\\nhn : n > 1\\n⊢ 0 ≤ n'")
    print()
    print("Expected: Extracted lemma should have (n : ℤ) and (hn : n > 1) as binders")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn_nonneg", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Extract binder names from result
        # This is simplified - we'd need to parse the AST to get binders properly
        # For now, check if the code contains the binders

        has_n_binder = "(n : ℤ)" in result_code or "(n:ℤ)" in result_code or "(n : ℤ)" in result_code.replace(" ", "")
        has_hn_binder = "(hn : n > 1)" in result_code or "(hn:n > 1)" in result_code

        print("Validation Results:")
        print(f"  Has (n : ℤ) binder: {has_n_binder}")
        print(f"  Has (hn : n > 1) binder: {has_hn_binder}")
        print()

        # VALIDATION RESULT
        if has_n_binder and has_hn_binder:
            print("✓ ASSUMPTION IS FALSE: Goal context fallback DOES include both n and hn")
            print("  Root Cause 3 is NOT a problem - fallback works correctly.")
        elif has_n_binder and not has_hn_binder:
            print("✗ ASSUMPTION IS TRUE: Goal context fallback includes n but NOT hn")
            print("  Root Cause 3 is a VALIDATED problem - only first variable is added.")
        elif not has_n_binder:
            print("✗ ASSUMPTION IS TRUE: Goal context fallback does NOT include n")
            print("  Root Cause 3 is a VALIDATED problem - fallback doesn't work correctly.")
        else:
            print("? UNKNOWN: Could not determine if binders are present")
            print("  Root Cause 3 needs further investigation.")

    except Exception as e:
        print(f"Error during extraction: {e}")
        print()
        print("NOTE: This test requires proper AST structure.")
        print("      The error indicates the test setup needs adjustment.")
        print("      Root Cause 3 assumption status: UNKNOWN (requires proper test setup)")

        # Still print assumption status based on code inspection
        print()
        print("Code inspection analysis:")
        print("  - Fallback condition: if not theorem_binders and goal_var_types (line 178)")
        print("  - Relevance check: __is_referenced_in(target, name) or name in deps (line 188)")
        print("  - First-name-only fallback: if not added_any, only first name added (lines 196-208)")
        print("  - Type reference check: subsequent names only if type references kept name (line 211)")
        print()
        print("Based on code structure:")
        print("  - If relevance check passes: all referenced variables should be added")
        print("  - If relevance check fails: only first name is added (lines 196-208)")
        print("  - Type reference check might add hn if n > 1 references n (line 211)")
        print()
        print("ASSUMPTION STATUS: UNKNOWN (requires working test setup)")


if __name__ == "__main__":
    test_assumption_rc3_fallback_with_empty_theorem_binders()
