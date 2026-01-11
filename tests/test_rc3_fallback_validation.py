"""Validation test for Root Cause 3: Goal context fallback logic.

This test validates whether the goal context fallback includes all theorem signature variables
when theorem_binders is empty.
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast


def test_rc3_fallback_includes_all_theorem_signature_variables() -> None:
    """
    VALIDATION TEST: Does goal context fallback include all theorem signature variables?

    Assumption: Goal context fallback may not include all theorem signature variables,
    especially when relevance check fails and only first name is kept.

    Test setup:
    - Create theorem AST where __extract_theorem_binders would return empty (no explicit binders)
    - Set sorries with goal context containing theorem signature variables (n, hn)
    - Extract a have statement that references n (so relevance check should pass for n)
    - Check if both n and hn are included in extracted lemma

    This tests the fallback logic at lines 178-216 in subgoal_rewriting.py.
    """
    # Create theorem AST where theorem_binders would be empty
    # Structure: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by ...
    # We'll use a structure where declSig doesn't have explicit binders
    # and the type is not properly parsed, so __extract_theorem_binders returns empty

    # Create theorem AST structure similar to test_revised_fix_plan_phase1.py
    # but with a have statement that references n
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {
                "val": "∀ n : ℤ, n > 1 → Prop",
                "info": {"leading": "", "trailing": " "},
            },  # Type as string - won't be parsed
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": " ", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
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
                                                        "args": [
                                                            {
                                                                "val": "hn_nonneg",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "n", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [
                                                            {"val": "sorry", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    }
                                                ],
                                            },
                                        ],
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
    # Goal context shows: n : ℤ, hn : n > 1 (theorem signature variables)
    sorries = [
        {
            "name": "hn_nonneg",
            "goal": "n : ℤ\nhn : n > 1\n⊢ 0 ≤ n",
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "proofState": 1,
        }
    ]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 3 - Goal context fallback")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  have hn_nonneg : 0 ≤ n := by sorry")
    print()
    print("Goal context: 'n : ℤ\\nhn : n > 1\\n⊢ 0 ≤ n'")
    print()
    print("Expected: Extracted lemma should have BOTH (n : ℤ) AND (hn : n > 1) as binders")
    print("  (since theorem_binders will be empty, fallback should add both from goal_var_types)")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn_nonneg", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Check if binders are present in the code
        has_n_binder = "(n : ℤ)" in result_code or "(n:ℤ)" in result_code or "(n : ℤ)" in result_code.replace(" ", "")
        has_hn_binder = "(hn : n > 1)" in result_code or "(hn:n > 1)" in result_code

        print("Validation Results:")
        print(f"  Has (n : ℤ) binder: {has_n_binder}")
        print(f"  Has (hn : n > 1) binder: {has_hn_binder}")
        print()

        # VALIDATION RESULT
        # Expected: Both binders should be present (the fix should work)
        assert has_n_binder, "Expected (n : ℤ) binder in extracted lemma - fallback should include n"
        assert has_hn_binder, "Expected (hn : n > 1) binder in extracted lemma - fallback should include hn"

        if has_n_binder and has_hn_binder:
            print("✓ ASSUMPTION IS FALSE: Goal context fallback DOES include both n and hn")
            print("  Root Cause 3 is NOT a problem - fallback works correctly.")
            print("  Both theorem signature variables are added when referenced.")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("NOTE: This test requires proper AST structure.")
        print("      The error indicates the test setup needs adjustment.")
        print("      Root Cause 3 assumption status: UNKNOWN (requires proper test setup)")
        raise  # Re-raise the exception so the test fails properly


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VALIDATION TEST FOR ROOT CAUSE 3 (GOAL CONTEXT FALLBACK)")
    print("=" * 70)
    print()
    print("This test validates whether goal context fallback includes all")
    print("theorem signature variables when theorem_binders is empty.")
    print()

    try:
        test_rc3_fallback_includes_all_theorem_signature_variables()
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print("RESULT: Assumption is FALSE - Root Cause 3 is NOT a problem")
        print("        Fallback correctly includes both n and hn")
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print("RESULT: Assumption is TRUE - Root Cause 3 is a VALIDATED problem")
        print(f"        Error: {e}")
        raise
    except Exception as e:
        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print("RESULT: UNKNOWN - Requires further investigation")
        print(f"        Error: {e}")
        raise
