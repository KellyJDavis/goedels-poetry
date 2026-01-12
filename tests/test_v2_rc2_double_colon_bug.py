"""Validation test for Root Cause 2: Double Colon Bug.

This test specifically reproduces the double colon bug (N : : ℕ) that appears
in the actual log file. The bug occurs when the type AST has a colon token that
isn't properly stripped before serialization.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast


def test_rc2_double_colon_bug_reproduction() -> None:
    """
    VALIDATION TEST: Does the double colon bug (N : : ℕ) occur?

    This test reproduces the exact scenario from the log file where
    let binding types are serialized with double colons.

    The bug occurs when:
    1. Value extraction fails (so we fall back to type annotation)
    2. Type AST is a typeSpec with a colon token
    3. The colon isn't properly stripped, causing (N : : ℕ) instead of (N : ℕ)
    """
    # Create theorem AST where value extraction will fail
    # This forces the fallback to create a type annotation
    # Use typeSpec structure with colon token to reproduce the bug
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "∀ n : ℤ, n > 1 → Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": " ", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # let N : ℕ := Int.toNat n
                            # Use typeSpec with colon - this structure can cause double colon if not stripped
                            {
                                "kind": "Lean.Parser.Tactic.tacticLet_",
                                "args": [
                                    {"val": "let", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.letDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.letIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.letId",
                                                        "args": [
                                                            {"val": "N", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    [],
                                                    # typeSpec with colon token - this is the problematic structure
                                                    [
                                                        {
                                                            "kind": "Lean.Parser.Term.typeSpec",
                                                            "args": [
                                                                {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                                {"val": "ℕ", "info": {"leading": "", "trailing": " "}},
                                                            ],
                                                        }
                                                    ],
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    # Use a complex value that might fail extraction, forcing fallback
                                                    {
                                                        "kind": "__value_container",
                                                        "args": [
                                                            {"val": "Int", "info": {"leading": "", "trailing": ""}},
                                                            {"val": ".", "info": {"leading": "", "trailing": ""}},
                                                            {"val": "toNat", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "n", "info": {"leading": "", "trailing": ""}},
                                                        ],
                                                    },
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                            # have hn0 : 0 ≤ n := by sorry
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
                                                            {"val": "hn0", "info": {"leading": "", "trailing": " "}}
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

    sorries = [
        {
            "name": "hn0",
            "goal": "n : ℤ\nhn : n > 1\nN : ℕ := Int.toNat n\n⊢ 0 ≤ n",
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "proofState": 1,
        }
    ]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 2 - Double Colon Bug Reproduction")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  let N : ℕ := Int.toNat n")
    print("  have hn0 : 0 ≤ n := by sorry")
    print()
    print("Using typeSpec structure with colon token to reproduce bug")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn0", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Check for double colon bug
        has_double_colon = "(N : : ℕ)" in result_code or "(N::ℕ)" in result_code or "N : : ℕ" in result_code
        has_single_colon = "(N : ℕ)" in result_code or "(N:ℕ)" in result_code
        has_N_binder = "N" in result_code and ("ℕ" in result_code or "Nat" in result_code)

        print("Validation Results:")
        print(f"  Has (N : ℕ) binder (single colon): {has_single_colon}")
        print(f"  Has (N : : ℕ) binder (double colon bug): {has_double_colon}")
        print(f"  Has N binder with type: {has_N_binder}")
        print()

        # VALIDATION
        if has_double_colon:
            print("✗ ASSUMPTION IS TRUE: Double colon bug EXISTS (N : : ℕ)")
            print("  Root Cause 2 is VALIDATED - double colon serialization bug exists")
            print("  The typeSpec colon token is not being properly stripped")
        elif has_single_colon and not has_double_colon:
            print("✓ ASSUMPTION IS FALSE: Double colon bug does NOT exist")
            print("  Root Cause 2 assumption is INVALIDATED - type extraction works correctly")
        else:
            print("? UNKNOWN: Could not determine double colon bug status")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("NOTE: This test requires proper AST structure.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS V2 - VALIDATION TEST: RC2 (Double Colon Bug)")
    print("=" * 70)
    print()

    test_rc2_double_colon_bug_reproduction()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
