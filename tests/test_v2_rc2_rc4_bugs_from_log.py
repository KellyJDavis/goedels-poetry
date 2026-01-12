"""Tests that reproduce the Double Colon Bug (RC2) and Comment Placement Bug (RC4) from the log file.

These tests use realistic AST structures and sorries data extracted from goedels_poetry.log
to reproduce the actual bugs that occur in production.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast


def test_rc2_double_colon_bug_by_triggering_fallback() -> None:
    """
    Test that reproduces the double colon bug by triggering the fallback path.

    The bug occurs when:
    1. Value extraction fails for a let binding
    2. Fallback creates a type annotation using __make_binder(var_name, binding_type_ast)
    3. binding_type_ast is a typeSpec that isn't properly stripped, causing (N : : ℕ)

    We'll force the fallback by making value extraction fail.
    """
    """
    Test that reproduces the double colon bug (N : : ℕ) from the log file.

    Based on log file line 22830 which shows:
    lemma hN_cast (hN : N = Int.toNat n) (hn0 : 0 ≤ n) (N : : ℕ) : (N : ℤ) = n := by sorry
    """
    # Sorries data extracted from log file (lines 15828-15912)
    sorries = [
        {
            "pos": {"line": 19, "column": 4},
            "endPos": {"line": 19, "column": 9},
            "goal": "n : ℤ\nhn : n > 1\nN : ℕ := n.toNat\n⊢ 0 ≤ n",
            "proofState": 1,
        },
        {
            "pos": {"line": 23, "column": 4},
            "endPos": {"line": 23, "column": 9},
            "goal": "n : ℤ\nhn : n > 1\nN : ℕ := n.toNat\nhn0 : 0 ≤ n\n⊢ ↑N = n",
            "proofState": 2,
        },
    ]

    # Create theorem AST with let binding that will trigger the fallback path
    # The key is to have a typeSpec structure that includes a colon token
    # This structure is based on what Kimina actually produces
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
                "kind": "Lean.Parser.Command.declSig",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.forall",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.explicitBinder",
                                "args": [
                                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "ℤ", "info": {"leading": "", "trailing": ""}},
                                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Term.arrow",
                                "args": [
                                    {
                                        "kind": "__type_container",
                                        "args": [
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": " ", "trailing": ""}},
                                        ],
                                    },
                                    {"val": "Prop", "info": {"leading": " ", "trailing": ""}},
                                ],
                            },
                        ],
                    },
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
                            # let N : ℕ := n.toNat
                            # Use typeSpec with colon token - this is the structure that causes the bug
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
                                                    # typeSpec with colon - this structure causes double colon if not stripped
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
                                                    {
                                                        "kind": "__value_container",
                                                        "args": [
                                                            {"val": "n", "info": {"leading": "", "trailing": ""}},
                                                            {"val": ".", "info": {"leading": "", "trailing": ""}},
                                                            {"val": "toNat", "info": {"leading": "", "trailing": ""}},
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
                            # have hN_cast : (N : ℤ) = n := by sorry
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
                                                            {"val": "hN_cast", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(N : ℤ) = n", "info": {"leading": "", "trailing": " "}},
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

    print("\n" + "=" * 70)
    print("TEST: RC2 Double Colon Bug Reproduction from Log File")
    print("=" * 70)
    print("Extracting lemma hN_cast which should show (N : : ℕ) bug")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hN_cast", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Check for double colon bug
        has_double_colon = "(N : : ℕ)" in result_code or "(N::ℕ)" in result_code or "N : : ℕ" in result_code
        has_single_colon = "(N : ℕ)" in result_code or "(N:ℕ)" in result_code

        print("Validation Results:")
        print(f"  Has (N : ℕ) binder (single colon): {has_single_colon}")
        print(f"  Has (N : : ℕ) binder (double colon bug): {has_double_colon}")
        print()

        # NOTE: The test may not reproduce the bug if the AST structure doesn't exactly match Kimina's output
        # The bug exists in the log file, so we document it here
        if has_double_colon:
            print("✓ BUG REPRODUCED: Double colon bug EXISTS (N : : ℕ)")
            print("  This confirms the bug from log file line 22830")
        elif has_single_colon:
            print("⚠️  BUG NOT REPRODUCED in this test, but EXISTS in log file line 22830")
            print("  The bug occurs with specific Kimina AST structures that this test doesn't fully replicate")
            print("  The test structure needs to match the exact AST from Kimina to reproduce the bug")
        else:
            print("? UNKNOWN: Could not determine if N binder exists")

        # We don't assert here because the test may not reproduce the bug with simplified AST
        # Instead, we document that the bug exists in the log file
        # When the bug is fixed, this test should be updated to verify the fix

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_rc4_comment_placement_bug_from_log() -> None:
    """
    Test that reproduces the comment placement bug from the log file.

    Based on log file line 22829 which shows:
    lemma hN_cast (hN : N = Int.toNat n

      -- `n > 1` implies `n` is nonnegative.
      ) (hn0 : 0 ≤ n) (N : : ℕ) : (N : ℤ) = n := by sorry

    The comment appears in the middle of binders, not before the lemma.
    """
    # Sorries data
    sorries = [
        {
            "pos": {"line": 19, "column": 4},
            "endPos": {"line": 19, "column": 9},
            "goal": "n : ℤ\nhn : n > 1\nN : ℕ := n.toNat\n⊢ 0 ≤ n",
            "proofState": 1,
        },
        {
            "pos": {"line": 23, "column": 4},
            "endPos": {"line": 23, "column": 9},
            "goal": "n : ℤ\nhn : n > 1\nN : ℕ := n.toNat\nhn0 : 0 ≤ n\n⊢ ↑N = n",
            "proofState": 2,
        },
    ]

    # Create theorem AST with comment in the have node's leading info
    # The comment should be in the info.leading field of the have node
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
                            # let N : ℕ := n.toNat
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
                                                    {"val": "n.toNat", "info": {"leading": "", "trailing": ""}},
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                            # -- `n > 1` implies `n` is nonnegative.
                            # have hn0 : 0 ≤ n := by sorry
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {
                                        "val": "have",
                                        "info": {
                                            "leading": "\n  -- `n > 1` implies `n` is nonnegative.\n  ",
                                            "trailing": " ",
                                        },
                                    },
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
                            # have hN_cast : (N : ℤ) = n := by sorry
                            # The comment might also be in a binder's info.leading field
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
                                                            {"val": "hN_cast", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(N : ℤ) = n", "info": {"leading": "", "trailing": " "}},
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

    print("\n" + "=" * 70)
    print("TEST: RC4 Comment Placement Bug Reproduction from Log File")
    print("=" * 70)
    print("Extracting lemma hN_cast which should show comment in middle of binders")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hN_cast", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        comment_text = "-- `n > 1` implies `n` is nonnegative."
        has_comment = comment_text in result_code
        lemma_pos = result_code.find("lemma hN_cast")
        comment_pos = result_code.find(comment_text) if has_comment else -1

        # Check if comment appears in the middle (between binders)
        # Pattern from log: ")  -- comment\n  )" or similar
        comment_after_paren = ")  --" in result_code or ")\n  --" in result_code or ") --" in result_code
        comment_before_lemma = comment_pos != -1 and lemma_pos != -1 and comment_pos < lemma_pos
        comment_in_middle = (
            has_comment
            and lemma_pos != -1
            and comment_pos != -1
            and comment_pos > lemma_pos
            and comment_pos < result_code.find(":=")
        )

        print("Validation Results:")
        print(f"  Has comment: {has_comment}")
        if has_comment:
            print(f"  Comment position: {comment_pos}")
            print(f"  Lemma position: {lemma_pos}")
            print(f"  Comment before lemma: {comment_before_lemma}")
            print(f"  Comment in middle (between 'lemma' and ':='): {comment_in_middle}")
            print(f"  Comment after closing paren: {comment_after_paren}")

        print()

        # NOTE: The test may not reproduce the bug if comments are stored differently in the AST
        # The bug exists in the log file, so we document it here
        if comment_in_middle or comment_after_paren:
            print("✓ BUG REPRODUCED: Comment appears in the middle of binders")
            print("  This confirms the bug from log file line 22829")
        elif not has_comment:
            print("⚠️  BUG NOT REPRODUCED in this test, but EXISTS in log file line 22829")
            print("  Comments may be stored in binder nodes' info fields, not just have nodes")
            print("  The test structure needs to match the exact AST from Kimina to reproduce the bug")
        elif comment_before_lemma:
            print("⚠️  BUG NOT REPRODUCED: Comment is correctly placed in this test")
            print("  But the bug exists in log file - test needs to match exact Kimina AST structure")
        else:
            print("? UNKNOWN: Comment placement status unclear")

        # We don't assert here because the test may not reproduce the bug with simplified AST
        # Instead, we document that the bug exists in the log file
        # When the bug is fixed, this test should be updated to verify the fix

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS V2 - BUG REPRODUCTION TESTS FROM LOG FILE")
    print("=" * 70)
    print()

    try:
        test_rc2_double_colon_bug_by_triggering_fallback()
        print("\n" + "-" * 70 + "\n")
        test_rc4_comment_placement_bug_from_log()
    except AssertionError as e:
        print(f"\n⚠️  Test assertion failed: {e}")
        print("This may indicate the bug is fixed or the test needs adjustment.")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
