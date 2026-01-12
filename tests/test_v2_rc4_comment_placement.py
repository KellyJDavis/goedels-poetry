"""Validation test for Root Cause 4: Comment Placement.

This test validates whether comments appear before extracted lemmas or in the middle.
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast


def test_rc4_comment_placement() -> None:
    """
    VALIDATION TEST: Are comments placed before extracted lemmas?

    Assumption: Comments appear in the middle of lemmas instead of before them.

    Test: Create theorem AST with a comment before a have statement and check
    if the comment appears before the extracted lemma or in the middle.
    """
    # Create theorem AST with comment before have statement:
    # theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by
    #   let N : ℕ := Int.toNat n
    #   -- `n > 1` implies `n` is nonnegative.
    #   have hn0 : 0 ≤ n := by sorry
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
                                                    [{"val": "ℕ", "info": {"leading": " ", "trailing": " "}}],
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Int.toNat n", "info": {"leading": "", "trailing": ""}},
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
    print("VALIDATION TEST: Root Cause 4 - Comment Placement")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  let N : ℕ := Int.toNat n")
    print("  -- `n > 1` implies `n` is nonnegative.")
    print("  have hn0 : 0 ≤ n := by sorry")
    print()
    print("Comment is stored in 'have' node's leading info field")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn0", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Check comment placement
        comment_text = "-- `n > 1` implies `n` is nonnegative."
        has_comment = comment_text in result_code
        comment_before_lemma = result_code.find(comment_text) < result_code.find("lemma hn0")
        comment_in_middle = (
            "lemma" in result_code
            and comment_text in result_code
            and result_code.find("lemma") < result_code.find(comment_text) < result_code.find(":=")
        )

        print("Validation Results:")
        print(f"  Has comment '{comment_text}': {has_comment}")
        if has_comment:
            comment_pos = result_code.find(comment_text)
            lemma_pos = result_code.find("lemma hn0")
            print(f"  Comment position: {comment_pos}")
            print(f"  Lemma position: {lemma_pos}")
            print(f"  Comment before lemma: {comment_before_lemma}")
            print(f"  Comment in middle (between 'lemma' and ':='): {comment_in_middle}")
            # Check if comment appears between binders (e.g., after a closing paren)
            comment_after_paren = ")  --" in result_code or ")\n  --" in result_code or ") --" in result_code
            print(f"  Comment after closing paren (in middle of binders): {comment_after_paren}")

        print()

        # VALIDATION
        if not has_comment:
            print("✗ ASSUMPTION IS TRUE: Comment is MISSING from extracted lemma")
            print("  Root Cause 4 is VALIDATED - comments are not extracted")
        elif comment_before_lemma and not comment_in_middle:
            print("✓ ASSUMPTION IS FALSE: Comment IS placed before lemma (correct)")
            print("  Root Cause 4 assumption is INVALIDATED - comment placement works")
        elif comment_in_middle or (has_comment and comment_pos > lemma_pos and comment_pos < result_code.find(":=")):
            print("✗ ASSUMPTION IS TRUE: Comment appears in the middle of lemma")
            print("  Root Cause 4 is VALIDATED - comment placement is incorrect")
        else:
            print("? UNKNOWN: Comment is present but placement is unclear")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("NOTE: This test requires proper AST structure.")


def test_rc4_comment_association() -> None:
    """
    VALIDATION TEST: Are comments associated with the correct have statements?

    Assumption: Comments may be associated with the wrong have statement.

    Test: Create theorem AST with multiple have statements, each with a comment,
    and check if each extracted lemma has the correct comment.
    """
    # Create theorem AST with two have statements, each with a comment:
    # theorem Test : Prop := by
    #   -- Comment 1
    #   have h1 : P1 := by sorry
    #   -- Comment 2
    #   have h2 : P2 := by sorry
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "Test", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": " ", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # -- Comment 1
                            # have h1 : P1 := by sorry
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {
                                        "val": "have",
                                        "info": {
                                            "leading": "\n  -- Comment 1\n  ",
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "P1", "info": {"leading": "", "trailing": " "}},
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
                            # -- Comment 2
                            # have h2 : P2 := by sorry
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {
                                        "val": "have",
                                        "info": {
                                            "leading": "\n  -- Comment 2\n  ",
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "P2", "info": {"leading": "", "trailing": " "}},
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
            "name": "h1",
            "goal": "⊢ P1",
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "proofState": 1,
        },
        {
            "name": "h2",
            "goal": "⊢ P2",
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "proofState": 1,
        },
    ]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 4 - Comment Association")
    print("=" * 70)
    print("Theorem: theorem Test : Prop := by")
    print("  -- Comment 1")
    print("  have h1 : P1 := by sorry")
    print("  -- Comment 2")
    print("  have h2 : P2 := by sorry")
    print()

    try:
        # Extract h1
        result_ast_h1 = _get_named_subgoal_rewritten_ast(theorem_ast, "h1", sorries)
        result_code_h1 = _ast_to_code(result_ast_h1)

        # Extract h2
        result_ast_h2 = _get_named_subgoal_rewritten_ast(theorem_ast, "h2", sorries)
        result_code_h2 = _ast_to_code(result_ast_h2)

        print("Extracted lemma h1:")
        print(result_code_h1)
        print()
        print("Extracted lemma h2:")
        print(result_code_h2)
        print()

        # Check comment association
        h1_has_comment1 = "-- Comment 1" in result_code_h1
        h1_has_comment2 = "-- Comment 2" in result_code_h1
        h2_has_comment1 = "-- Comment 1" in result_code_h2
        h2_has_comment2 = "-- Comment 2" in result_code_h2

        print("Validation Results:")
        print(f"  h1 has '-- Comment 1': {h1_has_comment1}")
        print(f"  h1 has '-- Comment 2': {h1_has_comment2}")
        print(f"  h2 has '-- Comment 1': {h2_has_comment1}")
        print(f"  h2 has '-- Comment 2': {h2_has_comment2}")
        print()

        # VALIDATION
        correct_association = h1_has_comment1 and not h1_has_comment2 and not h2_has_comment1 and h2_has_comment2
        wrong_association = h1_has_comment2 or h2_has_comment1

        if correct_association:
            print("✓ ASSUMPTION IS FALSE: Comments ARE associated with correct have statements")
            print("  Root Cause 4 assumption is INVALIDATED - comment association works")
        elif wrong_association:
            print("✗ ASSUMPTION IS TRUE: Comments are associated with WRONG have statements")
            print("  Root Cause 4 is VALIDATED - comment association is incorrect")
        else:
            print("? UNKNOWN: Comment association status is unclear")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("NOTE: This test requires proper AST structure.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS V2 - VALIDATION TEST: RC4 (Comment Placement)")
    print("=" * 70)
    print()

    test_rc4_comment_placement()
    test_rc4_comment_association()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
