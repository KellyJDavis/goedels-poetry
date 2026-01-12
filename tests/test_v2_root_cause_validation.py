"""Validation tests for Root Cause Analysis V2 assumptions.

These tests verify or falsify assumptions about why standalone lemmas are missing
theorem signature variables, let binding types, correct ordering, and comments.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util import _ast_to_code
from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
)
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name


def test_rc1_theorem_binders_with_let_bindings() -> None:
    """
    VALIDATION TEST: Are theorem_binders populated when let bindings are present?

    Assumption: theorem_binders may be empty or not populated when let bindings exist.

    Test: Create theorem AST with intro n hn, let N, and have hn0.
    Extract theorem_binders and check if they contain n and hn.
    """
    # Create theorem AST: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by intro n hn; let N : ℕ := Int.toNat n; have hn0 : 0 ≤ n := by sorry
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
                            # args[0]: binder (n : ℤ)
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
                            # args[1]: body (n > 1 → Prop)
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
                                                    # Type annotation: ℕ
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
                                                    {"val": "Int.toNat n", "info": {"leading": "", "trailing": ""}},
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

    # Goal context for hn0: should contain n, hn, N
    goal_var_types = {"n": "ℤ", "hn": "n > 1", "N": "ℕ"}

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 1 - theorem_binders with let bindings")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  let N : ℕ := Int.toNat n")
    print("  have hn0 : 0 ≤ n := by sorry")
    print()
    print(f"goal_var_types: {goal_var_types}")
    print()

    # Test: Extract theorem_binders
    theorem_binders = __extract_theorem_binders(theorem_ast, goal_var_types)
    binder_names = [__extract_binder_name(b) for b in theorem_binders if __extract_binder_name(b)]

    print(f"theorem_binders extracted: {len(theorem_binders)} binders")
    print(f"Binder names: {binder_names}")
    print()

    # VALIDATION
    has_n = "n" in binder_names
    has_hn = "hn" in binder_names

    if has_n and has_hn:
        print("✓ ASSUMPTION IS FALSE: theorem_binders DOES contain n and hn when let bindings are present")
        print("  Root Cause 1 assumption is INVALIDATED - theorem_binders extraction works correctly")
    elif has_n and not has_hn:
        print("✗ ASSUMPTION IS PARTIALLY TRUE: theorem_binders contains n but NOT hn")
        print("  Root Cause 1 is PARTIALLY VALIDATED - hn is missing")
    elif not has_n:
        print("✗ ASSUMPTION IS TRUE: theorem_binders does NOT contain n (or hn)")
        print("  Root Cause 1 is VALIDATED - theorem_binders extraction fails with let bindings")
    else:
        print("? UNKNOWN: Could not determine theorem_binders content")

    assert isinstance(theorem_binders, list), "theorem_binders should be a list"


def test_rc1_fallback_trigger_with_let_bindings() -> None:
    """
    VALIDATION TEST: Is fallback logic triggered when let bindings are present?

    Assumption: Fallback logic may not be triggered correctly when let bindings exist.

    Test: Create theorem AST where __extract_theorem_binders would return empty,
    and check if fallback logic populates theorem_binders.
    """
    # Create theorem AST where theorem_binders would be empty (type as string, not parsed)
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

    # Goal context for hn0: should contain n, hn, N
    goal_var_types = {"n": "ℤ", "hn": "n > 1", "N": "ℕ"}

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 1 - Fallback trigger with let bindings")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  let N : ℕ := Int.toNat n")
    print("  have hn0 : 0 ≤ n := by sorry")
    print()
    print("Note: Type is string (not parsed), so __extract_theorem_binders should return empty")
    print(f"goal_var_types: {goal_var_types}")
    print()

    # Test: Extract theorem_binders (should trigger fallback)
    theorem_binders = __extract_theorem_binders(theorem_ast, goal_var_types)
    binder_names = [__extract_binder_name(b) for b in theorem_binders if __extract_binder_name(b)]

    print(f"theorem_binders extracted: {len(theorem_binders)} binders")
    print(f"Binder names: {binder_names}")
    print()

    # VALIDATION
    has_n = "n" in binder_names
    has_hn = "hn" in binder_names
    is_empty = len(theorem_binders) == 0

    if is_empty:
        print("✗ ASSUMPTION IS TRUE: theorem_binders is empty, fallback NOT triggered")
        print("  Root Cause 1 is VALIDATED - fallback logic does not work with let bindings")
    elif has_n and has_hn:
        print("✓ ASSUMPTION IS FALSE: theorem_binders is populated, fallback IS triggered")
        print("  Root Cause 1 assumption is INVALIDATED - fallback works correctly")
    elif has_n and not has_hn:
        print("✗ ASSUMPTION IS PARTIALLY TRUE: theorem_binders has n but NOT hn")
        print("  Root Cause 1 is PARTIALLY VALIDATED - fallback partially works")
    else:
        print("? UNKNOWN: Could not determine fallback behavior")

    assert isinstance(theorem_binders, list), "theorem_binders should be a list"


def test_rc2_let_binding_type_extraction() -> None:
    """
    VALIDATION TEST: Are let binding types extracted correctly?

    Assumption: Let binding types are not extracted or incorrectly formatted.

    Test: Extract a have statement that depends on a let binding and check if
    the let binding type (N : ℕ) is included in the extracted lemma.

    IMPORTANT: This test must use a realistic AST structure where the type is
    a typeSpec node with a colon token, as this is what causes the double colon bug.
    """
    # Create theorem AST with let binding
    # The key is to use a typeSpec structure that includes a colon token,
    # which if not properly stripped will cause (N : : ℕ) instead of (N : ℕ)
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
                            # Use typeSpec structure with colon token to reproduce double colon bug
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
                                                    # Use typeSpec with colon token - this is what causes the bug
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
                                                    {"val": "Int.toNat n", "info": {"leading": "", "trailing": ""}},
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
    print("VALIDATION TEST: Root Cause 2 - Let binding type extraction")
    print("=" * 70)
    print("Theorem: theorem A303656 : ∀ n : ℤ, n > 1 → Prop := by")
    print("  let N : ℕ := Int.toNat n")
    print("  have hn0 : 0 ≤ n := by sorry")
    print()
    print("Goal context: 'n : ℤ\\nhn : n > 1\\nN : ℕ := Int.toNat n\\n⊢ 0 ≤ n'")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn0", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Check for let binding type
        has_n_type = "(n : ℤ)" in result_code or "(n:ℤ)" in result_code
        has_n_type_normalized = "(n : ℤ)" in result_code.replace(" ", "")
        has_N_type = "(N : ℕ)" in result_code or "(N:ℕ)" in result_code
        has_N_type_double_colon = "(N : : ℕ)" in result_code or "(N::ℕ)" in result_code
        has_hN_equality = "(hN : N" in result_code or "(hN:N" in result_code

        print("Validation Results:")
        print(f"  Has (n : ℤ) binder: {has_n_type or has_n_type_normalized}")
        print(f"  Has (N : ℕ) binder: {has_N_type}")
        print(f"  Has (N : : ℕ) (double colon bug): {has_N_type_double_colon}")
        print(f"  Has (hN : N = ...) equality: {has_hN_equality}")
        print()

        # VALIDATION
        if has_N_type and not has_N_type_double_colon:
            print("✓ ASSUMPTION IS FALSE: Let binding type IS extracted correctly (N : ℕ)")
            print("  Root Cause 2 assumption is INVALIDATED - type extraction works")
        elif has_N_type_double_colon:
            print("✗ ASSUMPTION IS TRUE: Let binding type has double colon bug (N : : ℕ)")
            print("  Root Cause 2 is VALIDATED - serialization bug exists")
        elif not has_N_type and has_hN_equality:
            print("✗ ASSUMPTION IS TRUE: Let binding type is MISSING (only equality hypothesis present)")
            print("  Root Cause 2 is VALIDATED - type annotation not created")
        else:
            print("? UNKNOWN: Could not determine let binding type extraction status")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("NOTE: This test requires proper AST structure.")


def test_rc3_binder_ordering() -> None:
    """
    VALIDATION TEST: Are binders ordered correctly (dependencies before dependents)?

    Assumption: Binders are not ordered by dependency.

    Test: Extract a have statement that depends on a let binding and check if
    N : ℕ appears before hN : N = Int.toNat n.
    """
    # Same setup as test_rc2_let_binding_type_extraction
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
    print("VALIDATION TEST: Root Cause 3 - Binder ordering")
    print("=" * 70)
    print("Expected order: (n : ℤ), (hn : n > 1), (N : ℕ), (hN : N = Int.toNat n)")
    print()

    try:
        result_ast = _get_named_subgoal_rewritten_ast(theorem_ast, "hn0", sorries)
        result_code = _ast_to_code(result_ast)

        print(f"Extracted lemma code:\n{result_code}\n")

        # Find positions of binders in the code
        n_pos = result_code.find("(n : ℤ)")
        if n_pos == -1:
            n_pos = result_code.find("(n:ℤ)")
        hn_pos = result_code.find("(hn : n > 1)")
        if hn_pos == -1:
            hn_pos = result_code.find("(hn:n > 1)")
        N_pos = result_code.find("(N : ℕ)")
        if N_pos == -1:
            N_pos = result_code.find("(N:ℕ)")
        hN_pos = result_code.find("(hN : N")
        if hN_pos == -1:
            hN_pos = result_code.find("(hN:N")

        print("Binder positions in code:")
        print(f"  (n : ℤ) at position: {n_pos}")
        print(f"  (hn : n > 1) at position: {hn_pos}")
        print(f"  (N : ℕ) at position: {N_pos}")
        print(f"  (hN : N = ...) at position: {hN_pos}")
        print()

        # VALIDATION: Check ordering
        # n should come before hn, N should come before hN
        n_before_hn = n_pos != -1 and hn_pos != -1 and n_pos < hn_pos
        N_before_hN = N_pos != -1 and hN_pos != -1 and N_pos < hN_pos
        n_before_N = n_pos != -1 and N_pos != -1 and n_pos < N_pos

        if n_before_hn and N_before_hN and n_before_N:
            print("✓ ASSUMPTION IS FALSE: Binders ARE ordered correctly")
            print("  Root Cause 3 assumption is INVALIDATED - ordering works correctly")
        elif not N_before_hN and N_pos != -1 and hN_pos != -1:
            print("✗ ASSUMPTION IS TRUE: Binders are NOT ordered correctly (hN before N)")
            print("  Root Cause 3 is VALIDATED - dependency ordering fails")
        else:
            print("? UNKNOWN: Could not determine binder ordering (some binders missing)")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS V2 - VALIDATION TESTS")
    print("=" * 70)
    print()
    print("These tests verify or falsify assumptions about root causes.")
    print()

    test_rc1_theorem_binders_with_let_bindings()
    test_rc1_fallback_trigger_with_let_bindings()
    test_rc2_let_binding_type_extraction()
    test_rc3_binder_ordering()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
