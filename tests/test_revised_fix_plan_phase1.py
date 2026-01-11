"""Tests for Phase 1: Fix Missing Type Annotations

Phase 1 fix: Removed skip at line 410-411 in subgoal_rewriting.py
This allows variables in variables_in_equality_hypotheses to be added as type annotations.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast


def test_let_binding_has_both_equality_hypothesis_and_type_annotation() -> None:
    """
    Test that let bindings result in BOTH equality hypothesis AND type annotation.

    Before fix: Only equality hypothesis (hx : x = value)
    After fix: Both equality hypothesis (hx : x = value) AND type annotation (x : ℕ)
    """
    # let x : ℕ := 42
    # have h1 : x > 0 := by sorry
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
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
                                                        "kind": "Lean.binderIdent",
                                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                    [],
                                                    [
                                                        {
                                                            "kind": "Lean.Parser.Term.typeSpec",
                                                            "args": [
                                                                {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                                {"val": "ℕ", "info": {"leading": "", "trailing": ""}},
                                                            ],
                                                        }
                                                    ],
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "42", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have", "info": {"leading": "\n  ", "trailing": " "}},
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
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
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
                                                            {
                                                                "val": "sorry",
                                                                "info": {"leading": "", "trailing": ""},
                                                            }
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
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ := 42\n⊢ x > 0",
            "proofState": 1,
        }
    ]

    result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)
    result_code = _ast_to_code(result)

    # Should have BOTH equality hypothesis AND type annotation
    has_equality_hypothesis = "hx" in result_code or "x  = 42" in result_code or "x = 42" in result_code
    has_type_annotation = "(x :" in result_code or "(x:" in result_code

    assert has_equality_hypothesis, f"Should have equality hypothesis (hx : x = 42). Got: {result_code}"
    assert has_type_annotation, f"Should have type annotation (x : ℕ). Got: {result_code}"

    # Verify both are present
    print("\nPhase 1 Test Result:")
    print(f"  Result code: {result_code}")
    print(f"  Has equality hypothesis: {has_equality_hypothesis}")
    print(f"  Has type annotation: {has_type_annotation}")
    print("  ✓ Both equality hypothesis AND type annotation are present")


def test_set_binding_has_both_equality_hypothesis_and_type_annotation() -> None:
    """
    Test that set bindings result in BOTH equality hypothesis AND type annotation.

    Before fix: Only equality hypothesis (hs : s = value)
    After fix: Both equality hypothesis (hs : s = value) AND type annotation (s : ℕ)
    """
    # set s : ℕ := x + 1
    # have h1 : s > 0 := by sorry
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticSet_",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.setDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.setIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [{"val": "s", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have", "info": {"leading": "\n  ", "trailing": " "}},
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
                                            {"val": "s", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
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
                                                            {
                                                                "val": "sorry",
                                                                "info": {"leading": "", "trailing": ""},
                                                            }
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
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "s : ℕ := x + 1\n⊢ s > 0",
            "proofState": 1,
        }
    ]

    result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)
    result_code = _ast_to_code(result)

    # Should have BOTH equality hypothesis AND type annotation
    has_equality_hypothesis = "hs" in result_code or "s  = x + 1" in result_code or "s = x + 1" in result_code
    has_type_annotation = "(s :" in result_code or "(s:" in result_code

    assert has_equality_hypothesis, f"Should have equality hypothesis (hs : s = x + 1). Got: {result_code}"
    assert has_type_annotation, f"Should have type annotation (s : ℕ). Got: {result_code}"

    # Verify both are present
    print("\nPhase 1 Test Result (set binding):")
    print(f"  Result code: {result_code}")
    print(f"  Has equality hypothesis: {has_equality_hypothesis}")
    print(f"  Has type annotation: {has_type_annotation}")
    print("  ✓ Both equality hypothesis AND type annotation are present")
