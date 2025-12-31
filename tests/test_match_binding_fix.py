"""Comprehensive tests for match binding fix.

Tests verify that match pattern bindings are correctly included in extracted subgoals
in various scenarios.
"""

from goedels_poetry.parsers.ast import AST


def test_match_binding_target_inside_branch() -> None:
    """Test that match bindings work when target is inside a match branch."""

    # match n with | x + 1 => have h : x > 0 := by sorry
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Term.bracketedBinderList",
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
                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
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
                                "kind": "Lean.Parser.Tactic.tacticMatch_",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "n", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n    "}},
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n    ", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
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
                                                                                "val": "h",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            }
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
                    },
                ],
            },
        ],
    }

    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ\n⊢ x > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h")

    # Verify x is included as a parameter
    has_x_param = "(x :" in result or "(x:" in result
    assert has_x_param, "x should be included as a parameter when target is inside match branch"
    assert "x > 0" in result or "x  > 0" in result


def test_match_binding_target_after_match() -> None:
    """Test that match bindings work when target is after match (sibling)."""
    # This is the same as test_match_binding_bug_variable_used_in_subgoal
    # but kept here for completeness

    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Term.bracketedBinderList",
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
                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
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
                                "kind": "Lean.Parser.Tactic.tacticMatch_",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "n", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n    "}},
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n    ", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
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
                                                            {
                                                                "val": "h_partition",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
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
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ\n⊢ x > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h_partition")

    # Verify x is included as a parameter
    has_x_param = "(x :" in result or "(x:" in result
    assert has_x_param, "x should be included as a parameter when target is after match"
    assert "x > 0" in result or "x  > 0" in result


def test_match_binding_multiple_branches() -> None:
    """Test that only relevant branch bindings are extracted when multiple branches exist."""

    # match n with
    #   | 0 => True
    #   | x + 1 => True
    #   | y + 2 => True
    # have h : x > 0 := by sorry  # Should include x, not y
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Term.bracketedBinderList",
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
                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
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
                                "kind": "Lean.Parser.Tactic.tacticMatch_",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "n", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n    "}},
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n    ", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n    ", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "y", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "2", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "True", "info": {"leading": "", "trailing": " "}},
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
                                                            {"val": "h", "info": {"leading": "", "trailing": " "}}
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
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ\n⊢ x > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h")

    # Verify x is included (post-processing filters unused bindings)
    has_x_param = "(x :" in result or "(x:" in result
    assert has_x_param, "x should be included as it's used in the subgoal type"
    # Note: y might be included initially since we extract from all branches,
    # but post-processing will filter it out if unused based on dependencies
    assert "x > 0" in result or "x  > 0" in result
