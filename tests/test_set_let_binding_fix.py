"""Comprehensive tests for the set and let binding fix.

This test suite verifies that the fix correctly handles both scenarios:
- Scenario A: Value extraction succeeds (variable needs to be included as parameter)
- Scenario B: Value extraction fails (defensive goal context parsing)
"""


class TestScenarioAValueExtractionSucceeds:
    """Tests for Scenario A: Value extraction succeeds."""

    def test_let_binding_variable_used_in_type(self) -> None:
        """Test that when value extraction succeeds, variable is included if used in type."""
        from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast

        # let x := 42
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
                                                            "args": [
                                                                {"val": "x", "info": {"leading": "", "trailing": ""}}
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "42", "info": {"leading": "", "trailing": " "}},
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
                "goal": "x : ℕ := 42\n⊢ x > 0",  # noqa: RUF001
                "proofState": 1,
            }
        ]

        result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)
        result_code = _ast_to_code(result)

        # Should have both equality hypothesis and variable as parameter
        assert "hx" in result_code or "x  = 42" in result_code or "x = 42" in result_code, (
            "Should have equality hypothesis"
        )
        assert "(x :" in result_code or "(x:" in result_code, "Should have x as parameter"
        assert "x > 0" in result_code or "x  > 0" in result_code, "Should have the type"

    def test_set_binding_variable_used_in_type(self) -> None:
        """Test that when value extraction succeeds, variable is included if used in type."""
        from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast

        # set s := x + 1
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
                                                            "args": [
                                                                {"val": "s", "info": {"leading": "", "trailing": ""}}
                                                            ],
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
                "goal": "s : ℕ := x + 1\n⊢ s > 0",  # noqa: RUF001
                "proofState": 1,
            }
        ]

        result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)
        result_code = _ast_to_code(result)

        # Should have both equality hypothesis and variable as parameter
        assert "hs" in result_code or "s  = x + 1" in result_code or "s = x + 1" in result_code, (
            "Should have equality hypothesis"
        )
        assert "(s :" in result_code or "(s:" in result_code, "Should have s as parameter"
        assert "s > 0" in result_code or "s  > 0" in result_code, "Should have the type"


class TestScenarioBValueExtractionFails:
    """Tests for Scenario B: Value extraction fails."""

    def test_set_binding_complex_value_goal_context_fallback(self) -> None:
        """Test that goal context is used when value extraction fails."""
        from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast

        # set odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)  # noqa: RUF003
        # have h_partition : evens ∪ odds = ... := by sorry  # noqa: RUF003
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
                                                            "args": [
                                                                {"val": "odds", "info": {"leading": "", "trailing": ""}}
                                                            ],
                                                        },
                                                        {
                                                            "kind": "Lean.Parser.Term.typeSpec",
                                                            "args": [
                                                                {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                                {
                                                                    "val": "Finset",
                                                                    "info": {"leading": "", "trailing": " "},
                                                                },
                                                                {"val": "ℕ", "info": {"leading": "", "trailing": ""}},  # noqa: RUF001
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
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
                                                {"val": "odds", "info": {"leading": "", "trailing": " "}},
                                                {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "Finset.range", "info": {"leading": "", "trailing": " "}},
                                                {"val": "10000", "info": {"leading": "", "trailing": " "}},
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
                "goal": "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\n⊢ odds = Finset.range 10000",  # noqa: RUF001
                "proofState": 1,
            }
        ]

        result = _get_named_subgoal_rewritten_ast(ast_dict, "h_partition", sorries)
        result_code = _ast_to_code(result)

        # Should have odds as parameter (value extraction fails, so uses goal context)
        assert "(odds :" in result_code or "(odds:" in result_code, "Should have odds as parameter"
        assert "Finset ℕ" in result_code or "Finset" in result_code, "Should have the type"  # noqa: RUF001

    def test_let_binding_complex_value_goal_context_fallback(self) -> None:
        """Test that goal context is used when value extraction fails for let."""
        from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast

        # let odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)  # noqa: RUF003
        # have h_partition : odds = ... := by sorry
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
                                                            "args": [
                                                                {"val": "odds", "info": {"leading": "", "trailing": ""}}
                                                            ],
                                                        },
                                                        {
                                                            "kind": "Lean.Parser.Term.typeSpec",
                                                            "args": [
                                                                {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                                {
                                                                    "val": "Finset",
                                                                    "info": {"leading": "", "trailing": " "},
                                                                },
                                                                {"val": "ℕ", "info": {"leading": "", "trailing": ""}},  # noqa: RUF001
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
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
                                                {"val": "odds", "info": {"leading": "", "trailing": " "}},
                                                {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "Finset.range", "info": {"leading": "", "trailing": " "}},
                                                {"val": "10000", "info": {"leading": "", "trailing": " "}},
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
                "goal": "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\n⊢ odds = Finset.range 10000",  # noqa: RUF001
                "proofState": 1,
            }
        ]

        result = _get_named_subgoal_rewritten_ast(ast_dict, "h_partition", sorries)
        result_code = _ast_to_code(result)

        # Should have odds as parameter (value extraction fails, so uses goal context)
        assert "(odds :" in result_code or "(odds:" in result_code, "Should have odds as parameter"
        assert "Finset ℕ" in result_code or "Finset" in result_code, "Should have the type"  # noqa: RUF001


class TestEdgeCases:
    """Edge case tests."""

    def test_multiple_set_let_bindings(self) -> None:
        """Test multiple set and let bindings are all included."""
        from goedels_poetry.parsers.util import _ast_to_code, _get_named_subgoal_rewritten_ast

        # set odds := ...
        # set evens := ...
        # let x := 42
        # have h : ... := by sorry
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
                                                            "args": [
                                                                {"val": "odds", "info": {"leading": "", "trailing": ""}}
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "value1", "info": {"leading": "", "trailing": " "}},
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "kind": "Lean.Parser.Tactic.tacticSet_",
                                    "args": [
                                        {"val": "set", "info": {"leading": "\n  ", "trailing": " "}},
                                        {
                                            "kind": "Lean.Parser.Term.setDecl",
                                            "args": [
                                                {
                                                    "kind": "Lean.Parser.Term.setIdDecl",
                                                    "args": [
                                                        {
                                                            "kind": "Lean.binderIdent",
                                                            "args": [
                                                                {
                                                                    "val": "evens",
                                                                    "info": {"leading": "", "trailing": ""},
                                                                }
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "value2", "info": {"leading": "", "trailing": " "}},
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "kind": "Lean.Parser.Tactic.tacticLet_",
                                    "args": [
                                        {"val": "let", "info": {"leading": "\n  ", "trailing": " "}},
                                        {
                                            "kind": "Lean.Parser.Term.letDecl",
                                            "args": [
                                                {
                                                    "kind": "Lean.Parser.Term.letIdDecl",
                                                    "args": [
                                                        {
                                                            "kind": "Lean.binderIdent",
                                                            "args": [
                                                                {"val": "x", "info": {"leading": "", "trailing": ""}}
                                                            ],
                                                        },
                                                    ],
                                                },
                                                {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "42", "info": {"leading": "", "trailing": " "}},
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
                                                {"val": "odds", "info": {"leading": "", "trailing": " "}},
                                                {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                                {"val": "evens", "info": {"leading": "", "trailing": " "}},
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
                "goal": "odds : Finset ℕ := value1\nevens : Finset ℕ := value2\nx : ℕ := 42\n⊢ odds = evens",  # noqa: RUF001
                "proofState": 1,
            }
        ]

        result = _get_named_subgoal_rewritten_ast(ast_dict, "h", sorries)
        result_code = _ast_to_code(result)

        # Should have all three variables
        assert "(odds :" in result_code or "(odds:" in result_code, "Should have odds"
        assert "(evens :" in result_code or "(evens:" in result_code, "Should have evens"
        assert "(x :" in result_code or "(x:" in result_code, "Should have x"
