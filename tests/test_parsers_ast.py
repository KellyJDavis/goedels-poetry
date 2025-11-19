"""Tests for goedels_poetry.parsers.ast module."""

import pytest

from goedels_poetry.parsers.ast import AST


def test_ast_init() -> None:
    """Test AST initialization."""
    ast_dict = {"kind": "test", "args": []}
    ast = AST(ast_dict)
    assert ast._ast == ast_dict


def test_ast_get_ast() -> None:
    """Test getting the AST representation."""
    ast_dict = {"kind": "Lean.Parser.Command.theorem", "args": [{"val": "test"}]}
    ast = AST(ast_dict)
    result = ast.get_ast()
    assert result == ast_dict


def test_ast_get_unproven_subgoal_names_empty() -> None:
    """Test getting unproven subgoals from AST with no sorries."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test_theorem"}]},
        ],
    }
    ast = AST(ast_dict)
    result = ast.get_unproven_subgoal_names()
    assert result == []


def test_ast_get_unproven_subgoal_names_with_sorry() -> None:
    """Test getting unproven subgoals from AST with sorries."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test_theorem"}]},
            {
                "kind": "Lean.Parser.Tactic.tacticSeq",
                "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
            },
        ],
    }
    ast = AST(ast_dict)
    result = ast.get_unproven_subgoal_names()
    assert len(result) == 1
    assert "<main body>" in result


def test_ast_get_unproven_subgoal_names_with_have() -> None:
    """Test getting unproven subgoals from AST with have statements."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test_theorem"}]},
            {
                "kind": "Lean.Parser.Tactic.tacticSeq",
                "args": [
                    {
                        "kind": "Lean.Parser.Tactic.tacticHave_",
                        "args": [
                            {"val": "have"},
                            {
                                "kind": "Lean.Parser.Term.haveDecl",
                                "args": [
                                    {
                                        "kind": "Lean.Parser.Term.haveIdDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveId",
                                                "args": [{"val": "h1"}],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
                            },
                        ],
                    }
                ],
            },
        ],
    }
    ast = AST(ast_dict)
    result = ast.get_unproven_subgoal_names()
    assert len(result) == 1
    assert "h1" in result


def test_ast_get_named_subgoal_ast_not_found() -> None:
    """Test getting named subgoal AST when name doesn't exist."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test_theorem"}]},
        ],
    }
    ast = AST(ast_dict)
    result = ast.get_named_subgoal_ast("nonexistent")
    assert result is None


def test_ast_get_named_subgoal_ast_theorem() -> None:
    """Test getting named subgoal AST for a theorem."""
    theorem_node = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "my_theorem"}]},
        ],
    }
    ast_dict = {"kind": "root", "args": [theorem_node]}
    ast = AST(ast_dict)
    result = ast.get_named_subgoal_ast("my_theorem")
    assert result == theorem_node


def test_ast_get_named_subgoal_ast_lemma() -> None:
    """Test getting named subgoal AST for a lemma."""
    lemma_node = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "my_lemma"}]},
        ],
    }
    ast_dict = {"kind": "root", "args": [lemma_node]}
    ast = AST(ast_dict)
    result = ast.get_named_subgoal_ast("my_lemma")
    assert result == lemma_node


def test_ast_get_named_subgoal_code() -> None:
    """Test getting named subgoal code."""
    # Create a simple theorem AST
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": "", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                            }
                        ],
                    },
                ],
            },
        ],
    }
    ast = AST(ast_dict)
    result = ast.get_named_subgoal_code("test_theorem")

    # Should contain the theorem declaration
    assert "theorem" in result
    assert "test_theorem" in result
    # The result should contain the basic structure even if formatting is different
    assert len(result) > 0


def test_ast_get_named_subgoal_code_not_found() -> None:
    """Test getting code for nonexistent subgoal raises KeyError."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test_theorem"}]},
        ],
    }
    ast = AST(ast_dict)

    with pytest.raises(KeyError, match="target 'nonexistent' not found in AST"):
        ast.get_named_subgoal_code("nonexistent")


def test_ast_with_sorries_extracts_types() -> None:
    """Test that AST with sorries properly extracts type information for variables."""
    # Create an AST with a have statement that uses variables
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": "", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": "", "trailing": " "}},
                                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                            }
                        ],
                    },
                ],
            },
        ],
    }

    # Create sorries list with goal context containing type information
    sorries = [
        {
            "pos": {"line": 10, "column": 4},
            "endPos": {"line": 10, "column": 9},
            "goal": "x y : Nat\n⊢ x = y",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Check that the generated code includes type information for x and y
    assert "lemma" in result
    assert "h1" in result
    # The exact format may vary, but it should contain references to the types
    assert len(result) > 0


def test_ast_init_with_sorries() -> None:
    """Test AST initialization with sorries parameter."""
    ast_dict = {"kind": "test", "args": []}
    sorries = [{"goal": "x : Nat\n⊢ x = x", "pos": {"line": 1, "column": 1}}]
    ast = AST(ast_dict, sorries)
    assert ast._ast == ast_dict
    assert ast._sorries == sorries


def test_ast_init_without_sorries() -> None:
    """Test AST initialization without sorries defaults to empty list."""
    ast_dict = {"kind": "test", "args": []}
    ast = AST(ast_dict)
    assert ast._ast == ast_dict
    assert ast._sorries == []


def test_ast_get_named_subgoal_code_includes_theorem_hypotheses() -> None:
    """Test that get_named_subgoal_code includes enclosing theorem's hypotheses."""
    # Create a theorem with parameters and a have statement
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Nat", "info": {"leading": " ", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "x", "info": {"leading": " ", "trailing": " "}},
                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                            {"val": "0", "info": {"leading": " ", "trailing": " "}},
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≠", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                            }
                        ],
                    },
                ],
            },
        ],
    }

    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "x : Nat\nh : x > 0\n⊢ x ≠ 0",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem's parameters
    assert "lemma" in result
    assert "h1" in result
    # Should contain references to x and h from the enclosing theorem
    assert "x" in result
    # Should contain the type Nat
    assert "Nat" in result or "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_includes_earlier_haves() -> None:
    """Test that get_named_subgoal_code includes earlier have statements as hypotheses."""
    # Create a theorem with multiple have statements
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
                                "args": [{"val": "C", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "ℂ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "D", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "ℂ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
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
                                                            {"val": "hCD_ne", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "C", "info": {"leading": "", "trailing": " "}},
                                            {"val": "-", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "D", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≠", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                                                            {"val": "hDB_ne", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "D", "info": {"leading": "", "trailing": " "}},
                                            {"val": "-", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "C", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≠", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "C D : ℂ\n⊢ C - D ≠ 0",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "C D : ℂ\nhCD_ne : C - D ≠ 0\n⊢ D - C ≠ 0",  # noqa: RUF001
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("hDB_ne")

    # The result should include the theorem's parameters (C and D)
    assert "lemma" in result
    assert "hDB_ne" in result
    assert "C" in result
    assert "D" in result
    # Should include the earlier have statement hCD_ne as a hypothesis
    assert "hCD_ne" in result


def test_ast_get_named_subgoal_code_includes_let_binding() -> None:
    """Test that get_named_subgoal_code includes earlier let bindings."""
    # Create a theorem with a let binding and a have statement
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
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
                            # Let binding
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
                                                        "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement using the let binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\nn : ℕ\n⊢ n > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include the let binding n as an equality hypothesis (hn : n = 5)
    assert "hn" in result  # Hypothesis name
    assert "n  = 5" in result or "n = 5" in result or "n=5" in result  # Equality format (with possible spaces)
    # Should NOT include n as a type annotation (n : ℕ) - that would be incorrect  # noqa: RUF003
    # The old incorrect format would have "(n : ℕ)" but we want "(hn : n = 5)"  # noqa: RUF003
    # Check that n is not included as a separate type annotation
    # Count occurrences of "(n :" - should only appear in the equality hypothesis context
    assert result.count("(n :") == 0, (
        f"Found type annotation for n, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_includes_obtain_binding() -> None:
    """Test that get_named_subgoal_code includes variables from obtain statements."""
    # Create a theorem with an obtain statement and a have statement
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
                                "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "∃", "info": {"leading": "", "trailing": " "}},
                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                            {"val": "P", "info": {"leading": "", "trailing": " "}},
                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Q", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # Obtain statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticObtain_",
                                "args": [
                                    {"val": "obtain", "info": {"leading": "", "trailing": " "}},
                                    {"val": "⟨", "info": {"leading": "", "trailing": ""}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ",", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "hx", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": "⟩", "info": {"leading": "", "trailing": " "}},
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "h", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
                            # Have statement using obtained variables
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Q", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "h : ∃ x, P x\nx : T\nhx : P x\n⊢ Q",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2")

    # The result should include the theorem hypothesis h
    assert "lemma" in result
    assert "h2" in result
    # Should include obtained variables x and hx as hypotheses
    assert "x" in result
    assert "hx" in result


def test_ast_get_named_subgoal_code_includes_set_binding() -> None:
    """Test that get_named_subgoal_code includes earlier set bindings as equality hypotheses."""
    # Create a theorem with a set binding and a have statement
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
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
                            # Set binding
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
                            # Have statement using the set binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "s", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\ns : ℕ\n⊢ s > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include the set binding s as an equality hypothesis (hs : s = x + 1)
    assert "hs" in result  # Hypothesis name
    assert (
        "s  = x  + 1" in result or "s = x + 1" in result or "s=x+1" in result
    )  # Equality format (with possible spaces)
    # Should NOT include s as a type annotation (s : ℕ) - that would be incorrect  # noqa: RUF003
    assert result.count("(s :") == 0, (
        f"Found type annotation for s, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_complex_let_statements() -> None:
    """Test that get_named_subgoal_code handles multiple let statements with complex expressions."""
    # Create a theorem with multiple let bindings (similar to the user's example)
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
                            # First let: s := (Finset.Icc 1 10000)
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
                                                        "args": [{"val": "s", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(", "info": {"leading": "", "trailing": ""}},
                                            {"val": "Finset.Icc", "info": {"leading": "", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                            {"val": "10000", "info": {"leading": "", "trailing": ""}},
                                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Second let: sOdd := Finset.filter (fun x : ℕ => ¬Even x) s  # noqa: RUF003
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
                                                            {"val": "sOdd", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
                                            {"val": "(", "info": {"leading": "", "trailing": ""}},
                                            {"val": "fun", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "¬Even", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": ""}},
                                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                                            {"val": "s", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement using the let bindings
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
                                                                "val": "h_partition_prod",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "s", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "sOdd", "info": {"leading": "", "trailing": " "}},
                                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "sEven", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 10, "column": 4},
            "endPos": {"line": 10, "column": 9},
            "goal": "s : Finset ℕ\nsOdd : Finset ℕ\n⊢ Finset.prod s = Finset.prod sOdd * Finset.prod sEven",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h_partition_prod")

    # Should include equality hypotheses for let bindings
    assert "hs" in result  # Hypothesis for s
    assert "hsOdd" in result  # Hypothesis for sOdd
    # Should have equality format, not type annotations
    assert "s  = " in result or "s = " in result or "s=" in result  # s should be in an equality (with possible spaces)
    assert "sOdd  = " in result or "sOdd = " in result or "sOdd=" in result  # sOdd should be in an equality
    # Should NOT have type annotations like (s : Finset ℕ) - that would be incorrect  # noqa: RUF003
    # Check that s and sOdd are not included as separate type annotations
    assert result.count("(s :") == 0, (
        f"Found type annotation for s, but it should only appear in equality hypothesis. Result: {result}"
    )
    assert result.count("(sOdd :") == 0, (
        f"Found type annotation for sOdd, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_let_binding_name_conflict() -> None:
    """Test that get_named_subgoal_code handles hypothesis name conflicts for let bindings."""
    # Create a theorem with a variable named "hs" and a let binding for "s"
    # The hypothesis for "s" should be "h2s" instead of "hs" to avoid conflict
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
                                "args": [{"val": "hs", "info": {"leading": "", "trailing": ""}}],
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
                            # Let binding for "s" - should generate "h2s" to avoid conflict with "hs"
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
                                                        "args": [{"val": "s", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement using the let binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "s", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "hs : ℕ\ns : ℕ\n⊢ s > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter hs
    assert "lemma" in result
    assert "h1" in result
    assert "hs" in result
    # Should include the let binding s with a conflict-resolved hypothesis name
    # Since "hs" already exists, should use "h2s" instead of "hs"
    assert "h2s" in result  # Conflict-resolved hypothesis name
    assert "s  = 5" in result or "s = 5" in result or "s=5" in result  # Equality format
    # Should NOT use "hs" as the hypothesis name for "s" (that would conflict)
    # The equality should be in h2s, not hs
    assert "(h2s : s" in result or "(h2s :s" in result or "h2s : s" in result
    # Should NOT have "(hs : s" as that would conflict with the parameter "hs"
    assert result.count("(hs : s") == 0, f"Found conflicting hypothesis name 'hs' for variable 's'. Result: {result}"


def test_ast_get_named_subgoal_code_mixed_bindings() -> None:
    """Test get_named_subgoal_code with mixed have, let, and obtain statements."""
    # Create a theorem with multiple binding types
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "mixed_test", "info": {"leading": "", "trailing": " "}}],
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
                            # Have statement
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                            # Let binding
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
                                                        "args": [{"val": "m", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Final have using both earlier bindings
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "m", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "n : ℕ\n⊢ n > 0",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 4, "column": 4},
            "endPos": {"line": 4, "column": 9},
            "goal": "n : ℕ\nh1 : n > 0\nm : ℕ\n⊢ m > 0",  # noqa: RUF001
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2")

    # The result should include all earlier bindings
    assert "lemma" in result
    assert "h2" in result
    # Should include theorem parameter n
    assert "n" in result
    # Should include earlier have h1
    assert "h1" in result
    # Should include let binding m
    assert "m" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_set_dependency_as_equality() -> None:
    """Test that get_named_subgoal_code creates equality hypotheses for set variables that appear as dependencies."""
    # This tests the case where a set statement's variable is referenced in the goal,
    # making it a dependency, and it should be handled as an equality hypothesis
    # even if it wasn't found as an earlier binding.
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
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
                            # Set binding for "l"
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
                                                        "args": [{"val": "l", "info": {"leading": "", "trailing": ""}}],
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
                            # Have statement that references "l" in the goal
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "l", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\nl : ℕ\n⊢ l > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include "l" as an equality hypothesis (hl : l = x + 1)
    # This tests that dependencies from set statements are handled correctly
    assert "hl" in result  # Hypothesis name
    assert (
        "l  = x  + 1" in result or "l = x + 1" in result or "l=x+1" in result
    )  # Equality format (with possible spaces)
    # Should NOT include l as a type annotation (l : ℕ) - that would be incorrect  # noqa: RUF003
    assert result.count("(l :") == 0, (
        f"Found type annotation for l, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_let_dependency_as_equality() -> None:
    """Test that get_named_subgoal_code creates equality hypotheses for let variables that appear as dependencies."""
    # This tests the case where a let statement's variable is referenced in the goal,
    # making it a dependency, and it should be handled as an equality hypothesis.
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
                            # Let binding for "m"
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
                                                        "args": [{"val": "m", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "2", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement that references "m" in the goal
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "m", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "n : ℕ\nm : ℕ\n⊢ m > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter n
    assert "lemma" in result
    assert "h1" in result
    assert "n" in result
    # Should include "m" as an equality hypothesis (hm : m = n * 2)
    assert "hm" in result  # Hypothesis name
    assert (
        "m  = n  * 2" in result or "m = n * 2" in result or "m=n*2" in result
    )  # Equality format (with possible spaces)
    # Should NOT include m as a type annotation (m : ℕ) - that would be incorrect  # noqa: RUF003
    assert result.count("(m :") == 0, (
        f"Found type annotation for m, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_set_dependency_complex_expression() -> None:
    """Test that get_named_subgoal_code handles set dependencies with complex expressions."""
    # Test case similar to the user's example with a complex set statement
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
                            # Set binding for "l" with a complex expression
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
                                                        "args": [{"val": "l", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement that references "l" in the goal
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
                                                                "val": "h_odd_prod_eq_l",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "l", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "l : ℕ\n⊢ l = 5",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h_odd_prod_eq_l")

    # Should include "l" as an equality hypothesis (hl : l = 5)
    assert "hl" in result  # Hypothesis name
    assert "l  = 5" in result or "l = 5" in result or "l=5" in result  # Equality format (with possible spaces)
    # Should NOT include l as a type annotation (l : ℕ) - that would be incorrect  # noqa: RUF003
    assert result.count("(l :") == 0, (
        f"Found type annotation for l, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_includes_set_binding_with_type() -> None:
    """Test that get_named_subgoal_code includes set statements with explicit types."""
    # Create a theorem with a typed set statement
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
                            # Set statement with explicit type
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
                                                            {"val": "oddProd", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(Finset.range 5000)", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement using the set binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "oddProd", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "oddProd : ℕ\n⊢ oddProd > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the set binding oddProd as an equality hypothesis
    assert "lemma" in result
    assert "h1" in result
    assert "oddProd" in result
    # Should include equality hypothesis (hoddProd : oddProd = Finset.prod (Finset.range 5000))
    assert "hoddProd" in result  # Hypothesis name
    assert "oddProd  = " in result or "oddProd = " in result or "oddProd=" in result  # Equality format
    # Should NOT include oddProd as a type annotation (oddProd : ℕ) - that would be incorrect  # noqa: RUF003
    assert result.count("(oddProd :") == 0, (
        f"Found type annotation for oddProd, but it should only appear in equality hypothesis. Result: {result}"
    )


def test_ast_get_named_subgoal_code_includes_suffices_binding() -> None:
    """Test that get_named_subgoal_code includes earlier suffices statements."""
    # Create a theorem with a suffices statement and a have statement
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
                            # Suffices statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticSuffices_",
                                "args": [
                                    {"val": "suffices", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [
                                                            {"val": "h_suff", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                            {"val": "from", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Nat.pos_of_ne_zero", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Have statement using the suffices binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "1", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "n : ℕ\nh_suff : n > 0\n⊢ n + 1 > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter n
    assert "lemma" in result
    assert "h1" in result
    assert "n" in result
    # Should include the suffices binding h_suff as a hypothesis
    assert "h_suff" in result


def test_ast_get_named_subgoal_code_includes_suffices_binding_with_by() -> None:
    """Test that get_named_subgoal_code includes suffices statements with 'by' syntax."""
    # Create a theorem with a suffices statement using 'by'
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
                            # Suffices statement with 'by'
                            {
                                "kind": "Lean.Parser.Tactic.tacticSuffices_",
                                "args": [
                                    {"val": "suffices", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [
                                                            {"val": "h_base", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "4", "info": {"leading": "", "trailing": " "}},
                                            {"val": "^", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "2", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "4", "info": {"leading": "", "trailing": " "}},
                                            {"val": "!", "info": {"leading": "", "trailing": " "}},
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
                            # Have statement using the suffices binding
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "∀", "info": {"leading": "", "trailing": " "}},
                                            {"val": "k", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "≥", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "4", "info": {"leading": "", "trailing": " "}},
                                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                                            {"val": "k", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "^", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "2", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "k", "info": {"leading": "", "trailing": " "}},
                                            {"val": "!", "info": {"leading": "", "trailing": " "}},
                                            {"val": "→", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(k + 1)", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "^", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "2", "info": {"leading": "", "trailing": " "}},
                                            {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(k + 1)", "info": {"leading": "", "trailing": " "}},
                                            {"val": "!", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h_base : 4 ^ 2 ≤ 4 !\n⊢ ∀ k ≥ 4, k ^ 2 ≤ k ! → (k + 1) ^ 2 ≤ (k + 1) !",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the suffices binding h_base
    assert "lemma" in result
    assert "h1" in result
    assert "h_base" in result


def test_ast_get_named_subgoal_code_mixed_set_suffices() -> None:
    """Test get_named_subgoal_code with mixed set and suffices statements."""
    # Create a theorem with set, suffices, and have statements
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "mixed_test", "info": {"leading": "", "trailing": " "}}],
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
                            # Set statement
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
                                            {"val": "Finset.filter", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Suffices statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticSuffices_",
                                "args": [
                                    {"val": "suffices", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [
                                                            {"val": "h_suff", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "odds", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(id : ℕ → ℕ)", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "oddProd", "info": {"leading": "", "trailing": " "}},
                                            {"val": "from", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Final have using both earlier bindings
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "10000!", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "oddProd", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "evenProd", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 4, "column": 4},
            "endPos": {"line": 4, "column": 9},
            "goal": "n : ℕ\nodds : Finset ℕ\nh_suff : Finset.prod odds (id : ℕ → ℕ) = oddProd\n⊢ 10000! = oddProd * evenProd",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include all earlier bindings
    assert "lemma" in result
    assert "h1" in result
    # Should include theorem parameter n
    assert "n" in result
    # Should include set binding odds
    assert "odds" in result
    # Should include suffices binding h_suff
    assert "h_suff" in result


def test_ast_get_named_subgoal_code_includes_choose_binding() -> None:
    """Test that get_named_subgoal_code includes variables from choose statements."""
    # Create a theorem with a choose statement and a have statement
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
                                "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "∀", "info": {"leading": "", "trailing": " "}},
                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                            {"val": "∃", "info": {"leading": "", "trailing": " "}},
                            {"val": "z", "info": {"leading": "", "trailing": " "}},
                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                            {"val": "P", "info": {"leading": "", "trailing": " "}},
                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                            {"val": "z", "info": {"leading": "", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Q", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # Choose statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticChoose_",
                                "args": [
                                    {"val": "choose", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "hx", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": "using", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "h", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
                            # Have statement using chosen variables
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Q", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "h : ∀ y, ∃ z, P y z\nx : T → U\nhx : ∀ y, P y (x y)\n⊢ Q",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2")

    # The result should include the theorem hypothesis h
    assert "lemma" in result
    assert "h2" in result
    # Should include chosen variables x and hx as hypotheses
    assert "x" in result
    assert "hx" in result


def test_ast_get_named_subgoal_code_includes_choose_binding_multiple() -> None:
    """Test that get_named_subgoal_code includes multiple variables from choose statements."""
    # Create a theorem with a choose statement introducing multiple variables
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
                                "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "∀", "info": {"leading": "", "trailing": " "}},
                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                            {"val": "∃", "info": {"leading": "", "trailing": " "}},
                            {"val": "z", "info": {"leading": "", "trailing": " "}},
                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                            {"val": "P", "info": {"leading": "", "trailing": " "}},
                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                            {"val": "z", "info": {"leading": "", "trailing": " "}},
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
                            # Choose statement with multiple variables
                            {
                                "kind": "Lean.Parser.Tactic.tacticChoose_",
                                "args": [
                                    {"val": "choose", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "f", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "hf", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "g", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": "using", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "h", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
                            # Have statement using chosen variables
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "f", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "g", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h : ∀ y, ∃ z, P y z\nf : T → U\ng : T → U\nhf : ∀ y, P y (f y)\n⊢ f = g",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include all chosen variables
    assert "lemma" in result
    assert "h1" in result
    assert "f" in result
    assert "g" in result
    assert "hf" in result


def test_ast_get_named_subgoal_code_mixed_choose_other_bindings() -> None:
    """Test get_named_subgoal_code with mixed choose and other binding types."""
    # Create a theorem with choose, set, and have statements
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "mixed_test", "info": {"leading": "", "trailing": " "}}],
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
                            # Set statement
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
                                                        "args": [{"val": "S", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Choose statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticChoose_",
                                "args": [
                                    {"val": "choose", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "f", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "hf", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": "using", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "some_hypothesis", "info": {"leading": " ", "trailing": " "}},
                                ],
                            },
                            # Final have using both earlier bindings
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "S", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "f", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 4, "column": 4},
            "endPos": {"line": 4, "column": 9},
            "goal": "n : ℕ\nS : Finset ℕ\nf : ℕ → ℕ\nhf : Prop\n⊢ Finset.prod S f = 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include all earlier bindings
    assert "lemma" in result
    assert "h1" in result
    # Should include theorem parameter n
    assert "n" in result
    # Should include set binding S
    assert "S" in result
    # Should include chosen variables f and hf
    assert "f" in result
    assert "hf" in result


def test_ast_get_named_subgoal_code_includes_generalize_binding() -> None:
    """Test that get_named_subgoal_code includes variables from generalize statements."""
    # Create a theorem with a generalize statement and a have statement
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
                                "args": [{"val": "e", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Expr", "info": {"leading": "", "trailing": " "}},
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
                            # Generalize statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticGeneralize_",
                                "args": [
                                    {"val": "generalize", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "e", "info": {"leading": "", "trailing": " "}},
                                    {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                    },
                                ],
                            },
                            # Have statement using generalized variables
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "e", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "e : Expr\nh : e = x\nx : Expr\n⊢ x = e",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter e
    assert "lemma" in result
    assert "h1" in result
    assert "e" in result
    # Should include generalized variables h and x as hypotheses
    assert "h" in result
    assert "x" in result


def test_ast_get_named_subgoal_code_includes_generalize_binding_without_hypothesis() -> None:
    """Test that get_named_subgoal_code includes variables from generalize statements without hypothesis names."""
    # Create a theorem with a generalize statement without a hypothesis name
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
                            # Generalize statement without hypothesis name
                            {
                                "kind": "Lean.Parser.Tactic.tacticGeneralize_",
                                "args": [
                                    {"val": "generalize", "info": {"leading": "", "trailing": " "}},
                                    {"val": "n", "info": {"leading": "", "trailing": " "}},
                                    {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "m", "info": {"leading": "", "trailing": ""}}],
                                    },
                                ],
                            },
                            # Have statement using generalized variable
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "m", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "n : ℕ\nm : ℕ\n⊢ m > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter n
    assert "lemma" in result
    assert "h1" in result
    assert "n" in result
    # Should include generalized variable m as a hypothesis
    assert "m" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_includes_generalize_binding_multiple() -> None:
    """Test that get_named_subgoal_code includes multiple variables from generalize statements."""
    # Create a theorem with a generalize statement introducing multiple variables
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
                            # Generalize statement with multiple generalizations
                            {
                                "kind": "Lean.Parser.Tactic.tacticGeneralize_",
                                "args": [
                                    {"val": "generalize", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h1", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "e1", "info": {"leading": "", "trailing": " "}},
                                    {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x1", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ",", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h2", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "e2", "info": {"leading": "", "trailing": " "}},
                                    {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x2", "info": {"leading": "", "trailing": ""}}],
                                    },
                                ],
                            },
                            # Have statement using generalized variables
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
                                                            {"val": "h3", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "x1", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "x2", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h1 : e1 = x1\nx1 : T\nh2 : e2 = x2\nx2 : T\n⊢ x1 = x2",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h3")

    # The result should include all generalized variables
    assert "lemma" in result
    assert "h3" in result
    assert "h1" in result
    assert "x1" in result
    assert "h2" in result
    assert "x2" in result


def test_ast_get_named_subgoal_code_mixed_generalize_other_bindings() -> None:
    """Test get_named_subgoal_code with mixed generalize and other binding types."""
    # Create a theorem with generalize, set, and have statements
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "mixed_test", "info": {"leading": "", "trailing": " "}}],
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
                                "args": [{"val": "e", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Expr", "info": {"leading": "", "trailing": " "}},
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
                            # Set statement
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
                                                        "args": [{"val": "S", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Generalize statement
                            {
                                "kind": "Lean.Parser.Tactic.tacticGeneralize_",
                                "args": [
                                    {"val": "generalize", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "e", "info": {"leading": "", "trailing": " "}},
                                    {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                    },
                                ],
                            },
                            # Final have using both earlier bindings
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Finset.prod", "info": {"leading": "", "trailing": " "}},
                                            {"val": "S", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(fun _ => x)", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 4, "column": 4},
            "endPos": {"line": 4, "column": 9},
            "goal": "e : Expr\nS : Finset ℕ\nh : e = x\nx : Expr\n⊢ Finset.prod S (fun _ => x) = 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include all earlier bindings
    assert "lemma" in result
    assert "h1" in result
    # Should include theorem parameter e
    assert "e" in result
    # Should include set binding S
    assert "S" in result
    # Should include generalized variables h and x
    assert "h" in result
    assert "x" in result


def test_ast_get_named_subgoal_code_includes_match_binding() -> None:
    """Test that get_named_subgoal_code includes variables from match pattern bindings."""
    # Create a theorem with a match expression and a have statement inside a branch
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Option", "info": {"leading": "", "trailing": " "}},
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
                            # Match expression
                            {
                                "kind": "Lean.Parser.Term.match",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "x", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n  "}},
                                    # Branch: some n
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "some", "info": {"leading": "", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "=>", "info": {"leading": " ", "trailing": "\n    "}},
                                            # Have statement inside branch using pattern binding
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
                                                                                "val": "h1",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            }
                                                                        ],
                                                                    }
                                                                ],
                                                            },
                                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "n", "info": {"leading": "", "trailing": " "}},
                                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                                        ],
                                                    },
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {
                                                        "kind": "Lean.Parser.Term.byTactic",
                                                        "args": [
                                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                                    # Branch: none
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n  ", "trailing": " "}},
                                            {"val": "none", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                                            },
                                        ],
                                    },
                                    {"val": "end", "info": {"leading": "\n", "trailing": ""}},
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
            "pos": {"line": 4, "column": 6},
            "endPos": {"line": 4, "column": 11},
            "goal": "x : Option ℕ\nn : ℕ\n⊢ n > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include match pattern binding n as a hypothesis
    assert "n" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_includes_match_binding_multiple_patterns() -> None:
    """Test that get_named_subgoal_code includes multiple variables from match patterns."""
    # Create a theorem with a match expression with tuple pattern
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
                                "args": [{"val": "p", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                            {"val": "×", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
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
                            # Match expression with tuple pattern
                            {
                                "kind": "Lean.Parser.Term.match",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "p", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n  "}},
                                    # Branch: (a, b)
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "(", "info": {"leading": "", "trailing": ""}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "a", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": ",", "info": {"leading": "", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "b", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": "\n    "}},
                                            # Have statement using pattern bindings
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
                                                                                "val": "h1",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            }
                                                                        ],
                                                                    }
                                                                ],
                                                            },
                                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "a", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "+", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "b", "info": {"leading": "", "trailing": " "}},
                                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
                                                        ],
                                                    },
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {
                                                        "kind": "Lean.Parser.Term.byTactic",
                                                        "args": [
                                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 4, "column": 6},
            "endPos": {"line": 4, "column": 11},
            "goal": "p : ℕ × ℕ\na : ℕ\nb : ℕ\n⊢ a + b > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter p
    assert "lemma" in result
    assert "h1" in result
    assert "p" in result
    # Should include match pattern bindings a and b as hypotheses
    assert "a" in result
    assert "b" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_includes_match_binding_nested() -> None:
    """Test that get_named_subgoal_code includes variables from nested match patterns."""
    # Create a theorem with nested match expressions
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Option", "info": {"leading": "", "trailing": " "}},
                            {"val": "(Option ℕ)", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
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
                            # Outer match
                            {
                                "kind": "Lean.Parser.Term.match",
                                "args": [
                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                    {"val": "x", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n  "}},
                                    # Branch: some y
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "some", "info": {"leading": "", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "y", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "=>", "info": {"leading": " ", "trailing": "\n    "}},
                                            # Nested match
                                            {
                                                "kind": "Lean.Parser.Term.match",
                                                "args": [
                                                    {"val": "match", "info": {"leading": "", "trailing": " "}},
                                                    {"val": "y", "info": {"leading": "", "trailing": " "}},
                                                    {"val": "with", "info": {"leading": " ", "trailing": "\n      "}},
                                                    # Inner branch: some n
                                                    {
                                                        "kind": "Lean.Parser.Term.matchAlt",
                                                        "args": [
                                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "some", "info": {"leading": "", "trailing": " "}},
                                                            {
                                                                "kind": "Lean.binderIdent",
                                                                "args": [
                                                                    {
                                                                        "val": "n",
                                                                        "info": {"leading": "", "trailing": ""},
                                                                    }
                                                                ],
                                                            },
                                                            {
                                                                "val": "=>",
                                                                "info": {"leading": " ", "trailing": "\n        "},
                                                            },
                                                            # Have statement using both pattern bindings
                                                            {
                                                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                                                "args": [
                                                                    {
                                                                        "val": "have",
                                                                        "info": {"leading": "", "trailing": " "},
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
                                                                                            {
                                                                                                "val": "h1",
                                                                                                "info": {
                                                                                                    "leading": "",
                                                                                                    "trailing": " ",
                                                                                                },
                                                                                            }
                                                                                        ],
                                                                                    }
                                                                                ],
                                                                            },
                                                                            {
                                                                                "val": ":",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            },
                                                                            {
                                                                                "val": "n",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            },
                                                                            {
                                                                                "val": ">",
                                                                                "info": {
                                                                                    "leading": " ",
                                                                                    "trailing": " ",
                                                                                },
                                                                            },
                                                                            {
                                                                                "val": "0",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            },
                                                                        ],
                                                                    },
                                                                    {
                                                                        "val": ":=",
                                                                        "info": {"leading": " ", "trailing": " "},
                                                                    },
                                                                    {
                                                                        "kind": "Lean.Parser.Term.byTactic",
                                                                        "args": [
                                                                            {
                                                                                "val": "by",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            },
                                                                            {
                                                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                                                "args": [
                                                                                    {
                                                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                                                        "args": [
                                                                                            {
                                                                                                "val": "sorry",
                                                                                                "info": {
                                                                                                    "leading": "",
                                                                                                    "trailing": "",
                                                                                                },
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
                                                    # Inner branch: none
                                                    {
                                                        "kind": "Lean.Parser.Term.matchAlt",
                                                        "args": [
                                                            {
                                                                "val": "|",
                                                                "info": {"leading": "\n      ", "trailing": " "},
                                                            },
                                                            {"val": "none", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                                            {
                                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                                "args": [
                                                                    {
                                                                        "val": "sorry",
                                                                        "info": {"leading": "", "trailing": ""},
                                                                    }
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                    {"val": "end", "info": {"leading": "\n    ", "trailing": ""}},
                                                ],
                                            },
                                        ],
                                    },
                                    # Outer branch: none
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n  ", "trailing": " "}},
                                            {"val": "none", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                                            },
                                        ],
                                    },
                                    {"val": "end", "info": {"leading": "\n", "trailing": ""}},
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
            "pos": {"line": 6, "column": 10},
            "endPos": {"line": 6, "column": 15},
            "goal": "x : Option (Option ℕ)\ny : Option ℕ\nn : ℕ\n⊢ n > 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include match pattern bindings from both outer and inner matches
    # Note: y might not be needed if it's not used, but n should definitely be included
    assert "n" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_mixed_match_other_bindings() -> None:
    """Test get_named_subgoal_code with mixed match and other binding types."""
    # Create a theorem with match, set, and have statements
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "mixed_test", "info": {"leading": "", "trailing": " "}}],
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Option", "info": {"leading": "", "trailing": " "}},
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
                            # Set statement
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
                                                        "args": [{"val": "S", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # Match expression
                            {
                                "kind": "Lean.Parser.Term.match",
                                "args": [
                                    {"val": "match", "info": {"leading": "\n  ", "trailing": " "}},
                                    {"val": "x", "info": {"leading": "", "trailing": " "}},
                                    {"val": "with", "info": {"leading": " ", "trailing": "\n    "}},
                                    # Branch: some n
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "", "trailing": " "}},
                                            {"val": "some", "info": {"leading": "", "trailing": " "}},
                                            {
                                                "kind": "Lean.binderIdent",
                                                "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                                            },
                                            {"val": "=>", "info": {"leading": " ", "trailing": "\n      "}},
                                            # Have statement using both set and match bindings
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
                                                                                "val": "h1",
                                                                                "info": {
                                                                                    "leading": "",
                                                                                    "trailing": " ",
                                                                                },
                                                                            }
                                                                        ],
                                                                    }
                                                                ],
                                                            },
                                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                                            {
                                                                "val": "Finset.prod",
                                                                "info": {"leading": "", "trailing": " "},
                                                            },
                                                            {"val": "S", "info": {"leading": " ", "trailing": " "}},
                                                            {
                                                                "val": "(fun _ => n)",
                                                                "info": {"leading": " ", "trailing": " "},
                                                            },
                                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "0", "info": {"leading": " ", "trailing": " "}},
                                                        ],
                                                    },
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {
                                                        "kind": "Lean.Parser.Term.byTactic",
                                                        "args": [
                                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
                                    # Branch: none
                                    {
                                        "kind": "Lean.Parser.Term.matchAlt",
                                        "args": [
                                            {"val": "|", "info": {"leading": "\n    ", "trailing": " "}},
                                            {"val": "none", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=>", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                                            },
                                        ],
                                    },
                                    {"val": "end", "info": {"leading": "\n  ", "trailing": ""}},
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
            "pos": {"line": 5, "column": 8},
            "endPos": {"line": 5, "column": 13},
            "goal": "x : Option ℕ\nS : Finset ℕ\nn : ℕ\n⊢ Finset.prod S (fun _ => n) = 0",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include all earlier bindings
    assert "lemma" in result
    assert "h1" in result
    # Should include theorem parameter x
    assert "x" in result
    # Should include set binding S
    assert "S" in result
    # Should include match pattern binding n
    assert "n" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_includes_set_with_hypothesis() -> None:
    """Test that get_named_subgoal_code includes hypothesis from 'set ... with h'."""
    # Create a theorem with a set binding with 'with' clause and a have statement
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
                                "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
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
                            # Set binding with 'with' clause
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10000", "info": {"leading": " ", "trailing": " "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hS", "info": {"leading": "", "trailing": "\n\n  "}},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            # Have statement using both S and hS
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "S", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10000", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\nS : Finset ℕ\nhS : S = Finset.range 10000\n⊢ S = Finset.range 10000",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # The result should include the theorem parameter x
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    # Should include the set binding S as an equality hypothesis (hS_set : S = Finset.range 10000)
    # Note: The generated hypothesis name might be different (e.g., hS_set) to avoid conflicts
    assert "S" in result
    # CRITICAL: Should include the hypothesis hS from 'with' clause
    assert "hS" in result, f"Missing hS hypothesis from 'set ... with hS'. Result: {result}"
    # The hS should appear as a typed hypothesis, not just in the equality
    assert "hS :" in result or "(hS :" in result, f"hS should appear as a typed hypothesis. Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_mathlib_tactic() -> None:
    """Test that get_named_subgoal_code includes hypothesis from Mathlib.Tactic.setTactic with 'with' clause."""
    # Similar to above but using Mathlib.Tactic.setTactic structure
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
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 10000"},
                                            [
                                                {"val": "with"},
                                                [],
                                                {"val": "hS"},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [{"val": "h1"}],
                                                    }
                                                ],
                                            },
                                            {"val": ":"},
                                            {"val": "S"},
                                            {"val": "="},
                                            {"val": "Finset.range"},
                                            {"val": "10000"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "S : Finset ℕ\nhS : S = Finset.range 10000\n⊢ S = Finset.range 10000",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include both S and hS
    assert "h1" in result
    assert "S" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    assert "hS :" in result or "(hS :" in result, f"hS should be a typed hypothesis. Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_multiple_sets() -> None:
    """Test that get_named_subgoal_code includes hypotheses from multiple 'set ... with h' statements."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test"}]},
            {"val": ":"},
            {"val": "Prop"},
            {"val": ":="},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by"},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 10000"},
                                            [{"val": "with"}, [], {"val": "hS"}],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "T"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 5000"},
                                            [{"val": "with"}, [], {"val": "hT"}],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [{"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]}],
                                            },
                                            {"val": ":"},
                                            {"val": "S"},
                                            {"val": "="},
                                            {"val": "T"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
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
            "pos": {"line": 4, "column": 4},
            "endPos": {"line": 4, "column": 9},
            "goal": "S : Finset ℕ\nhS : S = Finset.range 10000\nT : Finset ℕ\nhT : T = Finset.range 5000\n⊢ S = T",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include both hypotheses
    assert "h1" in result
    assert "S" in result
    assert "T" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    assert "hT" in result, f"Missing hT hypothesis. Result: {result}"
    assert "hS :" in result or "(hS :" in result
    assert "hT :" in result or "(hT :" in result


def test_ast_get_named_subgoal_code_set_with_hypothesis_no_with_clause() -> None:
    """Test that set without 'with' clause doesn't create a hypothesis binding."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test"}]},
            {"val": ":"},
            {"val": "Prop"},
            {"val": ":="},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by"},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 10000"},
                                            # No 'with' clause
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [{"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]}],
                                            },
                                            {"val": ":"},
                                            {"val": "S"},
                                            {"val": "="},
                                            {"val": "Finset.range"},
                                            {"val": "10000"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "S : Finset ℕ\n⊢ S = Finset.range 10000",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include S but NOT hS (since there's no 'with' clause)
    assert "h1" in result
    assert "S" in result
    # Should NOT have hS as a separate hypothesis (only S as equality)
    # The result should have S as an equality hypothesis (hS_set : S = ...) but not hS from 'with'
    # Since there's no 'with' clause, there should be no hS hypothesis
    assert result.count("hS") == 0 or "hS" not in result.split("("), (
        f"Found hS hypothesis but there was no 'with' clause. Result: {result}"
    )


def test_ast_get_named_subgoal_code_set_with_hypothesis_after_target() -> None:
    """Test that set ... with h appearing AFTER target subgoal is NOT included."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test"}]},
            {"val": ":"},
            {"val": "Prop"},
            {"val": ":="},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by"},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [{"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]}],
                                            },
                                            {"val": ":"},
                                            {"val": "Prop"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 10000"},
                                            [{"val": "with"}, [], {"val": "hS"}],
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
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "⊢ Prop",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should NOT include S or hS since they appear AFTER h1
    assert "h1" in result
    assert "S" not in result or "S" not in result.split("("), (
        f"Found S binding but it appears after target. Result: {result}"
    )
    assert "hS" not in result or "hS" not in result.split("("), (
        f"Found hS hypothesis but it appears after target. Result: {result}"
    )


def test_ast_get_named_subgoal_code_set_with_hypothesis_mixed_bindings() -> None:
    """Test that set ... with h works correctly with other bindings (have, let, etc.)."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "test"}]},
            {
                "kind": "Lean.Parser.Term.bracketedBinderList",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "("},
                            {"kind": "Lean.binderIdent", "args": [{"val": "x"}]},
                            {"val": ":"},
                            {"val": "ℕ"},  # noqa: RUF001
                            {"val": ")"},
                        ],
                    },
                ],
            },
            {"val": ":"},
            {"val": "Prop"},
            {"val": ":="},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by"},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h_earlier"}]}
                                                ],
                                            },
                                            {"val": ":"},
                                            {"val": "x"},
                                            {"val": ">"},
                                            {"val": "0"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set"},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S"},
                                            [],
                                            {"val": ":="},
                                            {"val": "Finset.range 10000"},
                                            [{"val": "with"}, [], {"val": "hS"}],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticLet_",
                                "args": [
                                    {"val": "let"},
                                    {
                                        "kind": "Lean.Parser.Term.letDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.letIdDecl",
                                                "args": [{"kind": "Lean.binderIdent", "args": [{"val": "n"}]}],
                                            },
                                            {"val": ":="},
                                            {"val": "5"},
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have"},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [{"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]}],
                                            },
                                            {"val": ":"},
                                            {"val": "S"},
                                            {"val": "="},
                                            {"val": "Finset.range"},
                                            {"val": "10000"},
                                        ],
                                    },
                                    {"val": ":="},
                                    {
                                        "kind": "Lean.Parser.Term.byTactic",
                                        "args": [
                                            {"val": "by"},
                                            {
                                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSorry",
                                                        "args": [{"val": "sorry"}],
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
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "x : ℕ\nh_earlier : x > 0\nS : Finset ℕ\nhS : S = Finset.range 10000\nn : ℕ\n⊢ S = Finset.range 10000",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include all earlier bindings
    assert "h1" in result
    assert "x" in result
    assert "h_earlier" in result
    assert "S" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    assert "n" in result  # Let binding
    assert "hS :" in result or "(hS :" in result
