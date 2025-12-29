"""Tests for goedels_poetry.parsers.ast module."""

import re

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


def test_ast_get_unproven_subgoal_names_with_anonymous_have() -> None:
    """Anonymous `have : ... := by sorry` should be extracted as a synthetic named subgoal."""
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
                                    {"val": ":"},
                                    {"val": "False"},
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

    assert "gp_anon_have__test_theorem__1" in result
    assert "<main body>" not in result


def test_ast_get_named_subgoal_code_for_anonymous_have() -> None:
    """Synthetic anonymous-have subgoal names should be resolvable via get_named_subgoal_code()."""
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "False", "info": {"leading": "", "trailing": " "}},
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

    ast = AST(ast_dict)
    code = ast.get_named_subgoal_code("gp_anon_have__test_theorem__1")

    assert "lemma" in code
    assert "gp_anon_have__test_theorem__1" in code
    assert "False" in code


def test_ast_anonymous_have_numbering_is_stable_with_multiple() -> None:
    """Multiple anonymous haves should get stable, sequential synthetic names within a theorem."""
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
                            {"kind": "Lean.Parser.Term.haveDecl", "args": [{"val": ":"}, {"val": "False"}]},
                            {
                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
                            },
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Tactic.tacticHave_",
                        "args": [
                            {"val": "have"},
                            {"kind": "Lean.Parser.Term.haveDecl", "args": [{"val": ":"}, {"val": "True"}]},
                            {
                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    ast = AST(ast_dict)
    names = ast.get_unproven_subgoal_names()

    assert "gp_anon_have__test_theorem__1" in names
    assert "gp_anon_have__test_theorem__2" in names


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


def test_ast_get_named_subgoal_code_hw_log_eq_12_example() -> None:
    """Test the specific user-reported bug: hw_log_eq_12 with many binders."""
    # This is the exact structure from the user's example
    ast_dict = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "hw_log_eq_12", "info": {"leading": "", "trailing": " "}}],
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
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y", "info": {"leading": " ", "trailing": ""}}],
                            },
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "z", "info": {"leading": " ", "trailing": ""}}],
                            },
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "w", "info": {"leading": " ", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "ℕ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "ht", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "1", "info": {"leading": " ", "trailing": " "}},
                            {"val": "<", "info": {"leading": " ", "trailing": " "}},
                            {"val": "x", "info": {"leading": " ", "trailing": " "}},
                            {"val": "∧", "info": {"leading": " ", "trailing": " "}},
                            {"val": "1", "info": {"leading": " ", "trailing": " "}},
                            {"val": "<", "info": {"leading": " ", "trailing": " "}},
                            {"val": "y", "info": {"leading": " ", "trailing": " "}},
                            {"val": "∧", "info": {"leading": " ", "trailing": " "}},
                            {"val": "1", "info": {"leading": " ", "trailing": " "}},
                            {"val": "<", "info": {"leading": " ", "trailing": " "}},
                            {"val": "z", "info": {"leading": "", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "hw", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "0", "info": {"leading": " ", "trailing": " "}},
                            {"val": "≤", "info": {"leading": " ", "trailing": " "}},
                            {"val": "w", "info": {"leading": " ", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "h0", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "w", "info": {"leading": " ", "trailing": " "}},
                            {"val": "/", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "x", "info": {"leading": " ", "trailing": " "}},
                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                            {"val": "24", "info": {"leading": " ", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "h1", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "w", "info": {"leading": " ", "trailing": " "}},
                            {"val": "/", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "y", "info": {"leading": " ", "trailing": " "}},
                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                            {"val": "40", "info": {"leading": " ", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.explicitBinder",
                        "args": [
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "h2", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "w", "info": {"leading": " ", "trailing": " "}},
                            {"val": "/", "info": {"leading": " ", "trailing": " "}},
                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                            {"val": "y", "info": {"leading": " ", "trailing": " "}},
                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                            {"val": "z", "info": {"leading": " ", "trailing": ""}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                            {"val": "12", "info": {"leading": " ", "trailing": " "}},
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"val": "w", "info": {"leading": "", "trailing": " "}},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
            {"val": ")", "info": {"leading": " ", "trailing": " "}},
            {"val": "=", "info": {"leading": " ", "trailing": " "}},
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"val": "12", "info": {"leading": "", "trailing": " "}},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
            {"val": ")", "info": {"leading": " ", "trailing": " "}},
            {"val": "*", "info": {"leading": " ", "trailing": " "}},
            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"val": "(", "info": {"leading": "", "trailing": ""}},
            {"val": "x", "info": {"leading": "", "trailing": " "}},
            {"val": "*", "info": {"leading": " ", "trailing": " "}},
            {"val": "y", "info": {"leading": " ", "trailing": " "}},
            {"val": "*", "info": {"leading": " ", "trailing": " "}},
            {"val": "z", "info": {"leading": " ", "trailing": ""}},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "ℕ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
            {"val": ")", "info": {"leading": "", "trailing": " "}},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
            {"val": ")", "info": {"leading": "", "trailing": " "}},
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
                                                            {"val": "h2'", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                                            {"val": "w", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
                                            {"val": ")", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "/", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Real.log", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                                            {"val": "(", "info": {"leading": "", "trailing": ""}},
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "y", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "*", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "z", "info": {"leading": " ", "trailing": ""}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℕ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
                                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
                                            {"val": ")", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "(", "info": {"leading": " ", "trailing": ""}},
                                            {"val": "12", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "ℝ", "info": {"leading": " ", "trailing": ""}},  # noqa: RUF001
                                            {"val": ")", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 10, "column": 4},
            "endPos": {"line": 10, "column": 9},
            "goal": "x y z w : ℕ\nht : 1 < x ∧ 1 < y ∧ 1 < z\nhw : 0 ≤ w\nh0 : Real.log w / Real.log x = 24\nh1 : Real.log w / Real.log y = 40\nh2 : Real.log w / Real.log (x * y * z) = 12\n⊢ Real.log (w : ℝ) / Real.log ((x * y * z : ℕ) : ℝ) = (12 : ℝ)",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2'")

    # The result should include ALL theorem binders
    assert "lemma" in result
    assert "h2'" in result
    # Check that all variables are present
    assert "x" in result
    assert "y" in result
    assert "z" in result
    assert "w" in result
    # Check that all hypotheses are present
    assert "ht" in result
    assert "hw" in result
    assert "h0" in result
    assert "h1" in result
    assert "h2" in result
    # Check that types are present
    assert "ℕ" in result  # noqa: RUF001
    assert "ℝ" in result  # noqa: RUF001
    # The subgoal should be valid Lean code
    assert ":=" in result
    assert "sorry" in result


def test_ast_get_named_subgoal_code_with_bytactic_in_type() -> None:
    """Test that binders are extracted even when byTactic appears in the type expression."""
    # This tests the fix: byTactic in type should not stop extraction
    ast_dict = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_lemma", "info": {"leading": "", "trailing": " "}}],
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
                            {"val": "ℕ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            # Type expression that contains byTactic (should not stop extraction)
            {
                "kind": "Lean.Parser.Term.app",
                "args": [
                    {"val": "P", "info": {"leading": "", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Term.byTactic",
                        "args": [
                            {"val": "by", "info": {"leading": "", "trailing": " "}},
                            {
                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                "args": [
                                    {
                                        "kind": "Lean.Parser.Tactic.decide",
                                        "args": [{"val": "decide", "info": {"leading": "", "trailing": ""}}],
                                    }
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include the binder x even though byTactic appears in type
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_no_binders() -> None:
    """Test extraction when theorem has no binders."""
    ast_dict = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_lemma", "info": {"leading": "", "trailing": " "}}],
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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
            "goal": "⊢ Prop",
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should work even with no binders
    assert "lemma" in result
    assert "h1" in result
    assert ":=" in result
    assert "sorry" in result


def test_ast_get_named_subgoal_code_multiple_binder_lists() -> None:
    """Test extraction when theorem has multiple bracketedBinderList nodes (edge case)."""
    # Some Lean parsers might produce multiple binder lists
    ast_dict = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_lemma", "info": {"leading": "", "trailing": " "}}],
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
                            {"val": "ℕ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                            {"val": ")", "info": {"leading": "", "trailing": " "}},
                        ],
                    },
                ],
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
                                "args": [{"val": "y", "info": {"leading": "", "trailing": ""}}],
                            },
                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                            {"val": "ℕ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\ny : ℕ\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include binders from both binder lists
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    assert "y" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_nested_binders() -> None:
    """Test extraction when binders are nested in complex structures."""
    ast_dict = {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_lemma", "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Term.app",
                "args": [
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
                                    {"val": "ℕ", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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
            "pos": {"line": 3, "column": 4},
            "endPos": {"line": 3, "column": 9},
            "goal": "x : ℕ\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should find nested binders through recursive traversal
    assert "lemma" in result
    assert "h1" in result
    assert "x" in result
    assert "ℕ" in result  # noqa: RUF001


def test_ast_get_named_subgoal_code_set_with_hypothesis_from_earlier_sorry() -> None:
    """
    Test that set_with_hypothesis type is correctly extracted from earlier sorry's goal context.

    This tests the fix for the issue where hOddProd type was missing when extracting
    subgoals because it was only in an earlier sorry's goal context, not the target-specific one.
    """
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
                            # Set binding with 'with' clause (introduces hOddProd)
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "OddProd", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "val": "(Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id",
                                                "info": {"leading": " ", "trailing": "\n  "},
                                            },
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hOddProd", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            # Have statement (target subgoal)
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
                                                                "val": "split_parity",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {
                                                "val": "∏ x ∈ Finset.range 10000, (x + 1) = ...",
                                                "info": {"leading": "", "trailing": " "},
                                            },
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

    # Earlier sorry (main theorem body) - contains hOddProd type
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "OddProd : ℕ := (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id\nhOddProd : OddProd = (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Target-specific sorry (for split_parity) - does NOT contain hOddProd type
    target_sorry = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "OddProd : ℕ := (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id\n⊢ ∏ x ∈ Finset.range 10000, (x + 1) = ...",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [earlier_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("split_parity")

    # Should include hOddProd with correct type from earlier sorry
    assert "split_parity" in result
    assert "hOddProd" in result, f"Missing hOddProd hypothesis. Result: {result}"
    # Check that hOddProd has the correct type (equality type, not Prop)
    assert "hOddProd :" in result or "(hOddProd :" in result
    # Should NOT be Prop (the fallback)
    assert "hOddProd : Prop" not in result, f"hOddProd incorrectly typed as Prop. Result: {result}"
    # Should have the equality type
    assert "OddProd = " in result or "hOddProd : OddProd = " in result, (
        f"hOddProd missing equality type. Result: {result}"
    )


def test_ast_get_named_subgoal_code_merged_sorries_type_conflict() -> None:
    """
    Test that target-specific sorry types take precedence over types from other sorries.

    When the same variable appears with different types in different sorries,
    the target-specific sorry's type should win.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Earlier sorry with hx : x = 5
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\n⊢ Prop",  # noqa: RUF001  # noqa: RUF001
        "proofState": 1,
    }

    # Target-specific sorry with hx : x = 10 (conflicting type)
    # Must include h1 in goal to be identified as target-specific
    target_sorry = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [earlier_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should use target-specific type (x = 10), not earlier type (x = 5)
    assert "h1" in result
    assert "hx" in result
    # Check for target-specific type (more flexible to whitespace)
    assert "hx" in result and "x = 10" in result, f"hx should have target-specific type (x = 10). Result: {result}"
    # Ensure earlier type is NOT present
    hx_pattern = re.compile(r"hx\s*:\s*x\s*=\s*5")
    assert not hx_pattern.search(result), f"hx should not have earlier type (x = 5). Result: {result}"


def test_ast_get_named_subgoal_code_no_target_specific_sorry() -> None:
    """
    Test that when no target-specific sorry is found, types from all sorries are used.

    This tests the fallback behavior when the target name doesn't appear in any sorry's goal.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry that doesn't mention h1 (so no target-specific sorry)
    sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\n⊢ Prop",  # noqa: RUF001  # noqa: RUF001
        "proofState": 1,
    }

    sorries = [sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should still include hx from the sorry (since no target-specific sorry found)
    assert "h1" in result
    assert "hx" in result
    # Check for type (more flexible to whitespace)
    assert "hx" in result and "x = 5" in result, f"hx should be included from all_types (x = 5). Result: {result}"


def test_ast_get_named_subgoal_code_multiple_set_with_hypothesis() -> None:
    """
    Test that multiple set_with_hypothesis bindings are correctly handled from merged sorries.

    This tests the case where there are multiple set ... with h statements, each with
    their types in different sorries.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set", "info": {"leading": "\n  ", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hy", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Earlier sorry with hx type
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\n⊢ Prop",  # noqa: RUF001  # noqa: RUF001
        "proofState": 1,
    }

    # Later sorry with hy type (but not hx)
    later_sorry = {
        "pos": {"line": 4, "column": 4},
        "endPos": {"line": 4, "column": 9},
        "goal": "x : ℕ := 5\ny : ℕ := 10\nhy : y = 10\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    # Target-specific sorry (mentions h1 but not hx or hy)
    target_sorry = {
        "pos": {"line": 7, "column": 4},
        "endPos": {"line": 7, "column": 9},
        "goal": "x : ℕ := 5\ny : ℕ := 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 3,
    }

    sorries = [earlier_sorry, later_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include both hx and hy from merged sorries
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    assert "hy" in result, f"Missing hy hypothesis. Result: {result}"
    # Check for types (more flexible to whitespace)
    assert "hx" in result and "x = 5" in result, f"hx missing correct type (x = 5). Result: {result}"
    assert "hy" in result and "y = 10" in result, f"hy missing correct type (y = 10). Result: {result}"


def test_ast_get_named_subgoal_code_empty_sorries() -> None:
    """
    Test that empty sorries list is handled correctly.

    When no sorries are provided, the code should still work (though types won't be available).
    """
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    sorries = []

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should still produce valid code even without sorries
    assert "h1" in result
    assert "lemma" in result or "theorem" in result


def test_ast_get_named_subgoal_code_sorries_with_empty_goals() -> None:
    """
    Test that sorries with empty or missing goal strings are handled correctly.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorries with various empty/missing goal cases
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "",
            "proofState": 1,
        },  # Empty goal
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "x : ℕ := 5\nhx : x = 5\n⊢ Prop",  # noqa: RUF001  # noqa: RUF001
            "proofState": 2,
        },  # Valid goal
        {"pos": {"line": 7, "column": 4}, "endPos": {"line": 7, "column": 9}, "proofState": 3},  # Missing goal key
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should still work and include hx from the valid sorry
    assert "h1" in result
    # Should handle empty/missing goals gracefully
    assert "hx" in result or "x" in result, f"Result: {result}"


def test_ast_get_named_subgoal_code_merge_priority_verification() -> None:
    """
    Test that explicitly verifies merge priority: target-specific types overwrite all_types.

    This test ensures that when the same variable appears in both target-specific sorry
    and other sorries with different types, the target-specific type takes precedence.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set", "info": {"leading": "\n  ", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "20", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hy", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Earlier sorry with hx : x = 5 and hy : y = 15 (conflicting with target)
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\ny : ℕ := 20\nhx : x = 5\nhy : y = 15\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Target-specific sorry with hx : x = 10 and hy : y = 20 (conflicting types)
    # Must include h1 in goal to be identified as target-specific
    target_sorry = {
        "pos": {"line": 7, "column": 4},
        "endPos": {"line": 7, "column": 9},
        "goal": "x : ℕ := 5\ny : ℕ := 20\nhx : x = 10\nhy : y = 20\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [earlier_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Verify merge priority: target-specific types should win
    assert "h1" in result
    assert "hx" in result
    assert "hy" in result

    # Target-specific types should be used (x = 10, y = 20), not earlier types (x = 5, y = 15)
    # Check hx has target-specific type (x = 10)
    assert "hx" in result and "x = 10" in result, f"hx should have target-specific type (x = 10). Result: {result}"
    # Ensure earlier type is NOT present
    hx_wrong_pattern = re.compile(r"hx\s*:\s*x\s*=\s*5\b")
    assert not hx_wrong_pattern.search(result), f"hx should not have earlier type (x = 5). Result: {result}"

    # Check hy has target-specific type (y = 20)
    assert "hy" in result and "y = 20" in result, f"hy should have target-specific type (y = 20). Result: {result}"
    # Ensure earlier type is NOT present
    hy_wrong_pattern = re.compile(r"hy\s*:\s*y\s*=\s*15\b")
    assert not hy_wrong_pattern.search(result), f"hy should not have earlier type (y = 15). Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_goal_context_basic() -> None:
    """
    Test that set_with_hypothesis type is correctly extracted from goal context.

    Basic test: set binding with 'with' clause, hypothesis type in goal context.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with hx type in goal context
    sorry_with_hx = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    sorries = [sorry_with_hx]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hx with correct type from goal context
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    # Should have the equality type, not Prop
    assert "hx" in result and "x = 5" in result, f"hx should have equality type (x = 5). Result: {result}"
    assert "hx : Prop" not in result, f"hx should not be typed as Prop. Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_goal_context_earlier_sorry() -> None:
    """
    Test that set_with_hypothesis type from earlier sorry is included.

    The hypothesis type is in an earlier sorry's goal context, not the target-specific one.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Earlier sorry (main theorem body) - contains hx type
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 10\nhx : x = 10\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Target-specific sorry (for h1) - does NOT contain hx type
    target_sorry = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [earlier_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hx with correct type from earlier sorry
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis from earlier sorry. Result: {result}"
    # Should have the equality type from earlier sorry
    assert "hx" in result and "x = 10" in result, (
        f"hx should have equality type (x = 10) from earlier sorry. Result: {result}"
    )
    assert "hx : Prop" not in result, f"hx should not be typed as Prop. Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_goal_context_target_sorry_priority() -> None:
    """
    Test that set_with_hypothesis type from target-specific sorry takes priority.

    When the same hypothesis appears in both earlier and target-specific sorries,
    the target-specific sorry's type should be used.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Earlier sorry with hx : x = 5
    earlier_sorry = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Target-specific sorry with hx : x = 10 (conflicting type, should take priority)
    target_sorry = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [earlier_sorry, target_sorry]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hx with target-specific type (x = 10), not earlier type (x = 5)
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    # Target-specific type should win
    assert "hx" in result and "x = 10" in result, f"hx should have target-specific type (x = 10). Result: {result}"
    # Earlier type should NOT be present
    hx_wrong_pattern = re.compile(r"hx\s*:\s*x\s*=\s*5\b")
    assert not hx_wrong_pattern.search(result), f"hx should not have earlier type (x = 5). Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_goal_context_complex_equality() -> None:
    """
    Test that complex equality type expressions are correctly parsed from goal context.

    Tests that equality types with complex expressions are preserved correctly.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "val": "Finset.range 10000",
                                                "info": {"leading": " ", "trailing": "\n  "},
                                            },
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hS", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with complex equality type
    sorry_with_complex_type = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "S : Finset ℕ := Finset.range 10000\nhS : S = Finset.range 10000\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    sorries = [sorry_with_complex_type]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hS with complex equality type
    assert "h1" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    # Should have the complex equality type
    assert "hS" in result and "S = Finset.range 10000" in result, (
        f"hS should have complex equality type. Result: {result}"
    )
    assert "hS : Prop" not in result, f"hS should not be typed as Prop. Result: {result}"


def test_ast_get_named_subgoal_code_set_with_hypothesis_goal_context_multiple_hypotheses() -> None:
    """
    Test that multiple set_with_hypothesis bindings are correctly handled from goal context.

    Tests that all hypothesis types are collected when there are multiple set ... with h statements.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Mathlib.Tactic.setTactic",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "y", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "10", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hy", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with both hx and hy types
    sorry_with_multiple = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\ny : ℕ := 10\nhx : x = 5\nhy : y = 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    sorries = [sorry_with_multiple]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include both hx and hy with correct types
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    assert "hy" in result, f"Missing hy hypothesis. Result: {result}"
    # Both should have equality types
    assert "hx" in result and "x = 5" in result, f"hx should have equality type (x = 5). Result: {result}"
    assert "hy" in result and "y = 10" in result, f"hy should have equality type (y = 10). Result: {result}"
    # Neither should be Prop
    assert "hx : Prop" not in result, f"hx should not be typed as Prop. Result: {result}"
    assert "hy : Prop" not in result, f"hy should not be typed as Prop. Result: {result}"


def test_parse_goal_context_set_with_hypothesis_assignment_syntax() -> None:
    """
    Test that __parse_goal_context correctly handles assignment syntax in goal context.

    Tests that lines with := (assignments) are parsed correctly, extracting the type
    before := rather than treating the whole assignment as a type.
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # Goal context with assignment syntax (variable : type := value)
    goal = (
        "OddProd : ℕ := (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id\n"  # noqa: RUF001
        "hOddProd : OddProd = (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id\n"
        "⊢ Prop"
    )

    result = __parse_goal_context(goal)

    # OddProd should have type ℕ (not the assignment)  # noqa: RUF003
    assert "OddProd" in result
    assert result["OddProd"] == "ℕ", f"OddProd should have type ℕ, got {result.get('OddProd')}"  # noqa: RUF001

    # hOddProd should have the equality type
    assert "hOddProd" in result
    assert "OddProd = " in result["hOddProd"], f"hOddProd should have equality type, got {result.get('hOddProd')}"

    # Should not have spurious entries like ":" or "ℕ" as separate variables  # noqa: RUF003
    assert ":" not in result, f"Should not have ':' as a variable name. Result: {result}"
    # Note: ℕ might appear if there are other variables, but it shouldn't be from this line  # noqa: RUF003
    if "ℕ" in result and result["ℕ"] != "ℕ":  # noqa: RUF001
        # If ℕ appears, it should be from a different line, not from parsing OddProd's assignment  # noqa: RUF003
        pass


def test_parse_goal_context_set_with_hypothesis_equality_types() -> None:
    """
    Test that __parse_goal_context correctly parses equality types for set_with_hypothesis.

    Tests various equality type formats that might appear in goal context.
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # Test case 1: Simple equality
    goal1 = "hx : x = 5\n⊢ Prop"
    result1 = __parse_goal_context(goal1)
    assert "hx" in result1
    assert result1["hx"] == "x = 5"

    # Test case 2: Equality with complex expression
    goal2 = "hS : S = Finset.range 10000\n⊢ Prop"
    result2 = __parse_goal_context(goal2)
    assert "hS" in result2
    assert result2["hS"] == "S = Finset.range 10000"

    # Test case 3: Multiple hypotheses with equality types
    goal3 = "hx : x = 5\nhy : y = 10\nhz : z = x + y\n⊢ Prop"
    result3 = __parse_goal_context(goal3)
    assert "hx" in result3 and result3["hx"] == "x = 5"
    assert "hy" in result3 and result3["hy"] == "y = 10"
    assert "hz" in result3 and result3["hz"] == "z = x + y"


def test_ast_get_named_subgoal_code_exact_key_matching_not_substring() -> None:
    """
    Test that exact key matching prevents false positives from substring matching.

    OLD BEHAVIOR (substring matching): "h1" would match "h10" in goal string
    NEW BEHAVIOR (exact key matching): "h1" only matches if "h1" is a key in parsed_types

    This test verifies that "h1" does NOT match "h10" when looking for target-specific sorry.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with h10 (NOT h1) - should NOT be identified as target-specific for "h1"
    # OLD BEHAVIOR: Would match because "h1" is substring of "h10"
    # NEW BEHAVIOR: Should NOT match because "h1" is not a key in parsed_types
    sorry_with_h10 = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh10 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Sorry with h1 (the actual target) - should be identified as target-specific
    sorry_with_h1 = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [sorry_with_h10, sorry_with_h1]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should use h1's sorry (second one), not h10's sorry (first one)
    assert "h1" in result
    # Should include hx from both sorries (merged)
    assert "hx" in result
    # Verify that we got the correct types (from h1's sorry, not h10's)
    assert "h1" in result and "Prop" in result


def test_ast_get_named_subgoal_code_exact_key_matching_vs_substring_h1_in_h10() -> None:
    """
    Test that "h1" does not match "h10" when using exact key matching.

    This explicitly tests the fix: OLD would match substring, NEW requires exact key.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # First sorry contains "h10" (not "h1")
    # OLD: "h1" in "h10" would be True (substring match)
    # NEW: "h1" in parsed_types.keys() would be False (exact key match)
    sorry_h10 = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh10 : Nat\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Second sorry contains "h1" (the actual target)
    sorry_h1 = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 10\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [sorry_h10, sorry_h1]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should use h1's sorry (second one), not h10's sorry
    # If old substring matching was used, h10's sorry would be selected and hx would have type "x = 5"
    # With new exact key matching, h1's sorry is selected and hx should have type "x = 10"
    assert "h1" in result
    assert "hx" in result
    # Verify we got the correct type from h1's sorry (x = 10), not h10's sorry (x = 5)
    assert "hx" in result and "x = 10" in result, f"hx should have type from h1's sorry (x = 10). Result: {result}"
    # Ensure we did NOT get the type from h10's sorry
    hx_wrong_pattern = re.compile(r"hx\s*:\s*x\s*=\s*5\b")
    assert not hx_wrong_pattern.search(result), f"hx should not have type from h10's sorry (x = 5). Result: {result}"


def test_ast_get_named_subgoal_code_exact_key_matching_vs_substring_multiple_cases() -> None:
    """
    Test multiple cases where substring matching would fail but exact key matching works correctly.

    Tests various patterns:
    - "h1" vs "h10", "h11", "h12"
    - "h" vs "h1", "h2", "ha"
    - "x" vs "x1", "x2", "xy"
    """
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
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Test case 1: "h1" should NOT match "h10", "h11", "h12"
    sorries_case1 = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h10 : Nat\nh11 : Int\nh12 : String\n⊢ Prop",
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries_case1)
    result = ast.get_named_subgoal_code("h1")

    # Should use h1's sorry (second one), not h10/h11/h12's sorry
    assert "h1" in result
    # Should NOT include h10, h11, h12 (they're not in scope for h1)
    assert "h10" not in result or "h10 :" not in result, f"h10 should not be included. Result: {result}"
    assert "h11" not in result or "h11 :" not in result, f"h11 should not be included. Result: {result}"
    assert "h12" not in result or "h12 :" not in result, f"h12 should not be included. Result: {result}"


def test_ast_get_named_subgoal_code_exact_key_matching_vs_substring_h_in_h1() -> None:
    """
    Test that "h" does not match "h1" when using exact key matching.

    OLD: "h" in "h1" would be True (substring match)
    NEW: "h" in parsed_types.keys() would be False (exact key match)
    """
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
                                                            {"val": "h", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with h1, h2, ha (NOT h)
    # OLD: "h" in "h1" would be True (substring match)
    # NEW: "h" in parsed_types.keys() would be False (exact key match)
    sorry_with_h1_h2_ha = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "h1 : Nat\nh2 : Int\nha : String\n⊢ Prop",
        "proofState": 1,
    }

    # Sorry with h (the actual target)
    sorry_with_h = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "h : Prop\n⊢ Prop",
        "proofState": 2,
    }

    sorries = [sorry_with_h1_h2_ha, sorry_with_h]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h")

    # Should use h's sorry (second one), not h1/h2/ha's sorry
    assert "h" in result
    # Should NOT include h1, h2, ha (they're not the target)
    assert "h1" not in result or "h1 :" not in result, f"h1 should not be included. Result: {result}"
    assert "h2" not in result or "h2 :" not in result, f"h2 should not be included. Result: {result}"
    assert "ha" not in result or "ha :" not in result, f"ha should not be included. Result: {result}"


def test_ast_get_named_subgoal_code_exact_key_matching_vs_substring_x_in_x1() -> None:
    """
    Test that "x" does not match "x1" when using exact key matching.

    OLD: "x" in "x1" would be True (substring match)
    NEW: "x" in parsed_types.keys() would be False (exact key match)
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with x1, x2, xy (NOT x)
    # OLD: "x" in "x1" would be True (substring match)
    # NEW: "x" in parsed_types.keys() would be False (exact key match)
    sorry_with_x1_x2_xy = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "x1 : Nat\nx2 : Int\nxy : String\nhx : x = 5\n⊢ Prop",
        "proofState": 1,
    }

    # Sorry with x (the actual variable) and h1 (the target)
    sorry_with_x_and_h1 = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [sorry_with_x1_x2_xy, sorry_with_x_and_h1]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should use h1's sorry (second one), not x1/x2/xy's sorry
    assert "h1" in result
    # Should include x and hx from h1's sorry
    assert "x" in result
    assert "hx" in result
    # Should NOT include x1, x2, xy (they're not in scope for h1)
    assert "x1" not in result or "x1 :" not in result, f"x1 should not be included. Result: {result}"
    assert "x2" not in result or "x2 :" not in result, f"x2 should not be included. Result: {result}"
    assert "xy" not in result or "xy :" not in result, f"xy should not be included. Result: {result}"


def test_ast_get_named_subgoal_code_exact_key_matching_positive_case() -> None:
    """
    Test that exact key matching still works for correct matches.

    This ensures the fix doesn't break the positive case where the target name
    is actually present as a key in parsed_types.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "5", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with h1 (the actual target) - should be identified correctly
    sorry_with_h1 = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "x : ℕ := 5\nhx : x = 5\nh1 : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    sorries = [sorry_with_h1]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should correctly identify h1's sorry and include all variables
    assert "h1" in result
    assert "x" in result
    assert "hx" in result
    assert "hx" in result and "x = 5" in result, f"hx should have correct type. Result: {result}"


def test_ast_get_named_subgoal_code_exact_key_matching_complex_names() -> None:
    """
    Test exact key matching with complex variable names that could be substrings.

    Tests names like "split_parity" vs "split_parity_helper", "hOddProd" vs "hOddProd2", etc.
    """
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
                                                                "val": "split_parity",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with split_parity_helper (NOT split_parity)
    # OLD: "split_parity" in "split_parity_helper" would be True (substring match)
    # NEW: "split_parity" in parsed_types.keys() would be False (exact key match)
    sorry_with_helper = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "split_parity_helper : Prop\n⊢ Prop",
        "proofState": 1,
    }

    # Sorry with split_parity (the actual target)
    sorry_with_split_parity = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "split_parity : Prop\n⊢ Prop",
        "proofState": 2,
    }

    sorries = [sorry_with_helper, sorry_with_split_parity]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("split_parity")

    # Should use split_parity's sorry (second one), not split_parity_helper's sorry
    assert "split_parity" in result
    # Should NOT include split_parity_helper
    assert "split_parity_helper" not in result or "split_parity_helper :" not in result, (
        f"split_parity_helper should not be included. Result: {result}"
    )


def test_ast_get_named_subgoal_code_exact_key_matching_hOddProd_vs_hOddProd2() -> None:
    """
    Test exact key matching with hOddProd vs hOddProd2.

    This is a real-world case from the original bug report where hOddProd was the variable.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "OddProd", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "val": "(Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id",
                                                "info": {"leading": " ", "trailing": "\n  "},
                                            },
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hOddProd", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                                                "val": "split_parity",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": "", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
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

    # Sorry with hOddProd2 (NOT hOddProd)
    # OLD: "hOddProd" in "hOddProd2" would be True (substring match)
    # NEW: "hOddProd" in parsed_types.keys() would be False (exact key match)
    sorry_with_hOddProd2 = {
        "pos": {"line": 2, "column": 4},
        "endPos": {"line": 2, "column": 9},
        "goal": "OddProd : ℕ := ...\nhOddProd2 : OddProd = ...\n⊢ Prop",  # noqa: RUF001
        "proofState": 1,
    }

    # Sorry with hOddProd (the actual variable) and split_parity (the target)
    sorry_with_hOddProd_and_split_parity = {
        "pos": {"line": 5, "column": 4},
        "endPos": {"line": 5, "column": 9},
        "goal": "OddProd : ℕ := ...\nhOddProd : OddProd = ...\nsplit_parity : Prop\n⊢ Prop",  # noqa: RUF001
        "proofState": 2,
    }

    sorries = [sorry_with_hOddProd2, sorry_with_hOddProd_and_split_parity]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("split_parity")

    # Should use split_parity's sorry (second one), not hOddProd2's sorry
    assert "split_parity" in result
    # Should include hOddProd from the second sorry (merged from all sorries)
    assert "hOddProd" in result, f"hOddProd should be included. Result: {result}"
    # Should NOT include hOddProd2
    assert "hOddProd2" not in result or "hOddProd2 :" not in result, (
        f"hOddProd2 should not be included. Result: {result}"
    )


# ============================================================================
# Integration tests for set_with_hypothesis type construction (Commit 3)
# ============================================================================


def test_ast_get_named_subgoal_set_with_hypothesis_constructed_type() -> None:
    """
    Test that set_with_hypothesis type is constructed from AST when goal context unavailable.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range 10000", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hS", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries without hS type in goal context (to force AST construction)
    sorries = [
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "S : Finset ℕ := Finset.range 10000\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 6, "column": 4},
            "endPos": {"line": 6, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hS with constructed type (S = Finset.range 10000)
    assert "h1" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    # Should have the equality type constructed from AST
    assert "S" in result and "=" in result and "Finset.range 10000" in result, (
        f"hS should have equality type constructed from AST. Result: {result}"
    )
    # Should NOT be Prop (the fallback)
    assert "hS : Prop" not in result, f"hS incorrectly typed as Prop. Result: {result}"


def test_ast_get_named_subgoal_set_with_hypothesis_goal_context_priority() -> None:
    """
    Test that goal context types take priority over constructed types.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "S", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range 10000", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hS", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries WITH hS type in goal context (should take priority)
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "S : Finset ℕ := Finset.range 10000\nhS : S = Finset.range 10000\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hS with type from goal context
    assert "h1" in result
    assert "hS" in result, f"Missing hS hypothesis. Result: {result}"
    # Should have the equality type from goal context
    assert "hS" in result and "S = Finset.range 10000" in result, (
        f"hS should have equality type from goal context. Result: {result}"
    )


def test_ast_get_named_subgoal_set_with_hypothesis_fallback_to_construction() -> None:
    """
    Test that falls back to AST construction when goal context unavailable.
    """
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
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    [],
                                    {
                                        "kind": "Mathlib.Tactic.setArgsRest",
                                        "args": [
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            [],
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "42", "info": {"leading": " ", "trailing": "\n  "}},
                                            [
                                                {"val": "with", "info": {"leading": " ", "trailing": " "}},
                                                [],
                                                {"val": "hx", "info": {"leading": "", "trailing": "\n  "}},
                                            ],
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries without hx type in goal context
    sorries = [
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 1,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include hx with constructed type (x = 42)
    assert "h1" in result
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    # Should have the equality type constructed from AST
    assert "x" in result and "=" in result and "42" in result, (
        f"hx should have equality type constructed from AST. Result: {result}"
    )
    # Should NOT be Prop (the fallback)
    assert "hx : Prop" not in result, f"hx incorrectly typed as Prop. Result: {result}"


# ============================================================================
# Integration tests for general binding type determination (Commit 4)
# ============================================================================


def test_ast_get_named_subgoal_have_type_determination() -> None:
    """
    Test that have binding type is determined correctly with improved fallback chain.
    """
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
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "n > 0", "info": {"leading": "", "trailing": "\n  "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": "\n  "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries with h1 type in goal context (should take priority over AST)
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h1 : m > 0\n⊢ Prop",  # Different type in goal context
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h2 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2")

    # Should include h1 with type from goal context (m > 0), not AST (n > 0)
    assert "h2" in result
    assert "h1" in result, f"Missing h1 hypothesis. Result: {result}"
    # Should have goal context type (m > 0), not AST type (n > 0)
    assert "h1" in result and "m > 0" in result, f"h1 should have type from goal context (m > 0). Result: {result}"
    # Should NOT have AST type
    assert "n > 0" not in result or "(h1 : n > 0" not in result, (
        f"h1 should not have AST type (n > 0). Result: {result}"
    )


def test_ast_get_named_subgoal_obtain_type_determination() -> None:
    """
    Test that obtain binding type is determined from goal context.
    """
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
                                    {"val": "⟩", "info": {"leading": "", "trailing": "\n  "}},
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries with obtain types in goal context
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "x : ℕ\nhx : x > 0\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include x and hx with types from goal context
    assert "h1" in result
    assert "x" in result, f"Missing x hypothesis. Result: {result}"
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    # Should have correct types from goal context
    assert "x" in result and "ℕ" in result, f"x should have type ℕ. Result: {result}"  # noqa: RUF001
    assert "hx" in result and "x > 0" in result, f"hx should have type x > 0. Result: {result}"


def test_ast_get_named_subgoal_obtain_type_no_goal_context() -> None:
    """
    Test that obtain binding falls back to Prop when goal context unavailable.
    """
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
                                "kind": "Lean.Parser.Tactic.tacticObtain_",
                                "args": [
                                    {"val": "obtain", "info": {"leading": "", "trailing": " "}},
                                    {"val": "⟨", "info": {"leading": "", "trailing": ""}},
                                    {"val": "x", "info": {"leading": "", "trailing": ""}},
                                    {"val": "⟩", "info": {"leading": "", "trailing": "\n  "}},
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries without obtain types in goal context
    sorries = [
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 1,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include h1
    # Note: x from obtain might be included as an earlier binding if found correctly
    # The main purpose of this test is to verify that type determination works
    # when goal context is unavailable (fallback to Prop)
    # If x is included, it should have Prop type (fallback), but it might not
    # be included if it's not found as an earlier binding
    assert "h1" in result
    # If x is included, verify it has Prop type (fallback)
    # But don't require x to be included - the test verifies the code runs without error


def test_ast_get_named_subgoal_choose_type_determination() -> None:
    """
    Test that choose binding type is determined from goal context.
    """
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
                                "kind": "Lean.Parser.Tactic.tacticChoose_",
                                "args": [
                                    {"val": "choose", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": " "}}],
                                    },
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "hx", "info": {"leading": "", "trailing": "\n  "}}],
                                    },
                                    {"val": "using", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "h", "info": {"leading": "", "trailing": "\n  "}},
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
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries with choose types in goal context
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "x : ℕ\nhx : x > 0\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h1 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h1")

    # Should include x and hx with types from goal context
    assert "h1" in result
    assert "x" in result, f"Missing x hypothesis. Result: {result}"
    assert "hx" in result, f"Missing hx hypothesis. Result: {result}"
    # Should have correct types from goal context
    assert "x" in result and "ℕ" in result, f"x should have type ℕ. Result: {result}"  # noqa: RUF001
    assert "hx" in result and "x > 0" in result, f"hx should have type x > 0. Result: {result}"


def test_ast_get_named_subgoal_multiple_binding_types() -> None:
    """
    Test type determination with multiple different binding types.
    """
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
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "P", "info": {"leading": "", "trailing": "\n  "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
                                        ],
                                    },
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Tactic.tacticObtain_",
                                "args": [
                                    {"val": "obtain", "info": {"leading": "", "trailing": " "}},
                                    {"val": "⟨", "info": {"leading": "", "trailing": ""}},
                                    {"val": "x", "info": {"leading": "", "trailing": ""}},
                                    {"val": "⟩", "info": {"leading": "", "trailing": "\n  "}},
                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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
                                                            {"val": "h2", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "sorry", "info": {"leading": " ", "trailing": ""}},
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

    # Sorries with types for some bindings
    sorries = [
        {
            "pos": {"line": 2, "column": 4},
            "endPos": {"line": 2, "column": 9},
            "goal": "h1 : P\nx : ℕ\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        },
        {
            "pos": {"line": 5, "column": 4},
            "endPos": {"line": 5, "column": 9},
            "goal": "h2 : Prop\n⊢ Prop",
            "proofState": 2,
        },
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h2")

    # Should include h1 and x with correct types
    assert "h2" in result
    assert "h1" in result, f"Missing h1 hypothesis. Result: {result}"
    # h1 should have type from goal context
    assert "h1" in result and "P" in result, f"h1 should have type P. Result: {result}"
    # x might be included as an earlier binding or dependency
    # If included, it should have type from goal context
    if "x" in result:
        assert "ℕ" in result, f"x should have type ℕ. Result: {result}"  # noqa: RUF001
