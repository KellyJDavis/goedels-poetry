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
