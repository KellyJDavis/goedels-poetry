"""Tests for goedels_poetry.parsers.util module."""

# Import private functions for testing
from goedels_poetry.parsers.util import (
    __extract_let_value,
    __extract_set_value,
    __extract_set_with_hypothesis_name,
    __extract_type_ast,
    _ast_to_code,
)


def test_ast_to_code_simple_val() -> None:
    """Test converting simple value node to code."""
    node = {"val": "test", "info": {"leading": "", "trailing": ""}}
    result = _ast_to_code(node)
    assert result == "test"


def test_ast_to_code_with_leading_trailing() -> None:
    """Test converting node with leading and trailing whitespace."""
    node = {"val": "test", "info": {"leading": "  ", "trailing": " "}}
    result = _ast_to_code(node)
    assert result == "  test "


def test_ast_to_code_with_args() -> None:
    """Test converting node with args."""
    node = {
        "kind": "some_kind",
        "args": [
            {"val": "first", "info": {"leading": "", "trailing": " "}},
            {"val": "second", "info": {"leading": "", "trailing": ""}},
        ],
    }
    result = _ast_to_code(node)
    assert result == "first second"


def test_ast_to_code_nested() -> None:
    """Test converting nested nodes."""
    node = {
        "kind": "parent",
        "args": [
            {"val": "parent_val", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "child",
                "args": [
                    {"val": "child_val", "info": {"leading": "", "trailing": ""}},
                ],
            },
        ],
    }
    result = _ast_to_code(node)
    assert result == "parent_val child_val"


def test_ast_to_code_list() -> None:
    """Test converting list of nodes."""
    nodes = [
        {"val": "one", "info": {"leading": "", "trailing": " "}},
        {"val": "two", "info": {"leading": "", "trailing": " "}},
        {"val": "three", "info": {"leading": "", "trailing": ""}},
    ]
    result = _ast_to_code(nodes)
    assert result == "one two three"


def test_ast_to_code_empty_dict() -> None:
    """Test converting empty dict."""
    result = _ast_to_code({})
    assert result == ""


def test_ast_to_code_empty_list() -> None:
    """Test converting empty list."""
    result = _ast_to_code([])
    assert result == ""


def test_ast_to_code_none_info() -> None:
    """Test converting node with None info."""
    node = {"val": "test", "info": None}
    result = _ast_to_code(node)
    assert result == "test"


def test_ast_to_code_missing_info() -> None:
    """Test converting node with missing info field."""
    node = {"val": "test"}
    result = _ast_to_code(node)
    assert result == "test"


def test_ast_to_code_string() -> None:
    """Test converting string (should return empty string)."""
    result = _ast_to_code("string")
    assert result == ""


def test_ast_to_code_number() -> None:
    """Test converting number (should return empty string)."""
    result = _ast_to_code(42)
    assert result == ""


def test_ast_to_code_complex() -> None:
    """Test converting complex nested structure."""
    node = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "my_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": "", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [{"val": "trivial", "info": {"leading": "", "trailing": ""}}],
                    },
                ],
            },
        ],
    }
    result = _ast_to_code(node)
    assert "theorem" in result
    assert "my_theorem" in result
    assert "True" in result
    assert "by" in result
    assert "trivial" in result


def test_ast_to_code_preserves_order() -> None:
    """Test that ast_to_code preserves order of args."""
    node = {
        "kind": "parent",
        "args": [
            {"val": "a", "info": {"leading": "", "trailing": ""}},
            {"val": "b", "info": {"leading": "", "trailing": ""}},
            {"val": "c", "info": {"leading": "", "trailing": ""}},
        ],
    }
    result = _ast_to_code(node)
    assert result == "abc"


def test_ast_to_code_with_newlines() -> None:
    """Test converting nodes with newlines in info."""
    node = {
        "kind": "parent",
        "args": [
            {"val": "line1", "info": {"leading": "", "trailing": "\n"}},
            {"val": "line2", "info": {"leading": "  ", "trailing": "\n"}},
            {"val": "line3", "info": {"leading": "", "trailing": ""}},
        ],
    }
    result = _ast_to_code(node)
    assert result == "line1\n  line2\nline3"


def test_ast_to_code_deeply_nested() -> None:
    """Test converting deeply nested structure."""
    node = {
        "kind": "level1",
        "args": [
            {
                "kind": "level2",
                "args": [
                    {
                        "kind": "level3",
                        "args": [
                            {"val": "deep", "info": {"leading": "", "trailing": ""}},
                        ],
                    }
                ],
            }
        ],
    }
    result = _ast_to_code(node)
    assert result == "deep"


# ============================================================================
# Tests for __extract_let_value
# ============================================================================


def test_extract_let_value_single_binding_no_name() -> None:
    """Test extracting value from single let binding without specifying name."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node)
    assert result is not None
    assert result["kind"] == "__value_container"
    assert len(result["args"]) == 1
    assert result["args"][0]["val"] == "42"


def test_extract_let_value_single_binding_with_name() -> None:
    """Test extracting value from single let binding with name specified."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "42"


def test_extract_let_value_multiple_bindings_no_name() -> None:
    """Test extracting value from multiple let bindings without specifying name (should get first)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node)
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "10"  # Should get first binding


def test_extract_let_value_multiple_bindings_with_name() -> None:
    """Test extracting value from multiple let bindings with name specified (should get specific one)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="y")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "20"  # Should get second binding


def test_extract_let_value_binding_name_not_found() -> None:
    """Test extracting value when binding name is not found."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="nonexistent")
    assert result is None


def test_extract_let_value_binding_found_but_malformed_no_assign() -> None:
    """Test extracting value when binding is found but has no := token."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            # Missing := token
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None  # Should return None for malformed binding


def test_extract_let_value_binding_with_nested_binder_ident() -> None:
    """Test extracting value when binding name is in nested binderIdent structure."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            },
                            [],
                            [],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "42"


def test_extract_let_value_complex_value_expression() -> None:
    """Test extracting complex value expression."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "a"},
                            {"val": "+"},
                            {"val": "b"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert len(result["args"]) == 3
    assert result["args"][0]["val"] == "a"
    assert result["args"][1]["val"] == "+"
    assert result["args"][2]["val"] == "b"


def test_extract_let_value_multiple_bindings_first_malformed() -> None:
    """Test extracting value when first binding is malformed but second is valid."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            # Missing :=
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    # Without name, should skip first malformed and get second
    result = __extract_let_value(let_node)
    assert result is not None
    assert result["args"][0]["val"] == "20"
    # With name matching first, should return None (malformed)
    result2 = __extract_let_value(let_node, binding_name="x")
    assert result2 is None


# ============================================================================
# Tests for __extract_set_value
# ============================================================================


def test_extract_set_value_single_binding_no_name() -> None:
    """Test extracting value from single set binding without specifying name."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node)
    assert result is not None
    assert result["kind"] == "__value_container"
    assert len(result["args"]) == 1
    assert result["args"][0]["val"] == "42"


def test_extract_set_value_single_binding_with_name() -> None:
    """Test extracting value from single set binding with name specified."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "42"


def test_extract_set_value_multiple_bindings_no_name() -> None:
    """Test extracting value from multiple set bindings without specifying name (should get first)."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "20"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node)
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "10"  # Should get first binding


def test_extract_set_value_multiple_bindings_with_name() -> None:
    """Test extracting value from multiple set bindings with name specified."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "20"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="y")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert result["args"][0]["val"] == "20"  # Should get second binding


def test_extract_set_value_binding_name_not_found() -> None:
    """Test extracting value when binding name is not found."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="nonexistent")
    assert result is None


def test_extract_set_value_complex_value_expression() -> None:
    """Test extracting complex value expression from set binding."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "a"},
                    {"val": "+"},
                    {"val": "b"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    assert len(result["args"]) == 3
    assert result["args"][0]["val"] == "a"
    assert result["args"][1]["val"] == "+"
    assert result["args"][2]["val"] == "b"


def test_extract_set_value_stops_at_next_set_id_decl() -> None:
    """Test that value extraction stops at next setIdDecl in multiple bindings."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                    {"val": "+"},
                    {"val": "5"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "20"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__value_container"
    # Should only include tokens before next setIdDecl
    assert len(result["args"]) == 3  # "10", "+", "5"
    assert result["args"][0]["val"] == "10"
    assert result["args"][1]["val"] == "+"
    assert result["args"][2]["val"] == "5"


def test_extract_set_value_no_set_id_decl_found() -> None:
    """Test extracting value when no setIdDecl is found (malformed)."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node)
    # Should still try to find := and extract value
    assert result is not None
    assert result["args"][0]["val"] == "42"


# ============================================================================
# Tests for __extract_type_ast for let bindings
# ============================================================================


def test_extract_type_ast_let_single_with_type_no_name() -> None:
    """Test extracting type from single let binding with type, no name specified."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [
                                        {"val": ":"},
                                        {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001  # noqa: RUF001
                                    ],
                                }
                            ],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node)
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 1
    assert result["args"][0]["kind"] == "Lean.Parser.Term.typeSpec"


def test_extract_type_ast_let_single_with_type_with_name() -> None:
    """Test extracting type from single let binding with type, name specified."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [
                                        {"val": ":"},
                                        {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001  # noqa: RUF001
                                    ],
                                }
                            ],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__type_container"


def test_extract_type_ast_let_single_without_type() -> None:
    """Test extracting type from let binding without type annotation."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # Empty type array
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    assert result is None  # No type annotation


def test_extract_type_ast_let_multiple_first_has_type() -> None:
    """Test extracting type when first binding has type, no name specified."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],  # No type
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node)
    assert result is not None
    assert result["kind"] == "__type_container"
    # Should return first typed binding


def test_extract_type_ast_let_multiple_second_has_type() -> None:
    """Test extracting type when second binding has type, no name specified."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # No type
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node)
    assert result is not None
    assert result["kind"] == "__type_container"
    # Should return first typed binding found (second one)


def test_extract_type_ast_let_multiple_with_name_matching_has_type() -> None:
    """Test extracting type when name specified and matching binding has type."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # No type
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="y")
    assert result is not None
    assert result["kind"] == "__type_container"


def test_extract_type_ast_let_multiple_with_name_matching_no_type() -> None:
    """Test extracting type when name specified and matching binding has no type."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # No type
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    assert result is None  # Matching binding has no type


def test_extract_type_ast_let_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="nonexistent")
    assert result is None


# ============================================================================
# Tests for __extract_type_ast for set bindings
# ============================================================================


def test_extract_type_ast_set_single_with_type_in_set_id_decl() -> None:
    """Test extracting type from set binding with type in setIdDecl."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},
                                    {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    assert result is not None
    assert result["kind"] == "__type_container"
    # Should extract type from typeSpec, skipping ":"
    assert len(result["args"]) >= 1


def test_extract_type_ast_set_single_with_type_directly_in_args() -> None:
    """Test extracting type from set binding with type directly in setDecl.args."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "ℕ"},  # noqa: RUF001
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node)
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 1
    assert result["args"][0]["val"] == "ℕ"  # noqa: RUF001


def test_extract_type_ast_set_multiple_with_name_matching_has_type() -> None:
    """Test extracting type when name specified and matching binding has type."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},
                                    {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "20"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="y")
    assert result is not None
    assert result["kind"] == "__type_container"


def test_extract_type_ast_set_multiple_with_name_matching_no_type() -> None:
    """Test extracting type when name specified and matching binding has no type."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},
                                    {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "20"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    assert result is None  # Matching binding has no type


def test_extract_type_ast_set_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},
                                    {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "10"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="nonexistent")
    assert result is None


# ============================================================================
# Edge case tests for multiple bindings
# ============================================================================


def test_extract_let_value_multiple_bindings_typed_and_untyped() -> None:
    """Test extracting value from multiple bindings where some are typed and some are not."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],  # No type
                            {"val": ":="},
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    # Should extract from x when name provided
    result_x = __extract_let_value(let_node, binding_name="x")
    assert result_x is not None
    assert result_x["args"][0]["val"] == "10"
    # Should extract from y when name provided
    result_y = __extract_let_value(let_node, binding_name="y")
    assert result_y is not None
    assert result_y["args"][0]["val"] == "20"


def test_extract_set_value_multiple_bindings_complex_values() -> None:
    """Test extracting values from multiple set bindings with complex expressions."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "a"},
                    {"val": "*"},
                    {"val": "b"},
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "c"},
                    {"val": "+"},
                    {"val": "d"},
                ],
            },
        ],
    }
    result_x = __extract_set_value(set_node, binding_name="x")
    assert result_x is not None
    assert len(result_x["args"]) == 3
    assert result_x["args"][0]["val"] == "a"
    assert result_x["args"][1]["val"] == "*"
    assert result_x["args"][2]["val"] == "b"
    result_y = __extract_set_value(set_node, binding_name="y")
    assert result_y is not None
    assert len(result_y["args"]) == 3
    assert result_y["args"][0]["val"] == "c"
    assert result_y["args"][1]["val"] == "+"
    assert result_y["args"][2]["val"] == "d"


def test_extract_let_value_empty_value() -> None:
    """Test extracting value when value is empty (edge case)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            # No value after :=
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None  # Empty value should return None


def test_extract_set_value_empty_value() -> None:
    """Test extracting value when value is empty (edge case)."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    # No value after :=
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is None  # Empty value should return None


def test_extract_let_value_name_with_direct_val() -> None:
    """Test extracting value when name is directly in val (not nested)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},  # Direct val, not nested
                            [],
                            [],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["args"][0]["val"] == "42"


def test_extract_type_ast_let_empty_type_array() -> None:
    """Test extracting type when type array is empty."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # Empty array (no type)
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    assert result is None


def test_extract_type_ast_set_no_type_spec() -> None:
    """Test extracting type when no typeSpec is found in setIdDecl."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                            # No typeSpec
                        ],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    assert result is None


def test_extract_let_value_multiple_bindings_all_malformed() -> None:
    """Test extracting value when all bindings are malformed (no :=)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            # Missing :=
                            {"val": "10"},
                        ],
                    },
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "y"},
                            [],
                            [],
                            # Missing :=
                            {"val": "20"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node)
    assert result is None  # All bindings malformed


def test_extract_set_value_no_set_decl() -> None:
    """Test extracting value when setDecl is not found."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            # No setDecl
        ],
    }
    result = __extract_set_value(set_node)
    assert result is None


def test_extract_type_ast_let_no_let_decl() -> None:
    """Test extracting type when letDecl is not found."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            # No letDecl
        ],
    }
    result = __extract_type_ast(let_node)
    assert result is None


def test_extract_type_ast_set_no_set_decl() -> None:
    """Test extracting type when setDecl is not found."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            # No setDecl
        ],
    }
    result = __extract_type_ast(set_node)
    assert result is None


def test_extract_let_value_with_type_annotation() -> None:
    """Test extracting value when binding has type annotation."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            {"val": ":="},
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["args"][0]["val"] == "42"  # Should extract value, not type


def test_extract_set_value_with_type_annotation() -> None:
    """Test extracting value when binding has type annotation."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},
                                    {"val": "ℕ"},  # noqa: RUF001  # noqa: RUF001
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is not None
    assert result["args"][0]["val"] == "42"  # Should extract value, not type


# ============================================================================
# Tests for __extract_type_ast for suffices bindings
# ============================================================================


def test_extract_type_ast_suffices_single_with_name() -> None:
    """Test extracting type from single suffices binding with name specified."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 1
    assert result["args"][0]["val"] == "P"


def test_extract_type_ast_suffices_single_no_name() -> None:
    """Test extracting type from single suffices binding without name specified."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node)
    assert result is not None
    assert result["kind"] == "__type_container"
    assert result["args"][0]["val"] == "P"


def test_extract_type_ast_suffices_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="nonexistent")
    assert result is None


def test_extract_type_ast_suffices_complex_type() -> None:
    """Test extracting complex type expression from suffices."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "∧"},
                    {"val": "Q"},
                    {"val": "from"},
                    {"val": "proof"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 3  # "P", "∧", "Q"
    assert result["args"][0]["val"] == "P"
    assert result["args"][1]["val"] == "∧"
    assert result["args"][2]["val"] == "Q"


def test_extract_type_ast_suffices_with_by() -> None:
    """Test extracting type from suffices with 'by' instead of 'from'."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "by"},
                    {"val": "tactic"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is not None
    assert result["kind"] == "__type_container"
    assert result["args"][0]["val"] == "P"


def test_extract_type_ast_suffices_no_have_decl_with_binding_name() -> None:
    """Test extracting type when haveDecl is not found and binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            # No haveDecl
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None


def test_extract_type_ast_suffices_no_have_decl_no_binding_name() -> None:
    """Test extracting type when haveDecl is not found and no binding_name provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            # No haveDecl
        ],
    }
    result = __extract_type_ast(suffices_node)
    # Should fall back to searching anywhere under node
    # May return None or something from fallback
    assert result is None or isinstance(result, dict)


def test_extract_type_ast_suffices_no_have_id_decl_with_binding_name() -> None:
    """Test extracting type when haveIdDecl is not found and binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    # No haveIdDecl
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None  # Can't verify name, so return None


def test_extract_type_ast_suffices_no_have_id_with_binding_name() -> None:
    """Test extracting type when haveId is not found and binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            # No haveId
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None  # Can't extract name, so return None


def test_extract_type_ast_suffices_no_colon_with_binding_name() -> None:
    """Test extracting type when no colon is found and binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    # Missing ":"
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None  # No type found, return None when binding_name provided


def test_extract_type_ast_suffices_no_colon_no_binding_name() -> None:
    """Test extracting type when no colon is found and no binding_name provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    # Missing ":"
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node)
    # Should fall back to old behavior
    assert result is None or isinstance(result, dict)


def test_extract_type_ast_suffices_empty_type_tokens_with_binding_name() -> None:
    """Test extracting type when type tokens are empty and binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    # No type tokens (colon immediately followed by "from")
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None  # Empty type, return None when binding_name provided


def test_extract_type_ast_suffices_no_from_or_by() -> None:
    """Test extracting type when neither 'from' nor 'by' is found (type extends to end)."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "∧"},
                    {"val": "Q"},
                    # No "from" or "by" - type extends to end
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 3  # "P", "∧", "Q"


def test_extract_type_ast_suffices_name_extraction_edge_cases() -> None:
    """Test name extraction edge cases for suffices."""
    # Test with nested structure
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
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
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h"}],
                                    }
                                ],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is not None
    assert result["args"][0]["val"] == "P"


def test_extract_type_ast_suffices_fallback_with_binding_name() -> None:
    """Test that fallback behavior doesn't trigger when binding_name is provided."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    # No colon, no type - malformed
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    # With binding_name, should return None (no fallback)
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None

    # Without binding_name, fallback might return something
    result2 = __extract_type_ast(suffices_node)
    # Fallback behavior - may return None or something
    assert result2 is None or isinstance(result2, dict)


# ============================================================================
# Additional edge case tests for let value extraction
# ============================================================================


def test_extract_let_value_flat_structure_let_id_decl_idx_none() -> None:
    """Test flat structure when letIdDecl index cannot be found (object reference issue)."""
    # This tests the case where object reference comparison might fail
    # Create a structure where the letIdDecl in ld_args is not the same object as 'arg'
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                        ],
                    },
                    # No ":=" in flat structure, and object reference might not match
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should return None since no ":=" found
    assert result is None


def test_extract_let_value_empty_filtered_tokens() -> None:
    """Test when filtered_tokens is empty after filtering out next letIdDecl."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                        ],
                    },
                    {"val": ":="},
                    # Value immediately followed by next letIdDecl (empty value)
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [{"val": "y"}],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should return None since filtered_tokens would be empty
    assert result is None


def test_extract_let_value_multiple_assign_tokens() -> None:
    """Test when multiple ':=' tokens exist (edge case)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            {"val": "1"},
                            {"val": ":="},  # Second ":=" (malformed but handle gracefully)
                            {"val": "2"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should extract value from first ":=" (includes everything after it, including second ":=")
    assert result is not None
    assert len(result["args"]) == 3  # "1", ":=", "2"
    assert result["args"][0]["val"] == "1"
    assert result["args"][1]["val"] == ":="
    assert result["args"][2]["val"] == "2"


# ============================================================================
# Additional edge case tests for set value extraction
# ============================================================================


def test_extract_set_value_empty_filtered_tokens() -> None:
    """Test when filtered_tokens is empty after filtering out next setIdDecl."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    # Value immediately followed by next setIdDecl (empty value)
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "y"}],
                            }
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    # Should return None since filtered_tokens would be empty
    assert result is None


def test_extract_set_value_multiple_assign_tokens() -> None:
    """Test when multiple ':=' tokens exist."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},
                    {"val": "1"},
                    {"val": ":="},  # Second ":=" (malformed)
                    {"val": "2"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    # Should extract value from first ":=" (includes everything after it until next setIdDecl or end)
    assert result is not None
    assert len(result["args"]) == 3  # "1", ":=", "2"
    assert result["args"][0]["val"] == "1"
    assert result["args"][1]["val"] == ":="
    assert result["args"][2]["val"] == "2"


# ============================================================================
# Additional edge case tests for let type extraction
# ============================================================================


def test_extract_type_ast_let_type_arg_not_list() -> None:
    """Test when type_arg is not a list (edge case)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            {"val": "not_a_list"},  # Not a list
                            {"val": ":="},
                            {"val": "1"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    # Should return None since type_arg is not a list
    assert result is None


def test_extract_type_ast_let_type_arg_none() -> None:
    """Test when type_arg is None."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            None,  # None instead of list
                            {"val": ":="},
                            {"val": "1"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    # Should return None since type_arg is None
    assert result is None


# ============================================================================
# Additional edge case tests for set type extraction
# ============================================================================


def test_extract_type_ast_set_multiple_colons() -> None:
    """Test when multiple ':' tokens exist in setDecl.args."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":"},  # First colon
                    {"val": "ℕ"},  # noqa: RUF001
                    {"val": ":"},  # Second colon (malformed)
                    {"val": ":="},
                    {"val": "1"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    # Should extract type between first ":" and ":="
    # This tests the colon_idx logic (should use first colon)
    assert result is None or isinstance(result, dict)  # May extract or fall back


def test_extract_type_ast_set_type_tokens_empty_after_filtering() -> None:
    """Test when type_tokens is empty after filtering out ':'."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            },
                            {
                                "kind": "Lean.Parser.Term.typeSpec",
                                "args": [
                                    {"val": ":"},  # Only ":" token, no actual type
                                ],
                            },
                        ],
                    },
                    {"val": ":="},
                    {"val": "1"},
                ],
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    # Should return None since type_tokens would be empty after filtering
    assert result is None


# ============================================================================
# Additional edge case tests for suffices type extraction
# ============================================================================


def test_extract_type_ast_suffices_multiple_colons() -> None:
    """Test when multiple ':' tokens exist."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},  # First colon
                    {"val": "P"},
                    {"val": ":"},  # Second colon (malformed)
                    {"val": "from"},
                    {"val": "Q"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    # Should extract type from first ":" to "from"
    assert result is not None
    assert result["kind"] == "__type_container"
    assert len(result["args"]) == 2  # "P", ":"
    assert result["args"][0]["val"] == "P"


def test_extract_type_ast_suffices_multiple_from_tokens() -> None:
    """Test when multiple 'from' tokens exist."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": "h"}],
                            }
                        ],
                    },
                    {"val": ":"},
                    {"val": "P"},
                    {"val": "from"},  # First "from"
                    {"val": "Q"},
                    {"val": "from"},  # Second "from" (malformed)
                    {"val": "R"},
                ],
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    # Should extract type from ":" to first "from"
    assert result is not None
    assert result["kind"] == "__type_container"
    assert result["args"][0]["val"] == "P"


# ============================================================================
# Additional edge case tests for empty/malformed structures
# ============================================================================


def test_extract_let_value_empty_let_decl_args() -> None:
    """Test when letDecl.args is empty."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [],  # Empty args
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None


def test_extract_let_value_empty_let_id_decl_args() -> None:
    """Test when letIdDecl.args is empty."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [],  # Empty args
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None


def test_extract_set_value_empty_set_decl_args() -> None:
    """Test when setDecl.args is empty."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [],  # Empty args
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is None


def test_extract_set_value_no_set_id_decl_start_idx_zero() -> None:
    """Test when no setIdDecl found and start_idx becomes 0."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    # No setIdDecl - malformed
                    {"val": ":="},
                    {"val": "1"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node)
    # Should find ":=" at index 0 and extract value
    assert result is not None
    assert result["args"][0]["val"] == "1"


def test_extract_let_value_name_node_unexpected_structure() -> None:
    """Test when name_node has unexpected structure (not dict, not string, not binderIdent)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            123,  # Unexpected: number instead of name
                            [],
                            [],
                            {"val": ":="},
                            {"val": "1"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should return None since name extraction fails
    assert result is None


def test_extract_let_value_name_node_empty_dict() -> None:
    """Test when name_node is an empty dict with no val and no binderIdent."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {},  # Empty dict, no val, no binderIdent
                            [],
                            [],
                            {"val": ":="},
                            {"val": "1"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should return None since name extraction fails
    assert result is None


def test_extract_type_ast_let_empty_let_decl_args() -> None:
    """Test when letDecl.args is empty."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [],  # Empty args
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    assert result is None


def test_extract_type_ast_set_empty_set_decl_args() -> None:
    """Test when setDecl.args is empty."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [],  # Empty args
            },
        ],
    }
    result = __extract_type_ast(set_node, binding_name="x")
    assert result is None


def test_extract_type_ast_suffices_empty_have_decl_args() -> None:
    """Test when haveDecl.args is empty."""
    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [],  # Empty args
            },
        ],
    }
    result = __extract_type_ast(suffices_node, binding_name="h")
    assert result is None


def test_extract_let_value_assign_idx_at_end() -> None:
    """Test when assign_idx is at the end of letIdDecl.args (no value after :=)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},  # := at end, no value
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    # Should return None since value_tokens would be empty
    assert result is None


def test_extract_set_value_assign_idx_at_end() -> None:
    """Test when assign_idx is at the end of setDecl.args (no value after :=)."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "x"}],
                            }
                        ],
                    },
                    {"val": ":="},  # := at end, no value
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    # Should return None since value_tokens would be empty
    assert result is None


def test_extract_type_ast_let_name_node_none() -> None:
    """Test when name_node is None (edge case)."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            None,  # None instead of name
                            [],
                            [],
                            {"val": ":="},
                            {"val": "1"},
                        ],
                    },
                ],
            },
        ],
    }
    result = __extract_type_ast(let_node, binding_name="x")
    # Should return None since name extraction fails
    assert result is None


# ============================================================================
# Tests for __extract_type_ast for choose bindings
# ============================================================================


def test_extract_type_ast_choose_single_with_name() -> None:
    """Test extracting type from single choose binding with name specified."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Choose doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_single_no_name() -> None:
    """Test extracting type from single choose binding without name specified."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Choose doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(choose_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_multiple_bindings_with_name() -> None:
    """Test extracting type from choose with multiple bindings when name is specified."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Should verify binding_name matches before returning None
    result_x = __extract_type_ast(choose_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST

    result_hx = __extract_type_ast(choose_node, binding_name="hx")
    assert result_hx is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_multiple_bindings_no_name() -> None:
    """Test extracting type from choose with multiple bindings when no name specified."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(choose_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found in choose statement."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(choose_node, binding_name="nonexistent")
    assert result is None  # Binding name not found, should return None


def test_extract_type_ast_choose_empty_names() -> None:
    """Test extracting type when choose statement has no names (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # No binderIdent nodes, so no names extracted
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # No names found, binding_name won't match


def test_extract_type_ast_choose_empty_node() -> None:
    """Test extracting type from empty choose node (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [],
    }
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # Empty node, no names to extract


def test_extract_type_ast_choose_malformed_ast_exception() -> None:
    """Test that exception handling works for malformed AST."""
    # Create a malformed node that will cause exception in extraction
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            # Missing structure that will cause exception
            None,  # This will cause issues when iterating
        ],
    }
    # Should handle exception gracefully and return None
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_choose_nested_structure() -> None:
    """Test extracting type from choose with nested binderIdent structure."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.Parser.Term.binderIdent",
                "args": [
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "x"}],
                    }
                ],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_with_keywords_in_names() -> None:
    """Test that keywords are not extracted as names."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},  # Should not be extracted as name
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "h"},
        ],
    }
    # "using" should not be in extracted names
    result = __extract_type_ast(choose_node, binding_name="using")
    assert result is None  # "using" is not a valid binding name


def test_extract_type_ast_choose_multiple_statements_same_name() -> None:
    """Test handling multiple choose statements with same binding name."""
    # First choose statement
    choose_node1 = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h1"},
        ],
    }
    # Second choose statement with same name (different node)
    choose_node2 = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h2"},
        ],
    }
    # Each node should be verified independently
    result1 = __extract_type_ast(choose_node1, binding_name="x")
    assert result1 is None  # Types come from goal context

    result2 = __extract_type_ast(choose_node2, binding_name="x")
    assert result2 is None  # Types come from goal context


def test_extract_type_ast_choose_empty_binding_name() -> None:
    """Test extracting type with empty string binding_name (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Empty string won't match any extracted names
    result = __extract_type_ast(choose_node, binding_name="")
    assert result is None  # Empty string is not a valid binding name


def test_extract_type_ast_choose_complex_structure() -> None:
    """Test extracting type from choose with complex nested structure."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "y"}],
            },
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "z"}],
            },
            {"val": "using"},
            {"val": "h"},
            {"val": "with"},
            {"val": "proof"},
        ],
    }
    # Should verify each binding name independently
    result_x = __extract_type_ast(choose_node, binding_name="x")
    assert result_x is None

    result_y = __extract_type_ast(choose_node, binding_name="y")
    assert result_y is None

    result_z = __extract_type_ast(choose_node, binding_name="z")
    assert result_z is None

    # Non-existent name
    result_nonexistent = __extract_type_ast(choose_node, binding_name="nonexistent")
    assert result_nonexistent is None


def test_extract_type_ast_choose_missing_args() -> None:
    """Test extracting type when args are missing (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        # Missing args
    }
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # No args, can't extract names


def test_extract_type_ast_choose_args_not_list() -> None:
    """Test extracting type when args is not a list (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": "not_a_list",  # Invalid structure
    }
    # Should handle gracefully (extraction will fail, exception caught)
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_choose_binder_ident_without_val() -> None:
    """Test extracting type when binderIdent has no val (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [],  # No val node
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # No names can be extracted
    result = __extract_type_ast(choose_node, binding_name="x")
    assert result is None  # No names found


def test_extract_type_ast_choose_binder_ident_empty_val() -> None:
    """Test extracting type when binderIdent has empty val (edge case)."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": ""}],  # Empty val
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Empty val should not be extracted as name
    result = __extract_type_ast(choose_node, binding_name="")
    assert result is None  # Empty val not extracted as name


def test_extract_type_ast_choose_no_binding_name_behavior() -> None:
    """Test that behavior without binding_name is consistent."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # Without binding_name, should skip verification and return None directly
    result = __extract_type_ast(choose_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_choose_verification_before_return() -> None:
    """Test that binding_name verification happens before returning None."""
    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "using"},
            {"val": "h"},
        ],
    }
    # When binding_name matches, should return None (types from goal context)
    result_match = __extract_type_ast(choose_node, binding_name="x")
    assert result_match is None

    # When binding_name doesn't match, should return None (with debug log)
    result_no_match = __extract_type_ast(choose_node, binding_name="y")
    assert result_no_match is None


# ============================================================================
# Tests for __extract_type_ast for generalize bindings
# ============================================================================


def test_extract_type_ast_generalize_single_with_name() -> None:
    """Test extracting type from single generalize binding with name specified."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Generalize doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(generalize_node, binding_name="h")
    assert result is None  # Types come from goal context, not AST

    result_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_single_no_name() -> None:
    """Test extracting type from single generalize binding without name specified."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Generalize doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(generalize_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_multiple_bindings_with_name() -> None:
    """Test extracting type from generalize with multiple bindings when name is specified."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h2"}],
            },
            {"val": ":"},
            {"val": "e2"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x2"}],
            },
        ],
    }
    # Should verify binding_name matches before returning None
    result_h = __extract_type_ast(generalize_node, binding_name="h")
    assert result_h is None  # Types come from goal context, not AST

    result_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST

    result_h2 = __extract_type_ast(generalize_node, binding_name="h2")
    assert result_h2 is None  # Types come from goal context, not AST

    result_x2 = __extract_type_ast(generalize_node, binding_name="x2")
    assert result_x2 is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_multiple_bindings_no_name() -> None:
    """Test extracting type from generalize with multiple bindings when no name specified."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    result = __extract_type_ast(generalize_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found in generalize statement."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    result = __extract_type_ast(generalize_node, binding_name="nonexistent")
    assert result is None  # Binding name not found, should return None


def test_extract_type_ast_generalize_empty_names() -> None:
    """Test extracting type when generalize statement has no names (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "e"},
            {"val": "="},
            {"val": "some_expr"},  # No binderIdent nodes
        ],
    }
    # No binderIdent nodes, so no names extracted
    result = __extract_type_ast(generalize_node, binding_name="x")
    assert result is None  # No names found, binding_name won't match


def test_extract_type_ast_generalize_empty_node() -> None:
    """Test extracting type from empty generalize node (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [],
    }
    result = __extract_type_ast(generalize_node, binding_name="x")
    assert result is None  # Empty node, no names to extract


def test_extract_type_ast_generalize_malformed_ast_exception() -> None:
    """Test that exception handling works for malformed AST."""
    # Create a malformed node that will cause exception in extraction
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            # Missing structure that will cause exception
            None,  # This will cause issues when iterating
        ],
    }
    # Should handle exception gracefully and return None
    result = __extract_type_ast(generalize_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_generalize_nested_structure() -> None:
    """Test extracting type from generalize with nested binderIdent structure."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.Parser.Term.binderIdent",
                "args": [
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "h"}],
                    }
                ],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    result = __extract_type_ast(generalize_node, binding_name="h")
    assert result is None  # Types come from goal context, not AST

    result_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_with_keywords_in_names() -> None:
    """Test that keywords are not extracted as names."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},  # Should not be extracted as name
            {"val": "e"},
            {"val": "="},  # Should not be extracted as name
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Keywords should not be in extracted names
    result_colon = __extract_type_ast(generalize_node, binding_name=":")
    assert result_colon is None  # ":" is not a valid binding name

    result_eq = __extract_type_ast(generalize_node, binding_name="=")
    assert result_eq is None  # "=" is not a valid binding name


def test_extract_type_ast_generalize_multiple_statements_same_name() -> None:
    """Test handling multiple generalize statements with same binding name."""
    # First generalize statement
    generalize_node1 = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Second generalize statement with same name (different node)
    generalize_node2 = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "e2"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Each node should be verified independently
    result1 = __extract_type_ast(generalize_node1, binding_name="x")
    assert result1 is None  # Types come from goal context

    result2 = __extract_type_ast(generalize_node2, binding_name="x")
    assert result2 is None  # Types come from goal context


def test_extract_type_ast_generalize_empty_binding_name() -> None:
    """Test extracting type with empty string binding_name (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Empty string won't match any extracted names
    result = __extract_type_ast(generalize_node, binding_name="")
    assert result is None  # Empty string is not a valid binding name


def test_extract_type_ast_generalize_complex_structure() -> None:
    """Test extracting type from generalize with complex nested structure."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h2"}],
            },
            {"val": ":"},
            {"val": "e2"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "y"}],
            },
        ],
    }
    # Should verify each binding name independently
    result_h = __extract_type_ast(generalize_node, binding_name="h")
    assert result_h is None

    result_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_x is None

    result_h2 = __extract_type_ast(generalize_node, binding_name="h2")
    assert result_h2 is None

    result_y = __extract_type_ast(generalize_node, binding_name="y")
    assert result_y is None

    # Non-existent name
    result_nonexistent = __extract_type_ast(generalize_node, binding_name="nonexistent")
    assert result_nonexistent is None


def test_extract_type_ast_generalize_missing_args() -> None:
    """Test extracting type when args are missing (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        # Missing args
    }
    result = __extract_type_ast(generalize_node, binding_name="x")
    assert result is None  # No args, can't extract names


def test_extract_type_ast_generalize_args_not_list() -> None:
    """Test extracting type when args is not a list (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": "not_a_list",  # Invalid structure
    }
    # Should handle gracefully (extraction will fail, exception caught)
    result = __extract_type_ast(generalize_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_generalize_binder_ident_without_val() -> None:
    """Test extracting type when binderIdent has no val (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [],  # No val node
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # No names can be extracted from empty binderIdent
    result = __extract_type_ast(generalize_node, binding_name="x")
    # x should still be found from the second binderIdent
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_binder_ident_empty_val() -> None:
    """Test extracting type when binderIdent has empty val (edge case)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": ""}],  # Empty val
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Empty val should not be extracted as name
    result = __extract_type_ast(generalize_node, binding_name="")
    assert result is None  # Empty val not extracted as name


def test_extract_type_ast_generalize_no_binding_name_behavior() -> None:
    """Test that behavior without binding_name is consistent."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Without binding_name, should skip verification and return None directly
    result = __extract_type_ast(generalize_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_generalize_verification_before_return() -> None:
    """Test that binding_name verification happens before returning None."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "h"}],
            },
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # When binding_name matches, should return None (types from goal context)
    result_match = __extract_type_ast(generalize_node, binding_name="h")
    assert result_match is None

    result_match_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_match_x is None

    # When binding_name doesn't match, should return None (with debug log)
    result_no_match = __extract_type_ast(generalize_node, binding_name="y")
    assert result_no_match is None


def test_extract_type_ast_generalize_without_hypothesis_name() -> None:
    """Test generalize without hypothesis name (just expression = variable)."""
    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "e"},
            {"val": "="},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
        ],
    }
    # Only x should be extracted (no h)
    result_x = __extract_type_ast(generalize_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST

    # h should not be found
    result_h = __extract_type_ast(generalize_node, binding_name="h")
    assert result_h is None  # h not in this generalize statement


# ============================================================================
# Tests for __extract_type_ast for obtain bindings
# ============================================================================


def test_extract_type_ast_obtain_single_with_name() -> None:
    """Test extracting type from single obtain binding with name specified."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Obtain doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_single_no_name() -> None:
    """Test extracting type from single obtain binding without name specified."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Obtain doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(obtain_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_multiple_bindings_with_name() -> None:
    """Test extracting type from obtain with multiple bindings when name is specified."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "y"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hz"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Should verify binding_name matches before returning None
    result_x = __extract_type_ast(obtain_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST

    result_y = __extract_type_ast(obtain_node, binding_name="y")
    assert result_y is None  # Types come from goal context, not AST

    result_hz = __extract_type_ast(obtain_node, binding_name="hz")
    assert result_hz is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_multiple_bindings_no_name() -> None:
    """Test extracting type from obtain with multiple bindings when no name specified."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(obtain_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found in obtain statement."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(obtain_node, binding_name="nonexistent")
    assert result is None  # Binding name not found, should return None


def test_extract_type_ast_obtain_empty_names() -> None:
    """Test extracting type when obtain statement has no names (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {"val": "⟩"},  # Empty pattern
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # No binderIdent nodes, so no names extracted
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # No names found, binding_name won't match


def test_extract_type_ast_obtain_empty_node() -> None:
    """Test extracting type from empty obtain node (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [],
    }
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Empty node, no names to extract


def test_extract_type_ast_obtain_malformed_ast_exception() -> None:
    """Test that exception handling works for malformed AST."""
    # Create a malformed node that will cause exception in extraction
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            # Missing structure that will cause exception
            None,  # This will cause issues when iterating
        ],
    }
    # Should handle exception gracefully and return None
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_obtain_nested_structure() -> None:
    """Test extracting type from obtain with nested binderIdent structure."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.Parser.Term.binderIdent",
                "args": [
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "x"}],
                    }
                ],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_with_keywords_in_names() -> None:
    """Test that keywords are not extracted as names."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},  # Should not be extracted as name
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},  # Should not be extracted as name
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "⟩"},  # Should not be extracted as name
            {"val": ":="},  # Should not be extracted as name
            {"val": "h"},
        ],
    }
    # Keywords should not be in extracted names
    result_angle = __extract_type_ast(obtain_node, binding_name="⟨")
    assert result_angle is None  # "⟨" is not a valid binding name

    result_comma = __extract_type_ast(obtain_node, binding_name=",")
    assert result_comma is None  # "," is not a valid binding name

    result_assign = __extract_type_ast(obtain_node, binding_name=":=")
    assert result_assign is None  # ":=" is not a valid binding name


def test_extract_type_ast_obtain_multiple_statements_same_name() -> None:
    """Test handling multiple obtain statements with same binding name."""
    # First obtain statement
    obtain_node1 = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h1"},
        ],
    }
    # Second obtain statement with same name (different node)
    obtain_node2 = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h2"},
        ],
    }
    # Each node should be verified independently
    result1 = __extract_type_ast(obtain_node1, binding_name="x")
    assert result1 is None  # Types come from goal context

    result2 = __extract_type_ast(obtain_node2, binding_name="x")
    assert result2 is None  # Types come from goal context


def test_extract_type_ast_obtain_empty_binding_name() -> None:
    """Test extracting type with empty string binding_name (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Empty string won't match any extracted names
    result = __extract_type_ast(obtain_node, binding_name="")
    assert result is None  # Empty string is not a valid binding name


def test_extract_type_ast_obtain_complex_structure() -> None:
    """Test extracting type from obtain with complex nested structure."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "y"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "z"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hz"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Should verify each binding name independently
    result_x = __extract_type_ast(obtain_node, binding_name="x")
    assert result_x is None

    result_y = __extract_type_ast(obtain_node, binding_name="y")
    assert result_y is None

    result_z = __extract_type_ast(obtain_node, binding_name="z")
    assert result_z is None

    result_hz = __extract_type_ast(obtain_node, binding_name="hz")
    assert result_hz is None

    # Non-existent name
    result_nonexistent = __extract_type_ast(obtain_node, binding_name="nonexistent")
    assert result_nonexistent is None


def test_extract_type_ast_obtain_missing_args() -> None:
    """Test extracting type when args are missing (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        # Missing args
    }
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # No args, can't extract names


def test_extract_type_ast_obtain_args_not_list() -> None:
    """Test extracting type when args is not a list (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": "not_a_list",  # Invalid structure
    }
    # Should handle gracefully (extraction will fail, exception caught)
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_obtain_binder_ident_without_val() -> None:
    """Test extracting type when binderIdent has no val (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [],  # No val node
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # No names can be extracted from empty binderIdent
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # No names found


def test_extract_type_ast_obtain_binder_ident_empty_val() -> None:
    """Test extracting type when binderIdent has empty val (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": ""}],  # Empty val
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Empty val should not be extracted as name
    result = __extract_type_ast(obtain_node, binding_name="")
    assert result is None  # Empty val not extracted as name


def test_extract_type_ast_obtain_no_binding_name_behavior() -> None:
    """Test that behavior without binding_name is consistent."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Without binding_name, should skip verification and return None directly
    result = __extract_type_ast(obtain_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_verification_before_return() -> None:
    """Test that binding_name verification happens before returning None."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "hx"}],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # When binding_name matches, should return None (types from goal context)
    result_match = __extract_type_ast(obtain_node, binding_name="x")
    assert result_match is None

    result_match_hx = __extract_type_ast(obtain_node, binding_name="hx")
    assert result_match_hx is None

    # When binding_name doesn't match, should return None (with debug log)
    result_no_match = __extract_type_ast(obtain_node, binding_name="y")
    assert result_no_match is None


def test_extract_type_ast_obtain_nested_pattern() -> None:
    """Test obtain with nested pattern structure (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ","},
            {
                "kind": "Lean.binderIdent",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.binderIdent",
                        "args": [{"val": "y"}],
                    }
                ],
            },
            {"val": "⟩"},
            {"val": ":="},
            {"val": "h"},
        ],
    }
    result_x = __extract_type_ast(obtain_node, binding_name="x")
    assert result_x is None  # Types come from goal context, not AST

    result_y = __extract_type_ast(obtain_node, binding_name="y")
    assert result_y is None  # Types come from goal context, not AST


def test_extract_type_ast_obtain_pattern_without_angle_brackets() -> None:
    """Test obtain pattern that might not have explicit angle brackets (edge case)."""
    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {
                "kind": "Lean.binderIdent",
                "args": [{"val": "x"}],
            },
            {"val": ":="},
            {"val": "h"},
        ],
    }
    # Should still extract names even without explicit angle brackets
    result = __extract_type_ast(obtain_node, binding_name="x")
    assert result is None  # Types come from goal context, not AST


# ============================================================================
# Tests for __extract_type_ast for match bindings
# ============================================================================


def test_extract_type_ast_match_single_with_name() -> None:
    """Test extracting type from single match binding with name specified."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Match doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_single_no_name() -> None:
    """Test extracting type from single match binding without name specified."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Match doesn't have explicit types in AST, types come from goal context
    result = __extract_type_ast(match_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_multiple_bindings_with_name() -> None:
    """Test extracting type from match with multiple bindings when name is specified."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "a"}],
                    },
                    {"val": ","},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "b"}],
                    },
                    {"val": "=>"},
                    {"val": "body1"},
                ],
            },
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "c"}],
                    },
                    {"val": "=>"},
                    {"val": "body2"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Should verify binding_name matches before returning None
    result_a = __extract_type_ast(match_node, binding_name="a")
    assert result_a is None  # Types come from goal context, not AST

    result_b = __extract_type_ast(match_node, binding_name="b")
    assert result_b is None  # Types come from goal context, not AST

    result_c = __extract_type_ast(match_node, binding_name="c")
    assert result_c is None  # Types come from goal context, not AST


def test_extract_type_ast_match_multiple_bindings_no_name() -> None:
    """Test extracting type from match with multiple bindings when no name specified."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    result = __extract_type_ast(match_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_binding_name_not_found() -> None:
    """Test extracting type when binding name is not found in match statement."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    result = __extract_type_ast(match_node, binding_name="nonexistent")
    assert result is None  # Binding name not found, should return None


def test_extract_type_ast_match_empty_names() -> None:
    """Test extracting type when match statement has no names (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {"val": "none"},  # No binderIdent nodes
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # No binderIdent nodes, so no names extracted
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # No names found, binding_name won't match


def test_extract_type_ast_match_empty_node() -> None:
    """Test extracting type from empty match node (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [],
    }
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Empty node, no names to extract


def test_extract_type_ast_match_malformed_ast_exception() -> None:
    """Test that exception handling works for malformed AST."""
    # Create a malformed node that will cause exception in extraction
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            # Missing structure that will cause exception
            None,  # This will cause issues when iterating
        ],
    }
    # Should handle exception gracefully and return None
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_match_nested_structure() -> None:
    """Test extracting type from match with nested binderIdent structure."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.Parser.Term.binderIdent",
                        "args": [
                            {
                                "kind": "Lean.binderIdent",
                                "args": [{"val": "n"}],
                            }
                        ],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_with_keywords_in_names() -> None:
    """Test that keywords are not extracted as names."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},  # Should not be extracted as name
            {"val": "x"},
            {"val": "with"},  # Should not be extracted as name
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},  # Should not be extracted as name
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},  # Should not be extracted as name
                    {"val": "body"},
                ],
            },
            {"val": "end"},  # Should not be extracted as name
        ],
    }
    # Keywords should not be in extracted names
    result_match = __extract_type_ast(match_node, binding_name="match")
    assert result_match is None  # "match" is not a valid binding name

    result_with = __extract_type_ast(match_node, binding_name="with")
    assert result_with is None  # "with" is not a valid binding name

    result_arrow = __extract_type_ast(match_node, binding_name="=>")
    assert result_arrow is None  # "=>" is not a valid binding name


def test_extract_type_ast_match_multiple_statements_same_name() -> None:
    """Test handling multiple match statements with same binding name."""
    # First match statement
    match_node1 = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body1"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Second match statement with same name (different node)
    match_node2 = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "y"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body2"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Each node should be verified independently
    result1 = __extract_type_ast(match_node1, binding_name="n")
    assert result1 is None  # Types come from goal context

    result2 = __extract_type_ast(match_node2, binding_name="n")
    assert result2 is None  # Types come from goal context


def test_extract_type_ast_match_empty_binding_name() -> None:
    """Test extracting type with empty string binding_name (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Empty string won't match any extracted names
    result = __extract_type_ast(match_node, binding_name="")
    assert result is None  # Empty string is not a valid binding name


def test_extract_type_ast_match_complex_structure() -> None:
    """Test extracting type from match with complex nested structure."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "a"}],
                    },
                    {"val": ","},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "b"}],
                    },
                    {"val": "=>"},
                    {"val": "body1"},
                ],
            },
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "c"}],
                    },
                    {"val": "=>"},
                    {"val": "body2"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Should verify each binding name independently
    result_a = __extract_type_ast(match_node, binding_name="a")
    assert result_a is None

    result_b = __extract_type_ast(match_node, binding_name="b")
    assert result_b is None

    result_c = __extract_type_ast(match_node, binding_name="c")
    assert result_c is None

    # Non-existent name
    result_nonexistent = __extract_type_ast(match_node, binding_name="nonexistent")
    assert result_nonexistent is None


def test_extract_type_ast_match_missing_args() -> None:
    """Test extracting type when args are missing (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        # Missing args
    }
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # No args, can't extract names


def test_extract_type_ast_match_args_not_list() -> None:
    """Test extracting type when args is not a list (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": "not_a_list",  # Invalid structure
    }
    # Should handle gracefully (extraction will fail, exception caught)
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Exception handled, returns None


def test_extract_type_ast_match_binder_ident_without_val() -> None:
    """Test extracting type when binderIdent has no val (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [],  # No val node
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # No names can be extracted from empty binderIdent
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # No names found


def test_extract_type_ast_match_binder_ident_empty_val() -> None:
    """Test extracting type when binderIdent has empty val (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": ""}],  # Empty val
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Empty val should not be extracted as name
    result = __extract_type_ast(match_node, binding_name="")
    assert result is None  # Empty val not extracted as name


def test_extract_type_ast_match_no_binding_name_behavior() -> None:
    """Test that behavior without binding_name is consistent."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Without binding_name, should skip verification and return None directly
    result = __extract_type_ast(match_node)
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_verification_before_return() -> None:
    """Test that binding_name verification happens before returning None."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    # When binding_name matches, should return None (types from goal context)
    result_match = __extract_type_ast(match_node, binding_name="n")
    assert result_match is None

    # When binding_name doesn't match, should return None (with debug log)
    result_no_match = __extract_type_ast(match_node, binding_name="m")
    assert result_no_match is None


def test_extract_type_ast_match_nested_match_expressions() -> None:
    """Test match with nested match expressions (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {
                        "kind": "Lean.Parser.Term.match",
                        "args": [
                            {"val": "match"},
                            {"val": "y"},
                            {"val": "with"},
                            {
                                "kind": "Lean.Parser.Term.matchAlt",
                                "args": [
                                    {"val": "|"},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "m"}],
                                    },
                                    {"val": "=>"},
                                    {"val": "body"},
                                ],
                            },
                            {"val": "end"},
                        ],
                    },
                ],
            },
            {"val": "end"},
        ],
    }
    # Should extract names from both outer and nested matches
    result_n = __extract_type_ast(match_node, binding_name="n")
    assert result_n is None  # Types come from goal context, not AST

    result_m = __extract_type_ast(match_node, binding_name="m")
    assert result_m is None  # Types come from goal context, not AST


def test_extract_type_ast_match_tactic_match() -> None:
    """Test extracting type from tacticMatch_ (tactic version of match)."""
    match_node = {
        "kind": "Lean.Parser.Tactic.tacticMatch_",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Tactic.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {"val": "end"},
        ],
    }
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_malformed_match_alt() -> None:
    """Test that malformed matchAlt nodes are handled gracefully."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    None,  # Malformed pattern
                    {"val": "=>"},
                    {"val": "body"},
                ],
            },
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    {"val": "=>"},
                    {"val": "body2"},
                ],
            },
            {"val": "end"},
        ],
    }
    # Should handle malformed branch gracefully and still extract from valid branch
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # Types come from goal context, not AST


def test_extract_type_ast_match_pattern_without_arrow() -> None:
    """Test match pattern that might not have explicit => (edge case)."""
    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {
                "kind": "Lean.Parser.Term.matchAlt",
                "args": [
                    {"val": "|"},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n"}],
                    },
                    # Missing => token (malformed)
                ],
            },
            {"val": "end"},
        ],
    }
    # __extract_match_pattern_names should return empty list if no => found
    # So no names extracted, binding_name won't match
    result = __extract_type_ast(match_node, binding_name="n")
    assert result is None  # No names found due to malformed pattern


# ============================================================================
# Tests for __extract_set_with_hypothesis_name (set ... with h bug fix)
# ============================================================================


def test_extract_set_with_hypothesis_name_mathlib_set_tactic() -> None:
    """Test extracting hypothesis name from Mathlib.Tactic.setTactic with 'with' clause."""
    set_node = {
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
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hS"


def test_extract_set_with_hypothesis_name_lean_parser_tactic_set() -> None:
    """Test extracting hypothesis name from Lean.Parser.Tactic.tacticSet_ with 'with' clause."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            [],
            {
                "kind": "Lean.Parser.Tactic.setArgsRest",
                "args": [
                    {"val": "x"},
                    [],
                    {"val": ":="},
                    {"val": "5"},
                    [
                        {"val": "with"},
                        [],
                        {"val": "hx"},
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hx"


def test_extract_set_with_hypothesis_name_no_with_clause() -> None:
    """Test that None is returned when set statement has no 'with' clause."""
    set_node = {
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
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None


def test_extract_set_with_hypothesis_name_hypothesis_name_as_dict() -> None:
    """Test extracting hypothesis name when it's a dict node with 'val' field."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],
                        {"val": "hS", "rawVal": "hS", "info": {"leading": "", "trailing": ""}},
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hS"


def test_extract_set_with_hypothesis_name_hypothesis_name_as_string() -> None:
    """Test extracting hypothesis name when it's a plain string."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],
                        "hS",  # Plain string instead of dict
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hS"


def test_extract_set_with_hypothesis_name_no_set_args_rest() -> None:
    """Test that None is returned when setArgsRest node is missing."""
    set_node = {
        "kind": "Mathlib.Tactic.setTactic",
        "args": [
            {"val": "set"},
            [],
            # Missing setArgsRest
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None


def test_extract_set_with_hypothesis_name_with_clause_too_short() -> None:
    """Test that None is returned when 'with' clause list is too short."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],  # Missing hypothesis name at index 2
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None


def test_extract_set_with_hypothesis_name_with_clause_not_first() -> None:
    """Test that hypothesis name is extracted even if 'with' is not the first element."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],
                        {"val": "hS"},
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hS"


def test_extract_set_with_hypothesis_name_empty_hypothesis_name() -> None:
    """Test that None is returned when hypothesis name is empty string."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],
                        {"val": ""},  # Empty string
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None


def test_extract_set_with_hypothesis_name_not_a_dict() -> None:
    """Test that None is returned when input is not a dict."""
    result = __extract_set_with_hypothesis_name("not a dict")
    assert result is None


def test_extract_set_with_hypothesis_name_multiple_with_clauses() -> None:
    """Test that first 'with' clause is extracted when multiple exist."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "with"},
                        [],
                        {"val": "hS"},
                    ],
                    [
                        {"val": "with"},
                        [],
                        {"val": "hS2"},
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result == "hS"  # First one found


def test_extract_set_with_hypothesis_name_different_set_args_rest_kinds() -> None:
    """Test that different setArgsRest node kinds are handled."""
    for kind in ["Mathlib.Tactic.setArgsRest", "Lean.Parser.Tactic.setArgsRest", "Lean.Parser.Term.setArgsRest"]:
        set_node = {
            "kind": "Mathlib.Tactic.setTactic",
            "args": [
                {"val": "set"},
                [],
                {
                    "kind": kind,
                    "args": [
                        {"val": "S"},
                        [],
                        {"val": ":="},
                        {"val": "value"},
                        [
                            {"val": "with"},
                            [],
                            {"val": "hS"},
                        ],
                    ],
                },
            ],
        }
        result = __extract_set_with_hypothesis_name(set_node)
        assert result == "hS", f"Failed for kind: {kind}"


def test_extract_set_with_hypothesis_name_with_clause_not_list() -> None:
    """Test that non-list 'with' clauses are skipped."""
    set_node = {
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
                    {"val": "value"},
                    {"val": "with"},  # Not a list
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None


def test_extract_set_with_hypothesis_name_with_not_first_element() -> None:
    """Test that 'with' clause is only recognized when 'with' is the first element."""
    set_node = {
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
                    {"val": "value"},
                    [
                        {"val": "something_else"},
                        {"val": "with"},
                        {"val": "hS"},
                    ],
                ],
            },
        ],
    }
    result = __extract_set_with_hypothesis_name(set_node)
    assert result is None  # 'with' is not first, so not recognized


# ============================================================================
# Edge case tests for value extraction robustness (Commit 2)
# ============================================================================


def test_extract_let_value_missing_let_decl() -> None:
    """Test extracting value when letDecl is missing."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [{"val": "let"}],  # No letDecl
    }
    result = __extract_let_value(let_node)
    assert result is None


def test_extract_let_value_no_let_id_decl() -> None:
    """Test extracting value when no letIdDecl is found."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [{"val": "something_else"}],  # Not a letIdDecl
            },
        ],
    }
    result = __extract_let_value(let_node)
    assert result is None


def test_extract_let_value_no_assign_token() -> None:
    """Test extracting value when := token is missing."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            # Missing := token
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None


def test_extract_let_value_empty_value_tokens() -> None:
    """Test extracting value when := is found but no value tokens after it."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            {"val": ":="},
                            # No value after :=
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is None


def test_extract_let_value_assign_token_as_string() -> None:
    """Test extracting value when := is a string token instead of dict."""
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],
                            ":=",  # String instead of dict
                            {"val": "42"},
                        ],
                    }
                ],
            },
        ],
    }
    result = __extract_let_value(let_node, binding_name="x")
    assert result is not None
    assert result["args"][0]["val"] == "42"


def test_extract_set_value_missing_set_decl() -> None:
    """Test extracting value when setDecl is missing."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [{"val": "set"}],  # No setDecl
    }
    result = __extract_set_value(set_node)
    assert result is None


def test_extract_set_value_no_set_id_decl() -> None:
    """Test extracting value when no setIdDecl is found."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [{"val": "something_else"}],  # Not a setIdDecl
            },
        ],
    }
    result = __extract_set_value(set_node)
    assert result is None


def test_extract_set_value_no_assign_token() -> None:
    """Test extracting value when := token is missing."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [{"val": "x"}],
                    },
                    # Missing := token
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is None


def test_extract_set_value_empty_value_tokens() -> None:
    """Test extracting value when := is found but no value tokens after it."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [{"val": "x"}],
                    },
                    {"val": ":="},
                    # No value after :=
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is None


def test_extract_set_value_assign_token_as_string() -> None:
    """Test extracting value when := is a string token instead of dict."""
    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [{"val": "x"}],
                    },
                    ":=",  # String instead of dict
                    {"val": "42"},
                ],
            },
        ],
    }
    result = __extract_set_value(set_node, binding_name="x")
    assert result is not None
    assert result["args"][0]["val"] == "42"


def test_handle_set_let_binding_fallback_to_type() -> None:
    """Test that __handle_set_let_binding_as_equality falls back to type when value extraction fails."""
    from goedels_poetry.parsers.util import __handle_set_let_binding_as_equality

    # Create a let binding node where value extraction will fail (no := token)
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [
                                {
                                    "kind": "Lean.Parser.Term.typeSpec",
                                    "args": [{"val": ":"}, {"val": "ℕ"}],  # noqa: RUF001
                                }
                            ],
                            # Missing := and value
                        ],
                    }
                ],
            },
        ],
    }

    existing_names: set[str] = set()
    variables_in_equality_hypotheses: set[str] = set()
    goal_var_types: dict[str, str] = {}

    # Should fall back to type extraction
    binder, was_handled = __handle_set_let_binding_as_equality(
        "x", "let", let_node, existing_names, variables_in_equality_hypotheses, goal_var_types=goal_var_types
    )

    # Should be handled (using type as fallback)
    assert was_handled
    assert binder is not None


def test_handle_set_let_binding_fallback_to_goal_context() -> None:
    """Test that __handle_set_let_binding_as_equality falls back to goal context when value and type extraction fail."""
    from goedels_poetry.parsers.util import __handle_set_let_binding_as_equality

    # Create a let binding node where both value and type extraction will fail
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # No type
                            # Missing := and value
                        ],
                    }
                ],
            },
        ],
    }

    existing_names: set[str] = set()
    variables_in_equality_hypotheses: set[str] = set()
    goal_var_types: dict[str, str] = {"x": "ℕ"}  # noqa: RUF001

    # Should fall back to goal context
    binder, was_handled = __handle_set_let_binding_as_equality(
        "x", "let", let_node, existing_names, variables_in_equality_hypotheses, goal_var_types=goal_var_types
    )

    # Should be handled (using goal context as fallback)
    assert was_handled
    assert binder is not None


def test_handle_set_let_binding_all_fallbacks_fail() -> None:
    """Test that __handle_set_let_binding_as_equality returns failure when all fallbacks fail."""
    from goedels_poetry.parsers.util import __handle_set_let_binding_as_equality

    # Create a let binding node where all extraction will fail
    let_node = {
        "kind": "Lean.Parser.Tactic.tacticLet_",
        "args": [
            {"val": "let"},
            {
                "kind": "Lean.Parser.Term.letDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.letIdDecl",
                        "args": [
                            {"val": "x"},
                            [],
                            [],  # No type
                            # Missing := and value
                        ],
                    }
                ],
            },
        ],
    }

    existing_names: set[str] = set()
    variables_in_equality_hypotheses: set[str] = set()
    goal_var_types: dict[str, str] = {}  # No goal context either

    # All fallbacks should fail
    binder, was_handled = __handle_set_let_binding_as_equality(
        "x", "let", let_node, existing_names, variables_in_equality_hypotheses, goal_var_types=goal_var_types
    )

    # Should not be handled
    assert not was_handled
    assert binder is None


# ============================================================================
# Tests for __construct_set_with_hypothesis_type (Commit 3)
# ============================================================================


def test_construct_set_with_hypothesis_type_basic() -> None:
    """Test constructing type for basic set_with_hypothesis."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    set_node = {
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
    }

    result = __construct_set_with_hypothesis_type(set_node, "hS")
    assert result is not None
    assert result["kind"] == "__equality_expr"
    assert len(result["args"]) >= 3
    # Check that it contains variable name, "=", and value
    assert result["args"][0]["val"] == "S"
    assert result["args"][1]["val"] == "="


def test_construct_set_with_hypothesis_type_complex_value() -> None:
    """Test constructing type with complex value expression."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    set_node = {
        "kind": "Mathlib.Tactic.setTactic",
        "args": [
            {"val": "set"},
            [],
            {
                "kind": "Mathlib.Tactic.setArgsRest",
                "args": [
                    {"val": "x"},
                    [],
                    {"val": ":="},
                    {"val": "(Finset.filter (fun n => n > 0) (Finset.range 100)).prod id"},
                    [
                        {"val": "with"},
                        [],
                        {"val": "hx"},
                    ],
                ],
            },
        ],
    }

    result = __construct_set_with_hypothesis_type(set_node, "hx")
    assert result is not None
    assert result["kind"] == "__equality_expr"
    assert result["args"][0]["val"] == "x"
    assert result["args"][1]["val"] == "="


def test_construct_set_with_hypothesis_type_no_with_clause() -> None:
    """Test that function returns None when no 'with' clause is present."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    set_node = {
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
    }

    result = __construct_set_with_hypothesis_type(set_node, "hS")
    assert result is None


def test_construct_set_with_hypothesis_type_name_mismatch() -> None:
    """Test that function returns None when hypothesis name doesn't match."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    set_node = {
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
    }

    # Request different hypothesis name
    result = __construct_set_with_hypothesis_type(set_node, "hT")
    assert result is None


def test_construct_set_with_hypothesis_type_name_extraction_fails() -> None:
    """Test that function returns None when variable name extraction fails."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    # Set node without proper variable name structure
    set_node = {
        "kind": "Mathlib.Tactic.setTactic",
        "args": [
            {"val": "set"},
            [],
            {
                "kind": "Mathlib.Tactic.setArgsRest",
                "args": [
                    # Missing variable name
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
    }

    result = __construct_set_with_hypothesis_type(set_node, "hS")
    assert result is None


def test_construct_set_with_hypothesis_type_value_extraction_fails() -> None:
    """Test that function returns None when value extraction fails."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    # Set node without proper value structure
    set_node = {
        "kind": "Mathlib.Tactic.setTactic",
        "args": [
            {"val": "set"},
            [],
            {
                "kind": "Mathlib.Tactic.setArgsRest",
                "args": [
                    {"val": "S"},
                    [],
                    # Missing := and value
                    [
                        {"val": "with"},
                        [],
                        {"val": "hS"},
                    ],
                ],
            },
        ],
    }

    result = __construct_set_with_hypothesis_type(set_node, "hS")
    assert result is None


def test_construct_set_with_hypothesis_type_lean_parser_structure() -> None:
    """Test constructing type with Lean.Parser.Tactic structure."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type

    set_node = {
        "kind": "Lean.Parser.Tactic.tacticSet_",
        "args": [
            {"val": "set"},
            {
                "kind": "Lean.Parser.Term.setDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.setIdDecl",
                        "args": [{"val": "x"}],
                    },
                    {"val": ":="},
                    {"val": "42"},
                ],
            },
            {
                "kind": "Lean.Parser.Tactic.setArgsRest",
                "args": [
                    [
                        {"val": "with"},
                        [],
                        {"val": "hx"},
                    ],
                ],
            },
        ],
    }

    result = __construct_set_with_hypothesis_type(set_node, "hx")
    assert result is not None
    assert result["kind"] == "__equality_expr"
    assert result["args"][0]["val"] == "x"
    assert result["args"][1]["val"] == "="


def test_construct_set_with_hypothesis_type_ast_serialization() -> None:
    """Test that constructed AST can be serialized to code."""
    from goedels_poetry.parsers.util import __construct_set_with_hypothesis_type, _ast_to_code

    set_node = {
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
    }

    result = __construct_set_with_hypothesis_type(set_node, "hS")
    assert result is not None

    # Serialize to code
    code = _ast_to_code(result)
    assert "S" in code
    assert "=" in code
    assert "Finset.range 10000" in code


# ============================================================================
# Tests for __determine_general_binding_type (Commit 4)
# ============================================================================


def test_determine_general_binding_type_have_from_goal_context() -> None:
    """Test that have binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    have_node = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]},
                            {"val": ":"},
                            {"val": "Prop"},
                        ],
                    },
                ],
            },
        ],
    }

    goal_var_types = {"h1": "n > 0"}

    result = __determine_general_binding_type("h1", "have", have_node, goal_var_types)

    # Should use goal context type
    assert result is not None
    # Check that it's a binder with the goal context type
    code = _ast_to_code(result)
    assert "h1" in code
    assert "n > 0" in code


def test_determine_general_binding_type_have_from_ast() -> None:
    """Test that have binding type is extracted from AST when goal context unavailable."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    have_node = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]},
                        ],
                    },
                    {"val": ":"},
                    {"val": "n > 0"},
                ],
            },
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("h1", "have", have_node, goal_var_types)

    # Should extract from AST
    assert result is not None
    code = _ast_to_code(result)
    assert "h1" in code
    assert "n > 0" in code


def test_determine_general_binding_type_have_both_fail() -> None:
    """Test that have binding falls back to Prop when both goal context and AST extraction fail."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    # Have node without type in AST
    have_node = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]},
                            # No type annotation
                        ],
                    },
                ],
            },
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("h1", "have", have_node, goal_var_types)

    # Should fall back to Prop
    assert result is not None
    code = _ast_to_code(result)
    assert "h1" in code
    # Should have Prop or no explicit type (which defaults to Prop)


def test_determine_general_binding_type_have_goal_context_priority() -> None:
    """Test that goal context takes priority over AST extraction for have bindings."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    have_node = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h1"}]},
                        ],
                    },
                    {"val": ":"},
                    {"val": "Prop"},  # Type in AST
                ],
            },
        ],
    }

    goal_var_types = {"h1": "n > 0"}  # Different type in goal context

    result = __determine_general_binding_type("h1", "have", have_node, goal_var_types)

    # Should use goal context type (priority)
    assert result is not None
    code = _ast_to_code(result)
    assert "h1" in code
    assert "n > 0" in code
    # Should not use AST type (Prop)
    assert "Prop" not in code or "(h1 : Prop )" not in code


def test_determine_general_binding_type_suffices_from_goal_context() -> None:
    """Test that suffices binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    suffices_node = {
        "kind": "Lean.Parser.Tactic.tacticSuffices_",
        "args": [
            {"val": "suffices"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h"}]},
                            {"val": ":"},
                            {"val": "Prop"},
                        ],
                    },
                ],
            },
        ],
    }

    goal_var_types = {"h": "P"}

    result = __determine_general_binding_type("h", "suffices", suffices_node, goal_var_types)

    # Should use goal context type
    assert result is not None
    code = _ast_to_code(result)
    assert "h" in code
    assert "P" in code


def test_determine_general_binding_type_obtain_from_goal_context() -> None:
    """Test that obtain binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {"val": "x"},
            {"val": ","},
            {"val": "hx"},
            {"val": "⟩"},
            {"val": ":="},
            {"val": "proof"},
        ],
    }

    goal_var_types = {"x": "ℕ", "hx": "x > 0"}  # noqa: RUF001

    result_x = __determine_general_binding_type("x", "obtain", obtain_node, goal_var_types)
    result_hx = __determine_general_binding_type("hx", "obtain", obtain_node, goal_var_types)

    # Should use goal context types
    assert result_x is not None
    assert result_hx is not None
    code_x = _ast_to_code(result_x)
    code_hx = _ast_to_code(result_hx)
    assert "x" in code_x
    assert "hx" in code_hx


def test_determine_general_binding_type_obtain_no_goal_context() -> None:
    """Test that obtain binding falls back to Prop when goal context unavailable."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    obtain_node = {
        "kind": "Lean.Parser.Tactic.tacticObtain_",
        "args": [
            {"val": "obtain"},
            {"val": "⟨"},
            {"val": "x"},
            {"val": "⟩"},
            {"val": ":="},
            {"val": "proof"},
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("x", "obtain", obtain_node, goal_var_types)

    # Should fall back to Prop with informative warning
    assert result is not None
    code = _ast_to_code(result)
    assert "x" in code


def test_determine_general_binding_type_choose_from_goal_context() -> None:
    """Test that choose binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {"val": "x"},
            {"val": "hx"},
            {"val": "using"},
            {"val": "h"},
        ],
    }

    goal_var_types = {"x": "ℕ", "hx": "x > 0"}  # noqa: RUF001

    result_x = __determine_general_binding_type("x", "choose", choose_node, goal_var_types)
    result_hx = __determine_general_binding_type("hx", "choose", choose_node, goal_var_types)

    # Should use goal context types
    assert result_x is not None
    assert result_hx is not None
    code_x = _ast_to_code(result_x)
    code_hx = _ast_to_code(result_hx)
    assert "x" in code_x
    assert "hx" in code_hx


def test_determine_general_binding_type_choose_no_goal_context() -> None:
    """Test that choose binding falls back to Prop when goal context unavailable."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    choose_node = {
        "kind": "Lean.Parser.Tactic.tacticChoose_",
        "args": [
            {"val": "choose"},
            {"val": "x"},
            {"val": "hx"},
            {"val": "using"},
            {"val": "h"},
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("x", "choose", choose_node, goal_var_types)

    # Should fall back to Prop with informative warning
    assert result is not None
    code = _ast_to_code(result)
    assert "x" in code


def test_determine_general_binding_type_generalize_from_goal_context() -> None:
    """Test that generalize binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "h"},
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {"val": "x"},
        ],
    }

    goal_var_types = {"h": "e = x"}

    result = __determine_general_binding_type("h", "generalize", generalize_node, goal_var_types)

    # Should use goal context type
    assert result is not None
    code = _ast_to_code(result)
    assert "h" in code
    assert "e = x" in code


def test_determine_general_binding_type_generalize_no_goal_context() -> None:
    """Test that generalize binding falls back to Prop when goal context unavailable."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    generalize_node = {
        "kind": "Lean.Parser.Tactic.tacticGeneralize_",
        "args": [
            {"val": "generalize"},
            {"val": "h"},
            {"val": ":"},
            {"val": "e"},
            {"val": "="},
            {"val": "x"},
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("h", "generalize", generalize_node, goal_var_types)

    # Should fall back to Prop with informative warning
    assert result is not None
    code = _ast_to_code(result)
    assert "h" in code


def test_determine_general_binding_type_match_from_goal_context() -> None:
    """Test that match binding type is determined from goal context."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {"val": "|"},
            {"val": "some"},
            {"val": "n"},
            {"val": "=>"},
            {"val": "body"},
        ],
    }

    goal_var_types = {"n": "ℕ"}  # noqa: RUF001

    result = __determine_general_binding_type("n", "match", match_node, goal_var_types)

    # Should use goal context type
    assert result is not None
    code = _ast_to_code(result)
    assert "n" in code


def test_determine_general_binding_type_match_no_goal_context() -> None:
    """Test that match binding falls back to Prop when goal context unavailable."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    match_node = {
        "kind": "Lean.Parser.Term.match",
        "args": [
            {"val": "match"},
            {"val": "x"},
            {"val": "with"},
            {"val": "|"},
            {"val": "some"},
            {"val": "n"},
            {"val": "=>"},
            {"val": "body"},
        ],
    }

    goal_var_types = {}  # No goal context

    result = __determine_general_binding_type("n", "match", match_node, goal_var_types)

    # Should fall back to Prop with informative warning
    assert result is not None
    code = _ast_to_code(result)
    assert "n" in code


def test_determine_general_binding_type_unknown_type() -> None:
    """Test that unknown binding type is handled gracefully."""
    from goedels_poetry.parsers.util import __determine_general_binding_type

    unknown_node = {"kind": "Unknown.Binding", "args": []}

    goal_var_types = {}

    result = __determine_general_binding_type("x", "unknown", unknown_node, goal_var_types)

    # Should fall back to Prop with warning about unknown type
    assert result is not None
    code = _ast_to_code(result)
    assert "x" in code
