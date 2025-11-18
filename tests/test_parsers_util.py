"""Tests for goedels_poetry.parsers.util module."""

# Import private functions for testing
from goedels_poetry.parsers.util import (
    __extract_let_value,
    __extract_set_value,
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
