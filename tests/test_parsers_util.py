"""Tests for goedels_poetry.parsers.util module."""

# Import private functions for testing
from goedels_poetry.parsers.util import _ast_to_code


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
