"""Tests for Phase 2: Variable Extraction.

This test suite verifies that variable extraction correctly:
1. Extracts all variables from check() responses
2. Distinguishes lemma parameters from proof body variables
3. Finds variable declarations in AST
4. Handles all variable declaration types (have, let, intro, obtain, etc.)
"""

# ruff: noqa: RUF001

from __future__ import annotations

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
    extract_lemma_parameters_from_ast,
    extract_variables_from_check_response,
    extract_variables_with_origin,
    find_variable_declaration_in_ast,
)

# Mark tests that require Kimina server as integration tests
pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def _create_ast_for_sketch(
    sketch: str, preamble: str = DEFAULT_IMPORTS, server_url: str = "http://localhost:8000", server_timeout: int = 60
) -> AST:
    """Create AST from sketch using Kimina server."""
    from kimina_client import KiminaClient

    from goedels_poetry.agents.util.common import remove_default_imports_from_ast
    from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response

    # Normalize sketch and combine with preamble
    normalized_sketch = sketch.strip()
    normalized_sketch = normalized_sketch if normalized_sketch.endswith("\n") else normalized_sketch + "\n"
    normalized_preamble = preamble.strip()
    full_text = combine_preamble_and_body(normalized_preamble, normalized_sketch)

    # Calculate body_start
    if not normalized_preamble:
        body_start = 0
    elif not normalized_sketch:
        body_start = len(full_text)
    else:
        body_start = len(normalized_preamble) + 2  # +2 for "\n\n"

    # Parse with Kimina
    client = KiminaClient(api_url=server_url, n_retries=3, http_timeout=server_timeout)
    ast_response = client.ast_code(full_text, timeout=server_timeout)
    parsed = parse_kimina_ast_code_response(ast_response)

    if parsed.get("error") is not None:
        raise ValueError(f"Failed to parse sketch for AST: {parsed['error']}")  # noqa: TRY003

    ast_without_imports = remove_default_imports_from_ast(parsed["ast"], preamble=preamble)
    return AST(ast_without_imports, sorries=parsed.get("sorries"), source_text=full_text, body_start=body_start)


def _create_mock_check_response(hypotheses: list[str]) -> dict:
    """Create a mock check response with unsolved goals."""
    # Format: each hypothesis on a new line WITHOUT indentation (they start at column 0)
    # The goal line (⊢) should come after all hypotheses with 2 spaces indentation
    hypothesis_lines = "\n".join(hypotheses)
    data = f"unsolved goals\n{hypothesis_lines}\n  ⊢ True"
    return {
        "errors": [
            {
                "severity": "error",
                "data": data,
            }
        ],
        "pass": False,
        "complete": False,
    }


class TestExtractVariablesFromCheckResponse:
    """Tests for extract_variables_from_check_response function."""

    def test_extract_simple_variable(self) -> None:
        """Test extraction of simple variable."""
        check_response = _create_mock_check_response(["x : ℤ"])
        variables = extract_variables_from_check_response(check_response)

        assert len(variables) == 1
        assert variables[0]["name"] == "x"
        assert variables[0]["type"] == "ℤ"
        assert variables[0]["hypothesis"] == "x : ℤ"
        assert variables[0]["source"] == "check_response"

    def test_extract_multiple_variables(self) -> None:
        """Test extraction of multiple variables."""
        check_response = _create_mock_check_response(["x : ℤ", "y : ℕ", "h : x > 0"])
        variables = extract_variables_from_check_response(check_response)

        assert len(variables) == 3
        names = {v["name"] for v in variables}
        assert names == {"x", "y", "h"}

    def test_extract_with_lemma_parameters(self) -> None:
        """Test that lemma parameters are correctly marked."""
        check_response = _create_mock_check_response(["x : ℤ", "y : ℕ", "z : ℤ"])
        lemma_parameters = {"x", "y"}
        variables = extract_variables_from_check_response(check_response, lemma_parameters)

        assert len(variables) == 3
        x_var = next(v for v in variables if v["name"] == "x")
        y_var = next(v for v in variables if v["name"] == "y")
        z_var = next(v for v in variables if v["name"] == "z")

        assert x_var["is_lemma_parameter"] is True
        assert y_var["is_lemma_parameter"] is True
        assert z_var["is_lemma_parameter"] is False
        assert z_var["is_proof_body_variable"] is True

    def test_extract_with_let_binding(self) -> None:
        """Test extraction of let binding variable."""
        check_response = _create_mock_check_response(["x := 5"])
        variables = extract_variables_from_check_response(check_response)

        assert len(variables) == 1
        assert variables[0]["name"] == "x"
        assert variables[0]["type"] is None  # Type inferred from value

    def test_extract_empty_response(self) -> None:
        """Test extraction from empty check response."""
        check_response = {"pass": True, "complete": True, "errors": []}
        variables = extract_variables_from_check_response(check_response)

        assert len(variables) == 0

    def test_extract_invalid_response(self) -> None:
        """Test extraction from invalid check response."""
        check_response = {"pass": True, "complete": True}
        variables = extract_variables_from_check_response(check_response)

        assert len(variables) == 0


class TestExtractLemmaParametersFromAst:
    """Tests for extract_lemma_parameters_from_ast function."""

    def test_extract_simple_parameters(self, kimina_server_url: str) -> None:
        """Test extraction of simple lemma parameters."""
        sketch = """theorem test (x : ℤ) (y : ℕ) : True := by sorry"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)
        parameters = extract_lemma_parameters_from_ast(ast)

        assert parameters == {"x", "y"}

    def test_extract_with_hypothesis(self, kimina_server_url: str) -> None:
        """Test extraction of lemma with hypothesis."""
        sketch = """theorem test (x : ℤ) (h : x > 0) : True := by sorry"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)
        parameters = extract_lemma_parameters_from_ast(ast)

        assert parameters == {"x", "h"}

    def test_extract_no_parameters(self, kimina_server_url: str) -> None:
        """Test extraction from lemma with no parameters."""
        sketch = """theorem test : True := by sorry"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)
        parameters = extract_lemma_parameters_from_ast(ast)

        assert parameters == set()

    def test_extract_complex_binders(self, kimina_server_url: str) -> None:
        """Test extraction with complex binder structures."""
        sketch = """theorem test (x y : ℤ) (h : x + y = 0) : True := by sorry"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)
        parameters = extract_lemma_parameters_from_ast(ast)

        assert "x" in parameters
        assert "y" in parameters
        assert "h" in parameters


class TestExtractVariablesWithOrigin:
    """Tests for extract_variables_with_origin function."""

    def test_distinguish_lemma_parameters(self, kimina_server_url: str) -> None:
        """Test that lemma parameters are distinguished from proof body variables."""
        sketch = """theorem test (x : ℤ) (y : ℕ) : True := by
  let z := 5
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        # Create check response with all variables
        check_response = _create_mock_check_response(["x : ℤ", "y : ℕ", "z : ℕ"])

        variables = extract_variables_with_origin(check_response, ast)

        x_var = next(v for v in variables if v["name"] == "x")
        y_var = next(v for v in variables if v["name"] == "y")
        z_var = next(v for v in variables if v["name"] == "z")

        assert x_var["is_lemma_parameter"] is True
        assert x_var["is_proof_body_variable"] is False
        assert y_var["is_lemma_parameter"] is True
        assert y_var["is_proof_body_variable"] is False
        assert z_var["is_lemma_parameter"] is False
        assert z_var["is_proof_body_variable"] is True

    def test_find_declaration_nodes(self, kimina_server_url: str) -> None:
        """Test that declaration nodes are found for proof body variables."""
        sketch = """theorem test : True := by
  have h : True := trivial
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response(["h : True"])

        variables = extract_variables_with_origin(check_response, ast)

        h_var = next(v for v in variables if v["name"] == "h")
        assert h_var["declaration_node"] is not None
        assert h_var["declaration_node"].get("kind") == "Lean.Parser.Tactic.tacticHave_"


class TestFindVariableDeclarationInAst:
    """Tests for find_variable_declaration_in_ast function."""

    def test_find_have_declaration(self, kimina_server_url: str) -> None:
        """Test finding have statement declaration."""
        sketch = """theorem test : True := by
  have h : True := trivial
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        decl_node = find_variable_declaration_in_ast(ast, "h")

        assert decl_node is not None
        assert decl_node.get("kind") == "Lean.Parser.Tactic.tacticHave_"

    def test_find_let_declaration(self, kimina_server_url: str) -> None:
        """Test finding let binding declaration."""
        sketch = """theorem test : True := by
  let x := 5
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        decl_node = find_variable_declaration_in_ast(ast, "x")

        assert decl_node is not None
        assert decl_node.get("kind") in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}

    def test_find_nonexistent_variable(self, kimina_server_url: str) -> None:
        """Test finding non-existent variable returns None."""
        sketch = """theorem test : True := by sorry"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        decl_node = find_variable_declaration_in_ast(ast, "nonexistent")

        assert decl_node is None


class TestIntegration:
    """Integration tests for Phase 2 variable extraction."""

    def test_complete_workflow(self, kimina_server_url: str) -> None:
        """Test complete workflow: extract all variables and distinguish origins."""
        sketch = """theorem test (x : ℤ) (y : ℕ) : True := by
  let z := x + 1
  have hz : z > 0 := by omega
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        # In a real scenario, we'd get this from kimina_client.check()
        # For testing, we create a mock response
        check_response = _create_mock_check_response(["x : ℤ", "y : ℕ", "z : ℤ", "hz : z > 0"])

        variables = extract_variables_with_origin(check_response, ast)

        # Verify all variables are extracted
        names = {v["name"] for v in variables}
        assert names == {"x", "y", "z", "hz"}

        # Verify lemma parameters
        lemma_params = {v["name"] for v in variables if v["is_lemma_parameter"]}
        assert lemma_params == {"x", "y"}

        # Verify proof body variables
        proof_vars = {v["name"] for v in variables if v["is_proof_body_variable"]}
        assert proof_vars == {"z", "hz"}

        # Verify declaration nodes are found
        z_var = next(v for v in variables if v["name"] == "z")
        hz_var = next(v for v in variables if v["name"] == "hz")
        assert z_var["declaration_node"] is not None
        assert hz_var["declaration_node"] is not None

    def test_with_intro_statement(self, kimina_server_url: str) -> None:
        """Test extraction with intro statement."""
        sketch = """theorem test : (x : ℤ) → x = x := by
  intro x
  sorry
"""
        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response(["x : ℤ"])

        variables = extract_variables_with_origin(check_response, ast)

        # x should be a proof body variable (from intro)
        x_var = next(v for v in variables if v["name"] == "x")
        assert x_var["is_lemma_parameter"] is False
        assert x_var["is_proof_body_variable"] is True
