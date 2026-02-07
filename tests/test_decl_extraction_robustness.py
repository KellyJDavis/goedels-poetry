from typing import Any

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
from goedels_poetry.config.kimina_server import KIMINA_LEAN_SERVER
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.decl_extraction import (
    extract_preamble_from_ast,
    extract_proof_body_from_ast,
    extract_signature_from_ast,
)


@pytest.fixture
def kimina_server_url(request: Any) -> str:
    """Fixture to get kimina server URL from environment or pytest config."""
    return str(request.config.getoption("--kimina-server-url", default=KIMINA_LEAN_SERVER["url"]))


def _create_ast(code: str, server_url: str) -> AST:
    from kimina_client import KiminaClient

    from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response

    client = KiminaClient(api_url=server_url)
    resp = client.ast_code(code)
    parsed = parse_kimina_ast_code_response(resp)
    if parsed.get("error"):
        pytest.fail(f"AST creation failed: {parsed['error']}")
    return AST(parsed["ast"], source_text=code)


def _normalise_sig(s: str) -> str:
    """Whitespace-normalise signature for comparison (plan 3.1, 7.5)."""
    return " ".join(s.strip().split())


@pytest.mark.usefixtures("skip_if_no_lean")
class TestDeclExtractionRobustness:
    """Tests for robust structural extraction from Lean 4 ASTs."""

    def test_signature_matching_and_structural_by(self, kimina_server_url: str) -> None:
        """Test matching by signature and extracting body after ':=' and 'by'."""
        code = f"""{DEFAULT_IMPORTS}
theorem target (n : Nat) : n = n := by
  -- Correct body
  rfl

theorem other (n : Nat) : n = n := by
  rfl
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "theorem target (n : Nat) : n = n"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body == "  -- Correct body\n  rfl"

    def test_last_occurrence_heuristic(self, kimina_server_url: str) -> None:
        """Test that the last occurrence of a matching signature is selected."""
        code = f"""{DEFAULT_IMPORTS}
theorem target : True := by
  -- First attempt (wrong)
  sorry

theorem target : True := by
  -- Second attempt (correct)
  trivial
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "theorem target : True"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body == "  -- Second attempt (correct)\n  trivial"

    def test_nested_block_prevention(self, kimina_server_url: str) -> None:
        """Test that nested proofs (e.g. in 'have') are not misidentified as main body."""
        code = f"""{DEFAULT_IMPORTS}
theorem target : True := by
  have h : 1 = 1 := by
    rfl
  trivial
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "theorem target : True"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body is not None
        assert "have h : 1 = 1 := by" in body
        assert "trivial" in body

    def test_preamble_extraction_and_ordering(self, kimina_server_url: str) -> None:
        """Test extraction of imports and opens with correct ordering."""
        # Use valid Lean 4: imports must come BEFORE opens
        code = """import Mathlib.Tactic.Basic
import Mathlib.Data.Nat.Basic
open Classical
open List

theorem t : True := trivial
"""
        ast = _create_ast(code, kimina_server_url)
        preamble = extract_preamble_from_ast(ast)

        lines = [line for line in preamble.split("\n") if line.strip()]
        # Imports must come first
        assert lines[0].startswith("import")
        assert lines[1].startswith("import")
        assert lines[2].startswith("open")
        assert lines[3].startswith("open")

    def test_signature_matching_alpha_equivalence(self, kimina_server_url: str) -> None:
        """Test that signature matching handles binder renaming (alpha-equivalence)."""
        code = f"""{DEFAULT_IMPORTS}
theorem target (x : Nat) : x = x := by rfl
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "theorem target (x : Nat) : x = x"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body == " rfl"

    def test_signature_matching_lemma_theorem_equivalence_target_is_lemma(self, kimina_server_url: str) -> None:
        """Allow matching lemma target against theorem proof declaration."""
        code = f"""{DEFAULT_IMPORTS}
theorem target : True := by
  trivial
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "lemma target : True"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body == "  trivial"

    def test_signature_matching_lemma_theorem_equivalence_target_is_theorem(self, kimina_server_url: str) -> None:
        """Allow matching theorem target against lemma proof declaration."""
        code = f"""{DEFAULT_IMPORTS}
lemma target : True := by
  trivial
"""
        ast = _create_ast(code, kimina_server_url)
        target_sig = "theorem target : True"
        body = extract_proof_body_from_ast(ast, target_sig)
        assert body == "  trivial"


@pytest.mark.usefixtures("skip_if_no_lean")
class TestExtractSignatureFromAst:
    """Unit tests for extract_signature_from_ast (plan 3.1)."""

    def test_case1_inline_sorry(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
theorem t : True := by sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t : True")

    def test_case2_newline_before_sorry(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
theorem t : True := by
  sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t : True")

    def test_case3_block_comment_before_sorry(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
theorem t : True := by
  /- block -/
  sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t : True")

    def test_case4_multiple_block_comments(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
theorem t : True := by
  /- one -/
  /- two -/
  sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t : True")

    def test_case5_docstring_stripped(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
/- doc -/ theorem t : True := by sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t : True")

    def test_case6_last_occurrence(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
theorem t1 : True := by sorry
theorem t2 : True := by sorry
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("theorem t2 : True")

    def test_case7_lemma(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
lemma L : 1 = 1 := by rfl
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is not None
        assert _normalise_sig(got) == _normalise_sig("lemma L : 1 = 1")

    def test_case8_no_theorem_returns_none(self, kimina_server_url: str) -> None:
        code = f"""{DEFAULT_IMPORTS}
"""
        ast = _create_ast(code, kimina_server_url)
        got = extract_signature_from_ast(ast)
        assert got is None
