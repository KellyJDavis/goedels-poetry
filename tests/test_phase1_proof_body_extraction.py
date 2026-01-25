"""Tests for Phase 1: Proof Body Extraction.

This test suite verifies that proof body extraction correctly includes
intro, have, let statements from have statements.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.high_level.subgoal_extraction_v2 import (
    _extract_proof_node_from_have,
    _extract_tactics_from_proof_node,
    extract_subgoal_with_check_responses,
)

# Mark tests that require Kimina server as integration tests
pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def _create_ast_for_sketch(
    sketch: str, preamble: str = DEFAULT_IMPORTS, server_url: str = "http://localhost:8000", server_timeout: int = 60
) -> AST:
    """
    Helper function to create an AST for a proof sketch in tests.
    """
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
    return {
        "errors": [
            {
                "severity": "error",
                "data": "unsolved goals\n" + "\n".join(f"  {h}" for h in hypotheses) + "\n  ⊢ True",
            }
        ],
        "pass": False,
        "complete": False,
    }


class TestExtractProofNodeFromHave:
    """Tests for _extract_proof_node_from_have function."""

    def test_extract_proof_node_from_have_by_tactic(self) -> None:
        """Test extraction from have with byTactic node."""
        # Create a mock have node with byTactic
        have_node = {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have"},
                {"val": "h"},
                {"val": ":"},
                {"val": "True"},
                {"val": ":="},
                {
                    "kind": "Lean.Parser.Term.byTactic",
                    "args": [
                        {"val": "by"},
                        {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [{"val": "trivial"}]},
                    ],
                },
            ],
        }

        result = _extract_proof_node_from_have(have_node)
        assert result is not None
        assert result.get("kind") == "Lean.Parser.Term.byTactic"

    def test_extract_proof_node_from_have_tactic_seq(self) -> None:
        """Test extraction from have with tacticSeq node."""
        # Create a mock have node with tacticSeq
        have_node = {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have"},
                {"val": "h"},
                {"val": ":"},
                {"val": "True"},
                {"val": ":="},
                {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [{"val": "trivial"}]},
            ],
        }

        result = _extract_proof_node_from_have(have_node)
        assert result is not None
        assert result.get("kind") == "Lean.Parser.Tactic.tacticSeq"

    def test_extract_proof_node_from_have_nested(self) -> None:
        """Test extraction from have with nested structure."""
        # Create a mock have node with nested proof structure
        have_node = {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have"},
                {"val": "h"},
                {"val": ":"},
                {"val": "True"},
                {"val": ":="},
                {
                    "kind": "Lean.Parser.Term.byTactic",
                    "args": [
                        {"val": "by"},
                        {
                            "kind": "Lean.Parser.Tactic.tacticSeq",
                            "args": [
                                {"kind": "Lean.Parser.Tactic.tacticIntro", "args": [{"val": "intro"}]},
                                {"val": "trivial"},
                            ],
                        },
                    ],
                },
            ],
        }

        result = _extract_proof_node_from_have(have_node)
        assert result is not None
        assert result.get("kind") in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}

    def test_extract_proof_node_from_have_anonymous(self) -> None:
        """Test extraction from anonymous have statements."""
        # Anonymous have (no name)
        have_node = {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have"},
                {"val": ":"},
                {"val": "True"},
                {"val": ":="},
                {
                    "kind": "Lean.Parser.Term.byTactic",
                    "args": [
                        {"val": "by"},
                        {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [{"val": "trivial"}]},
                    ],
                },
            ],
        }

        result = _extract_proof_node_from_have(have_node)
        assert result is not None

    def test_extract_proof_node_from_have_sorry(self) -> None:
        """Test extraction from have with sorry."""
        have_node = {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have"},
                {"val": "h"},
                {"val": ":"},
                {"val": "True"},
                {"val": ":="},
                {
                    "kind": "Lean.Parser.Term.byTactic",
                    "args": [
                        {"val": "by"},
                        {
                            "kind": "Lean.Parser.Tactic.tacticSeq",
                            "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
                        },
                    ],
                },
            ],
        }

        result = _extract_proof_node_from_have(have_node)
        assert result is not None

    def test_extract_proof_node_from_have_invalid_node(self) -> None:
        """Test extraction with invalid node."""
        # Not a have node
        result = _extract_proof_node_from_have({"kind": "Lean.Parser.Command.theorem"})
        assert result is None

        # Not a dict
        result = _extract_proof_node_from_have("not a dict")  # type: ignore[arg-type]
        assert result is None


class TestExtractTacticsFromProofNode:
    """Tests for _extract_tactics_from_proof_node function."""

    def test_extract_tactics_from_by_tactic(self) -> None:
        """Test tactics extraction from byTactic node."""
        # Create a mock byTactic node
        proof_node = {
            "kind": "Lean.Parser.Term.byTactic",
            "args": [
                {"val": "by"},
                {
                    "kind": "Lean.Parser.Tactic.tacticSeq",
                    "args": [
                        {"kind": "Lean.Parser.Tactic.tacticIntro", "args": [{"val": "intro"}, {"val": "z"}]},
                        {"val": "trivial"},
                    ],
                },
            ],
        }

        # Create a minimal AST for testing
        ast = AST({"kind": "root"}, source_text="", body_start=0)

        result = _extract_tactics_from_proof_node(proof_node, ast)
        # Should extract tactics without "by" keyword
        assert "by" not in result
        assert "intro" in result or "trivial" in result

    def test_extract_tactics_from_tactic_seq(self) -> None:
        """Test tactics extraction from tacticSeq node."""
        # Create a mock tacticSeq node
        proof_node = {
            "kind": "Lean.Parser.Tactic.tacticSeq",
            "args": [
                {"kind": "Lean.Parser.Tactic.tacticIntro", "args": [{"val": "intro"}, {"val": "z"}]},
                {"val": "trivial"},
            ],
        }

        ast = AST({"kind": "root"}, source_text="", body_start=0)

        result = _extract_tactics_from_proof_node(proof_node, ast)
        # Should extract tactics as-is (no "by" to remove)
        assert "intro" in result or "trivial" in result

    def test_extract_tactics_empty(self) -> None:
        """Test extraction from empty proof body."""
        proof_node = {
            "kind": "Lean.Parser.Term.byTactic",
            "args": [{"val": "by"}, {"kind": "Lean.Parser.Tactic.tacticSeq", "args": []}],
        }

        ast = AST({"kind": "root"}, source_text="", body_start=0)

        result = _extract_tactics_from_proof_node(proof_node, ast)
        assert result == "sorry"

    def test_extract_tactics_multiple_tactics(self) -> None:
        """Test extraction with multiple tactics."""
        proof_node = {
            "kind": "Lean.Parser.Term.byTactic",
            "args": [
                {"val": "by"},
                {
                    "kind": "Lean.Parser.Tactic.tacticSeq",
                    "args": [
                        {"kind": "Lean.Parser.Tactic.tacticIntro", "args": [{"val": "intro"}, {"val": "z"}]},
                        {
                            "kind": "Lean.Parser.Tactic.tacticHave_",
                            "args": [
                                {"val": "have"},
                                {"val": "hz"},
                                {"val": ":="},
                                {"val": "some_value"},
                            ],
                        },
                        {"val": "trivial"},
                    ],
                },
            ],
        }

        ast = AST({"kind": "root"}, source_text="", body_start=0)

        result = _extract_tactics_from_proof_node(proof_node, ast)
        # Should include all tactics
        assert len(result) > 0
        assert result != "sorry"

    def test_extract_tactics_invalid_node(self) -> None:
        """Test extraction with invalid node."""
        ast = AST({"kind": "root"}, source_text="", body_start=0)

        # Not a proof node
        result = _extract_tactics_from_proof_node({"kind": "Lean.Parser.Command.theorem"}, ast)
        assert result == "sorry"

        # Not a dict
        result = _extract_tactics_from_proof_node("not a dict", ast)  # type: ignore[arg-type]
        assert result == "sorry"


class TestExtractSubgoalWithCheckResponses:
    """Tests for extract_subgoal_with_check_responses function (integration tests)."""

    def test_extract_subgoal_includes_intro(self, kimina_server_url: str) -> None:
        """Test that intro statements are included in extracted subgoal."""
        sketch = """theorem test_intro : True := by
  have h : (z : ℤ) → z = z := by
    intro z
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        # Create check response with hypotheses
        check_response = _create_mock_check_response(["z : ℤ"])

        # Extract subgoal
        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should include intro statement
        assert "intro z" in result
        assert "lemma h" in result
        assert "(z : ℤ)" in result

    def test_extract_subgoal_includes_have(self, kimina_server_url: str) -> None:
        """Test that nested have statements are included."""
        sketch = """theorem test_have : True := by
  have h : True := by
    have hz : True := trivial
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response([])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should include nested have statement
        assert "have hz" in result or "hz" in result
        assert "lemma h" in result

    def test_extract_subgoal_includes_let(self, kimina_server_url: str) -> None:
        """Test that let statements are included."""
        sketch = """theorem test_let : True := by
  have h : True := by
    let z := 42
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response([])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should include let statement
        assert "let z" in result or "z" in result
        assert "lemma h" in result

    def test_extract_subgoal_includes_combined(self, kimina_server_url: str) -> None:
        """Test that combinations are included."""
        sketch = """theorem test_combined : True := by
  have h : (z : ℤ) → z = z := by
    intro z
    let x := z + 1
    have hx : x > z := by
      omega
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response(["z : ℤ"])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should include all tactics
        assert "intro z" in result or "intro" in result
        assert "lemma h" in result
        assert "(z : ℤ)" in result

    def test_extract_subgoal_preserves_sorry(self, kimina_server_url: str) -> None:
        """Test that existing sorry is preserved."""
        sketch = """theorem test_sorry : True := by
  have h : True := by
    trivial
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response([])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should preserve sorry
        assert "sorry" in result
        assert "lemma h" in result

    def test_extract_subgoal_adds_sorry_if_missing(self, kimina_server_url: str) -> None:
        """Test that sorry is added if not present."""
        sketch = """theorem test_add_sorry : True := by
  have h : True := by
    trivial
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response([])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should add sorry
        assert "sorry" in result
        assert "lemma h" in result

    def test_extract_subgoal_real_lean_code(self, kimina_server_url: str) -> None:
        """Test with real Lean code using Kimina server."""
        # Test case similar to numbertheory_4x3m7y3neq2003
        sketch = """theorem test_real : True := by
  have h : ∀ z : ZMod 7, z ^ 3 = 0 ∨ z ^ 3 = 1 ∨ z ^ 3 = 6 := by
    intro z
    have hz := cube_residue_mod7 z
    sorry
  sorry
"""

        ast = _create_ast_for_sketch(sketch, server_url=kimina_server_url)

        check_response = _create_mock_check_response(["z : ZMod 7"])

        result = extract_subgoal_with_check_responses(
            ast=ast,
            check_responses={"h": check_response},
            target_subgoal_identifier="h",
            target_subgoal_name="h",
        )

        # Should include intro and have statements
        assert "lemma h" in result
        # The binder format might vary, but z should be in the binders
        assert "z : ZMod 7" in result or "(z : ZMod 7)" in result
        # The proof body should be included (intro z, have hz := ...)
        assert "intro" in result.lower() or "have" in result.lower()
