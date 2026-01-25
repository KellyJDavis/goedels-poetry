"""Tests for goedels_poetry.agents.prover_agent module."""

import pytest

from goedels_poetry.agents.prover_agent import _parse_prover_response
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, LLMParsingError


def test_parse_prover_response_extracts_full_code_block() -> None:
    """Test that _parse_prover_response extracts the full Lean code block."""
    response = """Here's the proof:

```lean4
theorem test : True := by
  trivial
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert result == "theorem test : True := by\n  trivial"
    assert "theorem test" in result
    assert ":= by" in result


def test_parse_prover_response_includes_preamble_if_present() -> None:
    """Test that _parse_prover_response includes the preamble if present in the code block."""
    response = f"""```lean4
{DEFAULT_IMPORTS}theorem test : True := by
  trivial
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert DEFAULT_IMPORTS in result
    assert "theorem test : True := by" in result
    assert "  trivial" in result


def test_parse_prover_response_multiline_proof() -> None:
    """Test that _parse_prover_response handles multiline proofs by returning the full block."""
    response = """```lean4
theorem test : True := by
  have h : True := trivial
  exact h
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert "theorem test : True := by" in result
    assert "have h : True := trivial" in result
    assert "exact h" in result


def test_parse_prover_response_no_code_block() -> None:
    """Test that _parse_prover_response raises error when no code block found."""
    response = "This is just text, no code block"
    expected_preamble = DEFAULT_IMPORTS
    with pytest.raises(LLMParsingError):
        _parse_prover_response(response, expected_preamble)


def test_parse_prover_response_multiple_code_blocks() -> None:
    """Test that _parse_prover_response uses the last code block in full."""
    response = """First block:
```lean4
theorem first : True := by sorry
```

Second block:
```lean4
theorem second : True := by
  trivial
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert result == "theorem second : True := by\n  trivial"
    assert "theorem first" not in result
    assert "theorem second" in result


def test_parse_prover_response_no_by_pattern() -> None:
    """Test that _parse_prover_response handles code without := by pattern."""
    response = """```lean4
just some tactics
more tactics
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    # Should return the whole code block content
    assert result == "just some tactics\nmore tactics"


def test_parse_prover_response_with_comments() -> None:
    """Test that _parse_prover_response preserves the full block including comments."""
    response = """```lean4
theorem test : True := by
  -- This is a comment
  trivial
```"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert "theorem test : True := by" in result
    assert "-- This is a comment" in result
    assert "trivial" in result


def test_parse_prover_response_whitespace_variations() -> None:
    """Test that _parse_prover_response handles whitespace variations by returning the full block."""
    # Test with :=by (no spaces)
    response1 = """```lean4
theorem test : True :=by
  trivial
```"""
    expected_preamble = DEFAULT_IMPORTS
    result1 = _parse_prover_response(response1, expected_preamble)
    assert result1 == "theorem test : True :=by\n  trivial"

    # Test with :=  by  (multiple spaces)
    response2 = """```lean4
theorem test : True :=  by
  trivial
```"""
    result2 = _parse_prover_response(response2, expected_preamble)
    assert result2 == "theorem test : True :=  by\n  trivial"


def test_parse_prover_response_unclosed_final_code_block() -> None:
    """Test fallback when the final code block is missing closing ticks."""
    response = """Plan:
```lean4
theorem plan : True := by
  sorry
```

Final:
```lean4
theorem final : True := by
  trivial
"""
    expected_preamble = DEFAULT_IMPORTS
    result = _parse_prover_response(response, expected_preamble)
    assert result == "theorem final : True := by\n  trivial"
