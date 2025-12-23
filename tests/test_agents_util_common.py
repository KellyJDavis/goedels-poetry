"""Tests for goedels_poetry.agents.util.common module."""

from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    add_default_imports,
    combine_theorem_with_proof,
    get_error_str,
    load_prompt,
    remove_default_imports,
    strip_known_preamble,
)


def test_default_imports() -> None:
    """Test that DEFAULT_IMPORTS is defined correctly."""
    assert "import Mathlib" in DEFAULT_IMPORTS
    assert "import Aesop" in DEFAULT_IMPORTS
    assert "set_option maxHeartbeats 0" in DEFAULT_IMPORTS


def test_load_prompt_basic() -> None:
    """Test loading a basic prompt template."""
    # The prompt templates should exist in goedels_poetry/data/prompts/
    # Let's test with one of the known templates
    result = load_prompt("goedel-formalizer-v2")
    assert isinstance(result, str)
    assert len(result) > 0


def test_load_prompt_with_kwargs() -> None:
    """Test loading a prompt with template variables."""
    # Test with a prompt that uses variables
    result = load_prompt("goedel-prover-v2-initial", informal_statement="Test statement")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_error_str_empty_errors() -> None:
    """Test get_error_str with no errors."""
    code = "theorem test : True := by trivial"
    errors: list[dict] = []
    result = get_error_str(code, errors, error_thres=True)
    assert result == ""


def test_get_error_str_single_error() -> None:
    """Test get_error_str with a single error."""
    code = "line0\nline1\nline2\nline3\nline4\nline5\nline6"
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 0, "column": 5},
            "data": "Test error message",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "Test error message" in result
    assert "<error>" in result
    assert "</error>" in result
    # Line with error should be present (pos.line=0 + 2 = line2)
    assert "ne2" in result  # Part of line2 is in the error


def test_get_error_str_multiline_error() -> None:
    """Test get_error_str with an error spanning multiple lines."""
    code = "line0\nline1\nline2\nline3\nline4\nline5\nline6\nline7\nline8"
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 2, "column": 3},
            "data": "Multiline error",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "Multiline error" in result
    assert "<error>" in result
    assert "</error>" in result


def test_get_error_str_without_end_pos() -> None:
    """Test get_error_str with error missing endPos."""
    code = "line0\nline1\nline2\nline3\nline4\nline5"
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "data": "Error without endPos",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "Error without endPos" in result
    assert "<error>" in result
    assert "</error>" in result


def test_get_error_str_with_threshold() -> None:
    """Test get_error_str respects error threshold."""
    code = "\n".join([f"line{i}" for i in range(20)])
    errors = [
        {
            "pos": {"line": i, "column": 2},
            "endPos": {"line": i, "column": 5},
            "data": f"Error {i + 1}",
        }
        for i in range(15)
    ]

    # With threshold
    result_with_threshold = get_error_str(code, errors, error_thres=True)
    assert "Error 1:" in result_with_threshold
    assert "Error 8:" in result_with_threshold
    assert "Error 9:" not in result_with_threshold
    assert "[Omitted 7 more errors]" in result_with_threshold

    # Without threshold
    result_without_threshold = get_error_str(code, errors, error_thres=False)
    assert "Error 1:" in result_without_threshold
    assert "Error 15:" in result_without_threshold
    assert "[Omitted" not in result_without_threshold


def test_get_error_str_shows_context() -> None:
    """Test that get_error_str shows context lines around errors."""
    code = "line0\nline1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9"
    errors = [
        {
            "pos": {"line": 2, "column": 2},  # This will be line4 (2 + 2)
            "endPos": {"line": 2, "column": 5},
            "data": "Test error",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    # Should show 4 lines before the error
    # Line 4 is at index 4, so lines before would be 0, 1, 2, 3
    assert "line0" in result or "line1" in result  # Context lines


def test_get_error_str_formats_code_block() -> None:
    """Test that error strings are formatted as code blocks."""
    code = "line0\nline1\nline2\nline3\nline4\nline5"
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 0, "column": 5},
            "data": "Test error",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "```lean4" in result
    assert "```" in result


def test_get_error_str_multiple_errors() -> None:
    """Test get_error_str with multiple errors."""
    code = "\n".join([f"line{i}" for i in range(20)])
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 0, "column": 5},
            "data": "First error",
        },
        {
            "pos": {"line": 3, "column": 1},
            "endPos": {"line": 3, "column": 4},
            "data": "Second error",
        },
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "First error" in result
    assert "Error 2:" in result
    assert "Second error" in result


def test_get_error_str_truncated_multiline() -> None:
    """Test that very long multiline errors get truncated with threshold."""
    code = "\n".join([f"line{i}" for i in range(30)])
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 20, "column": 3},  # Very long error
            "data": "Very long error",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "Very long error" in result
    # Should have truncation marker for long errors
    if "line" in code.split("\n")[0]:
        # Check that it doesn't include all lines
        assert "--[Truncated]--" in result or result.count("line") < 20


def test_get_error_str_edge_case_first_line() -> None:
    """Test error on the first line after adjustment."""
    code = "line0\nline1\nline2\nline3"
    errors = [
        {
            "pos": {"line": -2, "column": 0},  # After +2 adjustment, will be line 0
            "endPos": {"line": -2, "column": 3},
            "data": "Error on adjusted first line",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    assert "Error 1:" in result
    assert "Error on adjusted first line" in result


def test_get_error_str_out_of_range_positions() -> None:
    """Test that get_error_str handles errors with out-of-range positions."""
    code = "line0\nline1\nline2"
    errors = [
        {
            "pos": {"line": 100, "column": 50},
            "endPos": {"line": 150, "column": 60},
            "data": "Out of range error",
        }
    ]
    result = get_error_str(code, errors, error_thres=False)

    assert "Error 1:" in result
    assert "Out of range error" in result
    assert "<error>" in result
    assert "</error>" in result


def test_get_error_str_next_line_included() -> None:
    """Test that the line after the error is included in context."""
    code = "line0\nline1\nline2\nline3\nline4\nline5\nline6"
    errors = [
        {
            "pos": {"line": 0, "column": 2},
            "endPos": {"line": 0, "column": 5},
            "data": "Test",
        }
    ]
    result = get_error_str(code, errors, error_thres=True)

    # Line after error (line3 = line at index 3 = pos.line + 2 + 1)
    assert "line3" in result


def test_add_default_imports() -> None:
    """Test that add_default_imports correctly adds the preamble."""
    code = "theorem test : True := by trivial"
    result = add_default_imports(code)
    assert result.startswith(DEFAULT_IMPORTS)
    assert code in result


def test_remove_default_imports_exact_match() -> None:
    """Test remove_default_imports with exact DEFAULT_IMPORTS prefix."""
    code = "theorem test : True := by trivial"
    code_with_imports = DEFAULT_IMPORTS + code
    result = remove_default_imports(code_with_imports)
    assert result == code


def test_remove_default_imports_no_imports() -> None:
    """Test remove_default_imports when code has no imports."""
    code = "theorem test : True := by trivial"
    result = remove_default_imports(code)
    assert result == code


def test_remove_default_imports_chatgpt_style_preamble() -> None:
    """Test remove_default_imports with ChatGPT/ChatOpenAI style preamble."""
    preamble = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

noncomputable section
open scoped Classical

"""
    code = "theorem test : True := by trivial"
    code_with_preamble = preamble + code
    result = remove_default_imports(code_with_preamble)
    assert result == code


def test_remove_default_imports_with_comments() -> None:
    """Test remove_default_imports with preamble containing comments."""
    preamble = """import Mathlib
import Aesop

-- This is a comment
set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""
    code = "theorem test : True := by trivial"
    code_with_preamble = preamble + code
    result = remove_default_imports(code_with_preamble)
    assert result == code


def test_remove_default_imports_with_multiline_comment() -> None:
    """Test remove_default_imports with multiline comment in preamble."""
    preamble = """import Mathlib
import Aesop

/-
This is a multiline comment
explaining the proof
-/

"""
    code = "theorem test : True := by trivial"
    code_with_preamble = preamble + code
    result = remove_default_imports(code_with_preamble)
    assert result == code


def test_remove_default_imports_preserves_theorem_comments() -> None:
    """Test that remove_default_imports doesn't remove comments that are part of the theorem."""
    preamble = """import Mathlib
import Aesop

open BigOperators Real Nat Topology Rat

"""
    code = """-- This comment is part of the theorem
theorem test : True := by trivial"""
    code_with_preamble = preamble + code
    result = remove_default_imports(code_with_preamble)
    # The comment should be preserved as it's part of the theorem
    assert "-- This comment is part of the theorem" in result


def test_strip_known_preamble_folds_new_header_into_body() -> None:
    existing_preamble = "import Mathlib"
    llm_code = """
import Mathlib

syntax (name := strange) "strange" term:41 : term

theorem weird : True := by
  trivial
"""
    body, matched = strip_known_preamble(llm_code.strip(), existing_preamble)
    assert matched is False
    assert body.startswith('import Mathlib\n\nsyntax (name := strange) "strange"')
    assert "theorem weird" in body


def test_strip_known_preamble_treats_header_only_snippet_as_body() -> None:
    existing_preamble = "import Mathlib"
    llm_code = """
syntax (name := strange) "strange" term:41 : term
"""
    body, matched = strip_known_preamble(llm_code.strip(), existing_preamble)
    assert matched is False
    assert body.startswith('syntax (name := strange) "strange"')


def test_strip_known_preamble_handles_empty_expected_preamble() -> None:
    existing_preamble = ""
    llm_code = """
-- no imports here
@[simp] theorem neat : True := by
  trivial
"""
    body, matched = strip_known_preamble(llm_code.strip(), existing_preamble)
    assert matched is True
    assert body.startswith("-- no imports here")
    assert "@[simp] theorem neat" in body


def test_remove_default_imports_empty_lines() -> None:
    """Test remove_default_imports handles multiple empty lines in preamble."""
    preamble = """import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


"""
    code = "theorem test : True := by trivial"
    code_with_preamble = preamble + code
    result = remove_default_imports(code_with_preamble)
    assert result == code


def test_remove_default_imports_round_trip() -> None:
    """Test that add_default_imports and remove_default_imports are inverses."""
    code = "theorem test : True := by trivial"
    with_imports = add_default_imports(code)
    without_imports = remove_default_imports(with_imports)
    assert without_imports == code


def test_combine_theorem_with_proof_single_line() -> None:
    """Test combining theorem with proof body (single line format)."""
    theorem = "theorem test : True := by sorry"
    proof_body = "trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True := by\ntrivial"


def test_combine_theorem_with_proof_multiline() -> None:
    """Test combining theorem with proof body (multiline format)."""
    theorem = "theorem test : True := by sorry"
    proof_body = """  have h : True := trivial
  exact h"""
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True := by\n  have h : True := trivial\n  exact h"


def test_combine_theorem_with_proof_no_spaces() -> None:
    """Test combining theorem with proof body (no spaces in := by sorry)."""
    theorem = "theorem test : True :=by sorry"
    proof_body = "trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True :=by\ntrivial"


def test_combine_theorem_with_proof_multiple_spaces() -> None:
    """Test combining theorem with proof body (multiple spaces)."""
    theorem = "theorem test : True :=  by   sorry"
    proof_body = "trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True :=  by\ntrivial"


def test_combine_theorem_with_proof_with_newlines() -> None:
    """Test combining theorem with proof body when original has newlines."""
    theorem = "theorem test : True :=\n  by\n  sorry"
    proof_body = "  trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True :=\n  by\n  trivial"


def test_combine_theorem_with_proof_empty_proof_body() -> None:
    """Test combining theorem with empty proof body."""
    theorem = "theorem test : True := by sorry"
    proof_body = ""
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == theorem


def test_combine_theorem_with_proof_no_sorry_pattern() -> None:
    """Test combining theorem when no := by sorry pattern is found."""
    theorem = "theorem test : True := by"
    proof_body = "trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert result == "theorem test : True := by\ntrivial"


def test_combine_theorem_with_proof_with_doc_comment() -> None:
    """Test combining theorem with proof body when theorem has doc comment."""
    theorem = """/-- This is a test theorem. -/
theorem test : True := by sorry"""
    proof_body = "trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert "/-- This is a test theorem. -/" in result
    assert "theorem test : True := by" in result
    assert "trivial" in result
    assert "sorry" not in result


def test_combine_theorem_with_proof_with_comment_before_sorry() -> None:
    """Test combining when comment appears between by and sorry."""
    theorem = """theorem test : True := by
    -- A comment to test parsing
    sorry"""
    proof_body = "  trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert "sorry" not in result
    assert result == "theorem test : True := by\n  trivial"


def test_combine_theorem_with_proof_comment_between_colon_eq_and_by() -> None:
    """Test combining when comment appears after := and before by."""
    theorem = """theorem test : True :=
  -- comment between := and by
  by
  sorry"""
    proof_body = "  trivial"
    result = combine_theorem_with_proof(theorem, proof_body)
    assert "sorry" not in result
    # Preserve comment block while inserting proof after `by`
    assert "comment between := and by" in result
    assert result.endswith("by\n  trivial")
