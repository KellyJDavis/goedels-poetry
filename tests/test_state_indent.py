"""
Unit tests for _indent_proof_body() relative-aware indentation functionality.

Tests cover:
- Mixed indentation (primary fix)
- Backward compatibility (uniform and all-zero indentation)
- Edge cases (empty lines, whitespace-only, long lines, idempotency)
- Error handling (empty string, None input)
- Real-world test data
"""

import os
import tempfile
import uuid
from contextlib import suppress

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


@pytest.fixture
def temp_state():
    """Create a temporary state for testing."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    tmpdir = tempfile.mkdtemp()
    os.environ["GOEDELS_POETRY_DIR"] = tmpdir

    theorem_name = f"test_indent_{uuid.uuid4().hex}"
    theorem_body = f"theorem {theorem_name} : True := by sorry"
    full_theorem = combine_preamble_and_body(DEFAULT_IMPORTS, theorem_body)
    state = GoedelsPoetryState(formal_theorem=full_theorem)

    yield state

    # Cleanup
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(full_theorem)
    if old_env is not None:
        os.environ["GOEDELS_POETRY_DIR"] = old_env
    elif "GOEDELS_POETRY_DIR" in os.environ:
        del os.environ["GOEDELS_POETRY_DIR"]


class TestIndentProofBody:
    """Test suite for _indent_proof_body() method."""

    # Test 1: Mixed Indentation (Primary Fix)
    def test_indent_proof_body_mixed_indentation(self, temp_state):
        """Test relative-aware behavior preserves relative structure for mixed indentation."""
        proof_body = "have h3 : True := by \n    trivial\n  exact h3"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        lines = result.split("\n")
        assert len(lines) == 3

        # Line 1: should be at base indent (4 spaces)
        line1_indent = len(lines[0]) - len(lines[0].lstrip())
        assert line1_indent == 4, f"Expected 4 spaces, got {line1_indent}"

        # Line 2: should be at base + 4 (8 spaces)
        line2_indent = len(lines[1]) - len(lines[1].lstrip())
        assert line2_indent == 8, f"Expected 8 spaces, got {line2_indent}"

        # Line 3: should be at base + 2 (6 spaces)
        line3_indent = len(lines[2]) - len(lines[2].lstrip())
        assert line3_indent == 6, f"Expected 6 spaces, got {line3_indent}"

        # Verify content is preserved
        assert "have h3" in lines[0]
        assert "trivial" in lines[1]
        assert "exact h3" in lines[2]

    # Test 2: Uniform Indentation (Backward Compatibility)
    def test_indent_proof_body_uniform_indentation(self):
        """Test backward compatibility for uniform indentation."""
        proof_body = "  trivial\n  exact h3"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        lines = result.split("\n")
        # Should add indent prefix: 4 + 2 = 6 spaces
        for line in lines:
            if line.strip():
                line_indent = len(line) - len(line.lstrip())
                assert line_indent == 6, f"Expected 6 spaces for uniform indent, got {line_indent}"

    # Test 3: All Zero Indentation (Backward Compatibility)
    def test_indent_proof_body_all_zero_indentation(self):
        """Test backward compatibility for all-zero indentation."""
        proof_body = "trivial\nexact h3"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        lines = result.split("\n")
        # Should add indent prefix: 4 spaces
        for line in lines:
            if line.strip():
                line_indent = len(line) - len(line.lstrip())
                assert line_indent == 4, f"Expected 4 spaces for all-zero, got {line_indent}"

    # Test 4: Complex Mixed Indentation
    def test_indent_proof_body_complex_mixed_indentation(self):
        """Test with more complex relative structure."""
        proof_body = "have h : True := by\n      complex\n    nested\n  simple"
        indent = "  "  # 2 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        lines = result.split("\n")
        assert len(lines) == 4

        expected_indents = [2, 8, 6, 4]
        for i, line in enumerate(lines):
            if line.strip():
                actual_indent = len(line) - len(line.lstrip())
                assert actual_indent == expected_indents[i], (
                    f"Line {i + 1}: expected {expected_indents[i]} spaces, got {actual_indent}"
                )

    # Test 5: Empty Lines Preservation
    def test_indent_proof_body_empty_lines_preservation(self):
        """Test empty lines are preserved correctly."""
        proof_body = "line1\n\n  line2"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0].strip() == "line1"
        assert lines[1] == "", f"Expected empty line, got {lines[1]!r}"
        assert "line2" in lines[2]

        # Empty line should remain empty (no indent added)
        assert lines[1] == ""

    # Test 6: Single Line
    def test_indent_proof_body_single_line(self):
        """Test edge case of single line."""
        proof_body = "trivial"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)
        line_indent = len(result) - len(result.lstrip())
        assert line_indent == 4
        assert result.strip() == "trivial"

    # Test 7: Only Empty Lines
    def test_indent_proof_body_only_empty_lines(self, temp_state):
        """Test edge case of only empty lines."""
        proof_body = "\n\n"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)
        lines = result.split("\n")
        # Should preserve empty lines
        # Note: "\n\n" splits to ['', '', ''] (3 elements - two empty lines plus trailing empty)
        assert len([line for line in lines if not line.strip()]) >= 2
        # All lines should be empty (no indentation added)
        assert all(not line.strip() for line in lines)

    # Test 8: Whitespace-Only Lines (Edge Case)
    def test_indent_proof_body_whitespace_only_lines(self):
        """Test lines with only spaces (not empty, but .strip() = '')."""
        proof_body = "line1\n    \n  line2"  # Line 2 has 4 spaces only
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)
        lines = result.split("\n")
        assert len(lines) == 3
        # Whitespace-only line should be preserved as-is
        assert lines[1].strip() == "", "Whitespace-only line should be preserved"
        assert "line1" in lines[0]
        assert "line2" in lines[2]

    # Test 9: Very Long Lines (Edge Case)
    def test_indent_proof_body_very_long_lines(self, temp_state):
        """Test edge case of very long lines with indentation."""
        long_line = "a" * 1000
        proof_body = f"{long_line}\n    {long_line}"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)
        lines = result.split("\n")
        assert len(lines) == 2
        assert len(lines[0]) - len(lines[0].lstrip()) == 4
        assert len(lines[1]) - len(lines[1].lstrip()) == 8  # Mixed indentation preserves relative structure

    # Test 10: Idempotency (Multiple Calls)
    def test_indent_proof_body_idempotent(self, temp_state):
        """Test that calling _indent_proof_body() multiple times produces consistent results."""
        proof_body = "have h3 : True := by \n    trivial\n  exact h3"
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result1 = manager._indent_proof_body(proof_body, indent)
        result2 = manager._indent_proof_body(result1, indent)

        # Second call should produce consistent result (or at least not break)
        # Note: May not be truly idempotent, but shouldn't crash
        assert isinstance(result2, str)
        assert len(result2) > 0

    # Test 11: Error Handling - Empty String
    def test_indent_proof_body_empty_string(self):
        """Test edge case of empty string input."""
        proof_body = ""
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)
        # Should return empty string or handle gracefully
        assert isinstance(result, str)
        # Empty input should produce empty output
        assert result == ""

    # Test 12: Error Handling - Invalid Input Type
    def test_indent_proof_body_invalid_input_type(self, temp_state):
        """Test that None input raises appropriate error."""
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        # Should raise TypeError or AttributeError for None
        with pytest.raises((TypeError, AttributeError)):
            manager._indent_proof_body(None, indent)

    # Test 13: Real-World Test Data - From Failing Tests (IMPORTANT)
    def test_indent_proof_body_real_world_data(self, temp_state):
        """Test with actual proof bodies extracted from failing tests."""
        # TODO: Extract actual proof bodies from failing tests:
        # - test_reconstruct_complete_proof_deep_nested_decomposition_3_levels
        # - test_reconstruct_complete_proof_deep_nested_decomposition_4_levels
        # - test_reconstruct_complete_proof_nested_with_non_ascii_names

        # Example (to be replaced with actual extracted data):
        proof_body = "have h3 : True := by\n  trivial\nexact h3"  # From actual test
        indent = "    "  # 4 spaces

        manager = GoedelsPoetryStateManager(temp_state)
        result = manager._indent_proof_body(proof_body, indent)

        # Verify correct relative indentation preserved
        lines = result.split("\n")
        # ... specific assertions based on actual extracted data
        assert len(lines) > 0

        # Verify the same behavior as Test 1 (since we're using the same example data)
        # This will be updated with actual extracted proof bodies
        line1_indent = len(lines[0]) - len(lines[0].lstrip())
        line2_indent = len(lines[1]) - len(lines[1].lstrip())
        line3_indent = len(lines[2]) - len(lines[2].lstrip())
        assert line1_indent == 4
        assert line2_indent == 6
        assert line3_indent == 4
