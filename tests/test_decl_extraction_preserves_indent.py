from __future__ import annotations

from goedels_poetry.parsers.util.foundation.decl_extraction import _extract_tactics_from_proof_node


def test_extract_tactics_from_byTactic_preserves_first_line_indent_when_by_trailing_has_no_spaces() -> None:
    """
    Regression test for proof-body extraction indentation.

    If Kimina's `by` token has trailing == "\\n" (no indentation spaces), and we `.strip()` the
    tactic body, the first tactic line can lose its indentation while later lines remain indented.
    That breaks relative indentation and can later make inlining fail with
    "All indentation strategies failed".
    """
    proof_node = {
        "kind": "Lean.Parser.Term.byTactic",
        "args": [
            {"val": "by", "info": {"trailing": "\n", "leading": ""}},
            {
                "kind": "Lean.Parser.Tactic.tacticSeq",
                # Use a simplified node: _ast_to_code serializes `val` verbatim.
                "val": "  have h₄ : True := by\n    trivial\n\n  apply h₄",
                "info": {"leading": "", "trailing": ""},
                "args": [],
            },
        ],
    }

    extracted = _extract_tactics_from_proof_node(proof_node)
    assert extracted.startswith("  have "), "Leading indentation of the first line must be preserved"
    assert "\n\n  apply h₄" in extracted, "Subsequent aligned lines must remain aligned"
