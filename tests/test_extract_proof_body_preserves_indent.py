from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


def test_extract_proof_body_ast_guided_preserves_leading_indent() -> None:
    """
    Regression test: do not strip leading indentation from a leaf `formal_proof`.

    Proof bodies often contain nested `have ... := by` blocks where subsequent tactics (e.g. `apply h₄`)
    must be aligned with the `have` line to be outside the nested `by` block. If we strip the first
    line's leading spaces but leave later lines indented, the relative structure breaks and
    reconstruction can fail with "All indentation strategies failed".
    """
    # Minimal manager; the helper under test doesn't depend on the stored state content.
    #
    # NOTE: GoedelsPoetryState has a filesystem side-effect (creates a theorem output dir).
    # Use a unique theorem string and clean up afterwards to keep this test repeatable.
    theorem = f"test_extract_proof_body_preserves_indent__{uuid.uuid4().hex}"
    try:
        manager = GoedelsPoetryStateManager(GoedelsPoetryState(informal_theorem=theorem))

        formal_proof = "  have h₄ : True := by\n    trivial\n\n  apply h₄"
        leaf = {
            "id": "x",
            "parent": None,
            "depth": 0,
            "formal_theorem": "theorem t : True := by sorry",
            "preamble": "",
            "syntactic": True,
            "formal_proof": formal_proof,
            "proved": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 0,
            "proof_history": [],
            "pass_attempts": 0,
            "hole_name": "hv_subst",
            "hole_start": 0,
            "hole_end": 0,
            "llm_lean_output": None,
        }

        extracted = manager._extract_proof_body_ast_guided(  # type: ignore[arg-type]
            leaf,
            kimina_client=MagicMock(),
            server_timeout=0,
        )

        assert extracted.startswith("  have "), "Leading indentation should be preserved for leaf proofs"
        # And ensure the later line remains aligned (still has 2 spaces).
        assert "\n  apply h₄" in extracted
    finally:
        GoedelsPoetryState.clear_theorem_directory(theorem)
