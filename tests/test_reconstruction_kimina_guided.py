"""
Kimina-backed integration test for Kimina-guided reconstruction selection.

This test intentionally constructs a case where the baseline reconstruction fails final
verification due to a formatting/indentation normalizer, but an alternative variant succeeds.
"""

# ruff: noqa: RUF001

from __future__ import annotations

from contextlib import suppress
from typing import cast

import pytest

# Try to import the required modules - skip all tests if imports fail
try:
    from kimina_client import KiminaClient  # noqa: F401

    from goedels_poetry.agents.proof_checker_agent import check_complete_proof
    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    IMPORTS_AVAILABLE = True
except Exception as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"Failed to import required modules: {e}"


if not IMPORTS_AVAILABLE:
    pytestmark = pytest.mark.skip(reason=SKIP_REASON)
else:
    pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def _with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def test_kimina_guided_reconstruction_recovers_from_baseline_failure(kimina_server_url: str) -> None:
    import uuid

    theorem_sig = f"theorem test_kimina_guided_{uuid.uuid4().hex} : True"
    theorem = _with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  have h_goal : True := by
    sorry
  exact h_goal
"""

        # Locate the `sorry` token for `h_goal`.
        sorry_start = sketch.index("sorry", sketch.index("have h_goal"))
        sorry_end = sorry_start + len("sorry")

        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Child proof body intentionally contains:
        # - an indented comment line
        # - a following tactic line that must remain indented under `by`
        #
        # The baseline reconstruction's comment-based indentation fixer snaps the tactic left,
        # breaking the `by` block and causing final verification to fail.
        # Note: `ring_nf` can simplify away equalities into `True`, so we use a `True` goal here.
        # This case reliably:
        # - fails under the baseline reconstruction (comment-based fixer dedents `ring_nf at h`)
        # - succeeds when the fixer is disabled (guided selection should pick that variant)
        child_proof = """have h_main : True := by
  have h : (0 : ℚ) = 0 := by
    rfl
  -- keep ring_nf inside the `by` block
  ring_nf at h
  trivial
exact h_main"""

        child = FormalTheoremProofState(
            parent=cast(TreeNode, root),
            depth=1,
            formal_theorem="lemma h_goal : True",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=child_proof,
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
            hole_name="h_goal",
            hole_start=sorry_start,
            hole_end=sorry_end,
        )

        root["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, root)

        manager = GoedelsPoetryStateManager(state)

        baseline = manager.reconstruct_complete_proof()
        ok0, _ = check_complete_proof(baseline, server_url=kimina_server_url, server_max_retries=3)
        assert not ok0, "Expected baseline reconstruction to fail final verification in this test case"

        guided, ok1, err1 = manager.reconstruct_complete_proof_kimina_guided(
            server_url=kimina_server_url, server_max_retries=3, max_candidates=12
        )
        assert ok1, f"Expected Kimina-guided reconstruction to succeed, but it failed:\n{err1}"
        assert "sorry" not in guided
        assert state.reconstruction_attempts >= 1
        assert state.reconstruction_strategy_used is not None

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
