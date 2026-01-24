# ruff: noqa: RUF001
from __future__ import annotations

from typing import cast

import pytest

from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
from goedels_poetry.parsers.ast import AST
from goedels_poetry.state import GoedelsPoetryStateManager
from goedels_poetry.util.tree import TreeNode

# Mark tests that require Kimina server as integration tests
# These tests use reconstruct_complete_proof() which now requires Kimina server for validation
pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def _create_ast_for_sketch(
    sketch: str, preamble: str = DEFAULT_IMPORTS, server_url: str = "http://localhost:8000", server_timeout: int = 60
):
    """
    Helper function to create an AST for a proof sketch in tests.

    This is needed because the new reconstruction implementation requires ASTs
    for all DecomposedFormalTheoremState nodes.
    """
    from kimina_client import KiminaClient

    from goedels_poetry.agents.util.common import combine_preamble_and_body, remove_default_imports_from_ast
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


def _mk_leaf(
    parent: TreeNode,
    *,
    hole_name: str,
    hole_start: int,
    hole_end: int,
    proof_body: str,
) -> FormalTheoremProofState:
    # Minimal leaf FormalTheoremProofState for reconstruction tests.
    return FormalTheoremProofState(
        parent=parent,
        depth=1,
        formal_theorem=f"lemma {hole_name} : True := by sorry",
        preamble=cast(DecomposedFormalTheoremState, parent)["preamble"],
        syntactic=True,
        formal_proof=proof_body,
        proved=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
        hole_name=hole_name,
        hole_start=hole_start,
        hole_end=hole_end,
        llm_lean_output=None,
    )


def test_reconstruction_fills_named_have_holes_by_offsets(kimina_server_url: str) -> None:
    # Parent sketch (body-only). Two named have subgoals, each a single `sorry`.
    parent_sketch = """theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
    (h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
  have hv : v = (1 / 3 : ℝ) * (b * h) := by
    simpa using h₁

  have hv' : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := by
    sorry

  have hcalc : (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) = 65 := by
    sorry

  calc
    v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := hv'
    _ = 65 := hcalc
"""

    # Locate the two `sorry` tokens (exact offsets are what AST-guided reconstruction records).
    hv_sorry_start = parent_sketch.index("sorry", parent_sketch.index("have hv'"))
    hv_sorry_end = hv_sorry_start + len("sorry")
    hcalc_sorry_start = parent_sketch.index("sorry", parent_sketch.index("have hcalc"))
    hcalc_sorry_end = hcalc_sorry_start + len("sorry")

    # Child proof bodies. `hcalc` includes a deliberately odd indentation on `exact h_main`
    # (this mirrors the failure mode seen in partial.log).
    hv_proof = "simpa [hv, h₂, h₃]"
    hcalc_proof = "\n".join([
        "have h_main : (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) = 65 := by",
        "    norm_num",
        "",
        "  -- close the goal",
        "  exact h_main",
    ])

    # Build a minimal state tree rooted at a decomposed node.
    # Avoid constructing a full GoedelsPoetryState here because it creates persistent
    # theorem output directories under the user's home directory.
    class _DummyState:
        _root_preamble = DEFAULT_IMPORTS
        formal_theorem_proof: TreeNode | None = None

    st = _DummyState()
    mgr = GoedelsPoetryStateManager(cast(object, st))  # runtime duck-typing

    root_ast = _create_ast_for_sketch(parent_sketch, DEFAULT_IMPORTS, kimina_server_url)
    root: DecomposedFormalTheoremState = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem mathd_algebra_478 ...",  # not used by reconstruction here
        preamble=DEFAULT_IMPORTS,
        proof_sketch=parent_sketch,
        syntactic=True,
        errors=None,
        ast=root_ast,
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
        llm_lean_output=None,
    )

    root["children"].append(
        cast(
            TreeNode,
            _mk_leaf(
                cast(TreeNode, root),
                hole_name="hv'",
                hole_start=hv_sorry_start,
                hole_end=hv_sorry_end,
                proof_body=hv_proof,
            ),
        )
    )
    root["children"].append(
        cast(
            TreeNode,
            _mk_leaf(
                cast(TreeNode, root),
                hole_name="hcalc",
                hole_start=hcalc_sorry_start,
                hole_end=hcalc_sorry_end,
                proof_body=hcalc_proof,
            ),
        )
    )

    st.formal_theorem_proof = cast(TreeNode, root)

    reconstructed = mgr.reconstruct_complete_proof(server_url=kimina_server_url)
    assert "sorry" not in reconstructed
    # Ensure the `exact h_main` line didn't become more-indented than the hole.
    assert "\n    exact h_main" in reconstructed
