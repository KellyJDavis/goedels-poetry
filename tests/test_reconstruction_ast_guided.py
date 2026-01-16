# ruff: noqa: RUF001
from __future__ import annotations

from typing import cast

from ast_test_utils import build_simple_ast, find_sorry_spans

from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, compute_source_hashes
from goedels_poetry.state import GoedelsPoetryStateManager
from goedels_poetry.util.tree import TreeNode


def _mk_leaf(
    parent: TreeNode,
    *,
    hole_name: str,
    hole_start: int,
    hole_end: int,
    proof_body: str,
) -> FormalTheoremProofState:
    # Minimal leaf FormalTheoremProofState for reconstruction tests.
    lemma_source = f"lemma {hole_name} : True := by{proof_body}"
    return FormalTheoremProofState(
        parent=parent,
        depth=1,
        formal_theorem=f"lemma {hole_name} : True := by sorry",
        preamble=cast(DecomposedFormalTheoremState, parent)["preamble"],
        syntactic=True,
        formal_proof=proof_body,
        proved=True,
        errors=None,
        ast=build_simple_ast(lemma_source),
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
        hole_name=hole_name,
        hole_start=hole_start,
        hole_end=hole_end,
    )


def test_reconstruction_fills_named_have_holes_by_offsets() -> None:
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
        "  norm_num",
        "",
        "-- close the goal",
        "  exact h_main",
    ])

    # Build a minimal state tree rooted at a decomposed node.
    # Avoid constructing a full GoedelsPoetryState here because it creates persistent
    # theorem output directories under the user's home directory.
    class _DummyState:
        _root_preamble = DEFAULT_IMPORTS
        formal_theorem_proof: TreeNode | None = None

    st = _DummyState()
    mgr = GoedelsPoetryStateManager(cast(object, st))  # type: ignore[arg-type]

    root: DecomposedFormalTheoremState = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem mathd_algebra_478 ...",  # not used by reconstruction here
        preamble=DEFAULT_IMPORTS,
        proof_sketch=parent_sketch,
        syntactic=True,
        errors=None,
        ast=build_simple_ast(
            parent_sketch,
            sorry_spans=[(hv_sorry_start, hv_sorry_end), (hcalc_sorry_start, hcalc_sorry_end)],
        ),
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
        hole_name=None,
        hole_start=None,
        hole_end=None,
        source_hash_raw=None,
        source_hash_normalized=None,
    )
    ast_text = root["ast"].get_body_text() if root.get("ast") is not None else parent_sketch
    raw_hash, normalized_hash = compute_source_hashes(ast_text)
    root["source_hash_raw"] = raw_hash
    root["source_hash_normalized"] = normalized_hash

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

    reconstructed = mgr.reconstruct_complete_proof()
    assert "sorry" not in reconstructed
    assert "exact h_main" in reconstructed


def test_reconstruction_prefers_canonical_sketch_text() -> None:
    canonical_sketch = """theorem recon_canonical (n : Nat) : n = n := by
  have h₁ : n = n := by
    sorry

  have h₂ : n = n := by
    sorry
  exact h₂
"""
    altered_sketch = canonical_sketch.replace("\n\n  have h₂", "\n  have h₂")

    sorry_spans = find_sorry_spans(canonical_sketch)
    (h1_start, h1_end), (h2_start, h2_end) = sorry_spans

    class _DummyState:
        _root_preamble = DEFAULT_IMPORTS
        formal_theorem_proof: TreeNode | None = None

    st = _DummyState()
    mgr = GoedelsPoetryStateManager(cast(object, st))  # type: ignore[arg-type]

    root: DecomposedFormalTheoremState = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem recon_canonical ...",
        preamble=DEFAULT_IMPORTS,
        proof_sketch=altered_sketch,
        syntactic=True,
        errors=None,
        ast=build_simple_ast(
            canonical_sketch,
            sorry_spans=[(h1_start, h1_end), (h2_start, h2_end)],
        ),
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
        hole_name=None,
        hole_start=None,
        hole_end=None,
        source_hash_raw=None,
        source_hash_normalized=None,
    )
    ast_text = root["ast"].get_body_text() if root.get("ast") is not None else canonical_sketch
    raw_hash, normalized_hash = compute_source_hashes(ast_text)
    root["source_hash_raw"] = raw_hash
    root["source_hash_normalized"] = normalized_hash

    root["children"].append(
        cast(
            TreeNode,
            _mk_leaf(
                cast(TreeNode, root),
                hole_name="h₁",
                hole_start=h1_start,
                hole_end=h1_end,
                proof_body="rfl",
            ),
        )
    )
    root["children"].append(
        cast(
            TreeNode,
            _mk_leaf(
                cast(TreeNode, root),
                hole_name="h₂",
                hole_start=h2_start,
                hole_end=h2_end,
                proof_body="rfl",
            ),
        )
    )

    st.formal_theorem_proof = cast(TreeNode, root)
    reconstructed = mgr.reconstruct_complete_proof()
    assert "sorry" not in reconstructed
    assert "\n\n  have h₂" in reconstructed
