"""Tests for proof composition with deep nested decomposition."""

from contextlib import suppress

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState


def with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def _annotate_hole_offsets(node: dict, sketch: str, *, hole_name: str, anchor: str) -> None:
    """
    Attach hole metadata to a manually-constructed proof-tree node for offset-based reconstruction tests.
    """
    base = sketch.index(anchor)
    i = sketch.find("sorry", base)
    if i == -1:
        raise ValueError("No `sorry` token found after anchor")  # noqa: TRY003
    node["hole_name"] = hole_name
    node["hole_start"] = i
    node["hole_end"] = i + len("sorry")


def test_reconstruct_complete_proof_deep_nested_decomposition_4_levels() -> None:
    """Test reconstruct_complete_proof with 4 levels of nested DecomposedFormalTheoremState."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_deep_4_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Level 0: Root
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  have a : A := by sorry
  exact a""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 1
        level1 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma a : A",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma a : A := by
  have b : B := by sorry
  exact b""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level1, str(root["proof_sketch"]), hole_name="a", anchor="have a")

        # Level 2
        level2 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level1),
            children=[],
            depth=2,
            formal_theorem="lemma b : B",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma b : B := by
  have c : C := by sorry
  exact c""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level2, str(level1["proof_sketch"]), hole_name="b", anchor="have b")

        # Level 3
        level3 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level2),
            children=[],
            depth=3,
            formal_theorem="lemma c : C",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma c : C := by
  have d : D := by sorry
  exact d""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level3, str(level2["proof_sketch"]), hole_name="c", anchor="have c")

        # Level 4: Leaf
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, level3),
            depth=4,
            formal_theorem="lemma d : D",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma d : D := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(leaf, str(level3["proof_sketch"]), hole_name="d", anchor="have d")

        # Build tree
        level3["children"].append(cast(TreeNode, leaf))
        level2["children"].append(cast(TreeNode, level3))
        level1["children"].append(cast(TreeNode, level2))
        root["children"].append(cast(TreeNode, level1))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "have a : A := by" in result
        assert "have b : B := by" in result
        assert "have c : C := by" in result
        assert "have d : D := by" in result
        assert "trivial" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
