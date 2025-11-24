"""Tests for proof composition with various edge cases and scenarios."""

from contextlib import suppress

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState


def with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def test_reconstruct_complete_proof_nested_with_non_ascii_names() -> None:
    """Test nested decomposition with non-ASCII names (unicode subscripts, Greek letters, etc.)."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_unicode_nested_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Root with unicode name
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  have α₁ : Q := by sorry
  exact α₁""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Child decomposed with Greek letter
        child_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma α₁ : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma α₁ : Q := by
  have β₂ : R := by sorry
  exact β₂""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Grandchild with another unicode name
        grandchild = FormalTheoremProofState(
            parent=cast(TreeNode, child_decomposed),
            depth=2,
            formal_theorem="lemma β₂ : R",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma β₂ : R := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        child_decomposed["children"].append(cast(TreeNode, grandchild))
        root["children"].append(cast(TreeNode, child_decomposed))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "have α₁ : Q := by" in result
        assert "have β₂ : R := by" in result
        assert "constructor" in result
        assert "exact α₁" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_let_statement() -> None:
    """Test reconstruct_complete_proof with 'let' statements in decomposition."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_let_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Sketch with let statement
        sketch = f"""{theorem_sig} := by
  let n : ℕ := 5
  have h : n > 0 := by sorry
  exact h"""  # noqa: RUF001

        decomposed = DecomposedFormalTheoremState(
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

        # Child proof that depends on the let binding
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h (n : ℕ) : n > 0",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h (n : ℕ) : n > 0 := by\n  omega",  # noqa: RUF001
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let n : ℕ := 5" in result  # noqa: RUF001
        assert "have h : n > 0 := by" in result
        assert "omega" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_obtain_statement() -> None:
    """Test reconstruct_complete_proof with 'obtain' statements in decomposition."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_obtain_{uuid.uuid4().hex} : Q"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Sketch with obtain statement
        sketch = f"""{theorem_sig} := by
  obtain ⟨x, hx⟩ : ∃ x, P x := by sorry
  have h : Q := by sorry
  exact h"""

        decomposed = DecomposedFormalTheoremState(
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

        # Child proof that depends on obtained variables
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h (x : T) (hx : P x) : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h (x : T) (hx : P x) : Q := by\n  exact hx",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "obtain ⟨x, hx⟩" in result
        assert "have h : Q := by" in result
        assert "exact hx" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        # The obtain's sorry should remain (it's not a have statement)
        # But the have's sorry should be replaced
        assert "have h : Q := by sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_let_and_have_nested() -> None:
    """Test reconstruct_complete_proof with 'let' and 'have' in nested decomposition."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_let_have_nested_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Root with let
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  let n : ℕ := 10
  have helper : n > 5 := by sorry
  exact helper""",  # noqa: RUF001
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Child decomposed with another let
        child_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma helper (n : ℕ) : n > 5",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma helper (n : ℕ) : n > 5 := by
  let m : ℕ := n + 1
  have h : m > 5 := by sorry
  exact h""",  # noqa: RUF001
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Grandchild proof
        grandchild = FormalTheoremProofState(
            parent=cast(TreeNode, child_decomposed),
            depth=2,
            formal_theorem="lemma h (n : ℕ) (m : ℕ) : m > 5",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h (n : ℕ) (m : ℕ) : m > 5 := by\n  omega",  # noqa: RUF001
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        child_decomposed["children"].append(cast(TreeNode, grandchild))
        root["children"].append(cast(TreeNode, child_decomposed))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let n : ℕ := 10" in result  # noqa: RUF001
        assert "have helper : n > 5 := by" in result
        assert "let m : ℕ := n + 1" in result  # noqa: RUF001
        assert "have h : m > 5 := by" in result
        assert "omega" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_mixed_bindings_deep_nested() -> None:
    """Test reconstruct_complete_proof with mixed let, obtain, and have in deep nested structure."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_mixed_deep_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Level 0: Root with let
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  let x : ℕ := 5
  have h1 : Q := by sorry
  exact h1""",  # noqa: RUF001
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 1: With obtain
        level1 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma h1 (x : ℕ) : Q",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h1 (x : ℕ) : Q := by
  obtain ⟨y, hy⟩ : ∃ y, R y := by sorry
  have h2 : S := by sorry
  exact h2""",  # noqa: RUF001
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 2: With let and have
        level2 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level1),
            children=[],
            depth=2,
            formal_theorem="lemma h2 (x : ℕ) (y : T) (hy : R y) : S",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h2 (x : ℕ) (y : T) (hy : R y) : S := by
  let z : ℕ := x + y
  have h3 : T := by sorry
  exact h3""",  # noqa: RUF001
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 3: Leaf
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, level2),
            depth=3,
            formal_theorem="lemma h3 (x : ℕ) (y : T) (hy : R y) (z : ℕ) : T",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h3 (x : ℕ) (y : T) (hy : R y) (z : ℕ) : T := by\n  trivial",  # noqa: RUF001
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        level2["children"].append(cast(TreeNode, leaf))
        level1["children"].append(cast(TreeNode, level2))
        root["children"].append(cast(TreeNode, level1))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let x : ℕ := 5" in result  # noqa: RUF001
        assert "have h1 : Q := by" in result
        assert "obtain ⟨y, hy⟩" in result
        assert "have h2 : S := by" in result
        assert "let z : ℕ := x + y" in result  # noqa: RUF001
        assert "have h3 : T := by" in result
        assert "trivial" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        # Only the obtain's sorry should remain
        assert "have h1 : Q := by sorry" not in result_no_imports
        assert "have h2 : S := by sorry" not in result_no_imports
        assert "have h3 : T := by sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_non_ascii_with_let_obtain() -> None:
    """Test reconstruct_complete_proof with non-ASCII names combined with let and obtain."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_unicode_bindings_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  let α : ℕ := 1
  obtain ⟨β, hβ⟩ : ∃ β, Q β := by sorry
  have γ : R := by sorry
  exact γ"""  # noqa: RUF001

        decomposed = DecomposedFormalTheoremState(
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

        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma γ (α : ℕ) (β : T) (hβ : Q β) : R",  # noqa: RUF001
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma γ (α : ℕ) (β : T) (hβ : Q β) : R := by\n  exact hβ",  # noqa: RUF001
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let α : ℕ := 1" in result  # noqa: RUF001
        assert "obtain ⟨β, hβ⟩" in result
        assert "have γ : R := by" in result  # noqa: RUF001
        assert "exact hβ" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "have γ : R := by sorry" not in result_no_imports  # noqa: RUF001

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_multiple_children_at_each_level() -> None:
    """Test reconstruct_complete_proof with multiple children at each level of nesting."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_multi_children_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Root with multiple haves
        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  have h1 : Q := by sorry
  have h2 : R := by sorry
  exact combine h1 h2""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # First child decomposed with multiple children
        child1_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma h1 : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h1 : Q := by
  have h1a : Q1 := by sorry
  have h1b : Q2 := by sorry
  exact combine h1a h1b""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Second child decomposed
        child2_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma h2 : R",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h2 : R := by
  have h2a : R1 := by sorry
  exact h2a""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Grandchildren for child1
        grandchild1a = FormalTheoremProofState(
            parent=cast(TreeNode, child1_decomposed),
            depth=2,
            formal_theorem="lemma h1a : Q1",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h1a : Q1 := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        grandchild1b = FormalTheoremProofState(
            parent=cast(TreeNode, child1_decomposed),
            depth=2,
            formal_theorem="lemma h1b : Q2",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h1b : Q2 := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        # Grandchild for child2
        grandchild2a = FormalTheoremProofState(
            parent=cast(TreeNode, child2_decomposed),
            depth=2,
            formal_theorem="lemma h2a : R1",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h2a : R1 := by\n  rfl",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        # Build tree
        child1_decomposed["children"].extend([cast(TreeNode, grandchild1a), cast(TreeNode, grandchild1b)])
        child2_decomposed["children"].append(cast(TreeNode, grandchild2a))
        root["children"].extend([cast(TreeNode, child1_decomposed), cast(TreeNode, child2_decomposed)])
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "have h1 : Q := by" in result
        assert "have h2 : R := by" in result
        assert "have h1a : Q1 := by" in result
        assert "have h1b : Q2 := by" in result
        assert "have h2a : R1 := by" in result
        assert "constructor" in result
        assert "trivial" in result
        assert "rfl" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_edge_case_empty_children() -> None:
    """Test reconstruct_complete_proof with DecomposedFormalTheoremState that has no children."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_empty_children_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Decomposed state with sketch but no children
        decomposed = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  sorry""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert theorem_sig in result
        # Should contain the sketch as-is since no children to replace
        assert "sorry" in result

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_edge_case_missing_proof() -> None:
    """Test reconstruct_complete_proof when a child FormalTheoremProofState has no formal_proof."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_missing_proof_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  have h : Q := by sorry
  exact h"""

        decomposed = DecomposedFormalTheoremState(
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

        # Child with no proof
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=None,  # Missing proof
            proved=False,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        # Should fall back to sorry when proof is missing
        assert "sorry" in result

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_edge_case_nested_missing_proof() -> None:
    """Test reconstruct_complete_proof with nested decomposition where inner child has no proof."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_nested_missing_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=f"""{theorem_sig} := by
  have h1 : Q := by sorry
  exact h1""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        child_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma h1 : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h1 : Q := by
  have h2 : R := by sorry
  exact h2""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Grandchild with no proof
        grandchild = FormalTheoremProofState(
            parent=cast(TreeNode, child_decomposed),
            depth=2,
            formal_theorem="lemma h2 : R",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=None,
            proved=False,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        child_decomposed["children"].append(cast(TreeNode, grandchild))
        root["children"].append(cast(TreeNode, child_decomposed))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        # Should fall back to sorry for missing proof
        assert "sorry" in result

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_edge_case_no_sketch() -> None:
    """Test reconstruct_complete_proof when DecomposedFormalTheoremState has no proof_sketch."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_no_sketch_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Decomposed state with no sketch
        decomposed = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=None,  # No sketch
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        # Should fall back to sorry
        assert "sorry" in result

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_edge_case_very_deep_nesting() -> None:
    """Test reconstruct_complete_proof with very deep nesting (5+ levels)."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_very_deep_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Build 5 levels deep
        levels = []
        for i in range(5):
            parent = levels[-1] if levels else None
            level = DecomposedFormalTheoremState(
                parent=cast(TreeNode, parent) if parent else None,
                children=[],
                depth=i,
                formal_theorem=f"lemma level{i} : Type{i}" if i > 0 else theorem,
                preamble=DEFAULT_IMPORTS,
                proof_sketch=f"""{"lemma " if i > 0 else ""}{theorem_sig if i == 0 else f"level{i}"} := by
  have level{i + 1} : Type{i + 1} := by sorry
  exact level{i + 1}"""
                if i < 4
                else f"lemma level{i} : Type{i} := by\n  sorry",
                syntactic=True,
                errors=None,
                ast=None,
                self_correction_attempts=1,
                decomposition_history=[],
            )
            levels.append(level)
            if parent:
                parent["children"].append(cast(TreeNode, level))

        # Add leaf
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, levels[-1]),
            depth=5,
            formal_theorem="lemma level5 : Type5",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma level5 : Type5 := by\n  rfl",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        levels[-1]["children"].append(cast(TreeNode, leaf))

        state.formal_theorem_proof = cast(TreeNode, levels[0])
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        # Check all levels are present (levels 1-4 are have statements, level 5 is the leaf proof)
        for i in range(4):
            assert f"have level{i + 1}" in result
        # Level 5 is a leaf node, so its proof (rfl) should be inlined into level 4's sorry
        assert "rfl" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
