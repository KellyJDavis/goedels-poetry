"""Tests for goedels_poetry.state module."""

# ruff: noqa: RUF001

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState


def with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def _annotate_hole_offsets(  # noqa: C901
    node: dict,
    sketch: str,
    *,
    hole_name: str,
    anchor: str | None = None,
    occurrence: int = 0,
) -> None:
    """
    Attach AST-style hole metadata (hole_name/hole_start/hole_end) to a proof-tree node for tests.

    These tests construct proof trees manually (without a Kimina AST), so we compute the `sorry`
    span by simple string offsets. Production code computes these from the AST.
    """

    def _find_sorry_token(text: str, start: int) -> int:
        """
        Find the next standalone `sorry` token at/after `start`.

        This intentionally skips occurrences inside comments/strings (e.g. `-- ... 'sorry' ...`)
        by requiring whitespace boundaries around the word.
        """
        i = start
        while True:
            i = text.find("sorry", i)
            if i == -1:
                raise ValueError("No standalone `sorry` token found")  # noqa: TRY003
            before = text[i - 1] if i > 0 else " "
            after = text[i + len("sorry")] if i + len("sorry") < len(text) else " "
            if before.isspace() and after.isspace():
                return i
            i += len("sorry")

    def _nth_sorry_token(text: str, start: int, n: int) -> int:
        i = start
        for _ in range(n + 1):
            i = _find_sorry_token(text, i)
            i += len("sorry")
        return i - len("sorry")

    if anchor is None and hole_name == "<main body>":
        # In sketches, the main-body `sorry` is typically the last `sorry` token in the body.
        # Use the last standalone token, not a mention inside a comment.
        positions: list[int] = []
        cursor = 0
        while True:
            try:
                pos = _find_sorry_token(sketch, cursor)
            except ValueError:
                break
            positions.append(pos)
            cursor = pos + len("sorry")
        if not positions:
            raise ValueError("No standalone `sorry` token found for <main body>")  # noqa: TRY003
        start = positions[-1]
    else:
        base = 0 if anchor is None else sketch.index(anchor)
        start = _nth_sorry_token(sketch, base, occurrence)
    end = start + len("sorry")

    node["hole_name"] = hole_name
    node["hole_start"] = start
    node["hole_end"] = end


def test_normalize_theorem() -> None:
    """Test theorem normalization."""
    # Test basic normalization
    assert GoedelsPoetryState._normalize_theorem("  Hello World  ") == "hello world"
    assert GoedelsPoetryState._normalize_theorem("Test Theorem") == "test theorem"
    assert GoedelsPoetryState._normalize_theorem("UPPERCASE") == "uppercase"
    assert GoedelsPoetryState._normalize_theorem("  \n\t  Mixed   Whitespace  \n  ") == "mixed   whitespace"


def test_hash_theorem() -> None:
    """Test theorem hashing."""
    # Same theorems should produce same hash
    hash1 = GoedelsPoetryState._hash_theorem("Test Theorem")
    hash2 = GoedelsPoetryState._hash_theorem("test theorem")  # Different case
    hash3 = GoedelsPoetryState._hash_theorem("  Test Theorem  ")  # Different whitespace
    assert hash1 == hash2 == hash3

    # Different theorems should produce different hashes
    hash4 = GoedelsPoetryState._hash_theorem("Different Theorem")
    assert hash1 != hash4

    # Hash should be 12 characters long
    assert len(hash1) == 12


def test_reconstruct_includes_root_signature_no_decomposition() -> None:
    """Ensure the final proof contains the root theorem signature when no decomposition occurs."""
    import uuid

    from goedels_poetry.agents.state import FormalTheoremProofState
    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem_sig = f"theorem includes_root_sig_nodecomp_{uuid.uuid4().hex} : True"
    theorem = with_default_preamble(f"{theorem_sig} := by sorry")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Store only tactics (as produced by the prover normally)
        leaf = FormalTheoremProofState(
            parent=None,
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
            hole_name=None,
            hole_start=None,
            hole_end=None,
        )
        state.formal_theorem_proof = leaf
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()
        assert result.startswith(DEFAULT_IMPORTS)
        assert theorem_sig in result
        assert ":= by" in result
        assert "trivial" in result
        # Root header must not duplicate any trailing 'sorry' from the original input
        assert ":= by sorry :=" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_includes_root_signature_shallow_decomposition() -> None:
    """Ensure the final proof contains the root theorem signature with shallow decomposition."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem includes_root_sig_shallow_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        parent: DecomposedFormalTheoremState = {
            "parent": None,
            "children": [],
            "depth": 0,
            "formal_theorem": theorem,
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": f"{theorem_sig} := by\n  have h : P := by sorry\n  exact h",
            "syntactic": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "decomposition_history": [],
        }

        child: FormalTheoremProofState = {
            "parent": cast(TreeNode, parent),
            "depth": 1,
            "formal_theorem": "have h : P := by sorry",
            "preamble": DEFAULT_IMPORTS,
            "syntactic": True,
            "formal_proof": "apply id; exact (by trivial)",
            "proved": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "proof_history": [],
            "pass_attempts": 0,
        }
        _annotate_hole_offsets(child, cast(str, parent["proof_sketch"]), hole_name="h", anchor="have h")

        parent["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, parent)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()
        assert result.startswith(DEFAULT_IMPORTS)
        assert theorem_sig in result
        assert ":= by" in result
        assert "apply id" in result
        assert ":= by sorry :=" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_includes_root_signature_deep_decomposition() -> None:
    """Ensure the final proof contains the root theorem signature with deep (nested) decomposition."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem includes_root_sig_deep_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        root: DecomposedFormalTheoremState = {
            "parent": None,
            "children": [],
            "depth": 0,
            "formal_theorem": theorem,
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": f"{theorem_sig} := by\n  have h1 : P := by sorry\n  exact h1",
            "syntactic": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "decomposition_history": [],
        }

        mid: DecomposedFormalTheoremState = {
            "parent": cast(TreeNode, root),
            "children": [],
            "depth": 1,
            "formal_theorem": "have h1 : P := by sorry",
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": "have h1 : P := by\n  have h2 : P := by sorry\n  exact h2",
            "syntactic": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "decomposition_history": [],
        }
        _annotate_hole_offsets(mid, cast(str, root["proof_sketch"]), hole_name="h1", anchor="have h1")

        leaf: FormalTheoremProofState = {
            "parent": cast(TreeNode, mid),
            "depth": 2,
            "formal_theorem": "have h2 : P := by sorry",
            "preamble": DEFAULT_IMPORTS,
            "syntactic": True,
            "formal_proof": "trivial",
            "proved": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "proof_history": [],
            "pass_attempts": 0,
        }
        _annotate_hole_offsets(leaf, cast(str, mid["proof_sketch"]), hole_name="h2", anchor="have h2")

        mid["children"].append(cast(TreeNode, leaf))
        root["children"].append(cast(TreeNode, mid))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()
        assert result.startswith(DEFAULT_IMPORTS)
        assert theorem_sig in result
        assert ":= by" in result
        assert "trivial" in result
        assert ":= by sorry :=" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)

    # End deep decomposition test


def test_list_checkpoints_neither_parameter() -> None:
    """Test list_checkpoints raises error when neither parameter is provided."""
    with pytest.raises(ValueError, match="Must specify either directory or theorem parameter"):
        GoedelsPoetryState.list_checkpoints()


def test_list_checkpoints_both_parameters() -> None:
    """Test list_checkpoints raises error when both parameters are provided."""
    with pytest.raises(ValueError, match="Cannot specify both directory and theorem parameters"):
        GoedelsPoetryState.list_checkpoints(directory="/nonexistent/test", theorem="Test")


def test_list_checkpoints_nonexistent_directory() -> None:
    """Test list_checkpoints returns empty list for nonexistent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = os.path.join(tmpdir, "nonexistent")
        checkpoints = GoedelsPoetryState.list_checkpoints(directory=nonexistent)
        assert checkpoints == []


def test_list_checkpoints_by_directory() -> None:
    """Test list_checkpoints lists checkpoints in a directory."""
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some checkpoint files
        checkpoint_files = [
            "goedels_poetry_state_20250101_120000_iter_0000.pkl",
            "goedels_poetry_state_20250101_130000_iter_0001.pkl",
            "goedels_poetry_state_20250101_140000_iter_0002.pkl",
        ]

        for i, filename in enumerate(checkpoint_files):
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write("dummy")
            # Set explicit modification times to ensure proper ordering
            # Add 1 second for each subsequent file
            mtime = time.time() - (len(checkpoint_files) - i - 1)
            os.utime(filepath, (mtime, mtime))

        # Create a non-checkpoint file that should be ignored
        with open(os.path.join(tmpdir, "other_file.txt"), "w") as f:
            f.write("dummy")

        checkpoints = GoedelsPoetryState.list_checkpoints(directory=tmpdir)

        # Should return all checkpoint files
        assert len(checkpoints) == 3

        # Should be sorted by modification time (newest first)
        # iter_0002.pkl was given the most recent modification time
        assert checkpoints[0].endswith("iter_0002.pkl")


def test_list_checkpoints_by_theorem() -> None:
    """Test list_checkpoints lists checkpoints for a theorem using the default directory."""
    import uuid

    theorem = with_default_preamble(f"theorem test_checkpoints_{uuid.uuid4().hex} : True := by sorry")

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    theorem_hash = GoedelsPoetryState._hash_theorem(theorem)

    # Get the default output directory
    from goedels_poetry import state as state_module

    output_dir = state_module._OUTPUT_DIR
    theorem_dir = os.path.join(output_dir, theorem_hash)

    try:
        Path(theorem_dir).mkdir(parents=True, exist_ok=True)

        # Create theorem.txt file
        with open(os.path.join(theorem_dir, "theorem.txt"), "w") as f:
            f.write(theorem)

        # Create checkpoint files
        for i in range(3):
            filename = f"goedels_poetry_state_2025010{i}_120000_iter_000{i}.pkl"
            with open(os.path.join(theorem_dir, filename), "w") as f:
                f.write("dummy")

        checkpoints = GoedelsPoetryState.list_checkpoints(theorem=theorem)
        assert len(checkpoints) == 3
    finally:
        # Clean up
        GoedelsPoetryState.clear_theorem_directory(theorem)


def test_clear_theorem_directory() -> None:
    """Test clearing a theorem directory."""
    import uuid

    theorem = with_default_preamble(f"theorem test_clear_{uuid.uuid4().hex} : True := by sorry")

    theorem_hash = GoedelsPoetryState._hash_theorem(theorem)
    from goedels_poetry import state as state_module

    output_dir = state_module._OUTPUT_DIR
    theorem_dir = os.path.join(output_dir, theorem_hash)

    try:
        Path(theorem_dir).mkdir(parents=True, exist_ok=True)

        # Create some files
        with open(os.path.join(theorem_dir, "test.txt"), "w") as f:
            f.write("test")

        # Directory should exist
        assert os.path.exists(theorem_dir)

        # Clear it
        result = GoedelsPoetryState.clear_theorem_directory(theorem)
        assert "Successfully cleared directory" in result
        assert theorem_dir in result

        # Directory should not exist
        assert not os.path.exists(theorem_dir)

        # Clearing again should indicate it doesn't exist
        result = GoedelsPoetryState.clear_theorem_directory(theorem)
        assert "Directory does not exist" in result
    finally:
        # Extra cleanup just in case
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_save_and_load() -> None:
    """Test saving and loading state."""
    import uuid

    theorem = with_default_preamble(f"theorem test_save_load_{uuid.uuid4().hex} : True := by sorry")

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        # Create a state
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Modify some state
        state.is_finished = True
        state.action_history = ["action1", "action2"]

        # Save it
        saved_path = state.save()
        assert os.path.exists(saved_path)
        assert "goedels_poetry_state_" in saved_path
        assert saved_path.endswith(".pkl")

        # Load it
        loaded_state = GoedelsPoetryState.load(saved_path)
        assert loaded_state.is_finished is True
        assert loaded_state.action_history == ["action1", "action2"]
    finally:
        # Clean up
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_load_latest() -> None:
    """Test loading the latest checkpoint."""
    import time
    import uuid

    theorem = with_default_preamble(f"theorem test_load_latest_{uuid.uuid4().hex} : True := by sorry")

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Save multiple times with different state
        # Add small delays to ensure different timestamps in fast CI environments
        state.action_history = ["first"]
        state.save()
        time.sleep(0.01)  # 10ms delay

        state.action_history = ["first", "second"]
        state.save()
        time.sleep(0.01)  # 10ms delay

        state.action_history = ["first", "second", "third"]
        state.save()

        # Load latest by theorem
        loaded = GoedelsPoetryState.load_latest(theorem=theorem)
        assert loaded is not None
        assert loaded.action_history == ["first", "second", "third"]

        # Load latest by directory
        theorem_hash = GoedelsPoetryState._hash_theorem(theorem)
        from goedels_poetry import state as state_module

        output_dir = state_module._OUTPUT_DIR
        theorem_dir = os.path.join(output_dir, theorem_hash)
        loaded2 = GoedelsPoetryState.load_latest(directory=theorem_dir)
        assert loaded2 is not None
        assert loaded2.action_history == ["first", "second", "third"]
    finally:
        # Clean up
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_load_latest_no_checkpoints() -> None:
    """Test load_latest returns None when no checkpoints exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = GoedelsPoetryState.load_latest(directory=tmpdir)
        assert loaded is None


def test_state_init_requires_one_argument() -> None:
    """Test that state initialization requires exactly one of formal_theorem or informal_theorem."""
    # Neither provided
    with pytest.raises(ValueError, match="Either 'formal_theorem' xor 'informal_theorem' must be provided"):
        GoedelsPoetryState()

    # Both provided
    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir
            with pytest.raises(ValueError, match="Only one of 'formal_theorem' or 'informal_theorem' can be provided"):
                GoedelsPoetryState(formal_theorem="test", informal_theorem="test")
        finally:
            if old_env is not None:
                os.environ["GOEDELS_POETRY_DIR"] = old_env
            elif "GOEDELS_POETRY_DIR" in os.environ:
                del os.environ["GOEDELS_POETRY_DIR"]


def test_state_init_creates_directory() -> None:
    """Test that state initialization creates output directory."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir
            theorem = with_default_preamble("theorem test_directory_creation : True := by sorry")
            state = GoedelsPoetryState(formal_theorem=theorem)

            # Directory should exist
            assert os.path.exists(state._output_dir)

            # theorem.txt should exist
            theorem_file = os.path.join(state._output_dir, "theorem.txt")
            assert os.path.exists(theorem_file)

            with open(theorem_file) as f:
                content = f.read()
            assert content == theorem

            # Clean up
            GoedelsPoetryState.clear_theorem_directory(theorem)
        finally:
            if old_env is not None:
                os.environ["GOEDELS_POETRY_DIR"] = old_env
            elif "GOEDELS_POETRY_DIR" in os.environ:
                del os.environ["GOEDELS_POETRY_DIR"]


def test_state_init_with_informal_theorem() -> None:
    """Test state initialization with informal theorem."""
    import uuid

    theorem = with_default_preamble(f"theorem test_formalization_{uuid.uuid4().hex} : True := by sorry")

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(informal_theorem=theorem)

        # Should have informal_formalizer_queue set
        assert state.informal_formalizer_queue is not None
        assert state.informal_formalizer_queue["informal_theorem"] == theorem

        # Should not have formal_theorem_proof set
        assert state.formal_theorem_proof is None
    finally:
        # Clean up
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_state_init_directory_exists_error() -> None:
    """Test that state initialization fails if directory already exists."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir
            theorem = with_default_preamble("theorem test_duplicate_directory : True := by sorry")

            # Create first state
            GoedelsPoetryState(formal_theorem=theorem)

            # Try to create second state with same theorem (should fail)
            with pytest.raises(FileExistsError, match="Directory for theorem already exists"):
                GoedelsPoetryState(formal_theorem=theorem)

            # Clean up
            GoedelsPoetryState.clear_theorem_directory(theorem)
        finally:
            if old_env is not None:
                os.environ["GOEDELS_POETRY_DIR"] = old_env
            elif "GOEDELS_POETRY_DIR" in os.environ:
                del os.environ["GOEDELS_POETRY_DIR"]


def test_save_increments_iteration() -> None:
    """Test that save increments the iteration counter."""
    import uuid

    theorem = with_default_preamble(f"theorem test_iteration_counter_{uuid.uuid4().hex} : True := by sorry")

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Initial iteration is 0
        assert state._iteration == 0

        # First save
        path1 = state.save()
        assert state._iteration == 1
        assert "iter_0000.pkl" in path1

        # Second save
        path2 = state.save()
        assert state._iteration == 2
        assert "iter_0001.pkl" in path2

        # Third save
        path3 = state.save()
        assert state._iteration == 3
        assert "iter_0002.pkl" in path3
    finally:
        # Clean up
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


# Tests for GoedelsPoetryStateManager


def test_state_manager_reason_property() -> None:
    """Test the reason property getter and setter."""
    import uuid

    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_reason_property_{uuid.uuid4().hex} : True := by sorry")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        # Initial reason should be None
        assert manager.reason is None

        # Set a reason
        manager.reason = "Test reason"
        assert manager.reason == "Test reason"

        # Update reason
        manager.reason = "Updated reason"
        assert manager.reason == "Updated reason"

        # Set to None
        manager.reason = None
        assert manager.reason is None
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_no_proof() -> None:
    """Test reconstruct_complete_proof when no proof exists."""
    import uuid

    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_no_proof_{uuid.uuid4().hex} : True := by sorry")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(informal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        # No proof tree exists
        result = manager.reconstruct_complete_proof()
        assert DEFAULT_IMPORTS in result
        assert "No proof available" in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_simple_leaf() -> None:
    """Test reconstruct_complete_proof with a simple FormalTheoremProofState."""
    import uuid

    from goedels_poetry.agents.state import FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_simple_leaf_{uuid.uuid4().hex} : True := by sorry")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a simple proof
        proof_state = FormalTheoremProofState(
            parent=None,
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=f"{theorem} := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )

        state.formal_theorem_proof = proof_state
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain the proof
        assert theorem in result
        assert ":= by" in result
        assert "trivial" in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_single_have() -> None:
    """Test reconstruct_complete_proof with a decomposed state containing one have statement."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_single_have_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with sketch
        sketch = f"""{theorem} := by
  have helper : Q := by sorry
  exact helper"""

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
            proof_history=[],
        )

        # Create child proof
        child_proof = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper : Q := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child_proof, sketch, hole_name="helper", anchor="have helper")

        decomposed["children"].append(cast(TreeNode, child_proof))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain main theorem
        assert theorem in result
        assert ":= by" in result

        # Should contain have with inline proof (not sorry)
        assert "have helper : Q := by" in result
        assert "constructor" in result

        # Should NOT contain sorry for helper
        lines = result.split("\n")
        have_line_idx = None
        for i, line in enumerate(lines):
            if "have helper" in line:
                have_line_idx = i
                break

        assert have_line_idx is not None
        # Check lines after have for sorry - should not find it before next statement
        for i in range(have_line_idx, min(have_line_idx + 5, len(lines))):
            if "exact helper" in lines[i]:
                break
            if i > have_line_idx and "sorry" in lines[i]:
                pytest.fail("Found sorry in have helper proof when it should be replaced")
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_multiple_haves() -> None:
    """Test reconstruct_complete_proof with multiple have statements."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_multi_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with multiple haves
        sketch = f"""{theorem} := by
  have helper1 : Q := by sorry
  have helper2 : R := by sorry
  exact combine helper1 helper2"""

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
            proof_history=[],
        )

        # Create first child proof
        child1 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper1 : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper1 : Q := by\n  intro x\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child1, sketch, hole_name="helper1", anchor="have helper1")

        # Create second child proof (with dependency)
        child2 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper2 (helper1 : Q) : R",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper2 (helper1 : Q) : R := by\n  cases helper1\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child2, sketch, hole_name="helper2", anchor="have helper2")

        decomposed["children"].extend([cast(TreeNode, child1), cast(TreeNode, child2)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain both haves with inline proofs
        assert "have helper1 : Q := by" in result
        assert "intro x" in result
        assert "have helper2 : R := by" in result
        assert "cases helper1" in result

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_handles_unicode_have_names() -> None:
    """Ensure have statements with unicode subscripts are inlined correctly."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_unicode_have_{uuid.uuid4()} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  have h₁ : P := by sorry
  have h₂ : P := by sorry
  exact h₂"""

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
            proof_history=[],
        )

        child1 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h₁ : P := by sorry",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h₁ : P := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child1, sketch, hole_name="h₁", anchor="have h₁")

        child2 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h₂ : P := by sorry",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h₂ : P := by\n  exact h₁",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child2, sketch, hole_name="h₂", anchor="have h₂")

        decomposed["children"].extend([cast(TreeNode, child1), cast(TreeNode, child2)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert "have h₁ : P := by" in result
        assert "have h₂ : P := by" in result
        assert "trivial" in result
        assert "exact h₁" in result
        assert "have h₁ : P := by sorry" not in result
        assert "have h₂ : P := by sorry" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_main_body() -> None:
    """Test reconstruct_complete_proof with main body proof replacement."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_main_body_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with have and main body sorry
        sketch = f"""{theorem} := by
  have helper : Q := by sorry
  sorry"""

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
            proof_history=[],
        )

        # Create have proof
        child_have = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper : Q := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child_have, sketch, hole_name="helper", anchor="have helper")

        # Create main body proof (no clear name, so it's the main body)
        child_main = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="theorem main_body : P",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="theorem main_body : P := by\n  apply helper\n  done",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child_main, sketch, hole_name="<main body>", anchor=None)

        decomposed["children"].extend([cast(TreeNode, child_have), cast(TreeNode, child_main)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain have with inline proof
        assert "have helper : Q := by" in result
        assert "constructor" in result

        # Should contain main body proof (not standalone sorry)
        assert "apply helper" in result
        assert "done" in result

        # Should NOT contain standalone sorry
        lines = result.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "sorry" and i > 0:
                # Check this isn't part of a have statement
                prev_lines = "\n".join(lines[max(0, i - 3) : i])
                if ":= by" not in prev_lines:
                    pytest.fail(f"Found standalone sorry at line {i}")
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_anonymous_have() -> None:
    """Reconstruction should inline proofs into anonymous `have : ... := by sorry` holes."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    decl = f"test_anon_have_{uuid.uuid4().hex}"
    theorem_sig = f"theorem {decl} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  have : False := by
    sorry
  exact this.elim"""

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
            proof_history=[],
        )

        # Child corresponds to the first anonymous have inside this theorem.
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem=f"lemma gp_anon_have__{decl}__1 : False := by sorry",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=f"lemma gp_anon_have__{decl}__1 : False := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child, sketch, hole_name=f"gp_anon_have__{decl}__1", anchor="have : False")

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert "have : False := by" in result
        assert "trivial" in result
        assert "exact this.elim" in result
        assert "sorry" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_proper_indentation() -> None:
    """Test that proof reconstruction maintains proper indentation."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_indent_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with indented have
        sketch = f"""{theorem} := by
  have helper : Q := by sorry
  exact helper"""

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
            proof_history=[],
        )

        # Create child with multi-line proof
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper : Q := by\n  intro x\n  cases x\n  constructor",
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

        lines = result.split("\n")

        # Find the have line
        have_line_idx = None
        for i, line in enumerate(lines):
            if "have helper" in line:
                have_line_idx = i
                # have should be indented with 2 spaces
                assert line.startswith("  have"), f"have line not properly indented: '{line}'"
                break

        assert have_line_idx is not None

        # Check that proof body lines are indented with 4 spaces (2 more than have)
        for i in range(have_line_idx + 1, min(have_line_idx + 5, len(lines))):
            line = lines[i]
            if line.strip() and "exact" not in line:
                # This should be part of the have proof, indented with 4 spaces
                assert line.startswith("    "), f"Proof body line not properly indented: '{line}'"
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_calc_with_comments_and_indented_child_proof() -> None:
    """
    Regression test for partial.log-style failures:

    - Parent sketch has comments between `:=` and `by`, and between `by` and `sorry`
    - Child proof is already indented and contains nested `have` + `calc`
    - Reconstruction must dedent+reindent safely (no layout break) and must not strip `have` binders
    """
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_reconstruct_calc_comments_{uuid.uuid4()} : (1 : ℕ) = 1")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = (
            f"{theorem} := by\n"
            "  have hv_subst : (1 : ℕ) = 1 :=\n"
            "    -- comment between ':=' and 'by'\n"
            "    by\n"
            "      -- comment between 'by' and 'sorry'\n"
            "      sorry\n"
            "  exact hv_subst\n"
        )

        decomposed: DecomposedFormalTheoremState = {
            "parent": None,
            "children": [],
            "depth": 0,
            "formal_theorem": theorem,
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": sketch,
            "syntactic": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "decomposition_history": [],
        }

        # Child proof is a tactic script (starts with `have`, not a lemma/theorem decl),
        # and is already indented to simulate prover output copied from a nested block.
        child_formal_proof = (
            "      have step₁ : (1 : ℕ) = 1 := by\n"
            "        rfl\n"
            "      have step₂ : (1 : ℕ) = 1 := by\n"
            "        calc\n"
            "          (1 : ℕ) = 1 := by rfl\n"
            "      exact step₂"
        )

        child: FormalTheoremProofState = {
            "parent": cast(TreeNode, decomposed),
            "depth": 1,
            "formal_theorem": "lemma hv_subst : (1 : ℕ) = 1",
            "preamble": DEFAULT_IMPORTS,
            "syntactic": True,
            "formal_proof": child_formal_proof,
            "proved": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "proof_history": [],
            "pass_attempts": 0,
        }
        _annotate_hole_offsets(child, sketch, hole_name="hv_subst", anchor="have hv_subst")

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)

        manager = GoedelsPoetryStateManager(state)
        result = manager.reconstruct_complete_proof()

        # `hv_subst` sorry should be replaced even with intervening comments/newlines.
        assert "have hv_subst : (1 : ℕ) = 1 :=" in result
        # Allow the word "sorry" to appear in comments, but not as an actual tactic placeholder.
        for line in result.split("\n"):
            assert line.strip() != "sorry"

        # Child proof must keep its `have` binders (no dangling `exact step₂` / missing `step₁`).
        assert "have step₁ : (1 : ℕ) = 1 := by" in result
        assert "have step₂ : (1 : ℕ) = 1 := by" in result
        assert "exact step₂" in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_normalizes_misindented_trailing_apply() -> None:
    """
    Regression test for partial.log-style inlined child proof bodies:

    - Child proof defines `have h_main : goal := by ...`
    - Child proof ends with a *misindented* `apply h_main` (common LLM failure)
    - Reconstruction should normalize indentation and turn the trailing `apply h_main` into
      `exact h_main` so the parent goal closes.
    """
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_reconstruct_apply_normalize_{uuid.uuid4().hex} : True"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        sketch = f"""{theorem_sig} := by
  have hv_subst : True := by sorry
  exact hv_subst"""

        parent: DecomposedFormalTheoremState = {
            "parent": None,
            "children": [],
            "depth": 0,
            "formal_theorem": theorem,
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": sketch,
            "syntactic": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "decomposition_history": [],
        }

        # Note the misindentation:
        # - `have h_main` is at indent 0
        # - inner proof is at indent 4
        # - trailing `apply h_main` is at indent 2 (invalid intermediate dedent level)
        # This mirrors the pattern in partial.log that causes "expected command".
        child_formal_proof = "have h_main : True := by\n    trivial\n  apply h_main\n"

        child: FormalTheoremProofState = {
            "parent": cast(TreeNode, parent),
            "depth": 1,
            "formal_theorem": "lemma hv_subst : True",
            "preamble": DEFAULT_IMPORTS,
            "syntactic": True,
            "formal_proof": child_formal_proof,
            "proved": True,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 1,
            "proof_history": [],
            "pass_attempts": 0,
        }
        _annotate_hole_offsets(child, sketch, hole_name="hv_subst", anchor="have hv_subst")

        parent["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, parent)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # The inlined proof should end with `exact h_main`, not `apply h_main`.
        assert "exact h_main" in result
        assert "apply h_main" not in result
        # And the hv_subst hole should no longer contain sorry.
        assert "have hv_subst : True := by sorry" not in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_nested_decomposition() -> None:
    """Test reconstruct_complete_proof with nested decomposed states."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_nested_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create parent decomposed state
        parent_sketch = f"""{theorem} := by
  have helper1 : Q := by sorry
  exact helper1"""

        parent = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=parent_sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
        )

        # Create child decomposed state (helper1 is also decomposed)
        child_sketch = """lemma helper1 : Q := by
  have subhelper : R := by sorry
  exact subhelper"""

        child_decomposed = DecomposedFormalTheoremState(
            parent=cast(TreeNode, parent),
            children=[],
            depth=1,
            formal_theorem="lemma helper1 : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=child_sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
        )
        _annotate_hole_offsets(child_decomposed, parent_sketch, hole_name="helper1", anchor="have helper1")

        # Create grandchild proof
        grandchild = FormalTheoremProofState(
            parent=cast(TreeNode, child_decomposed),
            depth=2,
            formal_theorem="lemma subhelper : R",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma subhelper : R := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(grandchild, child_sketch, hole_name="subhelper", anchor="have subhelper")

        child_decomposed["children"].append(cast(TreeNode, grandchild))
        parent["children"].append(cast(TreeNode, child_decomposed))
        state.formal_theorem_proof = cast(TreeNode, parent)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain main theorem
        assert theorem in result
        assert ":= by" in result

        # Should contain nested have statements
        assert "have helper1 : Q := by" in result
        assert "have subhelper : R := by" in result

        # Should contain the deepest proof
        assert "constructor" in result

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_with_dependencies_in_signature() -> None:
    """Test that reconstruction works when child has dependencies added to signature."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_deps_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state
        sketch = f"""{theorem} := by
  have cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {{0, 1, 8}} := by sorry
  have sum_not_3 : ∀ (s1 s2 : ℤ), s1 ∈ {{0, 1, 8}} → s2 ∈ {{0, 1, 8}} → (s1 + s2) % 9 ≠ 3 := by sorry
  sorry"""

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
            proof_history=[],
        )

        # Create first child
        child1 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {0, 1, 8}",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {0, 1, 8} := by\n  intro a\n  omega",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child1, sketch, hole_name="cube_mod9", anchor="have cube_mod9")

        # Create second child WITH DEPENDENCY in signature (as AST.get_named_subgoal_code does)
        child2 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma sum_not_3 (cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {0, 1, 8}) : ∀ (s1 s2 : ℤ), s1 ∈ {0, 1, 8} → s2 ∈ {0, 1, 8} → (s1 + s2) % 9 ≠ 3",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma sum_not_3 (cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {0, 1, 8}) : ∀ (s1 s2 : ℤ), s1 ∈ {0, 1, 8} → s2 ∈ {0, 1, 8} → (s1 + s2) % 9 ≠ 3 := by\n  intro s1 s2\n  omega",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child2, sketch, hole_name="sum_not_3", anchor="have sum_not_3")

        # Create main body
        child3 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="theorem main_body (cube_mod9 : ∀ (a : ℤ), (a^3) % 9 ∈ {0, 1, 8}) (sum_not_3 : ∀ (s1 s2 : ℤ), s1 ∈ {0, 1, 8} → s2 ∈ {0, 1, 8} → (s1 + s2) % 9 ≠ 3) : P",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="theorem main_body (cube_mod9 : ...) (sum_not_3 : ...) : P := by\n  apply sum_not_3\n  omega",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child3, sketch, hole_name="<main body>", anchor=None)

        decomposed["children"].extend([cast(TreeNode, child1), cast(TreeNode, child2), cast(TreeNode, child3)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should properly match cube_mod9 by name only
        assert "have cube_mod9" in result
        assert "intro a" in result

        # Should properly match sum_not_3 by name only (despite dependency in child signature)
        assert "have sum_not_3" in result
        assert "intro s1 s2" in result

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_empty_proof() -> None:
    """Test reconstruct_complete_proof when formal_proof is None."""
    import uuid

    from goedels_poetry.agents.state import FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_empty_{uuid.uuid4()} : True")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a proof state without formal_proof
        proof_state = FormalTheoremProofState(
            parent=None,
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=None,  # No proof yet
            proved=False,
            errors=None,
            ast=None,
            self_correction_attempts=0,
            proof_history=[],
            pass_attempts=0,
        )

        state.formal_theorem_proof = proof_state
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain theorem with sorry fallback
        assert theorem in result
        assert ":= by sorry" in result
    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_whitespace_robustness() -> None:
    """Test that reconstruct handles various whitespace variations in ':= by' patterns."""
    import uuid

    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem whitespace_test_{uuid.uuid4()} : True")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create child proofs with various whitespace patterns
        child1_proof = "lemma h1 : 1 = 1 :=  by rfl"  # Two spaces
        child2_proof = "lemma h2 : 2 = 2 :=by rfl"  # No space

        # Test _extract_tactics_after_by with various patterns
        manager = GoedelsPoetryStateManager(state)

        # Test with two spaces
        tactics1 = manager._extract_tactics_after_by(child1_proof)
        assert tactics1 == "rfl", f"Expected 'rfl', got '{tactics1}'"

        # Test with no space
        tactics2 = manager._extract_tactics_after_by(child2_proof)
        assert tactics2 == "rfl", f"Expected 'rfl', got '{tactics2}'"

        # Test with newline (unlikely but should handle)
        proof_with_newline = "lemma h3 : 3 = 3 :=\n  by rfl"
        tactics3 = manager._extract_tactics_after_by(proof_with_newline)
        assert tactics3 == "rfl", f"Expected 'rfl', got '{tactics3}'"

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_complete_proof_multiline_type_signatures() -> None:
    """Test that reconstruct handles multiline type signatures."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_multiline_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with multiline type signatures
        sketch = f"""{theorem} := by
  have helper1 :
    VeryLongType →
    AnotherType := by sorry
  have helper2 : SimpleType
    := by sorry
  exact combine helper1 helper2"""

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
            proof_history=[],
        )

        # Create first child proof with multiline type signature
        child1 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="""lemma helper1 :
            preamble=DEFAULT_IMPORTS,
  VeryLongType →
  AnotherType""",
            syntactic=True,
            formal_proof="""lemma helper1 :
  VeryLongType →
  AnotherType := by
  intro x
  constructor""",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child1, sketch, hole_name="helper1", anchor="have helper1")

        # Create second child proof with := on different line
        child2 = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper2 : SimpleType",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="""lemma helper2 : SimpleType
  := by
  constructor""",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child2, sketch, hole_name="helper2", anchor="have helper2")

        decomposed["children"].extend([cast(TreeNode, child1), cast(TreeNode, child2)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain both haves with inline proofs
        assert "have helper1 :" in result
        assert "intro x" in result
        assert "have helper2 : SimpleType" in result

        # Both constructors should be present
        assert result.count("constructor") == 2

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_extract_tactics_after_by_multiline_variations() -> None:
    """Test _extract_tactics_after_by with multiline ':= by' patterns."""
    import uuid

    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_multiline_by_{uuid.uuid4()} : True")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        # Test with := on one line, by on the next
        proof1 = """lemma h1 : VeryLongType →
  AnotherType :=
  by
    intro x
    constructor"""
        tactics1 = manager._extract_tactics_after_by(proof1)
        assert "intro x" in tactics1
        assert "constructor" in tactics1

        # Test with newline between := and by
        proof2 = """lemma h2 : SimpleType :=
by rfl"""
        tactics2 = manager._extract_tactics_after_by(proof2)
        assert tactics2.strip() == "rfl"

        # Test with everything on one line but multiline type
        proof3 = """lemma h3 :
  Type1 →
  Type2 := by exact h"""
        tactics3 = manager._extract_tactics_after_by(proof3)
        assert tactics3 == "exact h"

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_proof_with_apostrophe_identifiers() -> None:
    """Test proof reconstruction with identifiers containing apostrophes."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_apostrophe_proof_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with apostrophe in helper name
        sketch = f"""{theorem} := by
  have helper' : Q := by sorry
  exact helper'"""

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
            proof_history=[],
        )

        # Create child proof with apostrophe in name
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper' : Q",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper' : Q := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child, sketch, hole_name="helper'", anchor="have helper'")

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain have with apostrophe
        assert "have helper' : Q := by" in result
        assert "constructor" in result

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_reconstruct_proof_multiline_have_sorry() -> None:
    """Test complete proof reconstruction with multiline have statements."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem = with_default_preamble(f"theorem test_multiline_recon_{uuid.uuid4()} : P")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Create a decomposed state with multiline have
        sketch = f"""{theorem} := by
  have helper :
    VeryLongType := by
    sorry
  sorry"""

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
            proof_history=[],
        )

        # Create child proof for the have
        child_have = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma helper : VeryLongType",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma helper : VeryLongType := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child_have, sketch, hole_name="helper", anchor="have helper")

        # Create child proof for main body
        child_main = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="theorem main_body : P",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="theorem main_body : P := by\n  exact helper",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child_main, sketch, hole_name="<main body>", anchor=None)

        decomposed["children"].extend([cast(TreeNode, child_have), cast(TreeNode, child_main)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain the have statement
        assert "have helper :" in result
        assert "VeryLongType" in result

        # Should contain both proofs
        assert "constructor" in result
        assert "exact helper" in result

        # Should NOT contain any sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


# ============================================================================
# Comprehensive tests for proof composition with nested decomposition
# ============================================================================


def test_reconstruct_complete_proof_deep_nested_decomposition_3_levels() -> None:
    """Test reconstruct_complete_proof with 3 levels of nested DecomposedFormalTheoremState."""
    import uuid
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = f"theorem test_deep_3_{uuid.uuid4().hex} : P"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Level 0: Root decomposed state
        root_sketch = f"""{theorem_sig} := by
  have h1 : Q := by sorry
  exact h1"""

        root = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=0,
            formal_theorem=theorem,
            preamble=DEFAULT_IMPORTS,
            proof_sketch=root_sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )

        # Level 1: First child decomposed state
        level1_sketch = """lemma h1 : Q := by
  have h2 : R := by sorry
  exact h2"""

        level1 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, root),
            children=[],
            depth=1,
            formal_theorem="lemma h1 : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=level1_sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level1, root_sketch, hole_name="h1", anchor="have h1")

        # Level 2: Second child decomposed state
        level2_sketch = """lemma h2 : R := by
  have h3 : S := by sorry
  exact h3"""

        level2 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level1),
            children=[],
            depth=2,
            formal_theorem="lemma h2 : R",
            preamble=DEFAULT_IMPORTS,
            proof_sketch=level2_sketch,
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level2, level1_sketch, hole_name="h2", anchor="have h2")

        # Level 3: Leaf proof state
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, level2),
            depth=3,
            formal_theorem="lemma h3 : S",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h3 : S := by\n  constructor",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(leaf, level2_sketch, hole_name="h3", anchor="have h3")

        # Build tree
        level2["children"].append(cast(TreeNode, leaf))
        level1["children"].append(cast(TreeNode, level2))
        root["children"].append(cast(TreeNode, level1))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain all nested have statements
        assert "have h1 : Q := by" in result
        assert "have h2 : R := by" in result
        assert "have h3 : S := by" in result

        # Should contain the deepest proof
        assert "constructor" in result

        # Should NOT contain sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports

        # Verify proper nesting structure
        lines = result.split("\n")
        h1_idx = next((i for i, line in enumerate(lines) if "have h1" in line), None)
        h2_idx = next((i for i, line in enumerate(lines) if "have h2" in line), None)
        h3_idx = next((i for i, line in enumerate(lines) if "have h3" in line), None)
        assert h1_idx is not None and h2_idx is not None and h3_idx is not None
        assert h1_idx < h2_idx < h3_idx

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


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
        _annotate_hole_offsets(level1, cast(str, root["proof_sketch"]), hole_name="a", anchor="have a")

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
        _annotate_hole_offsets(level2, cast(str, level1["proof_sketch"]), hole_name="b", anchor="have b")

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
        _annotate_hole_offsets(level3, cast(str, level2["proof_sketch"]), hole_name="c", anchor="have c")

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
        _annotate_hole_offsets(leaf, cast(str, level3["proof_sketch"]), hole_name="d", anchor="have d")

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
        _annotate_hole_offsets(child_decomposed, cast(str, root["proof_sketch"]), hole_name="α₁", anchor="have α₁")

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
        _annotate_hole_offsets(
            grandchild, cast(str, child_decomposed["proof_sketch"]), hole_name="β₂", anchor="have β₂"
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

        # Child proof that depends on the let binding
        child = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma h (n : ℕ) : n > 0",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h (n : ℕ) : n > 0 := by\n  omega",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child, sketch, hole_name="h", anchor="have h")

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let n : ℕ := 5" in result
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
        _annotate_hole_offsets(child, sketch, hole_name="h", anchor="have h")

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
  exact helper""",
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
            formal_theorem="lemma helper (n : ℕ) : n > 5",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma helper (n : ℕ) : n > 5 := by
  let m : ℕ := n + 1
  have h : m > 5 := by sorry
  exact h""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(
            child_decomposed, cast(str, root["proof_sketch"]), hole_name="helper", anchor="have helper"
        )

        # Grandchild proof
        grandchild = FormalTheoremProofState(
            parent=cast(TreeNode, child_decomposed),
            depth=2,
            formal_theorem="lemma h (n : ℕ) (m : ℕ) : m > 5",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h (n : ℕ) (m : ℕ) : m > 5 := by\n  omega",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(grandchild, cast(str, child_decomposed["proof_sketch"]), hole_name="h", anchor="have h")

        child_decomposed["children"].append(cast(TreeNode, grandchild))
        root["children"].append(cast(TreeNode, child_decomposed))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let n : ℕ := 10" in result
        assert "have helper : n > 5 := by" in result
        assert "let m : ℕ := n + 1" in result
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
  exact h1""",
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
            formal_theorem="lemma h1 (x : ℕ) : Q",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h1 (x : ℕ) : Q := by
  obtain ⟨y, hy⟩ : ∃ y, R y := by sorry
  have h2 : S := by sorry
  exact h2""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level1, cast(str, root["proof_sketch"]), hole_name="h1", anchor="have h1")

        # Level 2: With let and have
        level2 = DecomposedFormalTheoremState(
            parent=cast(TreeNode, level1),
            children=[],
            depth=2,
            formal_theorem="lemma h2 (x : ℕ) (y : T) (hy : R y) : S",
            preamble=DEFAULT_IMPORTS,
            proof_sketch="""lemma h2 (x : ℕ) (y : T) (hy : R y) : S := by
  let z : ℕ := x + y
  have h3 : T := by sorry
  exact h3""",
            syntactic=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            decomposition_history=[],
        )
        _annotate_hole_offsets(level2, cast(str, level1["proof_sketch"]), hole_name="h2", anchor="have h2")

        # Level 3: Leaf
        leaf = FormalTheoremProofState(
            parent=cast(TreeNode, level2),
            depth=3,
            formal_theorem="lemma h3 (x : ℕ) (y : T) (hy : R y) (z : ℕ) : T",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma h3 (x : ℕ) (y : T) (hy : R y) (z : ℕ) : T := by\n  trivial",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(leaf, cast(str, level2["proof_sketch"]), hole_name="h3", anchor="have h3")

        level2["children"].append(cast(TreeNode, leaf))
        level1["children"].append(cast(TreeNode, level2))
        root["children"].append(cast(TreeNode, level1))
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let x : ℕ := 5" in result
        assert "have h1 : Q := by" in result
        assert "obtain ⟨y, hy⟩" in result
        assert "have h2 : S := by" in result
        assert "let z : ℕ := x + y" in result
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
  exact γ"""

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
            formal_theorem="lemma γ (α : ℕ) (β : T) (hβ : Q β) : R",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="lemma γ (α : ℕ) (β : T) (hβ : Q β) : R := by\n  exact hβ",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        _annotate_hole_offsets(child, sketch, hole_name="γ", anchor="have γ")

        decomposed["children"].append(cast(TreeNode, child))
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        assert result.startswith(DEFAULT_IMPORTS)
        assert "let α : ℕ := 1" in result
        assert "obtain ⟨β, hβ⟩" in result
        assert "have γ : R := by" in result
        assert "exact hβ" in result
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "have γ : R := by sorry" not in result_no_imports

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
        _annotate_hole_offsets(child1_decomposed, cast(str, root["proof_sketch"]), hole_name="h1", anchor="have h1")

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
        _annotate_hole_offsets(child2_decomposed, cast(str, root["proof_sketch"]), hole_name="h2", anchor="have h2")

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
        _annotate_hole_offsets(
            grandchild1a, cast(str, child1_decomposed["proof_sketch"]), hole_name="h1a", anchor="have h1a"
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
        _annotate_hole_offsets(
            grandchild1b, cast(str, child1_decomposed["proof_sketch"]), hole_name="h1b", anchor="have h1b"
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
        _annotate_hole_offsets(
            grandchild2a, cast(str, child2_decomposed["proof_sketch"]), hole_name="h2a", anchor="have h2a"
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
                _annotate_hole_offsets(
                    level,
                    cast(str, parent["proof_sketch"]),
                    hole_name=f"level{i}",
                    anchor=f"have level{i}",
                )

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
        _annotate_hole_offsets(leaf, cast(str, levels[-1]["proof_sketch"]), hole_name="<main body>", anchor=None)
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


# ============================================================================
# Regression: reconstruction should still succeed for the exact case from partial.log
# (now covered via offset-based hole replacement).
# ============================================================================


def test_replace_sorry_for_have_exact_partial_log_case() -> None:
    """Test the exact case from partial.log where reconstruction failed."""
    from typing import cast

    from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
    from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
    from goedels_poetry.state import GoedelsPoetryStateManager
    from goedels_poetry.util.tree import TreeNode

    theorem_sig = "theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65"
    theorem = with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # This is the exact sketch from partial.log
        sketch = f"""{theorem_sig} := by
  -- Reduce `v` to a concrete expression by rewriting with the given equalities.
  have hv_rewrite : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := by
    -- from `v = 1/3 * (b*h)` and `b=30`, `h=13/2`
    sorry

  -- Pure arithmetic evaluation of the concrete expression.
  have hcalc : ((1 / 3 : ℝ) * (30 * (13 / 2 : ℝ))) = (65 : ℝ) := by
    -- `norm_num` should solve this directly.
    sorry

  -- Conclude by chaining the equalities.
  calc
    v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := hv_rewrite
    _ = 65 := hcalc"""

        # Compute body-relative `sorry` spans (the offset-based reconstruction path).
        hv_sorry_start = sketch.index("sorry", sketch.index("have hv_rewrite"))
        hv_sorry_end = hv_sorry_start + len("sorry")
        hcalc_sorry_start = sketch.index("sorry", sketch.index("have hcalc"))
        hcalc_sorry_end = hcalc_sorry_start + len("sorry")

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
            proof_history=[],
        )

        # Create child proof for hv_rewrite (from partial.log)
        child_hv_rewrite = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma hv_rewrite (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) (h₃ : h = 13 / 2) : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ))",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="""have h₄ : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := by
  calc
    v = 1 / 3 * (b * h) := h₁
    _ = 1 / 3 * (30 * (13 / 2 : ℝ)) := by rw [h₂, h₃]
exact h₄""",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        child_hv_rewrite["hole_name"] = "hv_rewrite"
        child_hv_rewrite["hole_start"] = hv_sorry_start
        child_hv_rewrite["hole_end"] = hv_sorry_end

        # Create child proof for hcalc (from partial.log)
        child_hcalc = FormalTheoremProofState(
            parent=cast(TreeNode, decomposed),
            depth=1,
            formal_theorem="lemma hcalc (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h)) (h₂ : b = 30) (h₃ : h = 13 / 2) (hv_rewrite : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ))) : ((1 / 3 : ℝ) * (30 * (13 / 2 : ℝ))) = (65 : ℝ)",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof="""have h₄ : (30 : ℝ) * (13 / 2 : ℝ) = 195 := by norm_num
have h₅ : (1 / 3 : ℝ) * (195 : ℝ) = 65 := by norm_num
calc
  ((1 / 3 : ℝ) * (30 * (13 / 2 : ℝ))) = (1 / 3 : ℝ) * (195 : ℝ) := by rw [h₄]
  _ = 65 := by rw [h₅]""",
            proved=True,
            errors=None,
            ast=None,
            self_correction_attempts=1,
            proof_history=[],
            pass_attempts=0,
        )
        child_hcalc["hole_name"] = "hcalc"
        child_hcalc["hole_start"] = hcalc_sorry_start
        child_hcalc["hole_end"] = hcalc_sorry_end

        decomposed["children"].extend([cast(TreeNode, child_hv_rewrite), cast(TreeNode, child_hcalc)])
        state.formal_theorem_proof = cast(TreeNode, decomposed)
        manager = GoedelsPoetryStateManager(state)

        result = manager.reconstruct_complete_proof()

        # Should contain DEFAULT_IMPORTS
        assert result.startswith(DEFAULT_IMPORTS)

        # Should contain both have statements
        assert "have hv_rewrite :" in result
        assert "have hcalc :" in result

        # Should contain the proof bodies
        assert "calc" in result
        assert "v = 1 / 3 * (b * h) := h₁" in result
        assert "rw [h₂, h₃]" in result
        assert "norm_num" in result
        assert "rw [h₄]" in result
        assert "rw [h₅]" in result

        # Should preserve comments
        assert "-- from `v = 1/3 * (b*h)`" in result
        assert "-- `norm_num` should solve this directly." in result

        # Should NOT contain any sorry
        result_no_imports = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in result_no_imports, f"Found 'sorry' in reconstructed proof:\n{result_no_imports}"

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_extract_tactics_after_by_with_full_lemma() -> None:
    """Test that _extract_tactics_after_by correctly handles full lemma statements."""
    import uuid

    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_extract_{uuid.uuid4()} : True")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        # Test case 1: Normal tactics (should work as before)
        tactics_only = "calc\n  v = 1 / 3 * (b * h) := h₁\n  _ = 1 / 3 * (30 * (13 / 2 : ℝ)) := by rw [h₂, h₃]"
        result1 = manager._extract_tactics_after_by(tactics_only)
        assert result1 == tactics_only

        # Test case 2: Full lemma statement (should extract tactics)
        full_lemma = """lemma helper : Type := by
  constructor
  exact trivial"""
        result2 = manager._extract_tactics_after_by(full_lemma)
        assert "constructor" in result2
        assert "exact trivial" in result2
        assert "lemma helper" not in result2

        # Test case 3: Proof without := by (should return sorry if it looks like a lemma)
        lemma_without_by = "lemma helper : Type := sorry"
        result3 = manager._extract_tactics_after_by(lemma_without_by)
        assert result3 == "sorry"

        # Test case 4: Just tactics without := by (should return as-is)
        just_tactics = "constructor\nexact trivial"
        result4 = manager._extract_tactics_after_by(just_tactics)
        assert result4 == just_tactics

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)


def test_extract_tactics_after_by_nested_lemma() -> None:
    """Test that _extract_tactics_after_by handles nested lemma statements correctly."""
    import uuid

    from goedels_poetry.state import GoedelsPoetryStateManager

    theorem = with_default_preamble(f"theorem test_nested_{uuid.uuid4()} : True")

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        # Test case: Proof that contains a nested lemma statement
        nested_lemma = """have h₄ : v = (1 / 3 : ℝ) * (30 * (13 / 2 : ℝ)) := by
  calc
    v = 1 / 3 * (b * h) := h₁
    _ = 1 / 3 * (30 * (13 / 2 : ℝ)) := by rw [h₂, h₃]
exact h₄"""

        result = manager._extract_tactics_after_by(nested_lemma)
        # This is already a tactic script (starts with `have`), so we must NOT strip the `have` binder.
        # Stripping after the first `:= by` would produce a dangling `exact h₄` and break reconstruction.
        assert "have h₄" in result
        assert "calc" in result
        assert "exact h₄" in result

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
