"""Tests for goedels_poetry.state module."""

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

from goedels_poetry.state import GoedelsPoetryState


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

    # Hash should be hexadecimal
    assert all(c in "0123456789abcdef" for c in hash1)


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
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some checkpoint files
        checkpoint_files = [
            "goedels_poetry_state_20250101_120000_iter_0000.pkl",
            "goedels_poetry_state_20250101_130000_iter_0001.pkl",
            "goedels_poetry_state_20250101_140000_iter_0002.pkl",
        ]

        for filename in checkpoint_files:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write("dummy")

        # Create a non-checkpoint file that should be ignored
        with open(os.path.join(tmpdir, "other_file.txt"), "w") as f:
            f.write("dummy")

        checkpoints = GoedelsPoetryState.list_checkpoints(directory=tmpdir)

        # Should return all checkpoint files
        assert len(checkpoints) == 3

        # Should be sorted by modification time (newest first)
        # Since they were created in order, newest should be last in creation
        assert checkpoints[0].endswith("iter_0002.pkl")


def test_list_checkpoints_by_theorem() -> None:
    """Test list_checkpoints lists checkpoints for a theorem using the default directory."""
    import uuid

    theorem = f"Test Theorem For Checkpoints {uuid.uuid4()}"

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

    theorem = f"Test Theorem To Clear {uuid.uuid4()}"

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

    theorem = f"Test Save Load Theorem {uuid.uuid4()}"

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
    import uuid

    theorem = f"Test Load Latest Theorem {uuid.uuid4()}"

    # Clean up any existing directory first
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)

        # Save multiple times with different state
        state.action_history = ["first"]
        state.save()

        state.action_history = ["first", "second"]
        state.save()

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

            theorem = "Test Directory Creation"
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

    theorem = f"Prove that 3 cannot be written as sum of two cubes {uuid.uuid4()}."

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

            theorem = "Test Duplicate Directory"

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

    theorem = f"Test Iteration Counter {uuid.uuid4()}"

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
