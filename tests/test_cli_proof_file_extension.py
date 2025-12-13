"""Tests for CLI proof file extension based on validation result."""

import tempfile
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

from goedels_poetry.cli import (
    _handle_missing_header,
    _handle_processing_error,
    process_theorems_from_directory,
)
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


def test_proof_validation_result_field_exists():
    """Test that proof_validation_result field exists in GoedelsPoetryState."""
    import os

    old_env = os.environ.get("GOEDELS_POETRY_DIR")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir
            # Use unique theorem to avoid directory conflicts
            import uuid

            unique_theorem = f"import Mathlib.Data.Nat.Basic\n\ntheorem test_{uuid.uuid4().hex[:8]} : True := by sorry"

            # Clean up if it exists
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(unique_theorem)

            state = GoedelsPoetryState(formal_theorem=unique_theorem)
            assert hasattr(state, "proof_validation_result")
            assert state.proof_validation_result is None

            # Test that it can be set
            state.proof_validation_result = True
            assert state.proof_validation_result is True

            state.proof_validation_result = False
            assert state.proof_validation_result is False

            # Clean up
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(unique_theorem)
    finally:
        if old_env is not None:
            os.environ["GOEDELS_POETRY_DIR"] = old_env
        elif "GOEDELS_POETRY_DIR" in os.environ:
            del os.environ["GOEDELS_POETRY_DIR"]


def test_valid_proof_writes_to_proof_file(temp_dir):
    """Test that a valid proof writes to .proof file."""
    import uuid

    # Create a test theorem file with unique content to avoid directory conflicts
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    theorem_content = f"import Mathlib.Data.Nat.Basic\n\ntheorem {unique_name} : True := by sorry"
    theorem_file = temp_dir / f"{unique_name}.lean"
    theorem_file.write_text(theorem_content)

    # Setup mocks - patch where they're imported inside process_theorems_from_directory
    with (
        patch("goedels_poetry.framework.GoedelsPoetryFramework") as mock_framework_class,
        patch("goedels_poetry.state.GoedelsPoetryStateManager") as mock_sm_class,
        patch("goedels_poetry.state.GoedelsPoetryState") as mock_state_class,
        patch("goedels_poetry.framework.GoedelsPoetryConfig") as mock_config_class,
    ):
        mock_framework = MagicMock()
        mock_framework_class.return_value = mock_framework
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        # Mock state to avoid directory creation
        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        mock_state_manager = MagicMock()
        mock_state_manager.reason = "Proof completed successfully."
        mock_state_manager._state.proof_validation_result = True
        mock_state_manager.reconstruct_complete_proof.return_value = (
            f"import Mathlib.Data.Nat.Basic\n\ntheorem {unique_name} : True := by trivial"
        )
        mock_sm_class.return_value = mock_state_manager

        # Process the theorem
        process_theorems_from_directory(temp_dir, ".lean", is_formal=True)

        # Check that .proof file was created
        proof_file = temp_dir / f"{unique_name}.proof"
        assert proof_file.exists(), ".proof file should exist for valid proof"
        assert not (temp_dir / f"{unique_name}.failed-proof").exists(), ".failed-proof file should not exist"

        # Check content
        assert "trivial" in proof_file.read_text()


def test_invalid_proof_writes_to_failed_proof_file(temp_dir):
    """Test that an invalid proof writes to .failed-proof file."""
    # Create a test theorem file
    theorem_file = temp_dir / "test.lean"
    theorem_file.write_text("import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by sorry")

    # Setup mocks - patch where they're imported inside process_theorems_from_directory
    with (
        patch("goedels_poetry.framework.GoedelsPoetryFramework") as mock_framework_class,
        patch("goedels_poetry.state.GoedelsPoetryStateManager") as mock_sm_class,
        patch("goedels_poetry.state.GoedelsPoetryState") as mock_state_class,
        patch("goedels_poetry.framework.GoedelsPoetryConfig") as mock_config_class,
    ):
        mock_framework = MagicMock()
        mock_framework_class.return_value = mock_framework
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        # Setup mock state manager with invalid validation result
        mock_state_manager = MagicMock()
        mock_state_manager.reason = "Proof completed successfully."
        mock_state_manager._state.proof_validation_result = False
        mock_state_manager.reconstruct_complete_proof.return_value = (
            "import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by sorry"
        )
        mock_sm_class.return_value = mock_state_manager

        # Process the theorem
        process_theorems_from_directory(temp_dir, ".lean", is_formal=True)

        # Check that .failed-proof file was created
        failed_proof_file = temp_dir / "test.failed-proof"
        assert failed_proof_file.exists(), ".failed-proof file should exist for invalid proof"
        assert not (temp_dir / "test.proof").exists(), ".proof file should not exist"


def test_validation_exception_writes_to_failed_proof_file(temp_dir):
    """Test that validation exception (None result) writes to .failed-proof file."""
    # Create a test theorem file
    theorem_file = temp_dir / "test.lean"
    theorem_file.write_text("import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by sorry")

    # Setup mocks - patch where they're imported inside process_theorems_from_directory
    with (
        patch("goedels_poetry.framework.GoedelsPoetryFramework") as mock_framework_class,
        patch("goedels_poetry.state.GoedelsPoetryStateManager") as mock_sm_class,
        patch("goedels_poetry.state.GoedelsPoetryState") as mock_state_class,
        patch("goedels_poetry.framework.GoedelsPoetryConfig") as mock_config_class,
    ):
        mock_framework = MagicMock()
        mock_framework_class.return_value = mock_framework
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        # Setup mock state manager with None validation result (exception case)
        mock_state_manager = MagicMock()
        mock_state_manager.reason = "Proof completed successfully."
        mock_state_manager._state.proof_validation_result = None
        mock_state_manager.reconstruct_complete_proof.return_value = (
            "import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by trivial"
        )
        mock_sm_class.return_value = mock_state_manager

        # Process the theorem
        process_theorems_from_directory(temp_dir, ".lean", is_formal=True)

        # Check that .failed-proof file was created
        failed_proof_file = temp_dir / "test.failed-proof"
        assert failed_proof_file.exists(), ".failed-proof file should exist when validation exception occurs"
        assert not (temp_dir / "test.proof").exists(), ".proof file should not exist"


def test_non_successful_completion_writes_to_failed_proof_file(temp_dir):
    """Test that non-successful completion writes to .failed-proof file."""
    # Create a test theorem file
    theorem_file = temp_dir / "test.lean"
    theorem_file.write_text("import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by sorry")

    # Setup mocks - patch where they're imported inside process_theorems_from_directory
    with (
        patch("goedels_poetry.framework.GoedelsPoetryFramework") as mock_framework_class,
        patch("goedels_poetry.state.GoedelsPoetryStateManager") as mock_sm_class,
        patch("goedels_poetry.state.GoedelsPoetryState") as mock_state_class,
        patch("goedels_poetry.framework.GoedelsPoetryConfig") as mock_config_class,
    ):
        mock_framework = MagicMock()
        mock_framework_class.return_value = mock_framework
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        # Setup mock state manager with non-successful reason
        mock_state_manager = MagicMock()
        mock_state_manager.reason = "Proof failed: Maximum attempts exceeded."
        mock_sm_class.return_value = mock_state_manager

        # Process the theorem
        process_theorems_from_directory(temp_dir, ".lean", is_formal=True)

        # Check that .failed-proof file was created
        failed_proof_file = temp_dir / "test.failed-proof"
        assert failed_proof_file.exists(), ".failed-proof file should exist for non-successful completion"
        assert not (temp_dir / "test.proof").exists(), ".proof file should not exist"

        # Check content contains failure message
        content = failed_proof_file.read_text()
        assert "Proof failed" in content


def test_missing_header_writes_to_failed_proof_file(temp_dir):
    """Test that missing header writes to .failed-proof file."""
    theorem_file = temp_dir / "test.lean"
    theorem_file.write_text("theorem test : True := by sorry")

    _handle_missing_header(theorem_file)

    # Check that .failed-proof file was created
    failed_proof_file = temp_dir / "test.failed-proof"
    assert failed_proof_file.exists(), ".failed-proof file should exist for missing header"
    assert not (temp_dir / "test.proof").exists(), ".proof file should not exist"

    # Check content
    content = failed_proof_file.read_text()
    assert "Missing Lean header" in content


def test_processing_error_writes_to_failed_proof_file(temp_dir):
    """Test that processing error writes to .failed-proof file."""
    theorem_file = temp_dir / "test.lean"
    theorem_file.write_text("theorem test : True := by sorry")

    test_error = ValueError("Test error message")
    _handle_processing_error(theorem_file, test_error)

    # Check that .failed-proff file was created
    failed_proof_file = temp_dir / "test.failed-proof"
    assert failed_proof_file.exists(), ".failed-proof file should exist for processing error"
    assert not (temp_dir / "test.proof").exists(), ".proof file should not exist"

    # Check content
    content = failed_proof_file.read_text()
    assert "Error during processing" in content
    assert "Test error message" in content


def test_framework_stores_validation_result():
    """Test that framework.finish() stores validation result in state."""
    import os
    import tempfile
    import uuid

    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework

    old_env = os.environ.get("GOEDELS_POETRY_DIR")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir
            # Use unique theorem to avoid directory conflicts
            unique_theorem1 = f"import Mathlib.Data.Nat.Basic\n\ntheorem test_{uuid.uuid4().hex[:8]} : True := by sorry"

            # Clean up if exists
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(unique_theorem1)

            state = GoedelsPoetryState(formal_theorem=unique_theorem1)
            state_manager = GoedelsPoetryStateManager(state)
            state_manager.is_finished = True
            state_manager.reason = "Proof completed successfully."

            # Mock reconstruct_complete_proof to return a proof
            with (
                patch.object(
                    state_manager,
                    "reconstruct_complete_proof",
                    return_value="import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by trivial",
                ),
                patch("goedels_poetry.framework.check_complete_proof", return_value=(True, "")) as mock_check,
            ):
                config = GoedelsPoetryConfig()
                framework = GoedelsPoetryFramework(config, state_manager)
                framework.finish()

                # Check that validation result was stored
                assert state.proof_validation_result is True
                mock_check.assert_called_once()

            # Test with invalid proof
            unique_theorem2 = f"import Mathlib.Data.Nat.Basic\n\ntheorem test_{uuid.uuid4().hex[:8]} : True := by sorry"
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(unique_theorem2)
            state2 = GoedelsPoetryState(formal_theorem=unique_theorem2)
            state_manager2 = GoedelsPoetryStateManager(state2)
            state_manager2.is_finished = True
            state_manager2.reason = "Proof completed successfully."

            with (
                patch.object(
                    state_manager2,
                    "reconstruct_complete_proof",
                    return_value="import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by sorry",
                ),
                patch("goedels_poetry.framework.check_complete_proof", return_value=(False, "error message")),
            ):
                config = GoedelsPoetryConfig()
                framework2 = GoedelsPoetryFramework(config, state_manager2)
                framework2.finish()

                # Check that validation result was stored
                assert state2.proof_validation_result is False

            # Test with validation exception (should leave as None)
            unique_theorem3 = f"import Mathlib.Data.Nat.Basic\n\ntheorem test_{uuid.uuid4().hex[:8]} : True := by sorry"
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(unique_theorem3)
            state3 = GoedelsPoetryState(formal_theorem=unique_theorem3)
            state_manager3 = GoedelsPoetryStateManager(state3)
            state_manager3.is_finished = True
            state_manager3.reason = "Proof completed successfully."

            with (
                patch.object(
                    state_manager3,
                    "reconstruct_complete_proof",
                    return_value="import Mathlib.Data.Nat.Basic\n\ntheorem test : True := by trivial",
                ),
                patch("goedels_poetry.framework.check_complete_proof", side_effect=Exception("Validation error")),
            ):
                config = GoedelsPoetryConfig()
                framework3 = GoedelsPoetryFramework(config, state_manager3)
                framework3.finish()

                # Check that validation result was left as None (exception case)
                assert state3.proof_validation_result is None

            # Clean up
            for thm in [unique_theorem1, unique_theorem2, unique_theorem3]:
                with suppress(Exception):
                    GoedelsPoetryState.clear_theorem_directory(thm)
    finally:
        if old_env is not None:
            os.environ["GOEDELS_POETRY_DIR"] = old_env
        elif "GOEDELS_POETRY_DIR" in os.environ:
            del os.environ["GOEDELS_POETRY_DIR"]


def test_informal_theorem_processing(temp_dir):
    """Test that informal theorem processing also uses correct file extension."""
    # Create a test theorem file
    theorem_file = temp_dir / "test.txt"
    theorem_file.write_text("Prove that 1 + 1 = 2")

    # Setup mocks - patch where they're imported inside process_theorems_from_directory
    with (
        patch("goedels_poetry.framework.GoedelsPoetryFramework") as mock_framework_class,
        patch("goedels_poetry.state.GoedelsPoetryStateManager") as mock_sm_class,
        patch("goedels_poetry.state.GoedelsPoetryState") as mock_state_class,
        patch("goedels_poetry.framework.GoedelsPoetryConfig") as mock_config_class,
    ):
        mock_framework = MagicMock()
        mock_framework_class.return_value = mock_framework
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        # Setup mock state manager with valid validation result
        mock_state_manager = MagicMock()
        mock_state_manager.reason = "Proof completed successfully."
        mock_state_manager._state.proof_validation_result = True
        mock_state_manager.reconstruct_complete_proof.return_value = (
            "import Mathlib.Data.Nat.Basic\n\ntheorem test : 1 + 1 = 2 := by rfl"
        )
        mock_sm_class.return_value = mock_state_manager

        # Process the theorem
        process_theorems_from_directory(temp_dir, ".txt", is_formal=False)

        # Check that .proof file was created
        proof_file = temp_dir / "test.proof"
        assert proof_file.exists(), ".proof file should exist for valid proof"
        assert not (temp_dir / "test.failed-proof").exists(), ".failed-proof file should not exist"
