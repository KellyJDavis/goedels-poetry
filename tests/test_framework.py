"""Tests for goedels_poetry.framework module.

Note: These tests require Python 3.10+ due to kimina-client dependency compatibility.
Imports are deferred to test execution time to avoid collection errors on Python 3.9.
"""

import os
import sys
import tempfile
import uuid
from contextlib import suppress
from io import StringIO
from unittest.mock import patch

import pytest


def test_finish_called_when_is_finished_set_in_action() -> None:
    """
    Test that finish() is called even when is_finished is set inside an action method
    rather than by the supervisor.

    This is a unit test that simulates the bug scenario without requiring all the agent
    machinery or a real Lean server.
    """
    # Skip on Python < 3.10 due to kimina-client dependency issues
    if sys.version_info < (3, 10):
        pytest.skip("Requires Python 3.10+ due to kimina-client dependency syntax")

    # Import at test execution time to avoid collection errors on Python 3.9
    from unittest.mock import MagicMock

    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    theorem = f"theorem test_finish_{uuid.uuid4()} : True"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir

            # Create initial state
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Clear all queues so supervisor would normally return "finish"
            manager._state.proof_syntax_queue.clear()

            # Create framework with mocked LLMs (GitHub CI has no access to LLMs)
            mock_llm = MagicMock()
            config = GoedelsPoetryConfig(
                formalizer_agent_llm=mock_llm,
                prover_agent_llm=mock_llm,
                semantics_agent_llm=mock_llm,
                decomposer_agent_llm=mock_llm,
            )
            framework = GoedelsPoetryFramework(config=config, state_manager=manager)

            # Create a custom action that sets is_finished=True mid-execution
            # This simulates what happens in _handle_failed_sketch and other places
            def mock_action_that_sets_finished():
                manager._state.is_finished = True
                manager._state.reason = "Test: is_finished set inside action"

            # Patch the supervisor to return our custom action once, then it won't be called again
            call_count = [0]

            def mock_get_action(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "test_action"
                # This should never be reached due to our fix
                return "finish"

            # Add the test action to the framework
            framework.test_action = mock_action_that_sets_finished

            # Capture stdout
            captured_output = StringIO()
            with (
                patch("sys.stdout", captured_output),
                patch("goedels_poetry.agents.supervisor_agent.SupervisorAgent.get_action", mock_get_action),
            ):
                framework.run()

            # Verify finish() was called
            assert "finish" in manager._state.action_history, "finish should be in action history"
            assert manager._state.action_history[-1] == "finish", "finish should be the last action"
            assert manager.is_finished
            assert "Test: is_finished set inside action" in manager.reason

            # Verify finish message was printed
            output = captured_output.getvalue()
            assert "Proof process completed:" in output

        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

            if old_env is not None:
                os.environ["GOEDELS_POETRY_DIR"] = old_env
            elif "GOEDELS_POETRY_DIR" in os.environ:
                del os.environ["GOEDELS_POETRY_DIR"]


def test_finish_not_called_twice_when_supervisor_returns_finish() -> None:
    """
    Test that finish() is only called once when the supervisor naturally returns "finish".

    This ensures our fix doesn't cause finish() to be called twice in the normal success case.
    """
    # Skip on Python < 3.10 due to kimina-client dependency issues
    if sys.version_info < (3, 10):
        pytest.skip("Requires Python 3.10+ due to kimina-client dependency syntax")

    # Import at test execution time to avoid collection errors on Python 3.9
    from unittest.mock import MagicMock

    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    theorem = f"theorem test_normal_finish_{uuid.uuid4()} : True"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["GOEDELS_POETRY_DIR"] = tmpdir

            # Create state with empty queues - this will cause supervisor to return "finish" immediately
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Clear all queues so supervisor returns finish immediately
            manager._state.proof_syntax_queue.clear()

            # Create framework with mocked LLMs (GitHub CI has no access to LLMs)
            mock_llm = MagicMock()
            config = GoedelsPoetryConfig(
                formalizer_agent_llm=mock_llm,
                prover_agent_llm=mock_llm,
                semantics_agent_llm=mock_llm,
                decomposer_agent_llm=mock_llm,
            )
            framework = GoedelsPoetryFramework(config=config, state_manager=manager)

            # Capture output
            captured_output = StringIO()
            with patch("sys.stdout", captured_output):
                framework.run()

            # Verify finish() was called exactly once
            finish_count = manager._state.action_history.count("finish")
            assert finish_count == 1, f"finish should be called exactly once, but was called {finish_count} times"

            # Verify the success message
            assert manager.reason == "Proof completed successfully."

            output = captured_output.getvalue()
            # Should only have one completion message
            assert output.count("Proof process completed:") == 1

        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

            if old_env is not None:
                os.environ["GOEDELS_POETRY_DIR"] = old_env
            elif "GOEDELS_POETRY_DIR" in os.environ:
                del os.environ["GOEDELS_POETRY_DIR"]
