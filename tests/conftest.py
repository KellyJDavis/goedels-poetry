"""Test fixtures for goedels_poetry tests."""

import os
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def kimina_server_url() -> Generator[str, None, None]:
    """
    Fixture that provides a test Kimina Lean server URL.

    NOTE: Integration tests require a real, running Kimina Lean server.
    The server must be started manually before running these tests.

    To run integration tests:
    1. Start the Kimina server in a separate terminal:
       cd ../kimina-lean-server && python -m server

    2. Run the tests:
       make test-integration

    Yields
    ------
    str
        The base URL for the test server (e.g., "http://localhost:8000")
    """
    import httpx

    # Check if a real server is running
    server_url = os.getenv("KIMINA_SERVER_URL", "http://localhost:8000")

    # Try to connect to the server
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{server_url}/health", timeout=5.0)
            if response.status_code != 200:
                pytest.skip(
                    f"Kimina server at {server_url} is not healthy. "
                    "Start the server with: cd ../kimina-lean-server && python -m server"
                )
                return  # type: ignore[unreachable]
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        pytest.skip(
            f"Kimina server not running at {server_url}. "
            f"Error: {e}. "
            "Start the server with: cd ../kimina-lean-server && python -m server"
        )
        return  # type: ignore[unreachable]

    yield server_url


@pytest.fixture
def skip_if_no_lean() -> None:
    """
    Fixture that skips tests if Lean is not available.

    This is used for integration tests that require a working Lean installation.
    Tests using this fixture will be skipped in environments without Lean.
    """
    import shutil

    if not shutil.which("lake"):
        pytest.skip("Lean (lake) is not installed - skipping integration test")
