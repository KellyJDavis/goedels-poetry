"""Test fixtures for goedels_poetry tests."""

from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def kimina_server_url() -> Generator[str, None, None]:
    """
    Fixture that provides a test Kimina Lean server URL.

    Creates an in-process FastAPI TestClient for integration tests.
    This avoids the complexity of Docker or separate server processes in CI.

    Yields
    ------
    str
        The base URL for the test server (e.g., "http://testserver")
    """
    # Lazy import to avoid requiring fastapi for all tests
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed - skipping integration test")
        return  # type: ignore[unreachable]

    # Try to import kimina-lean-server modules
    # Note: kimina-lean-server is no longer a submodule. To run integration tests,
    # you need to install and run a Kimina Lean Server separately.
    try:
        from server.main import create_app  # type: ignore[import-not-found]
        from server.settings import Environment, Settings  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("kimina-lean-server not available - install separately to run integration tests")
        return  # type: ignore[unreachable]

    settings = Settings(
        _env_file=None,
        max_repls=5,
        max_repl_uses=10,
        init_repls={},
        database_url=None,
        environment=Environment.prod,
    )
    app = create_app(settings)

    # Create TestClient with base_url pointing to root
    # (the agents will append /api/check, /api/ast_code, etc.)
    with TestClient(app, base_url="http://testserver") as client:
        # Store the client on the app so we can access it if needed
        app.state.test_client = client  # type: ignore[attr-defined]
        yield "http://testserver"


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
