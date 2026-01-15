"""Test fixtures for goedels_poetry tests."""

import os
from collections.abc import Generator

import pytest
from ast_test_utils import build_simple_ast, find_sorry_spans


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


def _attach_ast_tree(node: object) -> None:
    if not isinstance(node, dict):
        return
    if "children" in node:
        sketch = node.get("proof_sketch")
        if node.get("ast") is None and isinstance(sketch, str) and "by" in sketch:
            node["ast"] = build_simple_ast(sketch, sorry_spans=find_sorry_spans(sketch))
        for child in node.get("children", []):
            _attach_ast_tree(child)
        return

    if node.get("ast") is not None:
        return
    proof = str(node.get("formal_proof") or "")
    formal_theorem = str(node.get("formal_theorem") or "")
    idx = formal_theorem.find(":=")
    theorem_sig = formal_theorem[:idx].rstrip() if idx != -1 else formal_theorem
    is_full = proof.lstrip().startswith(("theorem", "lemma", "example"))
    source = proof if is_full else f"{theorem_sig} := by{proof}"
    node["ast"] = build_simple_ast(source)


@pytest.fixture(autouse=True)
def attach_ast_to_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    from goedels_poetry.state import GoedelsPoetryStateManager

    original = GoedelsPoetryStateManager.reconstruct_complete_proof

    def _wrapped(self: GoedelsPoetryStateManager) -> str:
        root = getattr(self._state, "formal_theorem_proof", None)
        _attach_ast_tree(root)
        return original(self)

    monkeypatch.setattr(GoedelsPoetryStateManager, "reconstruct_complete_proof", _wrapped)
