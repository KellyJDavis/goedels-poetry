"""Test fixtures for goedels_poetry tests."""

import os
from collections.abc import Generator

import pytest
from ast_test_utils import build_simple_ast, find_sorry_spans

from goedels_poetry.agents.util.common import compute_source_hashes


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
        ast_obj = node.get("ast")
        if ast_obj is not None and hasattr(ast_obj, "get_body_text"):
            body_text = ast_obj.get_body_text()  # type: ignore[call-arg]
            raw_hash, normalized_hash = compute_source_hashes(body_text or "")
        elif isinstance(sketch, str):
            raw_hash, normalized_hash = compute_source_hashes(sketch)
        else:
            raw_hash, normalized_hash = compute_source_hashes("")
        node["source_hash_raw"] = raw_hash
        node["source_hash_normalized"] = normalized_hash
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
    raw_hash, normalized_hash = compute_source_hashes(source)
    node["source_hash_raw"] = raw_hash
    node["source_hash_normalized"] = normalized_hash


@pytest.fixture(autouse=True)
def attach_ast_to_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    from goedels_poetry.state import GoedelsPoetryStateManager

    original = GoedelsPoetryStateManager.reconstruct_complete_proof

    def _wrapped(self: GoedelsPoetryStateManager) -> str:
        root = getattr(self._state, "formal_theorem_proof", None)
        _attach_ast_tree(root)
        return original(self)

    monkeypatch.setattr(GoedelsPoetryStateManager, "reconstruct_complete_proof", _wrapped)


@pytest.fixture(autouse=True)
def stub_kimina_for_unit_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: C901
    if "kimina_server_url" in request.fixturenames or request.node.get_closest_marker("kimina"):
        return

    from goedels_poetry.agents.util.common import split_preamble_and_body

    def _stub_check(code: str, *, infotree=None, client=None) -> dict:
        _preamble, body = split_preamble_and_body(code)
        # If body is empty, treat the entire code as body (for cases like "have" statements)
        if not body:
            body = code
        if os.environ.get("GOEDELS_POETRY_DEBUG"):
            print(f"stub_check code_len={len(code)}, body_len={len(body)}, body={body[:100]!r}")
        sorries = []
        for line_idx, line in enumerate(body.splitlines(), start=1):
            start = line.find("sorry")
            while start != -1:
                sorries.append({
                    "pos": {"line": line_idx, "column": start},
                    "endPos": {"line": line_idx, "column": start + len("sorry")},
                    "goal": "",
                })
                start = line.find("sorry", start + 1)
        if os.environ.get("GOEDELS_POETRY_DEBUG"):
            print(f"stub_check found {len(sorries)} sorries")
        return {
            "sorries": sorries,
            "tactics": [],
            "errors": [],
            "warnings": [],
            "infos": [],
            "ast": {},
            "system_errors": None,
            "pass": True,
            "complete": not sorries,
        }

    def _stub_semantics():
        return {"column_is_byte": False, "line_base": 1}

    def _stub_match_sorries(self, ast, body_text, check_sorries, context, source_text=None):  # type: ignore[override]
        # Use body_text parameter instead of ast.get_body_text() to ensure we use the correct sketch
        # Parse body_text directly to find sorry holes
        lines = body_text.splitlines()
        line_starts: list[int] = [0]
        byte_count = 0
        for ch in body_text:
            if ch == "\n":
                line_starts.append(byte_count + len(ch.encode("utf-8")))
            byte_count += len(ch.encode("utf-8"))

        mapping: dict[str, list[tuple[int, int]]] = {}
        current_have: str | None = None
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("have "):
                parts = stripped.split()
                if len(parts) > 1:
                    name = parts[1].rstrip(":")
                    current_have = name
            # Find standalone "sorry" tokens (with word boundaries)
            start = 0
            while True:
                start = line.find("sorry", start)
                if start == -1:
                    break
                # Check if it's a standalone token (whitespace or start/end of line before and after)
                before = line[start - 1] if start > 0 else " "
                after_idx = start + len("sorry")
                after = line[after_idx] if after_idx < len(line) else " "
                if before.isspace() and after.isspace():
                    end = start + len("sorry")
                    line_start = line_starts[idx] if idx < len(line_starts) else 0
                    start_b = line_start + len(line[:start].encode("utf-8"))
                    end_b = line_start + len(line[:end].encode("utf-8"))
                    hole_name = current_have or "<main body>"
                    mapping.setdefault(hole_name, []).append((start_b, end_b))
                    current_have = None
                start = after_idx
        if os.environ.get("GOEDELS_POETRY_DEBUG"):
            print(f"stub match sorries body_text={body_text[:100]!r}, mapping={mapping}")
        return mapping

    def _stub_get_sorry_holes_by_name_bytes(self):  # type: ignore[override]
        text = self.get_body_text() or self._source_text or ""
        lines = text.splitlines()
        line_starts: list[int] = [0]
        byte_count = 0
        for ch in text:
            if ch == "\n":
                line_starts.append(byte_count + len(ch.encode("utf-8")))
            byte_count += len(ch.encode("utf-8"))

        mapping: dict[str, list[tuple[int, int]]] = {}
        current_have: str | None = None
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("have "):
                parts = stripped.split()
                if len(parts) > 1:
                    name = parts[1].rstrip(":")
                    current_have = name
            if "sorry" in line:
                start = line.find("sorry")
                end = start + len("sorry")
                line_start = line_starts[idx] if idx < len(line_starts) else 0
                start_b = line_start + len(line[:start].encode("utf-8"))
                end_b = line_start + len(line[:end].encode("utf-8"))
                hole_name = current_have or "<main body>"
                mapping.setdefault(hole_name, []).append((start_b, end_b))
                current_have = None
        return mapping

    monkeypatch.setattr("goedels_poetry.agents.util.kimina_server.check_code_with_infotree", _stub_check)
    monkeypatch.setattr("goedels_poetry.agents.util.kimina_server.detect_position_semantics", _stub_semantics)
    monkeypatch.setattr("goedels_poetry.state.check_code_with_infotree", _stub_check)
    monkeypatch.setattr("goedels_poetry.state.detect_position_semantics", _stub_semantics)
    monkeypatch.setattr(
        "goedels_poetry.parsers.ast.AST.get_sorry_holes_by_name_bytes", _stub_get_sorry_holes_by_name_bytes
    )
    monkeypatch.setattr(
        "goedels_poetry.state.GoedelsPoetryStateManager._match_check_sorries_to_ast", _stub_match_sorries
    )
