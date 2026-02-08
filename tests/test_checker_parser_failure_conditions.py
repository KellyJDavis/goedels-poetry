from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _FakeAstModuleResponse:
    """Minimal stand-in for kimina_client.models.AstModuleResponse."""

    results: list[dict]


class _FakeKiminaClient:
    """Minimal KiminaClient stand-in: returns queued ast_code responses."""

    def __init__(self, ast_code_results: list[dict]):
        self._queue = list(ast_code_results)

    def ast_code(self, _code: str, *, timeout: int | None = None):
        if not self._queue:
            raise AssertionError("Unexpected ast_code call: response queue exhausted")  # noqa: TRY003
        return _FakeAstModuleResponse(results=[self._queue.pop(0)])


def test_proof_checker_flags_kimina_ast_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If the proof otherwise 'passes' (proved=True, errors="") but ast_code is unusable,
    the checker should flip proved=False and set an error instead of letting the parser raise.
    """
    from goedels_poetry.agents import proof_checker_agent as mod

    proof_state = {
        "proved": True,
        "errors": "",
        "preamble": "import Foo",
        "formal_theorem": "theorem t : True := by trivial",
        "formal_proof": "dummy",
        "ast": {"dummy": True},
    }

    kimina = _FakeKiminaClient(
        ast_code_results=[
            # First call is for the proof/sketch body; error => unusable AST
            {"error": "parse failed", "ast": None},
        ]
    )

    mod._maybe_flag_downstream_parser_errors(  # type: ignore[attr-defined]
        proof_state=proof_state,
        kimina_client=kimina,
        server_timeout=1,
        raw_output="theorem t : True := by trivial",
    )

    assert proof_state["proved"] is False
    assert proof_state["formal_proof"] is None
    assert proof_state["ast"] is None
    assert isinstance(proof_state["errors"], str)
    assert "Kimina failed to parse proof" in proof_state["errors"]


def test_sketch_checker_flags_kimina_ast_parse_failure() -> None:
    from goedels_poetry.agents import sketch_checker_agent as mod

    theorem_state = {
        "syntactic": True,
        "errors": "",
        "preamble": "import Foo",
        "formal_theorem": "theorem t : True := by trivial",
        "proof_sketch": "dummy",
        "ast": {"dummy": True},
    }

    kimina = _FakeKiminaClient(
        ast_code_results=[
            {"error": "parse failed", "ast": None},
        ]
    )

    mod._maybe_flag_downstream_parser_errors(  # type: ignore[attr-defined]
        theorem_state=theorem_state,
        kimina_client=kimina,
        server_timeout=1,
        raw_output="theorem t : True := by trivial",
    )

    assert theorem_state["syntactic"] is False
    assert theorem_state["proof_sketch"] is None
    assert theorem_state["ast"] is None
    assert isinstance(theorem_state["errors"], str)
    assert "Kimina failed to parse proof" in theorem_state["errors"]


def test_proof_checker_flags_structural_extraction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If ast_code succeeds but structural extraction fails (extract_proof_body_from_ast returns None),
    the checker should flip proved=False and set a structural-extraction error.
    """
    from goedels_poetry.agents import proof_checker_agent as mod

    # Patch signature extraction so the structural check is exercised on dummy ASTs.
    monkeypatch.setattr(mod, "extract_signature_from_ast", lambda _ast: "theorem t : True")  # type: ignore[arg-type]
    monkeypatch.setattr(mod, "extract_proof_body_from_ast", lambda _ast, _sig: None)  # type: ignore[arg-type]

    proof_state = {
        "proved": True,
        "errors": "",
        "preamble": "import Foo",
        "formal_theorem": "theorem t : True := by trivial",
        "formal_proof": "dummy",
        "ast": {"dummy": True},
    }

    kimina = _FakeKiminaClient(
        ast_code_results=[
            # Proof body ast_code: usable AST
            {"error": None, "ast": {"commands": [{"kind": "dummy"}]}},
            # Formal theorem ast_code: usable AST (signature extraction is patched anyway)
            {"error": None, "ast": {"commands": [{"kind": "dummy"}]}},
        ]
    )

    mod._maybe_flag_downstream_parser_errors(  # type: ignore[attr-defined]
        proof_state=proof_state,
        kimina_client=kimina,
        server_timeout=1,
        raw_output="theorem t : True := by trivial",
    )

    assert proof_state["proved"] is False
    assert proof_state["formal_proof"] is None
    assert proof_state["ast"] is None
    assert "Structural extraction failed for target signature" in proof_state["errors"]
