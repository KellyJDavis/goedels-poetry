"""Proof-parser integration tests for formal_theorem with comments and := by sorry variants (plan 3.2-3.3)."""

from __future__ import annotations

import importlib
import sys
import types
import uuid
from typing import Any

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS


def _theorem_t_sorry() -> dict:
    """Minimal theorem t : True := by sorry for formal-theorem AST."""
    return {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "t", "info": {"leading": "", "trailing": " "}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                            }
                        ],
                    },
                ],
            },
        ],
    }


def _theorem_t_trivial() -> dict:
    """Minimal theorem t : True := by trivial for proof AST."""
    return {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "t", "info": {"leading": "", "trailing": " "}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticTrivial",
                                "args": [{"val": "trivial", "info": {"leading": "", "trailing": ""}}],
                            }
                        ],
                    },
                ],
            },
        ],
    }


def _clear_modules(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    for n in names:
        monkeypatch.delitem(sys.modules, n, raising=False)


def _install_kimina_stub(monkeypatch: pytest.MonkeyPatch, kimina_client_cls: type) -> None:
    _clear_modules(
        monkeypatch,
        [
            "kimina_client",
            "kimina_client.models",
            "goedels_poetry.agents.util.kimina_server",
            "goedels_poetry.agents.proof_parser_agent",
            "goedels_poetry.agents.sketch_parser_agent",
        ],
    )
    kimina_mod = types.ModuleType("kimina_client")
    kimina_mod.KiminaClient = kimina_client_cls
    models_mod = types.ModuleType("kimina_client.models")

    class _StubAstModuleResponse:
        def __init__(self, results: Any) -> None:
            self.results = results

    models_mod.AstModuleResponse = _StubAstModuleResponse
    models_mod.CheckResponse = object
    models_mod.CommandResponse = dict
    models_mod.Message = dict
    monkeypatch.setitem(sys.modules, "kimina_client", kimina_mod)
    monkeypatch.setitem(sys.modules, "kimina_client.models", models_mod)


class _OptionBAstResult:
    def __init__(self, ast: dict, sorries: list[dict] | None = None) -> None:
        self.module = "stub"
        self.error = None
        self.ast = ast
        self.sorries = sorries or []


class _OptionBAstResponse:
    def __init__(self, result: _OptionBAstResult) -> None:
        self.results = [result]


class _OptionBKiminaClient:
    """Option B stub: call-order tracking. First ast_code → formal AST, second → proof AST."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._n = 0

    def ast_code(self, _code: str) -> _OptionBAstResponse:
        self._n += 1
        if self._n == 1:
            return _OptionBAstResponse(_OptionBAstResult({"commands": [_theorem_t_sorry()]}))
        return _OptionBAstResponse(_OptionBAstResult({"commands": [_theorem_t_trivial()]}))


def _base_state(*, formal_theorem: str, llm_lean_output: str) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "parent": None,
        "depth": 0,
        "formal_theorem": formal_theorem,
        "preamble": DEFAULT_IMPORTS,
        "syntactic": True,
        "formal_proof": "",
        "proved": False,
        "errors": "",
        "ast": None,
        "self_correction_attempts": 0,
        "proof_history": [],
        "pass_attempts": 0,
        "llm_lean_output": llm_lean_output,
        "hole_name": None,
        "hole_start": None,
        "hole_end": None,
    }


@pytest.mark.parametrize(
    "formal_theorem,llm_lean_output",
    [
        ("/- doc -/ theorem t : True := by sorry", "theorem t : True := by\n  trivial"),
        ("theorem t : True := by sorry", "theorem t : True := by\n  trivial"),
        ("theorem t : True := by\n  sorry", "theorem t : True := by\n  trivial"),
        ("theorem t : True := by\n  /- comment -/\n  sorry", "theorem t : True := by\n  trivial"),
        ("theorem t : True := by\n  /- a -/\n  /- b -/\n  sorry", "theorem t : True := by\n  trivial"),
        ("theorem t : True := by\n  -- line\n  sorry", "theorem t : True := by\n  trivial"),
    ],
)
def test_parse_proof_formal_theorem_variants(
    monkeypatch: pytest.MonkeyPatch,
    formal_theorem: str,
    llm_lean_output: str,
) -> None:
    """Section 3.2: formal_theorem with docstring / := by sorry variants; proof body extracted."""
    _install_kimina_stub(monkeypatch, _OptionBKiminaClient)
    proof_parser_agent = importlib.import_module("goedels_poetry.agents.proof_parser_agent")
    state = _base_state(formal_theorem=formal_theorem, llm_lean_output=llm_lean_output)
    result = proof_parser_agent._parse_proof("url", 0, 0, state)  # type: ignore[arg-type]
    out = result["outputs"][0]
    assert out["formal_proof"] == "  trivial"


def test_parse_proof_mathd_algebra_478_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Section 3.3: mathd_algebra_478-style docstring + := by sorry; no structural extraction error."""
    formal = """/-- The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/
theorem t : True := by sorry"""
    proof = "theorem t : True := by\n  trivial"
    _install_kimina_stub(monkeypatch, _OptionBKiminaClient)
    proof_parser_agent = importlib.import_module("goedels_poetry.agents.proof_parser_agent")
    state = _base_state(formal_theorem=formal, llm_lean_output=proof)
    result = proof_parser_agent._parse_proof("url", 0, 0, state)  # type: ignore[arg-type]
    out = result["outputs"][0]
    assert out["formal_proof"] == "  trivial"
    assert "volume of a cone" in formal
