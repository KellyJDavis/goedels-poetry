from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS


def _named_have(name: str, prop: str) -> dict:
    """Minimal named `have` with a sorry proof."""
    return {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": name, "info": {"leading": "", "trailing": " "}}],
                            }
                        ],
                    },
                    {"val": ":", "info": {"leading": "", "trailing": " "}},
                    {"val": prop, "info": {"leading": "", "trailing": " "}},
                ],
            },
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


def _theorem_with_subgoals() -> dict:
    """A minimal theorem whose proof contains the subgoals we want to extract."""
    haves = [
        _named_have("hn_nonneg", "0 ≤ n"),
        _named_have("hm_coe", "m = n"),
        _named_have("hm_gt", "m > 1"),
        _named_have("nat_version", "True"),
    ]
    return {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {"kind": "Lean.Parser.Tactic.tacticSeq", "args": haves},
                ],
            },
        ],
    }


def _sorries() -> list[dict]:
    # Goal contexts mirroring the A303656-style decomposition seen in logs.
    return [
        {"goal": "n : Int\nhn : n > 1\n⊢ 0 ≤ n"},
        {"goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm : Nat := n.toNat\n⊢ ↑m = n"},
        {
            "goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm : Nat := n.toNat\nhm_coe : ↑m = n\n⊢ m > 1",
        },
        {
            "goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm✝ : Nat := n.toNat\nhm_coe : ↑m✝ = n\nhm_gt : m✝ > 1\nm : Nat\nhm : m > 1\n⊢ ∃ a b c d, ↑m = ↑a ^ 2 + ↑b ^ 2 + 3 ^ c + 5 ^ d",
        },
    ]


def _clear_modules(monkeypatch: pytest.MonkeyPatch, module_names: list[str]) -> None:
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _install_kimina_stub(monkeypatch: pytest.MonkeyPatch, kimina_client_cls: type) -> None:
    """
    Install a stub kimina_client module (and its .models) before importing agents to
    avoid pulling the real dependency (which requires Python 3.12+ TypedDict).
    Also clears previously imported agent modules so they reload with the stub.
    """
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

    class _StubAstModuleResponse:  # minimal placeholder
        def __init__(self, results: Any) -> None:
            self.results = results

    models_mod.AstModuleResponse = _StubAstModuleResponse
    models_mod.CheckResponse = object
    models_mod.CommandResponse = dict
    models_mod.Message = dict

    monkeypatch.setitem(sys.modules, "kimina_client", kimina_mod)
    monkeypatch.setitem(sys.modules, "kimina_client.models", models_mod)


class _DummyAstResult:
    def __init__(self) -> None:
        self.module = "dummy"
        self.error = None
        self.ast = _theorem_with_subgoals()
        self.sorries = _sorries()


class _DummyAstResponse:
    def __init__(self) -> None:
        self.results = [_DummyAstResult()]


def test_parse_ast_code_response_threads_sorries(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_kimina_stub(monkeypatch, _DummyKiminaClient)
    kimina_server = importlib.import_module("goedels_poetry.agents.util.kimina_server")
    parsed = kimina_server.parse_kimina_ast_code_response(_DummyAstResponse())
    assert "sorries" in parsed
    assert parsed["sorries"] == _sorries()


class _DummyKiminaClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # signature-compatible
        pass

    def ast_code(self, _code: str) -> _DummyAstResponse:
        return _DummyAstResponse()


def _assert_binders_present(code: str) -> None:
    assert "(n : Int)" in code
    assert "(hn : n > 1)" in code


def test_sketch_parser_passes_goal_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_kimina_stub(monkeypatch, _DummyKiminaClient)
    sketch_parser_agent = importlib.import_module("goedels_poetry.agents.sketch_parser_agent")

    state = {
        "parent": None,
        "children": [],
        "depth": 0,
        "formal_theorem": "theorem A303656 : True := by sorry",
        "preamble": DEFAULT_IMPORTS,
        "proof_sketch": "theorem A303656 : True := by sorry",
        "syntactic": True,
        "errors": "",
        "ast": None,
        "self_correction_attempts": 0,
        "decomposition_history": [],
        "search_queries": None,
        "search_results": None,
        "hole_name": None,
        "hole_start": None,
        "hole_end": None,
    }

    result = sketch_parser_agent._parse_sketch("url", 0, 0, state)  # type: ignore[arg-type]
    assert result["outputs"][0]["ast"] is not None

    code_hn = result["outputs"][0]["ast"].get_named_subgoal_code("hn_nonneg")
    _assert_binders_present(code_hn)


def test_proof_parser_passes_goal_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_kimina_stub(monkeypatch, _DummyKiminaClient)
    proof_parser_agent = importlib.import_module("goedels_poetry.agents.proof_parser_agent")

    state = {
        "parent": None,
        "depth": 0,
        "formal_theorem": "theorem A303656 : True := by sorry",
        "preamble": DEFAULT_IMPORTS,
        "syntactic": True,
        "formal_proof": "",
        "proved": False,
        "errors": "",
        "ast": None,
        "self_correction_attempts": 0,
        "proof_history": [],
        "pass_attempts": 0,
        "hole_name": None,
        "hole_start": None,
        "hole_end": None,
    }

    result = proof_parser_agent._parse_proof("url", 0, 0, state)  # type: ignore[arg-type]
    assert result["outputs"][0]["ast"] is not None

    code_hn = result["outputs"][0]["ast"].get_named_subgoal_code("hn_nonneg")
    _assert_binders_present(code_hn)
