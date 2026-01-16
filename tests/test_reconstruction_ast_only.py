from __future__ import annotations

from typing import Any

from goedels_poetry.parsers.ast import AST
from goedels_poetry.state import GoedelsPoetryStateManager


def _make_token(val: str, start_b: int, end_b: int) -> dict[str, Any]:
    return {"val": val, "info": {"pos": [start_b, end_b]}}


def test_decl_subtree_detection() -> None:
    ast_dict = {"kind": "Lean.Parser.Command.declValSimple", "args": []}
    ast = AST(ast_dict, source_text="theorem t : True := by\n  trivial\n", body_start=0)
    manager = GoedelsPoetryStateManager.__new__(GoedelsPoetryStateManager)
    subtree = manager._find_decl_subtree(ast.get_ast())
    assert subtree is not None
    assert subtree.get("kind") == "Lean.Parser.Command.declValSimple"


def test_boundary_mismatch_rejects_extraction() -> None:
    source = "theorem t : True := by\n  exact trivial\n"
    by_token = _make_token("by", source.index("by"), source.index("by") + 2)
    # Mismatched token span ("exact" token points to wrong location)
    bad_token = _make_token("exact trivial", 0, 5)
    tactic_seq = {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [bad_token]}
    by_tactic = {"kind": "Lean.Parser.Term.byTactic", "args": [by_token, tactic_seq]}
    ast = AST({"kind": "Lean.Parser.Command.declValSimple", "args": [by_tactic]}, source_text=source, body_start=0)
    manager = GoedelsPoetryStateManager.__new__(GoedelsPoetryStateManager)
    mode = manager._reconstruction_mode_strict()
    context = manager.ReconstructionContext(mode_id=mode.mode_id)
    body = manager._extract_proof_body_from_ast(ast, mode, context)
    assert body is None
    assert context.boundary_mismatch is True
