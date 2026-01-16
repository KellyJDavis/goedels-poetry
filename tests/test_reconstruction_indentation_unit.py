from __future__ import annotations

from goedels_poetry.parsers.ast import AST
from goedels_poetry.state import GoedelsPoetryStateManager


def test_layout_sensitive_rejects_misaligned_indent() -> None:
    sketch = "have h : True := by sorry\n"
    start = sketch.index("sorry")
    end = start + len("sorry")
    layout_ast = AST({"kind": "calc"}, source_text="by", body_start=0)
    proof_body = "calc\n  1 = 1 := by\n      rfl"
    manager = GoedelsPoetryStateManager.__new__(GoedelsPoetryStateManager)
    mode = manager._reconstruction_mode_strict()
    context = manager.ReconstructionContext(mode_id=mode.mode_id)
    replacement = manager._format_body_for_hole(sketch, start, end, proof_body, mode, context, {"ast": layout_ast})
    assert replacement is None
    assert context.layout_sensitive is True


def test_normalization_only_on_uniform_indent() -> None:
    sketch = "have h : True := by sorry\n"
    start = sketch.index("sorry")
    end = start + len("sorry")
    proof_body = "  have h1 : True := by\n  trivial\n  exact h1"
    manager = GoedelsPoetryStateManager.__new__(GoedelsPoetryStateManager)
    mode = manager._reconstruction_mode_strict()
    context = manager.ReconstructionContext(mode_id=mode.mode_id)
    replacement = manager._format_body_for_hole(sketch, start, end, proof_body, mode, context, {"ast": None})
    assert replacement is not None
    assert "\n  trivial" in replacement
