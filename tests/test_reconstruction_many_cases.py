from __future__ import annotations

from contextlib import suppress
from typing import cast

import pytest

from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager
from goedels_poetry.util.tree import TreeNode


def _build_case(idx: int) -> tuple[DecomposedFormalTheoremState, list[FormalTheoremProofState]]:
    names = ["h1", "h2", "h₁", "hα", "hβ"]  # noqa: RUF001
    name = names[idx % len(names)]
    mode = idx % 3
    theorem_name = f"t_case_{idx}"

    if mode == 0:
        sketch = f"theorem {theorem_name} : True := by\n  have {name} : True := by\n    sorry\n  exact {name}\n"
        children = [
            FormalTheoremProofState(
                parent=None,
                depth=1,
                formal_theorem=f"lemma {name} : True",
                preamble=DEFAULT_IMPORTS,
                syntactic=True,
                formal_proof="by\n  trivial",
                proved=True,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name=name,
                hole_start=None,
                hole_end=None,
                source_hash_raw=None,
                source_hash_normalized=None,
            )
        ]
    elif mode == 1:
        name2 = f"{name}_2"
        sketch = (
            f"theorem {theorem_name} : True := by\n"
            f"  have {name} : True := by\n"
            f"    sorry\n"
            f"  have {name2} : True := by\n"
            f"    sorry\n"
            f"  exact {name2}\n"
        )
        children = [
            FormalTheoremProofState(
                parent=None,
                depth=1,
                formal_theorem=f"lemma {name} : True",
                preamble=DEFAULT_IMPORTS,
                syntactic=True,
                formal_proof="by\n  trivial",
                proved=True,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name=name,
                hole_start=None,
                hole_end=None,
                source_hash_raw=None,
                source_hash_normalized=None,
            ),
            FormalTheoremProofState(
                parent=None,
                depth=1,
                formal_theorem=f"lemma {name2} : True",
                preamble=DEFAULT_IMPORTS,
                syntactic=True,
                formal_proof="by\n  trivial",
                proved=True,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name=name2,
                hole_start=None,
                hole_end=None,
                source_hash_raw=None,
                source_hash_normalized=None,
            ),
        ]
    else:
        sketch = f"theorem {theorem_name} : True := by\n  sorry\n"
        children = [
            FormalTheoremProofState(
                parent=None,
                depth=1,
                formal_theorem=f"lemma {theorem_name}_main : True",
                preamble=DEFAULT_IMPORTS,
                syntactic=True,
                formal_proof="by\n  trivial",
                proved=True,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name="<main body>",
                hole_start=None,
                hole_end=None,
                source_hash_raw=None,
                source_hash_normalized=None,
            )
        ]

    root = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem=f"theorem {theorem_name} : True",
        preamble=DEFAULT_IMPORTS,
        proof_sketch=sketch,
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
        hole_name=None,
        hole_start=None,
        hole_end=None,
        source_hash_raw=None,
        source_hash_normalized=None,
    )

    for child in children:
        child["parent"] = cast(TreeNode, root)
    root["children"] = [cast(TreeNode, child) for child in children]
    return root, children


@pytest.mark.parametrize("idx", list(range(60)))
def test_reconstruction_many_cases(idx: int) -> None:
    root, _children = _build_case(idx)
    formal = combine_preamble_and_body(DEFAULT_IMPORTS, str(root["formal_theorem"]))
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(formal)
    state = GoedelsPoetryState(formal_theorem=formal)
    try:
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)
        result = manager.reconstruct_complete_proof()
        body = result[len(DEFAULT_IMPORTS) :]
        assert "sorry" not in body
    finally:
        GoedelsPoetryState.clear_theorem_directory(formal)
