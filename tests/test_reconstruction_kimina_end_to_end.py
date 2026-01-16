from __future__ import annotations

from contextlib import suppress
from typing import cast

import pytest
from kimina_client import KiminaClient

from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    combine_preamble_and_body,
    compute_source_hashes,
    remove_default_imports_from_ast,
    split_preamble_and_body,
)
from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response, parse_kimina_check_response
from goedels_poetry.parsers.ast import AST
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager
from goedels_poetry.util.tree import TreeNode


def _ast_from_code(client: KiminaClient, code: str) -> AST:
    ast_response = client.ast_code(code)
    parsed = parse_kimina_ast_code_response(ast_response)
    ast_without_imports = remove_default_imports_from_ast(parsed["ast"], preamble=DEFAULT_IMPORTS)
    preamble, body = split_preamble_and_body(code)
    body_start = code.find(body, len(preamble)) if preamble and body else 0
    return AST(ast_without_imports, sorries=parsed.get("sorries"), source_text=code, body_start=body_start)


@pytest.mark.kimina
@pytest.mark.parametrize(
    "body, children",
    [
        (
            "theorem t1 : True := by\n  have h1 : True := by\n    sorry\n  exact h1\n",
            [("h1", "by\n  trivial")],
        ),
        (
            "theorem t2 : True := by\n  have hα : True := by\n    sorry\n  exact hα\n",  # noqa: RUF001
            [("hα", "by\n  trivial")],  # noqa: RUF001
        ),
        (
            "theorem t3 : True := by\n  sorry\n",
            [("<main body>", "by\n  trivial")],
        ),
    ],
)
def test_reconstruction_kimina_end_to_end(
    kimina_server_url: str,
    body: str,
    children: list[tuple[str, str]],
) -> None:
    client = KiminaClient(api_url=kimina_server_url, http_timeout=60, n_retries=1)
    code = combine_preamble_and_body(DEFAULT_IMPORTS, body)
    ast = _ast_from_code(client, code)
    body_text = ast.get_body_text() or body
    raw_hash, normalized_hash = compute_source_hashes(body_text)

    root = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem=body.splitlines()[0].strip(),
        preamble=DEFAULT_IMPORTS,
        proof_sketch=body,
        syntactic=True,
        errors=None,
        ast=ast,
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
        hole_name=None,
        hole_start=None,
        hole_end=None,
        source_hash_raw=raw_hash,
        source_hash_normalized=normalized_hash,
    )

    for name, proof_body in children:
        proof_code = f"lemma {name} : True := by\n  trivial\n"
        proof_ast = _ast_from_code(client, combine_preamble_and_body(DEFAULT_IMPORTS, proof_code))
        child = FormalTheoremProofState(
            parent=cast(TreeNode, root),
            depth=1,
            formal_theorem=f"lemma {name} : True",
            preamble=DEFAULT_IMPORTS,
            syntactic=True,
            formal_proof=proof_body,
            proved=True,
            errors=None,
            ast=proof_ast,
            self_correction_attempts=0,
            proof_history=[],
            pass_attempts=0,
            hole_name=name,
            hole_start=None,
            hole_end=None,
            source_hash_raw=None,
            source_hash_normalized=None,
        )
        root["children"].append(cast(TreeNode, child))

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(code)
    state = GoedelsPoetryState(formal_theorem=code)
    try:
        state.formal_theorem_proof = cast(TreeNode, root)
        manager = GoedelsPoetryStateManager(state)
        result = manager.reconstruct_complete_proof()

        check = client.check(result, timeout=60, show_progress=False)
        parsed_check = parse_kimina_check_response(check)
        assert not parsed_check["errors"]
        assert not parsed_check["sorries"]
    finally:
        GoedelsPoetryState.clear_theorem_directory(code)
