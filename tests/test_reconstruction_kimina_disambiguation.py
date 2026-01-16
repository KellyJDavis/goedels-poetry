from __future__ import annotations

from typing import cast

import pytest
from kimina_client import KiminaClient

from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    combine_preamble_and_body,
    remove_default_imports_from_ast,
    split_preamble_and_body,
)
from goedels_poetry.agents.util.kimina_server import (
    check_code_with_infotree,
    detect_position_semantics,
    parse_kimina_ast_code_response,
)
from goedels_poetry.parsers.ast import AST
from goedels_poetry.state import GoedelsPoetryStateManager


def _ast_from_code(client: KiminaClient, code: str) -> AST:
    ast_response = client.ast_code(code)
    parsed = parse_kimina_ast_code_response(ast_response)
    ast_without_imports = remove_default_imports_from_ast(parsed["ast"], preamble=DEFAULT_IMPORTS)
    preamble, body = split_preamble_and_body(code)
    body_start = code.find(body, len(preamble)) if preamble and body else 0
    return AST(ast_without_imports, sorries=parsed.get("sorries"), source_text=code, body_start=body_start)


@pytest.mark.kimina
def test_detect_position_semantics(kimina_server_url: str) -> None:
    client = KiminaClient(api_url=kimina_server_url, http_timeout=60, n_retries=1)
    semantics = detect_position_semantics(client)
    assert "column_is_byte" in semantics
    assert "line_base" in semantics


@pytest.mark.kimina
def test_disambiguate_multiple_sorries(kimina_server_url: str) -> None:
    client = KiminaClient(api_url=kimina_server_url, http_timeout=60, n_retries=1)
    body = "theorem t : True := by\n  have h1 : True := by\n    sorry\n  have h2 : True := by\n    sorry\n  exact h2\n"
    code = combine_preamble_and_body(DEFAULT_IMPORTS, body)
    ast = _ast_from_code(client, code)
    manager = GoedelsPoetryStateManager.__new__(GoedelsPoetryStateManager)
    parsed_check = check_code_with_infotree(code, infotree=None)
    parsed = cast(
        dict,
        manager._match_check_sorries_to_ast(
            ast,
            ast.get_body_text() or body,
            cast(list, parsed_check.get("sorries", [])),
            manager.ReconstructionContext(mode_id="strict"),
            source_text=code,
        ),
    )
    assert "h1" in parsed
    assert "h2" in parsed
