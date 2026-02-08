from __future__ import annotations

from collections.abc import Callable

from kimina_client import KiminaClient

from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response


def actionable_suffix(parsed: dict, code_preview: str, *, preview_len: int = 200) -> str:
    """
    Always include something actionable: error when present, else a short preview.

    This is used in both parser agents (raised exceptions) and checker agents (correctable errors)
    to surface parse/extraction failures in a consistent, LLM-friendly way.
    """
    err = parsed.get("error")
    if err:
        return f"; error: {err}"
    return f"; preview: {code_preview[:preview_len]!r}"


def compute_body_start(normalized_preamble: str, normalized_body: str, combined: str) -> int:
    """
    Compute the body_start offset used to construct AST metadata.

    Mirrors the parser agents' logic: locate `normalized_body` within `combined` after the preamble.
    """
    if normalized_preamble and normalized_body:
        idx = combined.find(normalized_body, len(normalized_preamble))
        return idx if idx != -1 else len(normalized_preamble)
    return 0


def ast_code_parsed(
    kimina_client: KiminaClient,
    code: str,
    *,
    server_timeout: int | None = None,
    log_label: str | None = None,
    log_fn: Callable[[str, dict], None] | None = None,
) -> tuple[dict | None, str | None]:
    """
    Call Kimina `ast_code` and parse the response into our normalized dict form.

    Returns (parsed_dict, None) on success, or (None, error_repr) on exception.
    """
    try:
        if server_timeout is None:
            ast_code_response = kimina_client.ast_code(code)
        else:
            ast_code_response = kimina_client.ast_code(code, timeout=server_timeout)
    except Exception as e:
        return None, repr(e)

    parsed = parse_kimina_ast_code_response(ast_code_response)
    if log_fn is not None and log_label is not None:
        log_fn(log_label, parsed)
    return parsed, None
