from __future__ import annotations

from typing import Any, cast

from kimina_client import KiminaClient
from kimina_client.models import (
    AstModuleResponse,
    CheckResponse,
    CommandResponse,
    Infotree,
    Message,
)

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body, split_preamble_and_body
from goedels_poetry.config.kimina_server import KIMINA_LEAN_SERVER


def parse_kimina_check_response(check_response: CheckResponse) -> dict:
    """
    Parses the passed Kimina CheckResponse into a dict used by Goedel-Prover-V2

    Parameters
    ----------
    check_response: CheckResponse
        The Kimina CheckResponse to parse

    Returns
    -------
    dict
        A dict used by Goedel-Prover-V2
    """
    response: CommandResponse = cast(
        CommandResponse, check_response.results[0].response
    )  # TODO: Is this the right element?
    ast_responses: dict = {}
    parsed_response: dict = {
        "sorries": response.get("sorries", []),
        "tactics": response.get("tactics", []),
        "errors": [m for m in cast(list[Message], response.get("messages", [])) if m.get("severity") == "error"],
        "warnings": [m for m in cast(list[Message], response.get("messages", [])) if m.get("severity") == "warning"],
        "infos": [m for m in cast(list[Message], response.get("messages", [])) if m.get("severity") == "info"],
        "ast": ast_responses,
        "system_errors": None,
    }
    parsed_response["pass"] = not parsed_response["errors"]
    parsed_response["complete"] = (
        parsed_response["pass"]
        and not parsed_response["sorries"]
        and not any(
            "declaration uses 'sorry'" in warning["data"] or "failed" in warning["data"]
            for warning in parsed_response["warnings"]
        )
    )
    return parsed_response


def parse_kimina_ast_code_response(ast_code_response: AstModuleResponse) -> dict:
    """
    Parses the passed Kimina AstModuleResponse into a dict representing the response.

    Parameters
    ----------
    ast_code_response: AstModuleResponse
        The Kimina AstModuleResponse to parse

    Returns
    -------
    dict
        A dict representing the response
    """
    response = ast_code_response.results[0]  # TODO: Is this the right element?

    def _maybe_get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    error = _maybe_get(response, "error")
    parsed_response = {
        "module": _maybe_get(response, "module"),
        "error": error,
        "ast": _maybe_get(response, "ast") if error is None else None,
    }

    # Thread through goal-context data (sorries) and any additional diagnostic fields if present.
    # Kimina includes `sorries` on AST responses for incomplete proofs; downstream subgoal extraction
    # relies on this goal context to reconstruct standalone subgoals.
    sorries = _maybe_get(response, "sorries") or _maybe_get(_maybe_get(response, "response"), "sorries")
    if sorries is not None:
        parsed_response["sorries"] = sorries

    # Preserve optional diagnostic fields when available (safe, best-effort).
    for field in ("messages", "warnings", "infos", "tactics"):
        value = _maybe_get(response, field)
        if value is not None:
            parsed_response[field] = value

    return parsed_response


_POSITION_SEMANTICS: dict[str, int | bool] | None = None


def get_kimina_client() -> KiminaClient:
    return KiminaClient(
        api_url=KIMINA_LEAN_SERVER["url"],
        http_timeout=KIMINA_LEAN_SERVER["timeout"],
        n_retries=KIMINA_LEAN_SERVER["max_retries"],
    )


def detect_position_semantics(client: KiminaClient | None = None) -> dict[str, int | bool]:
    """
    Detect whether Kimina uses character-based or byte-based columns, and line base.
    """
    global _POSITION_SEMANTICS
    if _POSITION_SEMANTICS is not None:
        return _POSITION_SEMANTICS

    client = client or get_kimina_client()
    body = "theorem gp_pos_semantics : True := by\n  have h₁ : True := by\n    trivial\n  sorry\n"
    code = combine_preamble_and_body(DEFAULT_IMPORTS, body)
    check_response = client.check(code, timeout=KIMINA_LEAN_SERVER["timeout"], infotree=None, show_progress=False)
    parsed = parse_kimina_check_response(check_response)
    sorries = parsed.get("sorries", [])
    if not sorries:
        _POSITION_SEMANTICS = {"column_is_byte": False, "line_base": 1}
        return _POSITION_SEMANTICS

    pos = sorries[0].get("pos", {})
    pos_line = int(pos.get("line", 1))
    pos_col = int(pos.get("column", 0))

    _preamble, body_only = split_preamble_and_body(code)
    lines = body_only.splitlines()
    sorry_line_idx = 0
    for idx, line in enumerate(lines):
        if "sorry" in line:
            sorry_line_idx = idx
            break
    line_base = 1 if pos_line == sorry_line_idx + 1 else 0 if pos_line == sorry_line_idx else 1

    line_text = lines[sorry_line_idx] if lines else ""
    char_col = line_text.find("sorry")
    if char_col == -1:
        char_col = 0
    byte_col = len(line_text[:char_col].encode("utf-8"))

    if pos_col == byte_col and pos_col != char_col:
        column_is_byte = True
    elif pos_col == char_col and pos_col != byte_col:
        column_is_byte = False
    else:
        column_is_byte = False

    _POSITION_SEMANTICS = {"column_is_byte": column_is_byte, "line_base": line_base}
    return _POSITION_SEMANTICS


def check_code_with_infotree(
    code: str,
    *,
    infotree: Infotree | None = None,
    client: KiminaClient | None = None,
) -> dict:
    client = client or get_kimina_client()
    check_response = client.check(
        code,
        timeout=KIMINA_LEAN_SERVER["timeout"],
        infotree=infotree,
        show_progress=False,
    )
    return parse_kimina_check_response(check_response)


def extract_hypotheses_from_check_response(parsed_check_response: dict) -> list[str]:  # noqa: C901
    """
    Extract hypothesis strings from "unsolved goals" error messages in a parsed check response.

    Parameters
    ----------
    parsed_check_response: dict
        A parsed check response from parse_kimina_check_response(). Should have an "errors" key
        containing a list of Message dicts with "severity" == "error".

    Returns
    -------
    list[str]
        List of hypothesis strings in order (e.g., ["n : Z", "hn : n > 1"]).
        The order is preserved from all "unsolved goals" messages in the order they appear.

    Raises
    ------
    ValueError
        If no error message contains "unsolved goals" in its "data" field.
    """
    hypotheses: list[str] = []
    found_unsolved_goals = False

    errors = parsed_check_response.get("errors", [])
    if not isinstance(errors, list):
        raise TypeError("parsed_check_response['errors'] must be a list")  # noqa: TRY003

    # Search all error messages in order
    for error_msg in errors:
        if not isinstance(error_msg, dict):
            continue
        if error_msg.get("severity") != "error":
            continue

        data = error_msg.get("data", "")
        if not isinstance(data, str):
            continue

        # Check if this is an "unsolved goals" message
        if not data.startswith("unsolved goals"):
            continue

        found_unsolved_goals = True
        lines = data.splitlines()

        # Process lines after "unsolved goals" (first line)
        for line in lines[1:]:
            stripped = line.strip()
            # Stop at the goal line (starts with ⊢ or \u22a2)
            if stripped.startswith("⊢") or stripped.startswith("\u22a2"):
                break
            # Skip blank lines for robustness (kimina server shouldn't produce them, but be safe)
            if stripped:
                hypotheses.append(stripped)

    if not found_unsolved_goals:
        raise ValueError('No error message contains "unsolved goals" in its data field')  # noqa: TRY003

    return hypotheses
