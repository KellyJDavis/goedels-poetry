from __future__ import annotations

from typing import Any, cast

# Note: This import may fail on Python 3.11 due to a compatibility issue between
# kimina-ast-client and Pydantic 2.11.9. kimina-ast-client uses typing.TypedDict
# but Pydantic 2.11.9 requires typing_extensions.TypedDict on Python < 3.12.
# This is a known issue with kimina-ast-client that needs to be fixed upstream.
# If you encounter this error, you can:
# 1. Use Python 3.10, 3.12, or 3.13 instead
# 2. Wait for kimina-ast-client to release a fix
# 3. Report the issue to kimina-ast-client maintainers
from kimina_client.models import AstModuleResponse, CheckResponse, CommandResponse, Message

from goedels_poetry.parsers.util.hypothesis_extraction import extract_hypotheses_from_unsolved_goals_data


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


def extract_hypotheses_from_check_response(parsed_check_response: dict) -> list[str]:
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
        hypotheses.extend(extract_hypotheses_from_unsolved_goals_data(data))

    if not found_unsolved_goals:
        raise ValueError('No error message contains "unsolved goals" in its data field')  # noqa: TRY003

    return hypotheses
