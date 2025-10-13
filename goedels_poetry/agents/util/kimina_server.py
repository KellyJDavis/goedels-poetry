from __future__ import annotations

import re
from typing import cast

from kimina_client.models import AstModuleResponse, CheckResponse, CommandResponse, Message

from goedels_poetry.agents.util.common import LLMParsingError


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


def parse_semantic_check_response(response: str) -> str:
    """
    Parses the passed symantic response into a string used by  Goedel-Prover-V2

    Parameters
    ----------
    response: str
        The semantic check response from the server.

    Returns
    -------
    str:
        The parsed judgement string.

    Raises
    ------
    LLMParsingError
        If no judgement is found in the response.
    """
    pattern = r"Judgement:\s*(.+)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    if not matches:
        raise LLMParsingError("Failed to extract judgement from LLM response", response)  # noqa: TRY003
    return cast(str, matches[-1]).strip()


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
    parsed_response = {
        "module": response.module,
        "error": response.error,
        "ast": response.ast if response.error is None else None,
    }
    return parsed_response
