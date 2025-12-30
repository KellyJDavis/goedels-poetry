"""Debug utilities for logging LLM prompts, responses, and Kimina server responses."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Create console for debug output
_debug_console = Console()

# Check if debug mode is enabled
_DEBUG_ENABLED = os.environ.get("GOEDELS_POETRY_DEBUG", "").lower() in ("1", "true", "yes")


def _get_timestamp() -> str:
    """
    Get the current date and time formatted as a string.

    Returns
    -------
    str
        Formatted timestamp string (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_debug_enabled() -> bool:
    """
    Check if debug mode is enabled via the GOEDELS_POETRY_DEBUG environment variable.

    Returns
    -------
    bool
        True if debug mode is enabled, False otherwise.
    """
    return _DEBUG_ENABLED


def log_llm_prompt(agent_name: str, prompt: str, prompt_name: str | None = None) -> None:
    """
    Log an LLM prompt if debug mode is enabled.

    Parameters
    ----------
    agent_name : str
        The name of the agent (e.g., "FORMALIZER_AGENT", "PROVER_AGENT")
    prompt : str
        The prompt content to be sent to the LLM
    prompt_name : str, optional
        The name of the prompt template (e.g., "goedel-formalizer-v2"), by default None
    """
    if not _DEBUG_ENABLED:
        return

    timestamp = _get_timestamp()
    title_parts = [f"[bold yellow]{agent_name}[/bold yellow]"]
    if prompt_name:
        title_parts.append(f"prompt: {prompt_name}")
    else:
        title_parts.append("prompt")
    title_parts.append(f"[dim]{timestamp}[/dim]")
    title = " - ".join(title_parts)

    # Try to detect if prompt contains Lean code
    if "```lean" in prompt or "theorem" in prompt or "lemma" in prompt:
        # Display as Lean syntax
        syntax = Syntax(prompt, "lean", theme="monokai", line_numbers=False)
        _debug_console.print(Panel(syntax, title=title, border_style="yellow"))
    else:
        # Display as regular text
        _debug_console.print(Panel(prompt, title=title, border_style="yellow"))


def log_llm_response(agent_name: str, response: str, response_type: str = "response") -> None:
    """
    Log an LLM response if debug mode is enabled.

    Parameters
    ----------
    agent_name : str
        The name of the agent (e.g., "FORMALIZER_AGENT_LLM", "PROVER_AGENT_LLM")
    response : str
        The response content from the LLM
    response_type : str, optional
        The type of response (e.g., "response", "parsed"), by default "response"
    """
    if not _DEBUG_ENABLED:
        return

    timestamp = _get_timestamp()
    title = f"[bold cyan]{agent_name}[/bold cyan] - {response_type} - [dim]{timestamp}[/dim]"

    # Try to detect if response is Lean code
    if "```lean" in response or "theorem" in response or "lemma" in response:
        # Display as Lean syntax
        syntax = Syntax(response, "lean", theme="monokai", line_numbers=False)
        _debug_console.print(Panel(syntax, title=title, border_style="cyan"))
    else:
        # Display as regular text
        _debug_console.print(Panel(response, title=title, border_style="cyan"))


def log_kimina_response(operation: str, response: dict[str, Any]) -> None:
    """
    Log a Kimina server response if debug mode is enabled.

    Parameters
    ----------
    operation : str
        The operation performed (e.g., "check", "ast_code")
    response : dict
        The parsed response from the Kimina server
    """
    if not _DEBUG_ENABLED:
        return

    timestamp = _get_timestamp()
    title = f"[bold magenta]KIMINA_SERVER[/bold magenta] - {operation} - [dim]{timestamp}[/dim]"

    # Format the response nicely
    import json

    formatted_response = json.dumps(response, indent=2, default=str)
    syntax = Syntax(formatted_response, "json", theme="monokai", line_numbers=False)
    _debug_console.print(Panel(syntax, title=title, border_style="magenta"))


def log_vectordb_response(operation: str, response: list[Any]) -> None:
    """
    Log a vector database response if debug mode is enabled.

    Parameters
    ----------
    operation : str
        The operation performed (e.g., "search")
    response : list[dict]
        The search results from the vector database
    """
    if not _DEBUG_ENABLED:
        return

    timestamp = _get_timestamp()
    title = f"[bold green]VECTOR_DB[/bold green] - {operation} - [dim]{timestamp}[/dim]"

    # Format the response nicely
    import json

    formatted_response = json.dumps(response, indent=2, default=str)
    syntax = Syntax(formatted_response, "json", theme="monokai", line_numbers=False)
    _debug_console.print(Panel(syntax, title=title, border_style="green"))


def log_debug_message(message: str, style: str = "yellow") -> None:
    """
    Log a general debug message if debug mode is enabled.

    Parameters
    ----------
    message : str
        The debug message to log
    style : str, optional
        The style to apply to the message, by default "yellow"
    """
    if not _DEBUG_ENABLED:
        return

    timestamp = _get_timestamp()
    _debug_console.print(f"[{style}][DEBUG][/{style}] [dim]{timestamp}[/dim] {message}")
