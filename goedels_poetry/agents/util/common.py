from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


class LLMParsingError(Exception):
    """
    Exception raised when the LLM returns a response that cannot be parsed.

    This typically occurs when the LLM fails to return code in the expected format
    (e.g., missing code blocks or malformed responses).
    """

    def __init__(self, message: str, response: str) -> None:
        """
        Initialize the LLMParsingError.

        Parameters
        ----------
        message : str
            A short description of what failed to parse
        response : str
            The full LLM response that failed to parse
        """
        self.response = response
        super().__init__(f"{message}: {response}")


# Create Environment for loading prompts
_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../../data/prompts")),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


DEFAULT_IMPORTS = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
)


def _is_lean_declaration_line(s: str) -> bool:
    """Check if a line is a Lean declaration."""
    return (
        s.startswith("theorem ")
        or s.startswith("def ")
        or s.startswith("lemma ")
        or s.startswith("example ")
        or s.startswith("axiom ")
        or s.startswith("inductive ")
        or s.startswith("structure ")
        or s.startswith("class ")
    )


def _is_preamble_line(s: str) -> bool:
    """Check if a line is a preamble line (import, open, etc.)."""
    return (
        s.startswith("import ")
        or s.startswith("open ")
        or s.startswith("set_option ")
        or s.startswith("noncomputable ")
        or s == ""
    )


def _has_additional_preamble(remainder: str) -> bool:
    """Check if remainder starts with additional preamble lines."""
    return any(
        remainder.startswith(prefix) for prefix in ["import ", "open ", "set_option ", "noncomputable ", "--", "/-"]
    )


def _is_doc_comment(lines: list[str], i: int) -> bool:
    """Check if a comment at line i is a doc comment (precedes a declaration)."""
    for j in range(i + 1, len(lines)):
        next_stripped = lines[j].strip()
        if next_stripped != "":
            return _is_lean_declaration_line(next_stripped)
    return False


def _find_preamble_end(lines: list[str]) -> int:
    """Find the line index where the preamble ends."""
    skip_until = 0
    in_multiline_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle multiline comments
        if stripped.startswith("/-"):
            in_multiline_comment = True
            skip_until = i + 1
            continue
        if in_multiline_comment:
            if stripped.endswith("-/") or stripped == "-/":
                in_multiline_comment = False
            skip_until = i + 1
            continue

        # Check if this is actual Lean code
        if _is_lean_declaration_line(stripped):
            break

        # Check if this is a doc comment
        if stripped.startswith("--"):
            if _is_doc_comment(lines, i):
                break
            skip_until = i + 1
            continue

        # Check if this is a preamble line
        if _is_preamble_line(stripped):
            skip_until = i + 1
        else:
            break

    return skip_until


def add_default_imports(code: str) -> str:
    """
    Add DEFAULT_IMPORTS prefix to the given code.

    Parameters
    ----------
    code: str
        The code to add DEFAULT_IMPORTS to.

    Returns
    -------
    str
        The code with DEFAULT_IMPORTS prefix.
    """
    return DEFAULT_IMPORTS + code


def remove_default_imports(code: str) -> str:
    """
    Remove DEFAULT_IMPORTS prefix from the given code (up to whitespace).
    Also removes common variations of import preambles that LLMs might generate.

    Parameters
    ----------
    code: str
        The code to remove DEFAULT_IMPORTS from.

    Returns
    -------
    str
        The code without DEFAULT_IMPORTS prefix.
    """
    normalized_imports = DEFAULT_IMPORTS.strip()
    normalized_code = code.strip()

    # First, try exact match with DEFAULT_IMPORTS only if there's nothing after it
    # that looks like additional preamble
    if normalized_code.startswith(normalized_imports):
        remainder = normalized_code[len(normalized_imports) :].strip()
        if remainder and not _has_additional_preamble(remainder):
            return remainder

    # Try to remove all preamble patterns
    lines = normalized_code.split("\n")
    skip_until = _find_preamble_end(lines)

    if skip_until > 0:
        result = "\n".join(lines[skip_until:]).strip()
        if result and not result.startswith("--"):
            return result

    return code


def remove_default_imports_from_ast(ast: dict[str, Any] | None) -> dict[str, Any]:
    """
    Remove DEFAULT_IMPORTS related nodes from the parsed AST.

    The AST returned by Kimina includes all the declarations from DEFAULT_IMPORTS.
    We need to remove the import statements and declarations that come from DEFAULT_IMPORTS.

    Parameters
    ----------
    ast: dict[str, Any] | None
        The AST to remove DEFAULT_IMPORTS from.

    Returns
    -------
    dict[str, Any]
        The AST without DEFAULT_IMPORTS nodes. If None, returns empty dict.
    """
    if ast is None:
        return {}

    # The AST is a dict. If it contains a list of commands, we need to skip
    # the ones that correspond to DEFAULT_IMPORTS.
    # DEFAULT_IMPORTS contains:
    # 1. import Mathlib
    # 2. import Aesop
    # 3. set_option maxHeartbeats 0
    # 4. open BigOperators Real Nat Topology Rat
    # So we skip the first 4 commands

    # Check if the AST is a list at the top level (older format)
    if isinstance(ast, list):
        num_imports_to_skip = 4
        if len(ast) > num_imports_to_skip:
            return {"commands": ast[num_imports_to_skip:]}
        return {"commands": ast}

    # If it's a dict with a "commands" key, filter that
    if "commands" in ast and isinstance(ast["commands"], list):
        num_imports_to_skip = 4
        filtered_ast = ast.copy()
        if len(ast["commands"]) > num_imports_to_skip:
            filtered_ast["commands"] = ast["commands"][num_imports_to_skip:]
        return filtered_ast

    # Otherwise, return as-is (might be a different AST structure or already filtered)
    return ast


def load_prompt(name: str, **kwargs: str) -> str:
    """
    Load a template from the prompts directory and renders
    it with the given kwargs.

    Parameters
    ----------
    name: str
        The name of the template to load, without the .md extension.
    **kwargs: dict
        The kwargs to render the template with.

    Returns
    -------
    str
        The rendered template.
    """
    return _env.get_template(f"{name}.md").render(**kwargs)


def get_error_str(code: str, errors: list[dict], error_thres: bool) -> str:  # noqa: C901
    """
    Given the code and errors from the previous proof attempt, this function returns a string
    summarizing the error. This string is in the formate expected by Goedel-Prover-V2.

    Parameters
    ----------
    code: str
        The code from the previous proof attempt
    errors: list[dict]
        A list of dicts in the the errors member format returned from parse_kimina_check_response()
    error_thres: bool
        A bool indicating if the number of errors should be capped at 8

    Returns
    -------
    str:
        A string summarizing the errors in a format expected by Goedel-Prover-V2.
    """
    err_str = ""
    code_lines = code.split("\n")
    # token_lengths = [len(line) + 1 for line in code_lines]

    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error["pos"]["line"] + 2  # Originally -1 was here, but Kimina requires +2
        start_col = error["pos"]["column"]

        if error.get("endPos", None) is None:  # Originally get() wasn't used by Kimina requires it
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error["endPos"]["line"] + 2  # Originally -1 was here, but Kimina requires +2
            end_col = error["endPos"]["column"]

        # start_char_pos = sum(token_lengths[:start_line]) + start_col
        # end_char_pos = sum(token_lengths[:end_line]) + end_col

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"

            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(" "))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"

            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            error_code += (
                code_lines[start_line][:start_col]
                + "<error>"
                + code_lines[start_line][start_col:end_col]
                + "</error>"
                + code_lines[start_line][end_col:]
                + "\n"
            )
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"

        err_str += error_code
        err_str += "\n```\n"
        err_str += f"\nError Message: {error['data']}\n"

    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"

    return err_str
