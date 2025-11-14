from __future__ import annotations

import os
import re
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


_MANDATORY_PREAMBLE_LINES: tuple[str, ...] = ("set_option maxHeartbeats 0",)


def _count_preamble_commands(preamble: str) -> int:
    """Count non-empty, non-comment lines in a preamble block."""
    if not preamble:
        return 0
    return sum(1 for line in preamble.splitlines() if line.strip() and not line.strip().startswith("--"))


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


def _is_doc_comment(lines: list[str], i: int) -> bool:
    """Check if a comment at line i is a doc comment (precedes a declaration)."""
    for j in range(i + 1, len(lines)):
        next_stripped = lines[j].strip()
        if next_stripped != "":
            return _is_lean_declaration_line(next_stripped)
    return False


def _handle_complete_comment_line(line: str, lines: list[str], line_idx: int) -> tuple[int | None, bool]:
    """
    Handle a line that contains a complete comment (/- ... -/).

    Returns
    -------
    tuple[int | None, bool]
        (split_index if body starts here, should_continue)
    """
    stripped = line.strip()
    comment_start = stripped.find("/-")
    comment_end = stripped.find("-/", comment_start)
    if comment_end == -1:
        return None, False

    # Found a complete comment on this line
    if _is_doc_comment(lines, line_idx):
        # This doc comment precedes a declaration, so it's part of the body
        return line_idx, False

    # Not a doc comment - process the rest of the line after the comment
    after_comment = stripped[comment_end + 2 :].strip()
    if after_comment:
        # There's content after the comment, process it
        if _is_lean_declaration_line(after_comment):
            return None, False  # Stop processing
        if _is_preamble_line(after_comment):
            return line_idx + 1, True  # Continue, skip this line
        return None, False  # Stop processing
    # Just a comment, skip this line
    return line_idx + 1, True  # Continue, skip this line


def _handle_multiline_comment(stripped: str, in_multiline_comment: bool, line_idx: int) -> tuple[bool, int]:
    """
    Handle multiline comment state.

    Returns
    -------
    tuple[bool, int]
        (new_in_multiline_comment, skip_until)
    """
    if stripped.startswith("/-") and not in_multiline_comment:
        return True, line_idx + 1

    if in_multiline_comment:
        if stripped.endswith("-/") or stripped == "-/":
            return False, line_idx + 1
        return True, line_idx + 1

    return in_multiline_comment, line_idx


def _process_preamble_line(stripped: str, lines: list[str], line_idx: int) -> tuple[int | None, bool]:
    """
    Process a single line to determine if it's part of the preamble.

    Returns
    -------
    tuple[int | None, bool]
        (split_index if body starts here, is_preamble_line)
    """
    # Check if this is actual Lean code
    if _is_lean_declaration_line(stripped):
        return None, False

    # Check if this is a doc comment (single-line -- style)
    if stripped.startswith("--"):
        if _is_doc_comment(lines, line_idx):
            return line_idx, False  # Body starts here
        return None, True  # Preamble line

    # Check if this is a preamble line
    if _is_preamble_line(stripped):
        return None, True  # Preamble line

    return None, False  # Not a preamble line


def _find_preamble_end(lines: list[str]) -> int:
    """Find the line index where the preamble ends."""
    skip_until = 0
    in_multiline_comment = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle complete single-line comments
        if "/-" in stripped:
            result, should_continue = _handle_complete_comment_line(stripped, lines, i)
            if result is not None and not should_continue:
                return result
            if should_continue:
                skip_until = result if result is not None else skip_until
                continue
            if result is None:
                # Starts with /- but no closing -/ on this line - multiline comment
                in_multiline_comment, skip_until = _handle_multiline_comment(stripped, in_multiline_comment, i)
                continue

        in_multiline_comment, new_skip = _handle_multiline_comment(stripped, in_multiline_comment, i)
        if in_multiline_comment or new_skip != i:
            skip_until = new_skip
            continue

        # Process the line to determine if it's preamble or body
        split_idx, is_preamble = _process_preamble_line(stripped, lines, i)
        if split_idx is not None:
            return split_idx
        if is_preamble:
            skip_until = i + 1
        else:
            break

    return skip_until


def _normalize_block(block: str) -> str:
    """Normalize a multi-line string for comparison."""
    if not block:
        return ""
    lines = block.splitlines()
    normalized = [line.rstrip() for line in lines]
    return "\n".join(normalized).strip()


def split_preamble_and_body(code: str) -> tuple[str, str]:
    """Split Lean code into preamble and body parts."""
    if not code:
        return "", ""

    lines = code.split("\n")
    split_index = _find_preamble_end(lines)

    preamble_lines = lines[:split_index]
    body_lines = lines[split_index:]

    preamble = "\n".join(preamble_lines).strip("\n")
    body = "\n".join(body_lines).strip()

    return preamble, body


def combine_preamble_and_body(preamble: str, body: str) -> str:
    """Combine a preamble and body into Lean code."""
    normalized_preamble = preamble.strip()
    normalized_body = body.strip()

    if not normalized_preamble:
        return normalized_body
    if not normalized_body:
        return normalized_preamble

    return f"{normalized_preamble}\n\n{normalized_body}"


def strip_known_preamble(code: str, expected_preamble: str) -> tuple[str, bool]:
    """Remove a known preamble from code if it matches after normalization."""
    preamble, body = split_preamble_and_body(code)
    if _normalize_block(preamble) == _normalize_block(expected_preamble):
        return body, True

    if not preamble.strip() and not _normalize_block(expected_preamble):
        return body, True

    return code, False


def ensure_mandatory_preamble(preamble: str) -> str:
    """Ensure required Lean directives are present in a preamble."""
    lines = preamble.split("\n") if preamble else []
    existing = {line.strip() for line in lines if line.strip()}
    additions = [line for line in _MANDATORY_PREAMBLE_LINES if line not in existing]

    if not additions:
        return preamble

    if lines and lines[-1].strip():
        lines.append("")

    lines.extend(additions)
    return "\n".join(lines)


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
    preamble, body = split_preamble_and_body(code)
    if preamble.strip():
        return body
    return code


def remove_default_imports_from_ast(ast: dict[str, Any] | None, preamble: str = DEFAULT_IMPORTS) -> dict[str, Any]:
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

    skip_default_imports = _normalize_block(preamble) == _normalize_block(DEFAULT_IMPORTS)
    num_imports_to_skip = _count_preamble_commands(DEFAULT_IMPORTS) if skip_default_imports else 0

    # The AST is a dict. If it contains a list of commands, we need to skip
    # the ones that correspond to DEFAULT_IMPORTS.
    # DEFAULT_IMPORTS currently expands to four commands (two imports, a set_option, and an open statement).
    # We compute the exact count dynamically so updates to DEFAULT_IMPORTS stay in sync.

    # Check if the AST is a list at the top level (older format)
    if isinstance(ast, list):
        if len(ast) > num_imports_to_skip:
            return {"commands": ast[num_imports_to_skip:]}
        return {"commands": ast}

    # If it's a dict with a "commands" key, filter that
    if "commands" in ast and isinstance(ast["commands"], list):
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
    if not code_lines:
        code_lines = [""]

    def clamp_line(idx: int) -> int:
        return max(0, min(idx, len(code_lines) - 1))

    def clamp_col(idx: int, line: str) -> int:
        return max(0, min(idx, len(line)))

    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        raw_start_line = error["pos"]["line"] + 2  # Kimina requires +2
        start_line = clamp_line(raw_start_line)
        start_col = error["pos"]["column"]

        if error.get("endPos", None) is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            raw_end_line = error["endPos"]["line"] + 2
            end_line = clamp_line(raw_end_line)
            end_line = max(end_line, start_line)
            end_col = error["endPos"]["column"]

        start_col = clamp_col(start_col, code_lines[start_line])
        end_col = clamp_col(end_col, code_lines[end_line])
        if end_line == start_line and end_col < start_col:
            end_col = start_col

        err_str += f"\nError {i + 1}:\n"
        err_str += "\nCorresponding Code:\n```lean4\n"

        error_code = ""

        for ii in range(-4, 0):
            line_idx = start_line + ii
            if 0 <= line_idx < len(code_lines):
                error_code += f"{code_lines[line_idx]}\n"

        if start_line != end_line:
            start_line_text = code_lines[start_line]
            error_code += start_line_text[:start_col] + "<error>" + start_line_text[start_col:] + "\n"

            middle_indices = [idx for idx in range(start_line + 1, end_line) if 0 <= idx < len(code_lines)]
            if not error_thres:
                for idx in middle_indices:
                    error_code += f"{code_lines[idx]}\n"
            else:
                show_line = 6
                for idx in middle_indices[:show_line]:
                    error_code += f"{code_lines[idx]}\n"
                if len(middle_indices) > show_line:
                    last_shown_idx = middle_indices[show_line - 1] if show_line > 0 else middle_indices[0]
                    last_line_text = code_lines[last_shown_idx]
                    leading_spaces = len(last_line_text) - len(last_line_text.lstrip(" "))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"

            end_line_text = code_lines[end_line]
            error_code += end_line_text[:end_col] + "</error>" + end_line_text[end_col:] + "\n"
        else:
            line_text = code_lines[start_line]
            error_code += (
                line_text[:start_col]
                + "<error>"
                + line_text[start_col:end_col]
                + "</error>"
                + line_text[end_col:]
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


def combine_theorem_with_proof(theorem_statement: str, proof_body: str) -> str:
    """
    Combine a theorem statement (with `:= by sorry` or `:= sorry`) with a proof body.

    Parameters
    ----------
    theorem_statement: str
        The theorem statement that ends with `:= by sorry` or `:= sorry`
    proof_body: str
        The proof body (tactics after `:= by`, already properly indented)

    Returns
    -------
    str
        The theorem statement with `sorry` replaced by the proof body
    """
    if not proof_body:
        return theorem_statement

    # Try := by sorry first (most common pattern in MOBench files)
    pattern1 = r":=(\s*)by(\s+)sorry"
    match1 = re.search(pattern1, theorem_statement, re.DOTALL)
    if match1:
        before = theorem_statement[: match1.start()]
        after = theorem_statement[match1.end() :]
        whitespace_before_by = match1.group(1)
        # Preserve whitespace before "by", add newline after "by" for proof body
        return f"{before}:={whitespace_before_by}by\n{proof_body}{after}"

    # Try := sorry (without "by") - used in compfiles
    # Prefer theorem/example declarations over def/abbrev when multiple := sorry patterns exist
    theorem_sorry_pattern = r"(theorem|example)\s+[a-zA-Z0-9_']+.*?:=\s*sorry"
    theorem_sorry_match = re.search(theorem_sorry_pattern, theorem_statement, re.DOTALL)
    if theorem_sorry_match:
        # Found a theorem/example with := sorry, replace that one
        # Find := sorry within the matched declaration (pattern ensures it's at the end)
        decl_text = theorem_sorry_match.group(0)
        sorry_match = re.search(r":=\s*sorry\s*$", decl_text, re.MULTILINE)
        if sorry_match:
            decl_start = theorem_sorry_match.start()
            before = theorem_statement[: decl_start + sorry_match.start()]
            after = theorem_statement[theorem_sorry_match.end() :]
            return f"{before}:= by\n{proof_body}{after}"

    # Fallback: find any := sorry pattern
    pattern2 = r":=(\s+)sorry"
    match2 = re.search(pattern2, theorem_statement, re.DOTALL)
    if match2:
        before = theorem_statement[: match2.start()]
        after = theorem_statement[match2.end() :]
        return f"{before}:= by\n{proof_body}{after}"

    # Fallback: append after := by if present
    if re.search(r":=\s*by\s*$", theorem_statement, re.MULTILINE):
        return f"{theorem_statement}\n{proof_body}"

    # Last resort: append
    return f"{theorem_statement}\n{proof_body}"
