from __future__ import annotations

import os
import re
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from goedels_poetry.agents.state import APISearchResponseTypedDict


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
    return sum(1 for line in preamble.splitlines() if line.strip())


_DECLARATION_KEYWORDS: tuple[str, ...] = (
    "theorem",
    "lemma",
    "def",
    "definition",
    "example",
    "instance",
    "inductive",
    "coinductive",
    "structure",
    "class",
    "abbrev",
    "opaque",
    "constant",
    "axiom",
    "mutual",
    "deriving",
)

_DECLARATION_MODIFIERS: tuple[str, ...] = (
    "private",
    "protected",
    "unsafe",
    "scoped",
    "local",
    "noncomputable",
    "partial",
)


def normalize_escape_sequences(content: str) -> str:
    """
    Convert literal escape sequences to their actual characters.

    This function converts common escape sequences that appear as literal
    two-character sequences (e.g., \\n, \\t) in strings to their actual
    character equivalents (newline, tab, etc.). This is necessary because
    some input sources may contain escape sequences as literal text rather than
    actual escape sequences.

    The function handles escaped backslashes (\\\\ -> \\) correctly by
    processing them appropriately to avoid double conversion.

    Parameters
    ----------
    content: str
        The content that may contain literal escape sequences

    Returns
    -------
    str
        The content with escape sequences converted to actual characters
    """
    # Use a character-based approach to handle escaped backslashes correctly
    # Process character by character to distinguish \\n from \n
    result = []
    i = 0
    while i < len(content):
        if content[i] == "\\" and i + 1 < len(content):
            next_char = content[i + 1]
            if next_char == "\\":
                # Escaped backslash: preserve as single backslash
                result.append("\\")
                i += 2
            elif next_char == "n":
                # Literal \n: convert to actual newline
                result.append("\n")
                i += 2
            elif next_char == "t":
                # Literal \t: convert to actual tab
                result.append("\t")
                i += 2
            elif next_char == "r":
                # Literal \r: convert to actual carriage return
                result.append("\r")
                i += 2
            else:
                # Backslash followed by something else: keep as-is
                result.append(content[i])
                i += 1
        else:
            result.append(content[i])
            i += 1
    return "".join(result)


def _normalize_block(block: str) -> str:
    """Normalize a multi-line string for comparison."""
    if not block:
        return ""
    lines = block.splitlines()
    normalized = [line.rstrip() for line in lines]
    return "\n".join(normalized).strip()


def _is_identifier_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_" or ch == "'"


def _has_word_prefix(code: str, idx: int) -> bool:
    return idx > 0 and _is_identifier_char(code[idx - 1])


def _has_word_suffix(code: str, idx: int) -> bool:
    return idx < len(code) and _is_identifier_char(code[idx])


def _starts_with_keyword(code: str, idx: int, keyword: str) -> bool:
    if not code.startswith(keyword, idx):
        return False
    end_idx = idx + len(keyword)
    if _has_word_prefix(code, idx):
        return False
    return not _has_word_suffix(code, end_idx)


def _skip_whitespace(code: str, idx: int) -> int:
    while idx < len(code) and code[idx] in " \t\r\n":
        idx += 1
    return idx


def _skip_line(code: str, idx: int) -> int:
    while idx < len(code) and code[idx] != "\n":
        idx += 1
    if idx < len(code):
        idx += 1
    return idx


def _skip_line_comment(code: str, idx: int) -> int:
    return _skip_line(code, idx)


def _handle_line_comment(code: str, idx: int) -> tuple[int, int | None]:
    """
    Skip a single-line comment and determine if it marks body start.

    Returns
    -------
    tuple[int, int | None]
        (next_index, body_start if comment belongs to body)
    """
    next_idx = _skip_line_comment(code, idx)
    if _next_token_is_decl_or_attribute(code, next_idx):
        return next_idx, idx
    return next_idx, None


def _skip_block_comment(code: str, idx: int) -> int:
    depth = 0
    i = idx
    while i < len(code):
        if code.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if code.startswith("-/", i):
            depth -= 1
            i += 2
            if depth == 0:
                break
            continue
        i += 1
    return i


def _skip_whitespace_and_comments(code: str, idx: int) -> int:
    """
    Skip whitespace plus non-doc comments and return the next index.
    """
    i = idx
    while i < len(code):
        i = _skip_whitespace(code, i)
        if i >= len(code):
            return i
        if code.startswith("--", i):
            i = _skip_line_comment(code, i)
            continue
        if code.startswith("/-", i):
            i = _skip_block_comment(code, i)
            continue
        break
    return i


def _starts_with_declaration(code: str, idx: int) -> bool:
    return any(_starts_with_keyword(code, idx, keyword) for keyword in _DECLARATION_KEYWORDS)


def _consume_modifier(code: str, idx: int) -> int | None:
    """
    If a declaration modifier starts at idx, return the index immediately after it.
    """
    for keyword in _DECLARATION_MODIFIERS:
        if _starts_with_keyword(code, idx, keyword):
            return idx + len(keyword)
    return None


def _handle_modifiers(code: str, idx: int) -> tuple[bool, int, int | None]:
    """
    Consume consecutive modifiers and determine if they lead into a declaration/attribute.

    Returns
    -------
    tuple[bool, int, int | None]
        (handled, next_idx, body_start)
    """
    original_idx = idx
    i = idx
    consumed = False
    while True:
        next_i = _consume_modifier(code, i)
        if next_i is None:
            break
        consumed = True
        i = _skip_whitespace(code, next_i)

    if not consumed:
        return False, idx, None

    lookahead = _skip_whitespace_and_comments(code, i)
    if lookahead >= len(code):
        return False, idx, None
    if code.startswith("@[", lookahead) or _starts_with_declaration(code, lookahead):
        return True, lookahead, original_idx
    return False, idx, None


def _next_token_is_decl_or_attribute(code: str, idx: int) -> bool:
    """
    Determine if the next meaningful token starts a declaration or attribute.
    """
    next_idx = _skip_whitespace_and_comments(code, idx)
    if next_idx >= len(code):
        return False
    if code.startswith("@[", next_idx):
        return True
    return _starts_with_declaration(code, next_idx)


def _handle_doc_comment(code: str, idx: int) -> tuple[int, int | None]:
    """
    Skip a block comment and determine whether it should remain in the header.

    Returns
    -------
    tuple[int, bool]
        (new_idx, should_continue_scanning)
    """
    start = idx
    is_doc = idx + 2 < len(code) and code[idx + 2] in ("-", "!")
    next_idx = _skip_block_comment(code, idx)
    if is_doc and _next_token_is_decl_or_attribute(code, next_idx):
        return next_idx, start
    return next_idx, None


def _process_token(code: str, idx: int) -> tuple[bool, int, int | None]:
    """
    Process the token that starts at idx.

    Returns
    -------
    tuple[bool, int, int | None]
        (handled, next_idx, body_start)
    """
    if code.startswith("--", idx):
        next_idx, body_start = _handle_line_comment(code, idx)
        return True, next_idx, body_start

    if code.startswith("/-", idx):
        next_idx, body_start = _handle_doc_comment(code, idx)
        if body_start is not None:
            return True, next_idx, body_start
        return True, next_idx, None

    if code.startswith("@[", idx):
        return True, idx, idx

    handled_modifier, modifier_next_idx, body_start = _handle_modifiers(code, idx)
    if handled_modifier:
        return True, modifier_next_idx, body_start

    if _starts_with_declaration(code, idx):
        return True, idx, idx

    return False, idx + 1, None


def _find_body_start(code: str) -> int | None:
    if not code:
        return None

    idx = 0
    n = len(code)
    while idx < n:
        idx = _skip_whitespace(code, idx)
        if idx >= n:
            return None

        handled, next_idx, body_start = _process_token(code, idx)
        if body_start is not None:
            return body_start
        if handled:
            idx = next_idx
            continue
        idx += 1

    return None


def split_preamble_and_body(code: str) -> tuple[str, str]:
    """Split Lean code into preamble and body parts."""
    if not code:
        return "", ""

    body_start = _find_body_start(code)

    if body_start is None:
        preamble = code.strip("\n")
        return preamble, ""

    preamble = code[:body_start].rstrip("\n")
    body = code[body_start:].strip()
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
    normalized_preamble = _normalize_block(preamble)
    normalized_expected = _normalize_block(expected_preamble)

    if normalized_preamble == normalized_expected:
        return body, True

    if not normalized_preamble:
        return body or code, not normalized_expected

    # Preamble differs: fold it into the body so header commands are preserved locally.
    folded_body = f"{preamble.strip()}\n\n{body.strip()}" if body.strip() else preamble.strip()
    return folded_body, False


def strip_known_preamble_loose(code: str, expected_preamble: str) -> tuple[str, bool]:
    """
    Legacy alias retained for backwards compatibility.
    """
    return strip_known_preamble(code, expected_preamble)


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

    # Try := by sorry first (most common pattern in MOBench files).
    # Allow whitespace and comments between `by` and `sorry` so stubs like
    # `by\n    -- comment\n    sorry` are handled.
    pattern1 = r":=(?P<pre_by>(?:\s|--[^\n]*\n|/-.*?-/)*?)by(?:(?:\s|--[^\n]*\n|/-.*?-/)*)sorry"
    match1 = re.search(pattern1, theorem_statement, re.DOTALL)
    if match1:
        before = theorem_statement[: match1.start()]
        after = theorem_statement[match1.end() :]
        pre_by = match1.group("pre_by")
        # Preserve any whitespace/comments between := and by, add newline after "by" for proof body
        return f"{before}:={pre_by}by\n{proof_body}{after}"

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


def _is_valid_theorem_result(result: dict[str, Any]) -> bool:
    """
    Check if a result dictionary has the required fields for a theorem.

    Parameters
    ----------
    result: dict[str, Any]
        A result dictionary from the search results.

    Returns
    -------
    bool
        True if the result has all required fields, False otherwise.
    """
    if not isinstance(result, dict):
        return False

    primary_declaration = result.get("primary_declaration")
    if not isinstance(primary_declaration, dict):
        return False

    lean_name = primary_declaration.get("lean_name")
    return bool(lean_name)


def _filter_valid_theorems(results: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    """
    Filter results to only include valid theorems and limit to max_count.

    Parameters
    ----------
    results: list[dict[str, Any]]
        List of result dictionaries from search results.
    max_count: int
        Maximum number of valid results to return.

    Returns
    -------
    list[dict[str, Any]]
        List of valid theorem results, limited to max_count.
    """
    valid_results = []
    for result in results:
        if _is_valid_theorem_result(result):
            valid_results.append(result)
            if len(valid_results) >= max_count:
                break
    return valid_results


def _format_single_theorem(idx: int, result: dict[str, Any]) -> str:
    """
    Format a single theorem result into a string.

    Parameters
    ----------
    idx: int
        The index number of the theorem (1-based).
    result: dict[str, Any]
        A valid theorem result dictionary.

    Returns
    -------
    str
        Formatted string for the theorem.
    """
    primary_declaration = result.get("primary_declaration", {})
    lean_name = primary_declaration.get("lean_name", "")
    informal_description = result.get("informal_description", "")
    statement_text = result.get("statement_text", "")

    return (
        f"{idx}. {lean_name}\n"
        f"Name of the Theorem: {lean_name}\n"
        f"Informal Description of the Theorem: {informal_description}\n"
        f"Formal Statement of the Theorem:\n"
        f"```lean4\n{statement_text}\n```\n\n"
    )


def _format_query_section(query: str, valid_results: list[dict[str, Any]]) -> str:
    """
    Format a query section with its theorems.

    Parameters
    ----------
    query: str
        The search query string.
    valid_results: list[dict[str, Any]]
        List of valid theorem results for this query.

    Returns
    -------
    str
        Formatted string for the query section with all its theorems.
    """
    query_header = (
        f"When querying a store of proven theorems that may be of use to the task at hand "
        f'with the query "{query}" we found the following theorems ordered from those we think '
        f"to be of most utility to those we think will be of less utility\n\n"
    )

    theorem_sections = [_format_single_theorem(idx, result) for idx, result in enumerate(valid_results, start=1)]

    return query_header + "".join(theorem_sections)


def parse_semantic_check_response(response: str) -> str:
    """
    Parses the passed semantic response into a string used by Goedel-Prover-V2.

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
    from typing import cast

    pattern = r"Judgement:\s*(.+)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    if not matches:
        raise LLMParsingError("Failed to extract judgement from LLM response", response)  # noqa: TRY003
    return cast(str, matches[-1]).strip()


def _format_theorem_hints_section(search_results: list[APISearchResponseTypedDict] | None) -> str:
    """
    Format search results into a theorem hints section for prompts.

    Parameters
    ----------
    search_results: list[APISearchResponseTypedDict] | None
        List of search results from the vector database. None indicates results have not been retrieved yet.

    Returns
    -------
    str
        A formatted string containing theorem hints, or a message indicating no helpful theorems were found.
    """
    # Handle None or empty search_results
    if search_results is None or not search_results:
        return "No helpful theorems were found in the search results."

    sections: list[str] = []
    max_theorems_per_query = 5

    # Process each search result (each corresponds to a query)
    for search_result in search_results:
        query = search_result.get("query", "")
        results = search_result.get("results", [])

        # Skip queries with no results
        if not results:
            continue

        # Filter and limit results to those with required fields
        valid_results = _filter_valid_theorems(results, max_theorems_per_query)

        # Skip if no valid results after filtering
        if not valid_results:
            continue

        # Format the section for this query
        query_section = _format_query_section(query, valid_results)
        sections.append(query_section)

    # If no sections were created, return message
    if not sections:
        return "No helpful theorems were found in the search results."

    # Join all sections with double newline for clear separation between queries
    return "\n\n".join(sections).strip()
