import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Create Environment for loading prompts
_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "data/prompts")),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


DEFAULT_IMPORTS = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
)


def load_prompt(name: str, **kwargs) -> str:
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


def get_error_str(code: str, errors: list[dict], error_thres: bool):  # noqa: C901
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
