"""Goal context parsing utilities."""


def __parse_goal_context_line(line: str) -> dict[str, str] | None:
    """
    Parse a single line from goal context to extract variable type declarations.

    Parameters
    ----------
    line: str
        A single line from the goal context (already stripped)

    Returns
    -------
    Optional[dict[str, str]]
        Dictionary mapping variable names to their types, or None if line doesn't contain a declaration
    """
    # Check if line contains a type declaration (has colon)
    if ":" not in line:
        return None

    # Handle assignment syntax (name : type := value)
    # Split at ":=" first if present, then extract type
    if " := " in line:
        # For assignments, we want the type part before ":="
        # Format: "name : type := value"
        # Split at ":=" to separate declaration from value
        assign_parts = line.split(" := ", 1)
        if len(assign_parts) == 2:
            # Take the part before ":=" which contains "name : type"
            line = assign_parts[0].strip()

    # Split at the first colon to separate name(s) from type.
    # Goal-context lines like "hq : ∃ q : Nat, ..." should preserve the leading quantifier,
    # so prefer split, not rsplit.
    parts = line.split(":", 1)
    if len(parts) != 2:
        return None

    names_part = parts[0].strip()
    type_part = parts[1].strip()

    # Skip if no names or no type
    if not names_part or not type_part:
        return None

    # Handle multiple variables with same type (e.g., "O A C B D : Complex")
    # Filter out empty strings and whitespace-only strings
    # Also filter out ":" tokens that might appear if parsing went wrong
    names = [n.strip() for n in names_part.split() if n.strip() and n.strip() != ":"]

    # Validate names are non-empty after filtering
    if not names:
        return None

    # Return dictionary mapping names to type
    return dict.fromkeys(names, type_part)


def __parse_goal_context(goal: str, *, stop_at_name: str | None = None) -> dict[str, str]:  # noqa: C901
    r"""
    Parse the goal string to extract variable type declarations.

    - Splits on lines, stopping at the turnstile.
    - Handles assignment lines `name : type := value` by keeping only the type.
    - Optionally stops when encountering `stop_at_name`, collecting only earlier names on that line.
    """

    var_types: dict[str, str] = {}
    if not isinstance(goal, str):
        return var_types

    lines = goal.split("\n")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("⊢"):
            break

        if " := " in line:
            parts = line.split(" := ", 1)
            if len(parts) == 2:
                line = parts[0].strip()

        line_types = __parse_goal_context_line(line)
        if not line_types:
            continue

        if stop_at_name and stop_at_name in line_types:
            for nm, typ in line_types.items():
                if nm == stop_at_name:
                    break
                var_types[nm] = typ
            break

        var_types.update(line_types)

    return var_types
