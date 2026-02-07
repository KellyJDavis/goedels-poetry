"""Hypothesis extraction utilities for subgoal extraction."""


def extract_hypotheses_from_unsolved_goals_data(data: str) -> list[str]:
    """
    Extract hypothesis strings from a Lean "unsolved goals" message.

    Lean (and thus Kimina) pretty-prints local contexts with newlines/indentation for long types.
    Those indented continuation lines belong to the preceding hypothesis and must be merged.

    Parameters
    ----------
    data: str
        The message text, typically from a Kimina message dict's "data" field.

    Returns
    -------
    list[str]
        Hypothesis strings (single-line), in order, stopping before the goal line (⊢ / \u22a2).
        Returns [] if the message is not an "unsolved goals" message.
    """
    if not data.startswith("unsolved goals"):
        return []

    lines = data.splitlines()
    hypotheses: list[str] = []
    current_parts: list[str] = []

    for line in lines[1:]:
        stripped = line.strip()

        # Stop at the goal line (starts with ⊢ or \u22a2).
        if stripped.startswith("⊢") or stripped.startswith("\u22a2"):
            break

        # Skip blanks and multi-goal case headers.
        if not stripped:
            continue
        if stripped.startswith("case "):
            continue

        # Continuation lines are indented.
        if line[:1].isspace():
            if current_parts:
                current_parts.append(stripped)
            else:
                # Defensive: ignore orphaned continuation lines.
                continue
            continue

        # New hypothesis starts.
        if current_parts:
            hypotheses.append(" ".join(current_parts))
        current_parts = [stripped]

    if current_parts:
        hypotheses.append(" ".join(current_parts))

    return hypotheses


def parse_hypothesis_strings_to_binders(hypotheses: list[str]) -> list[str]:
    """
    Convert hypothesis strings from "unsolved goals" messages to Lean binder strings.

    Hypothesis strings from "unsolved goals" are already in the correct format for Lean binders.
    They just need to be wrapped in parentheses. No parsing on `:` or `:=` is needed.

    This works for all hypothesis types:
    - Simple types: "n : Z" → "(n : Z)"
    - Complex types: "hn : n > 1" → "(hn : n > 1)"
    - Equality types: "hN_coe : ↑N = n" → "(hN_coe : ↑N = n)"
    - Default values: "N : N := n.toNat" → "(N : N := n.toNat)"

    Parameters
    ----------
    hypotheses: list[str]
        List of hypothesis strings from "unsolved goals" messages (e.g., ["n : Z", "hn : n > 1"]).

    Returns
    -------
    list[str]
        List of binder strings with parentheses (e.g., ["(n : Z)", "(hn : n > 1)"]).
        The order is preserved from the input list.
    """
    # Strip leading/trailing whitespace so we don't synthesize binders like `(hx : 1 < x )`
    # when the hypothesis type string ends with trailing spaces.
    return [f"({hypothesis.strip()})" for hypothesis in hypotheses]
