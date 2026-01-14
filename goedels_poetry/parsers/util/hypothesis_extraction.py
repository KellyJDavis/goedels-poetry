"""Hypothesis extraction utilities for subgoal extraction."""


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
    return [f"({hypothesis})" for hypothesis in hypotheses]
