"""Extract names from various binding types (let, set, obtain, choose, generalize, match, suffices)."""

import logging

from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node


def __extract_let_name(let_node: dict) -> str | None:
    """
    Extract the variable name from a let binding node.

    Returns None if the name cannot be extracted, with debug logging for failures.
    """
    if not isinstance(let_node, dict):
        logging.debug("__extract_let_name: let_node is not a dict")
        return None

    # Look for letIdDecl or letId patterns
    let_id = __find_first(
        let_node,
        lambda n: n.get("kind") in {"Lean.Parser.Term.letId", "Lean.Parser.Term.letIdDecl", "Lean.binderIdent"},
    )
    if not let_id:
        logging.debug("__extract_let_name: Could not find letId/letIdDecl/binderIdent in let_node")
        return None

    val_node = __find_first(let_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if not val_node:
        logging.debug("__extract_let_name: Could not find val node with non-empty string in letId")
        return None

    val = val_node.get("val")
    if val is None:
        logging.debug("__extract_let_name: val node exists but val is None")
        return None

    return str(val)


def __extract_obtain_names(obtain_node: dict) -> list[str]:  # noqa: C901
    """
    Extract variable names from an obtain statement.
    obtain ⟨x, y, hz⟩ := proof extracts [x, y, hz]

    Note: This function extracts all binderIdent nodes from the pattern,
    which correctly captures all destructured bindings. Names after ":="
    are references, not bindings, but may be included for dependency tracking.

    Returns empty list if no names found, with debug logging for failures.
    """
    if not isinstance(obtain_node, dict):
        logging.debug("__extract_obtain_names: obtain_node is not a dict")
        return []

    names: list[str] = []

    # Look for pattern/rcases pattern which contains the destructured names
    # Common patterns: binderIdent nodes within the obtain structure
    def collect_names(n: Node) -> None:
        if isinstance(n, dict):
            # Look for binder identifiers
            if n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                val_node = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if val_node and val_node["val"]:
                    name = val_node["val"]
                    # Avoid collecting keywords or special symbols
                    if name not in {"obtain", ":=", ":", "(", ")", "⟨", "⟩", ","}:
                        names.append(name)
            # Recurse
            for v in n.values():
                collect_names(v)
        elif isinstance(n, list):
            for item in n:
                collect_names(item)

    collect_names(obtain_node)
    if not names:
        logging.debug("__extract_obtain_names: No names extracted from obtain_node (may be unnamed binding)")
    return names


def __extract_choose_names(choose_node: dict) -> list[str]:  # noqa: C901
    """
    Extract variable names from a choose statement.
    choose x hx using h extracts [x, hx]

    Note: This function extracts all binderIdent nodes, which may include
    names from the "using" clause. For dependency tracking purposes, this
    is acceptable as it ensures all referenced names are included.

    Returns empty list if no names found, with debug logging for failures.
    """
    if not isinstance(choose_node, dict):
        logging.debug("__extract_choose_names: choose_node is not a dict")
        return []

    names: list[str] = []

    # Look for binderIdent nodes within the choose structure
    # The structure is: choose x hx using h
    def collect_names(n: Node) -> None:
        if isinstance(n, dict):
            # Look for binder identifiers
            if n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                val_node = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if val_node and val_node["val"]:
                    name = val_node["val"]
                    # Avoid collecting keywords or special symbols
                    if name not in {"choose", "using", ":=", ":", "(", ")", ",", "⟨", "⟩"}:
                        names.append(name)
            # Recurse
            for v in n.values():
                collect_names(v)
        elif isinstance(n, list):
            for item in n:
                collect_names(item)

    collect_names(choose_node)
    if not names:
        logging.debug("__extract_choose_names: No names extracted from choose_node (may be unnamed binding)")
    return names


def __extract_generalize_names(generalize_node: dict) -> list[str]:  # noqa: C901
    """
    Extract variable names from a generalize statement.
    generalize h : e = x extracts [h, x]
    generalize e = x extracts [x]
    generalize h : e = x, h2 : e2 = x2 extracts [h, x, h2, x2]

    Returns empty list if no names found, with debug logging for failures.
    """
    if not isinstance(generalize_node, dict):
        logging.debug("__extract_generalize_names: generalize_node is not a dict")
        return []

    names: list[str] = []

    # Look for binderIdent nodes within the generalize structure
    # The structure is: generalize h : e = x or generalize e = x
    def collect_names(n: Node) -> None:
        if isinstance(n, dict):
            # Look for binder identifiers
            if n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                val_node = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if val_node and val_node["val"]:
                    name = val_node["val"]
                    # Avoid collecting keywords or special symbols
                    if name not in {"generalize", ":=", ":", "(", ")", ",", "=", "⟨", "⟩"}:
                        names.append(name)
            # Recurse
            for v in n.values():
                collect_names(v)
        elif isinstance(n, list):
            for item in n:
                collect_names(item)

    collect_names(generalize_node)
    if not names:
        logging.debug("__extract_generalize_names: No names extracted from generalize_node (may be unnamed binding)")
    return names


def __extract_match_names(match_node: dict) -> list[str]:  # noqa: C901
    """
    Extract variable names from all match pattern branches.
    match x with | some n => ... | (a, b) => ... extracts [n, a, b] from all branches

    Note: This extracts names from all branches, including nested match expressions.
    Match pattern bindings are scoped to their branch, but we collect all names
    to verify if a binding_name exists anywhere in the match structure.

    Returns empty list if no names found, with debug logging for failures.
    """
    # Input validation: ensure match_node is a dict
    if not isinstance(match_node, dict):
        logging.debug("__extract_match_names: match_node is not a dict")
        return []

    names: list[str] = []

    # Find all matchAlt nodes in the match expression
    def find_match_alts(n: Node) -> None:
        if isinstance(n, dict):
            if n.get("kind") in {"Lean.Parser.Term.matchAlt", "Lean.Parser.Tactic.matchAlt"}:
                # Extract names from this branch
                # Handle exceptions for malformed matchAlt nodes gracefully
                try:
                    branch_names = __extract_match_pattern_names(n)
                    names.extend(branch_names)
                except (KeyError, IndexError, TypeError, AttributeError) as e:
                    # Log and skip this branch, continue with others
                    logging.debug(
                        f"Exception extracting names from matchAlt branch: {e}, skipping",
                        exc_info=True,
                    )
            # Recurse
            for v in n.values():
                find_match_alts(v)
        elif isinstance(n, list):
            for item in n:
                find_match_alts(item)

    find_match_alts(match_node)
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    if not unique_names:
        logging.debug(
            "__extract_match_names: No names extracted from match_node (may be unnamed bindings or no matchAlt branches)"
        )
    return unique_names


def __extract_match_pattern_names(match_alt_node: dict) -> list[str]:  # noqa: C901
    """
    Extract variable names from a match pattern (only from the pattern part, before =>).
    match x with | some n => ... extracts [n] from the pattern
    match x with | (a, b) => ... extracts [a, b] from the pattern
    """
    names: list[str] = []

    # The matchAlt structure is: [|, pattern, =>, body]
    # We only want to extract from the pattern part (before =>)
    args = match_alt_node.get("args", [])
    arrow_idx = None

    # Find the => token to separate pattern from body
    for i, arg in enumerate(args):
        if (isinstance(arg, dict) and arg.get("val") == "=>") or (isinstance(arg, str) and arg == "=>"):
            arrow_idx = i
            break

    # If we found =>, only extract from args before it (the pattern part)
    # Otherwise, return empty list (safer than extracting from all args which might include body)
    # No => found likely means malformed AST, but safer to return empty than extract from body
    pattern_args = args[:arrow_idx] if arrow_idx is not None else []

    # Look for binderIdent nodes within the pattern part only
    def collect_names(n: Node) -> None:
        if isinstance(n, dict):
            # Look for binder identifiers
            if n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                val_node = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if val_node and val_node["val"]:
                    name = val_node["val"]
                    # Avoid collecting keywords or special symbols
                    if name not in {
                        "match",
                        "with",
                        "|",
                        "=>",
                        ":=",
                        ":",
                        "(",
                        ")",
                        ",",
                        "⟨",
                        "⟩",
                        "end",
                        "some",
                        "none",
                    }:
                        names.append(name)
            # Recurse
            for v in n.values():
                collect_names(v)
        elif isinstance(n, list):
            for item in n:
                collect_names(item)
        elif isinstance(n, str):
            # Skip string tokens (they're not bindings)
            pass

    # Only collect from the pattern part
    for arg in pattern_args:
        collect_names(arg)

    return names


def __extract_set_name(set_node: dict) -> str | None:
    """
    Extract the variable name from a set statement node.
    set x := value or set x : Type := value

    Returns None if the name cannot be extracted, with debug logging for failures.
    """
    if not isinstance(set_node, dict):
        logging.debug("__extract_set_name: set_node is not a dict")
        return None

    # Look for setIdDecl or similar patterns
    # The structure is similar to let: [set_keyword, setDecl, ...]
    set_id = __find_first(
        set_node,
        lambda n: n.get("kind") in {"Lean.Parser.Term.setId", "Lean.Parser.Term.setIdDecl", "Lean.binderIdent"},
    )
    if not set_id:
        logging.debug("__extract_set_name: Could not find setId/setIdDecl/binderIdent in set_node")
        return None

    if set_id:
        val_node = __find_first(set_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
        if val_node:
            val = val_node.get("val")
            return str(val) if val is not None else None

    # Alternative: look for the name directly in args, similar to let
    # Try to find a binderIdent in the first few args
    args = set_node.get("args", [])
    for arg in args[:3]:  # Check first few args
        if isinstance(arg, dict):
            binder_ident = __find_first(
                arg, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}
            )
            if binder_ident:
                val_node = __find_first(binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
                if val_node and val_node.get("val") not in {"set", ":=", ":"}:
                    return str(val_node.get("val"))

    return None


def __extract_set_with_hypothesis_name(set_node: dict) -> str | None:
    """
    Extract the hypothesis name from a set statement with a 'with' clause.
    set x := value with h extracts "h"

    The AST structure for set ... with h is:
    - setTactic.args second element = setArgsRest
    - setArgsRest.args fifth element = list containing "with", empty list, and "h"
    - The hypothesis name is at the third element of that list

    Also handles Mathlib.Tactic.setTactic structure:
    - setTactic.args second element = setArgsRest (Mathlib.Tactic.setArgsRest)
    - setArgsRest.args fifth element = list containing "with", empty list, and "h"

    Returns None if no 'with' clause is present or if the name cannot be extracted.
    """
    if not isinstance(set_node, dict):
        logging.debug("__extract_set_with_hypothesis_name: set_node is not a dict")
        return None

    # Look for setArgsRest node (can be Mathlib.Tactic.setArgsRest or similar)
    set_args_rest = __find_first(
        set_node,
        lambda n: n.get("kind")
        in {
            "Mathlib.Tactic.setArgsRest",
            "Lean.Parser.Tactic.setArgsRest",
            "Lean.Parser.Term.setArgsRest",
        },
    )

    if not set_args_rest or not isinstance(set_args_rest, dict):
        # No with clause present
        return None

    # Look for the "with" clause in setArgsRest.args
    # The structure is: [variable_name, [], ":=", value, ["with", [], hypothesis_name]]
    args = set_args_rest.get("args", [])

    # Search for a list that starts with "with"
    for arg in args:
        if (
            isinstance(arg, list)
            and len(arg) >= 3
            and isinstance(arg[0], dict)
            and arg[0].get("val") == "with"
            and len(arg) > 2
        ):
            # The hypothesis name should be at index 2
            hypothesis_name_node = arg[2]
            if isinstance(hypothesis_name_node, dict):
                hypothesis_name = hypothesis_name_node.get("val")
                if isinstance(hypothesis_name, str) and hypothesis_name:
                    return hypothesis_name
            elif isinstance(hypothesis_name_node, str):
                return hypothesis_name_node if hypothesis_name_node else None

    return None


def __extract_suffices_name(suffices_node: dict) -> str | None:
    """
    Extract the hypothesis name from a suffices statement node.
    suffices h : P from Q or suffices h : P by ...
    """
    # Look for haveIdDecl or similar pattern (suffices uses similar structure to have)
    have_id_decl = __find_first(suffices_node, lambda n: n.get("kind") == "Lean.Parser.Term.haveIdDecl")
    if have_id_decl:
        have_id = __find_first(have_id_decl, lambda n: n.get("kind") == "Lean.Parser.Term.haveId")
        if have_id:
            val_node = __find_first(have_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
            if val_node:
                val = val_node.get("val")
                return str(val) if val is not None else None

    # Alternative: look for binderIdent in args
    args = suffices_node.get("args", [])
    for arg in args:
        if isinstance(arg, dict):
            binder_ident = __find_first(
                arg, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}
            )
            if binder_ident:
                val_node = __find_first(binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
                if val_node and val_node.get("val") not in {"suffices", "from", "by", ":=", ":"}:
                    return str(val_node.get("val"))

    return None
