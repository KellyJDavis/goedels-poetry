"""Extract type ASTs from various node types."""

import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node
from ..foundation.kind_utils import __is_decl_command_kind, __normalize_kind
from ..names_and_bindings.binding_name_extraction import (
    __extract_choose_names,
    __extract_generalize_names,
    __extract_match_names,
    __extract_obtain_names,
)

# Note: __TYPE_KIND_CANDIDATES is defined here (not in constants) as it's specific to type extraction

# Type kind candidates for type extraction
__TYPE_KIND_CANDIDATES = {
    "Lean.Parser.Term.typeSpec",
    "Lean.Parser.Term.forall",
    "Lean.Parser.Term.arrow",
    "Lean.Parser.Term.exists",
    "Lean.Parser.Term.existsContra",
    "Lean.Parser.Term.typeAscription",
    "Lean.Parser.Term.app",
    "Lean.Parser.Term.bracketedBinderList",
    "Lean.Parser.Term.paren",
}


def __extract_exists_witness_binder(type_node: Node) -> dict | None:
    """
    For an existential type node, try to extract the explicit/implicit/instance binder
    representing the witness. Returns a binder AST or None.

    NOTE: This function only extracts the FIRST witness binder. For multiple witnesses,
    use __extract_all_exists_witness_binders instead.
    """
    if not isinstance(type_node, dict):
        return None
    kind = __normalize_kind(type_node.get("kind", ""))
    if kind not in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"}:
        return None
    witness_binder = __find_first(
        type_node,
        lambda n: isinstance(n, dict)
        and n.get("kind")
        in {
            "Lean.Parser.Term.explicitBinder",
            "Lean.Parser.Term.implicitBinder",
            "Lean.Parser.Term.instBinder",
            "Lean.Parser.Term.strictImplicitBinder",
        },
    )
    return deepcopy(witness_binder) if witness_binder is not None else None


def __find_all_nodes(node: Any, predicate: Callable[[Any], bool]) -> list[Any]:
    """
    Find all nodes in the AST that match the predicate.
    Similar to __find_first but returns all matches.
    """
    results = []
    if isinstance(node, dict):
        if predicate(node):
            results.append(node)
        for val in node.values():
            results.extend(__find_all_nodes(val, predicate))
    elif isinstance(node, list):
        for item in node:
            results.extend(__find_all_nodes(item, predicate))
    return results


def __extract_all_exists_witness_binders(exists_type_ast: dict) -> list[dict]:
    """
    Extract all witness binders from an existential type AST.

    For "∃ c d : Nat, P", returns binders for both 'c' and 'd'.
    For "∃ x : T, P", returns a binder for 'x'.

    Parameters
    ----------
    exists_type_ast: dict
        The existential type AST node (kind should be "Lean.Parser.Term.exists" or "Lean.Parser.Term.existsContra")

    Returns
    -------
    list[dict]
        List of all witness binder AST nodes (deep copied)
    """
    if not isinstance(exists_type_ast, dict):
        return []
    kind = __normalize_kind(exists_type_ast.get("kind", ""))
    if kind not in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"}:
        return []

    # Find all binder nodes (not just the first one)
    witness_binders = __find_all_nodes(
        exists_type_ast,
        lambda n: isinstance(n, dict)
        and n.get("kind")
        in {
            "Lean.Parser.Term.explicitBinder",
            "Lean.Parser.Term.implicitBinder",
            "Lean.Parser.Term.instBinder",
            "Lean.Parser.Term.strictImplicitBinder",
        },
    )

    # Deep copy all binders
    return [deepcopy(binder) for binder in witness_binders]


def __extract_type_ast(node: Any, binding_name: str | None = None) -> dict | None:  # noqa: C901
    """
    Extract type AST from a node (theorem, have, let, set, suffices, choose, obtain, generalize, etc.).

    Parameters
    ----------
    node: Any
        The AST node to extract type from
    binding_name: Optional[str]
        For let/set/suffices/choose/obtain/generalize/match bindings, if provided, only extract type
        from the binding matching this name. If None, extract from the first binding found.
        For choose/obtain/generalize/match, types come from goal context (not AST), so this parameter
        is used for verification only.

    Returns
    -------
    Optional[dict]
        The type AST, or None if not found. For choose/obtain/generalize/match, always returns None
        as types come from goal context, not the AST.
    """
    if not isinstance(node, dict):
        return None
    k = __normalize_kind(node.get("kind", ""))
    # top-level decl (common place: args[2] often contains the signature)
    if __is_decl_command_kind(k):
        args = node.get("args", [])
        # Prefer the first arg that looks like a type, skip colon tokens.
        for arg in args:
            if isinstance(arg, dict):
                if arg.get("val") == ":":
                    continue
                if arg.get("kind") in __TYPE_KIND_CANDIDATES:
                    return deepcopy(arg)
        cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES and n.get("val") != ":")
        return deepcopy(cand) if cand is not None else None
    # have: look for haveDecl then extract the type specification
    if k == "Lean.Parser.Tactic.tacticHave_":
        have_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.haveDecl")
        if have_decl and isinstance(have_decl, dict):
            # The structure is: [haveIdDecl, ":", type_tokens...]
            # Note: ":=" is at the parent tacticHave_ level, not in haveDecl
            # We need to collect everything after ":"
            hd_args = have_decl.get("args", [])
            # Find index of ":"
            colon_idx = None
            for i, arg in enumerate(hd_args):
                if isinstance(arg, dict) and arg.get("val") == ":":
                    colon_idx = i
                    break

            # Extract all type tokens after colon
            if colon_idx is not None and colon_idx + 1 < len(hd_args):
                type_tokens = hd_args[colon_idx + 1 :]
                if type_tokens:
                    # Wrap in a container to preserve structure
                    return {"kind": "__type_container", "args": type_tokens}

            # Fallback to old behavior
            if len(hd_args) > 1 and isinstance(hd_args[1], dict):
                return deepcopy(hd_args[1])
            cand = __find_first(have_decl, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
            return deepcopy(cand) if cand is not None else None
    # let: extract type from let binding (if explicitly typed)
    # let x : T := value or let x := value (inferred type)
    if k in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
        # Look for letDecl which contains type information
        let_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.letDecl")
        if let_decl and isinstance(let_decl, dict):
            ld_args = let_decl.get("args", [])
            # Iterate through letDecl.args to find all letIdDecl nodes
            # Structure: letDecl.args[i] = letIdDecl
            # Inside letIdDecl: args[0]=name, args[1]=[], args[2]=type_array_or_empty, args[3]=":=", args[4]=value
            matched_binding = False
            for arg in ld_args:
                if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.letIdDecl":
                    let_id_decl_args = arg.get("args", [])
                    # If binding_name is provided, check if this letIdDecl matches
                    if binding_name is not None:
                        # Extract name from letIdDecl.args[0]
                        extracted_name = None
                        if len(let_id_decl_args) > 0:
                            name_node = let_id_decl_args[0]
                            # name_node might be a dict with "val", a binderIdent node, or a string
                            if isinstance(name_node, dict):
                                if name_node.get("val"):
                                    extracted_name = name_node.get("val")
                                else:
                                    # Look for binderIdent inside
                                    binder_ident = __find_first(
                                        name_node,
                                        lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"},
                                    )
                                    if binder_ident:
                                        val_node = __find_first(
                                            binder_ident,
                                            lambda n: isinstance(n.get("val"), str) and n.get("val") != "",
                                        )
                                        if val_node:
                                            extracted_name = val_node.get("val")
                            elif isinstance(name_node, str):
                                # Direct string name (unlikely but handle it)
                                extracted_name = name_node
                        # Skip this letIdDecl if name doesn't match
                        if extracted_name != binding_name:
                            continue
                        matched_binding = True
                    # Check args[2] which contains the type annotation (if present)
                    # args[2] is either [] (no type) or [typeSpec] (with type)
                    if len(let_id_decl_args) > 2:
                        type_arg = let_id_decl_args[2]
                        # If type_arg is a non-empty array, it contains the type
                        if isinstance(type_arg, list) and len(type_arg) > 0:
                            # The type is in args[2] as an array containing typeSpec
                            return {"kind": "__type_container", "args": type_arg}
                    # If binding_name was provided and we matched, but no type found, return None
                    # (don't continue searching other bindings)
                    if binding_name is not None and matched_binding:
                        return None
                    # If no type found in this letIdDecl and no specific binding requested, continue to next one
            # If binding_name was provided but no match found, log a warning and return None
            if binding_name is not None and not matched_binding:
                logging.debug(
                    f"Could not find let binding '{binding_name}' in node when extracting type, returning None"
                )
                return None
    # obtain: types are inferred from the source, not explicitly in the syntax
    # We rely on goal context for obtain types
    if k == "Lean.Parser.Tactic.tacticObtain_":
        # obtain doesn't have explicit type annotations in the syntax
        # Types must come from goal context
        # However, if binding_name is provided, verify it matches one of the obtained names
        if binding_name is not None:
            try:
                obtained_names = __extract_obtain_names(node)
                if binding_name not in obtained_names:
                    logging.debug(
                        f"Could not find obtain binding '{binding_name}' in node when extracting type, returning None"
                    )
                    return None
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # If extraction fails due to malformed AST, log and return None
                logging.debug(
                    f"Exception extracting obtain names for binding '{binding_name}': {e}, returning None",
                    exc_info=True,
                )
                return None
        # Types come from goal context, not AST, so return None
        return None
    # choose: types are inferred from the source, not explicitly in the syntax
    # We rely on goal context for choose types
    if k == "Lean.Parser.Tactic.tacticChoose_":
        # choose doesn't have explicit type annotations in the syntax
        # Types must come from goal context
        # However, if binding_name is provided, verify it matches one of the chosen names
        if binding_name is not None:
            try:
                chosen_names = __extract_choose_names(node)
                if binding_name not in chosen_names:
                    logging.debug(
                        f"Could not find choose binding '{binding_name}' in node when extracting type, returning None"
                    )
                    return None
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # If extraction fails due to malformed AST, log and return None
                logging.debug(
                    f"Exception extracting choose names for binding '{binding_name}': {e}, returning None",
                    exc_info=True,
                )
                return None
        # Types come from goal context, not AST, so return None
        return None
    # generalize: types are inferred from the source, not explicitly in the syntax
    # We rely on goal context for generalize types
    if k == "Lean.Parser.Tactic.tacticGeneralize_":
        # generalize doesn't have explicit type annotations in the syntax
        # Types must come from goal context
        # However, if binding_name is provided, verify it matches one of the generalized names
        if binding_name is not None:
            try:
                generalized_names = __extract_generalize_names(node)
                if binding_name not in generalized_names:
                    logging.debug(
                        f"Could not find generalize binding '{binding_name}' in node when extracting type, returning None"
                    )
                    return None
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # If extraction fails due to malformed AST, log and return None
                logging.debug(
                    f"Exception extracting generalize names for binding '{binding_name}': {e}, returning None",
                    exc_info=True,
                )
                return None
        # Types come from goal context, not AST, so return None
        return None
    # match: types are inferred from the pattern matching, not explicitly in the syntax
    # We rely on goal context for match pattern bindings
    if k in {"Lean.Parser.Term.match", "Lean.Parser.Tactic.tacticMatch_"}:
        # match pattern bindings don't have explicit type annotations in the syntax
        # Types must come from goal context
        # However, if binding_name is provided, verify it matches one of the match pattern names
        if binding_name is not None:
            try:
                match_names = __extract_match_names(node)
                if binding_name not in match_names:
                    logging.debug(
                        f"Could not find match binding '{binding_name}' in node when extracting type, returning None"
                    )
                    return None
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                # If extraction fails due to malformed AST, log and return None
                logging.debug(
                    f"Exception extracting match names for binding '{binding_name}': {e}, returning None",
                    exc_info=True,
                )
                return None
        # Types come from goal context, not AST, so return None
        return None
    # set: extract type from set binding (if explicitly typed)
    # set x : T := value or set x := value (inferred type)
    if k == "Lean.Parser.Tactic.tacticSet_":
        # Look for setDecl which contains type information
        set_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.setDecl")
        if set_decl and isinstance(set_decl, dict):
            sd_args = set_decl.get("args", [])
            # Structure for set: setDecl.args = [setIdDecl, ":=", value, ...]
            # First check if type is directly in setDecl.args (between ":" and ":=")
            # But only if we're not looking for a specific binding
            if binding_name is None:
                colon_idx = None
                assign_idx = None
                for i, arg in enumerate(sd_args):
                    if isinstance(arg, dict):
                        if arg.get("val") == ":" and colon_idx is None:
                            colon_idx = i
                        elif arg.get("val") == ":=":
                            assign_idx = i
                            break

                # Extract type if found directly in setDecl.args
                if colon_idx is not None and assign_idx is not None and assign_idx > colon_idx + 1:
                    type_tokens = sd_args[colon_idx + 1 : assign_idx]
                    if type_tokens:
                        return {"kind": "__type_container", "args": type_tokens}

            # Fallback: check inside setIdDecl nodes for type annotation
            # Similar to let, the type might be nested inside setIdDecl
            # Check all setIdDecl nodes in case of multiple bindings
            matched_binding = False
            for arg in sd_args:
                if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.setIdDecl":
                    # If binding_name is provided, check if this setIdDecl matches
                    if binding_name is not None:
                        # Extract name from setIdDecl
                        extracted_name = None
                        binder_ident = __find_first(
                            arg, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}
                        )
                        if binder_ident:
                            val_node = __find_first(
                                binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != ""
                            )
                            if val_node:
                                extracted_name = val_node.get("val")
                        # Skip this setIdDecl if name doesn't match
                        if extracted_name != binding_name:
                            continue
                        matched_binding = True
                    # Look for typeSpec inside setIdDecl
                    type_spec = __find_first(arg, lambda n: n.get("kind") == "Lean.Parser.Term.typeSpec")
                    if type_spec:
                        # Extract type from typeSpec
                        ts_args = type_spec.get("args", [])
                        # Skip the ":" token and get the actual type
                        type_tokens = [a for a in ts_args if not (isinstance(a, dict) and a.get("val") == ":")]
                        if type_tokens:
                            return {"kind": "__type_container", "args": type_tokens}
                    # If binding_name was provided and we matched, but no type found, return None
                    # (don't continue searching other bindings)
                    if binding_name is not None and matched_binding:
                        return None
            # If binding_name was provided but no match found, log a warning and return None
            if binding_name is not None and not matched_binding:
                logging.debug(
                    f"Could not find set binding '{binding_name}' in node when extracting type, returning None"
                )
                return None
        # Fallback: try to find type in the node structure (only if no specific binding requested)
        if binding_name is None:
            cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
            return deepcopy(cand) if cand is not None else None
        return None
    # suffices: extract type from suffices statement (similar to have)
    # suffices h : P from Q or suffices h : P by ...
    if k == "Lean.Parser.Tactic.tacticSuffices_":
        # Look for haveDecl (suffices uses similar structure to have)
        have_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.haveDecl")
        if have_decl and isinstance(have_decl, dict):
            hd_args = have_decl.get("args", [])

            # If binding_name is provided, verify it matches this suffices statement
            matched_binding = False
            if binding_name is not None:
                # Extract name from haveIdDecl/haveId (similar to __extract_suffices_name)
                extracted_name = None
                have_id_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.haveIdDecl")
                if have_id_decl:
                    have_id = __find_first(have_id_decl, lambda n: n.get("kind") == "Lean.Parser.Term.haveId")
                    if have_id:
                        val_node = __find_first(have_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
                        if val_node:
                            extracted_name = val_node.get("val")

                # If name doesn't match, return None
                if extracted_name != binding_name:
                    logging.debug(
                        f"Could not find suffices binding '{binding_name}' in node when extracting type, returning None"
                    )
                    return None
                matched_binding = True

            # Find index of ":"
            colon_idx = None
            for i, arg in enumerate(hd_args):
                if isinstance(arg, dict) and arg.get("val") == ":":
                    colon_idx = i
                    break

            # Extract all type tokens after colon (before "from" or "by")
            if colon_idx is not None and colon_idx + 1 < len(hd_args):
                # Find where the type ends (either "from" or "by" or end of args)
                type_end_idx = len(hd_args)
                for i in range(colon_idx + 1, len(hd_args)):
                    arg = hd_args[i]
                    if isinstance(arg, dict):
                        val = arg.get("val", "")
                        if val in {"from", "by"}:
                            type_end_idx = i
                            break

                type_tokens = hd_args[colon_idx + 1 : type_end_idx]
                if type_tokens:
                    return {"kind": "__type_container", "args": type_tokens}

            # If binding_name was provided and we matched, but no type found, return None
            # (don't fall back to old behavior - this binding has no type)
            if binding_name is not None and matched_binding:
                return None

            # Fallback to old behavior (only if no specific binding requested)
            if len(hd_args) > 1 and isinstance(hd_args[1], dict):
                return deepcopy(hd_args[1])
            cand = __find_first(have_decl, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
            return deepcopy(cand) if cand is not None else None
        # If binding_name was provided but no haveDecl found, return None
        if binding_name is not None:
            logging.debug(
                f"Could not find suffices binding '{binding_name}' in node when extracting type (no haveDecl), returning None"
            )
            return None
    # fallback: search anywhere under node (only if no specific binding requested)
    if binding_name is None:
        cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
        return deepcopy(cand) if cand is not None else None
    return None


def __strip_type_container(type_ast: dict) -> Any:
    """
    Handle __type_container by stripping leading colons while preserving all tokens.

    For multi-token types like "x > 0" or "x = y", we need to preserve all tokens.
    For single-token types wrapped in typeSpec (like "Nat" from "let N : Nat := ..."),
    we extract just the type token.
    """
    args = type_ast.get("args", [])
    if not args:
        return deepcopy(type_ast)

    # Skip leading colon tokens
    non_colon_args = []
    for arg in args:
        if isinstance(arg, dict) and arg.get("val") == ":":
            continue  # Skip colon tokens
        non_colon_args.append(arg)

    if not non_colon_args:
        # All args were colons, return container as-is
        return deepcopy(type_ast)

    # Check if first non-colon arg is a typeSpec (from let bindings)
    first_arg = non_colon_args[0]
    if isinstance(first_arg, dict) and first_arg.get("kind") == "Lean.Parser.Term.typeSpec":
        # Extract the type from typeSpec (skip the colon token inside)
        type_from_spec = __strip_leading_colon(first_arg)
        # If there are more args after the typeSpec, combine them
        if len(non_colon_args) > 1:
            # Multiple tokens: return a new container with the extracted type + remaining args
            return {"kind": "__type_container", "args": [type_from_spec, *non_colon_args[1:]]}
        else:
            # Single typeSpec: return just the extracted type
            return type_from_spec

    # Multiple tokens or single non-typeSpec token: preserve all tokens
    if len(non_colon_args) == 1:
        # Single token: return it directly (after stripping if needed)
        return __strip_leading_colon(non_colon_args[0])
    else:
        # Multiple tokens: return a new container with all of them
        return {"kind": "__type_container", "args": [deepcopy(arg) for arg in non_colon_args]}


def __strip_leading_colon(type_ast: Any) -> Any:
    """If the AST begins with a ':' token (typeSpec style), return the inner type AST instead."""
    if not isinstance(type_ast, dict):
        return deepcopy(type_ast)
    args = type_ast.get("args", [])
    # Handle our custom __type_container by stripping its inner payload (if any)
    if type_ast.get("kind") == "__type_container":
        return __strip_type_container(type_ast)
    # If this node itself is a 'typeSpec', often args include colon token (val=":") then the type expression.
    if type_ast.get("kind") == "Lean.Parser.Term.typeSpec" and args:
        # find the first arg that is not the colon token
        for arg in args:
            if isinstance(arg, dict) and arg.get("val") == ":":
                continue
            # return first non-colon arg (deepcopy)
            return deepcopy(arg)
    # Otherwise, if first arg is a colon token, return second
    if args and isinstance(args[0], dict) and args[0].get("val") == ":":  # noqa: SIM102
        if len(args) > 1:
            return deepcopy(args[1])
    # Handle val field with leading colon (e.g., {"val": ": Nat", ...})
    # This prevents double colons when serializing binders
    if "val" in type_ast and isinstance(type_ast["val"], str):
        # Strip whitespace first, then colons, then whitespace again
        # This handles cases like "  :  Nat  " -> "Nat"
        val = type_ast["val"].lstrip().lstrip(":").strip()
        if val != type_ast["val"]:
            result = deepcopy(type_ast)
            result["val"] = val
            return result
    # Nothing to strip: return a deepcopy of original
    return deepcopy(type_ast)
