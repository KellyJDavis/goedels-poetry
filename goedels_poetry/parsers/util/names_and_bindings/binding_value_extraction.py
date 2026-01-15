"""Extract values from binding types (let, set)."""

import logging

from ..foundation.ast_walkers import __find_first


def __extract_let_value(let_node: dict, binding_name: str | None = None) -> dict | None:  # noqa: C901
    """
    Extract the value expression from a let binding node.
    Returns the AST of the value expression (everything after :=).

    Parameters
    ----------
    let_node: dict
        The let binding node (tacticLet_ or let node)
    binding_name: Optional[str]
        If provided, only extract value from the letIdDecl matching this name.
        If None, extract from the first letIdDecl found.

    Returns
    -------
    Optional[dict]
        The value AST wrapped in __value_container, or None if not found.

    Notes
    -----
    This function handles various AST structures:
    - Nested structures where := is inside letIdDecl.args
    - Flat structures where := is at letDecl level
    - Multiple bindings in a single let statement
    - Typed and untyped bindings
    """
    if not isinstance(let_node, dict):
        logging.debug("__extract_let_value: let_node is not a dict")
        return None

    # Look for letDecl which contains the value
    let_decl = __find_first(let_node, lambda n: n.get("kind") == "Lean.Parser.Term.letDecl")
    if not let_decl or not isinstance(let_decl, dict):
        logging.debug(
            f"__extract_let_value: Could not find letDecl in node (kind: {let_node.get('kind')}, "
            f"binding_name: {binding_name})"
        )
        return None

    ld_args = let_decl.get("args", [])
    if not ld_args:
        logging.debug("__extract_let_value: letDecl.args is empty")
        return None
    # Iterate through letDecl.args to find all letIdDecl nodes
    # Structure: letDecl.args[i] = letIdDecl
    # Inside letIdDecl: args[0]=name, args[1]=[], args[2]=type_or_empty, args[3]=":=", args[4]=value
    matched_binding = False
    found_binding_names: list[str] = []  # Track found names for better error messages
    for arg in ld_args:
        if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.letIdDecl":
            let_id_decl_args = arg.get("args", [])
            # Extract name from letIdDecl.args[0] for matching and error reporting
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
                                binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != ""
                            )
                            if val_node:
                                extracted_name = val_node.get("val")
                elif isinstance(name_node, str):
                    # Direct string name (unlikely but handle it)
                    extracted_name = name_node

            # If binding_name is provided, check if this letIdDecl matches
            if binding_name is not None:
                if extracted_name:
                    found_binding_names.append(extracted_name)
                # Skip this letIdDecl if name doesn't match
                if extracted_name != binding_name:
                    continue
                matched_binding = True

            # Find ":=" - check both inside letIdDecl.args (nested) and at letDecl level (flat)
            assign_idx = None
            # First try: look inside letIdDecl.args
            for i, lid_arg in enumerate(let_id_decl_args):
                if isinstance(lid_arg, dict) and lid_arg.get("val") == ":=":
                    assign_idx = i
                    break
                # Also check for string ":="
                if isinstance(lid_arg, str) and lid_arg == ":=":
                    assign_idx = i
                    break

            if assign_idx is not None and assign_idx + 1 < len(let_id_decl_args):
                # Found ":=" inside letIdDecl, extract value from there
                value_tokens = let_id_decl_args[assign_idx + 1 :]
                if value_tokens:
                    return {"kind": "__value_container", "args": value_tokens}
                else:
                    logging.debug(
                        f"__extract_let_value: Found ':=' at index {assign_idx} but no value tokens after it "
                        f"(binding: {extracted_name or binding_name})"
                    )
            else:
                # Second try: look for ":=" at letDecl level after this letIdDecl (flat structure)
                # Find the index of this letIdDecl in ld_args
                let_id_decl_idx = None
                for i, ld_arg in enumerate(ld_args):
                    if ld_arg is arg:  # Same object reference
                        let_id_decl_idx = i
                        break

                if let_id_decl_idx is not None:
                    # Search for ":=" after this letIdDecl
                    for i in range(let_id_decl_idx + 1, len(ld_args)):
                        ld_arg = ld_args[i]
                        if isinstance(ld_arg, dict) and ld_arg.get("val") == ":=":
                            # Found ":=", extract value tokens after it
                            value_tokens = ld_args[i + 1 :]
                            # Stop at next letIdDecl if present (for multiple bindings)
                            filtered_tokens = []
                            for token in value_tokens:
                                if isinstance(token, dict) and token.get("kind") == "Lean.Parser.Term.letIdDecl":
                                    break
                                filtered_tokens.append(token)
                            if filtered_tokens:
                                return {"kind": "__value_container", "args": filtered_tokens}
                            else:
                                logging.debug(
                                    f"__extract_let_value: Found ':=' at letDecl level but no value tokens "
                                    f"(binding: {extracted_name or binding_name})"
                                )
                            break
                        # Also check for string ":="
                        if isinstance(ld_arg, str) and ld_arg == ":=":
                            value_tokens = ld_args[i + 1 :]
                            filtered_tokens = []
                            for token in value_tokens:
                                if isinstance(token, dict) and token.get("kind") == "Lean.Parser.Term.letIdDecl":
                                    break
                                filtered_tokens.append(token)
                            if filtered_tokens:
                                return {"kind": "__value_container", "args": filtered_tokens}
                            break
                        # If we hit another letIdDecl before finding ":=", something's wrong
                        if isinstance(ld_arg, dict) and ld_arg.get("kind") == "Lean.Parser.Term.letIdDecl":
                            break

            # If binding_name was provided and we matched, but no ":=" found, return None
            # (don't continue searching other bindings - this binding is malformed)
            if binding_name is not None and matched_binding:
                logging.debug(
                    f"__extract_let_value: Binding '{binding_name}' matched but no ':=' token found "
                    f"(letIdDecl.args length: {len(let_id_decl_args)})"
                )
                return None
            # If we found a letIdDecl but no ":=" and no specific binding requested,
            # continue to next one (shouldn't happen in well-formed AST, but be defensive)

    # If binding_name was provided but no match found, log a debug message with available names
    if binding_name is not None and not matched_binding:
        if found_binding_names:
            logging.debug(
                f"__extract_let_value: Could not find let binding '{binding_name}' in node. "
                f"Available bindings: {found_binding_names}"
            )
        else:
            logging.debug(
                f"__extract_let_value: Could not find let binding '{binding_name}' in node. "
                "No letIdDecl nodes found or names could not be extracted."
            )
    elif binding_name is None and not matched_binding:
        # No binding_name provided but no letIdDecl found or processed
        logging.debug("__extract_let_value: No letIdDecl nodes found in letDecl.args")
    return None


def __extract_set_value(set_node: dict, binding_name: str | None = None) -> dict | None:  # noqa: C901
    """
    Extract the value expression from a set statement node.
    Returns the AST of the value expression (everything after :=).

    Parameters
    ----------
    set_node: dict
        The set binding node (tacticSet_ node)
    binding_name: Optional[str]
        If provided, only extract value from the setIdDecl matching this name.
        If None, extract from the first setIdDecl found.

    Returns
    -------
    Optional[dict]
        The value AST wrapped in __value_container, or None if not found.

    Notes
    -----
    This function handles various AST structures:
    - Flat structures where := is after setIdDecl
    - Multiple bindings in a single set statement
    - Typed and untyped bindings
    - setArgsRest structures with 'with' clauses
    """
    if not isinstance(set_node, dict):
        logging.debug("__extract_set_value: set_node is not a dict")
        return None

    # Look for setDecl which contains the value
    set_decl = __find_first(set_node, lambda n: n.get("kind") == "Lean.Parser.Term.setDecl")
    if not set_decl or not isinstance(set_decl, dict):
        logging.debug(
            f"__extract_set_value: Could not find setDecl in node (kind: {set_node.get('kind')}, "
            f"binding_name: {binding_name})"
        )
        return None
    sd_args = set_decl.get("args", [])
    if not sd_args:
        logging.debug("__extract_set_value: setDecl.args is empty")
        return None

    # Structure for set is flatter than let:
    # setDecl.args = [setIdDecl, ":=", value, ...]
    # OR if multiple bindings: [setIdDecl1, ":=", value1, setIdDecl2, ":=", value2, ...]
    # Find the matching setIdDecl if binding_name is provided
    target_set_id_decl_idx = None
    matched_binding = False
    found_binding_names: list[str] = []  # Track found names for better error messages
    if binding_name is not None:
        for i, arg in enumerate(sd_args):
            if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.setIdDecl":
                # Extract name from setIdDecl by looking for binderIdent inside
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
                # Also check if name is directly in setIdDecl.args[0]
                if not extracted_name:
                    set_id_decl_args = arg.get("args", [])
                    if set_id_decl_args and isinstance(set_id_decl_args[0], dict):
                        if set_id_decl_args[0].get("val"):
                            extracted_name = set_id_decl_args[0].get("val")
                    elif set_id_decl_args and isinstance(set_id_decl_args[0], str):
                        extracted_name = set_id_decl_args[0]

                if extracted_name:
                    found_binding_names.append(extracted_name)
                if extracted_name == binding_name:
                    target_set_id_decl_idx = i
                    matched_binding = True
                    break

    # If binding_name was provided but no match found, return None immediately
    if binding_name is not None and not matched_binding:
        if found_binding_names:
            logging.debug(
                f"__extract_set_value: Could not find set binding '{binding_name}' in node. "
                f"Available bindings: {found_binding_names}"
            )
        else:
            logging.debug(
                f"__extract_set_value: Could not find set binding '{binding_name}' in node. "
                "No setIdDecl nodes found or names could not be extracted."
            )
        return None

    # Find ":=" token - either after target setIdDecl or first one if no target
    assign_idx = None
    if target_set_id_decl_idx is not None:
        # Start searching from the index after the target setIdDecl
        # The ":=" should be immediately after the setIdDecl
        start_idx = target_set_id_decl_idx + 1
    else:
        # When no specific binding requested, find first setIdDecl, then search for ":=" after it
        first_set_id_decl_idx = None
        for i, arg in enumerate(sd_args):
            if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.setIdDecl":
                first_set_id_decl_idx = i
                break
        start_idx = first_set_id_decl_idx + 1 if first_set_id_decl_idx is not None else 0

    for i in range(start_idx, len(sd_args)):
        arg = sd_args[i]
        if isinstance(arg, dict) and arg.get("val") == ":=":
            assign_idx = i
            break
        # Also check for string ":="
        if isinstance(arg, str) and arg == ":=":
            assign_idx = i
            break
        # If we're looking for a specific binding and hit another setIdDecl, stop
        if (
            binding_name is not None
            and target_set_id_decl_idx is not None
            and i > target_set_id_decl_idx
            and isinstance(arg, dict)
            and arg.get("kind") == "Lean.Parser.Term.setIdDecl"
        ):
            # We've passed the target binding without finding ":=", something's wrong
            logging.debug(
                f"__extract_set_value: Passed target setIdDecl at index {target_set_id_decl_idx} "
                f"without finding ':=' token (binding: {binding_name})"
            )
            break
        # If no specific binding requested and we hit another setIdDecl, stop
        # (we should only extract from the first binding)
        if (
            binding_name is None
            and isinstance(arg, dict)
            and arg.get("kind") == "Lean.Parser.Term.setIdDecl"
            and i > start_idx
        ):
            # We've passed the first binding, stop here
            break

    # Extract value tokens after ":="
    if assign_idx is not None and assign_idx + 1 < len(sd_args):
        value_tokens = sd_args[assign_idx + 1 :]
        # Stop at next setIdDecl if present (for multiple bindings)
        # Filter out any setIdDecl nodes that might appear after the value
        filtered_tokens = []
        for token in value_tokens:
            if isinstance(token, dict) and token.get("kind") == "Lean.Parser.Term.setIdDecl":
                # We've hit the next binding, stop here
                break
            filtered_tokens.append(token)
        if filtered_tokens:
            # Wrap in a container to preserve structure
            return {"kind": "__value_container", "args": filtered_tokens}
        else:
            logging.debug(
                f"__extract_set_value: Found ':=' at index {assign_idx} but no value tokens after it "
                f"(binding: {binding_name or 'first'})"
            )
    elif assign_idx is None:
        binding_info = f"binding: {binding_name}" if binding_name else "first binding"
        logging.debug(
            f"__extract_set_value: Could not find ':=' token after setIdDecl ({binding_info}, "
            f"start_idx: {start_idx}, sd_args length: {len(sd_args)})"
        )

    return None
