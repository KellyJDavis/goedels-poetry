import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

Node = dict[str, Any] | list[Any]

__ANON_HAVE_NAME_PREFIX = "gp_anon_have__"

__DECL_KIND_ALIASES: dict[str, str] = {
    # Kimina can emit either fully-qualified kinds or unqualified ones for top-level commands.
    # Normalize both forms so downstream code can match reliably.
    "theorem": "Lean.Parser.Command.theorem",
    "lemma": "Lean.Parser.Command.lemma",
    "def": "Lean.Parser.Command.def",
}


def __normalize_kind(kind: str | None) -> str:
    if not isinstance(kind, str):
        return ""
    return __DECL_KIND_ALIASES.get(kind, kind)


def __is_decl_command_kind(kind: str | None) -> bool:
    k = __normalize_kind(kind)
    return k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}


def __is_theorem_or_lemma_kind(kind: str | None) -> bool:
    k = __normalize_kind(kind)
    return k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}


def __sanitize_lean_ident_fragment(s: str) -> str:
    """
    Best-effort sanitizer to keep synthetic identifiers readable and safe.

    Lean identifiers can be unicode-rich, but for synthetic names we keep to a conservative
    subset to avoid surprising tokenization issues in downstream text processing.
    """
    if not isinstance(s, str) or not s:
        return "unknown"
    # Keep ASCII letters/digits/underscore; replace everything else with '_'
    return "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in s)


def __collect_anonymous_haves(ast: Node) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    """
    Collect anonymous `have : ... := ...` tactic nodes and assign stable synthetic names.

    Naming scheme (1-based index per enclosing theorem/lemma/def):
      gp_anon_have__<theorem_name>__<idx>

    Returns
    -------
    (by_id, by_name):
      - by_id: maps `id(node)` to synthetic name
      - by_name: maps synthetic name to the original dict node
    """
    by_id: dict[int, str] = {}
    by_name: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}

    def rec(n: Any, current_decl: str | None) -> None:
        if isinstance(n, dict):
            k = __normalize_kind(n.get("kind", ""))
            # Track enclosing decl name for stable per-declaration numbering
            if __is_decl_command_kind(k):
                decl = _extract_decl_id_name(n) or "unknown_decl"
                current_decl = __sanitize_lean_ident_fragment(decl)
                counters.setdefault(current_decl, 0)

            if k == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(n)
                if not have_name:
                    decl_key = current_decl or "unknown_decl"
                    counters.setdefault(decl_key, 0)
                    counters[decl_key] += 1
                    synthetic = f"{__ANON_HAVE_NAME_PREFIX}{decl_key}__{counters[decl_key]}"
                    by_id[id(n)] = synthetic
                    # Only record the first occurrence if somehow duplicated (defensive)
                    by_name.setdefault(synthetic, n)

            for v in n.values():
                rec(v, current_decl)
        elif isinstance(n, list):
            for it in n:
                rec(it, current_decl)

    rec(ast, None)
    return by_id, by_name


# ---------------------------
# AST structure validation
# ---------------------------
def _validate_ast_structure(ast: Node, raise_on_error: bool = False) -> bool:  # noqa: C901
    """
    Validate that the AST has a basic valid structure.

    The AST from kimina-lean-server can be:
    - A top-level dict with "header" and "commands" fields (from AstExport.lean)
    - A dict representing a single node with "kind" field
    - A list of nodes
    - Any nested combination of the above

    Parameters
    ----------
    ast: Node
        The AST node to validate
    raise_on_error: bool
        If True, raise ValueError on invalid structure. If False, return False.

    Returns
    -------
    bool
        True if the AST structure appears valid, False otherwise.

    Raises
    ------
    ValueError
        If raise_on_error is True and the AST structure is invalid.
    """
    if ast is None:
        if raise_on_error:
            raise ValueError("AST cannot be None")  # noqa: TRY003
        return False

    # AST must be a dict or list
    if not isinstance(ast, dict | list):
        if raise_on_error:
            raise TypeError(f"AST must be a dict or list, got {type(ast).__name__}")  # noqa: TRY003
        return False

    # If it's a dict, check for expected structure
    if isinstance(ast, dict):
        # Top-level AST from AstExport has "header" and/or "commands"
        if "header" in ast or "commands" in ast:
            # Validate commands if present
            if "commands" in ast:
                commands = ast["commands"]
                if not isinstance(commands, list):
                    if raise_on_error:
                        raise TypeError("AST 'commands' field must be a list")  # noqa: TRY003
                    return False
            return True

        # Node-level AST should have "kind" field (though some nodes might not)
        # We're lenient here - if it's a dict, we consider it potentially valid
        # The actual structure will be validated during traversal
        return True

    # If it's a list, validate that all elements are valid nodes
    if isinstance(ast, list):
        for item in ast:
            if not isinstance(item, dict | list):
                if raise_on_error:
                    raise TypeError(f"AST list contains invalid item type: {type(item).__name__}")  # noqa: TRY003
                return False
            # Recursively validate nested structures (with depth limit to avoid infinite recursion)
            if isinstance(item, dict) and ("header" in item or "commands" in item or "kind" in item):
                # This looks like a valid node, continue
                pass
            elif isinstance(item, dict | list):
                # Nested structure - validate recursively but limit depth
                # For now, we'll be lenient and just check it's a dict/list
                pass

    return True


# ---------------------------
# Safe nested value extraction helpers
# ---------------------------
def _extract_nested_value(node: dict, path: list[int | str], default: Any = None) -> Any:
    """
    Safely extract value from nested structure using a path.

    Based on Lean's Syntax AST structure from AstExport.lean:
    - Syntax nodes have structure: {"kind": kind, "args": args, "info": info}
    - Atoms/Idents have structure: {"val": val, "info": info}
    - Path can mix string keys and integer indices

    Parameters
    ----------
    node: dict
        Starting node (must be a dict)
    path: list[Union[int, str]]
        List of keys/indices to traverse, e.g., ["args", 1, "args", 0, "val"]
    default: Any
        Default value to return if path doesn't exist

    Returns
    -------
    Any
        The value at the path, or default if path doesn't exist

    Examples
    --------
    >>> node = {"args": [{"val": "test"}]}
    >>> _extract_nested_value(node, ["args", 0, "val"])
    'test'
    >>> _extract_nested_value(node, ["args", 1, "val"], "default")
    'default'
    """
    if not isinstance(node, dict):
        return default

    current = node
    for step in path:
        if isinstance(step, int):
            # Integer index - access list or dict by position
            if isinstance(current, list):
                if step < 0 or step >= len(current):
                    return default
                current = current[step]
            elif isinstance(current, dict):
                # For dicts, convert to list of values (order may vary)
                values = list(current.values())
                if step < 0 or step >= len(values):
                    return default
                current = values[step]
            else:
                return default
        else:
            # String key - access dict
            if not isinstance(current, dict) or step not in current:
                return default
            current = current[step]

    return current


def _extract_decl_id_name(node: dict[str, Any]) -> str | None:
    """
    Extract the name from a Lean.Parser.Command.declId node.

    Structure (based on Lean parser grammar):
    - theorem/lemma node: {"kind": "Lean.Parser.Command.theorem", "args": [..., declId, ...]}
    - declId: {"kind": "Lean.Parser.Command.declId", "args": [name_node, ...]}
    - name_node: {"val": "theorem_name", "info": {...}} (from Syntax.ident or Syntax.atom)

    Origin: Based on Lean's parser grammar where declId is the first argument after
    the theorem/lemma keyword, and the name is the first argument of declId.

    Parameters
    ----------
    node: dict[str, Any]
        A theorem or lemma node

    Returns
    -------
    Optional[str]
        The theorem/lemma name, or None if not found
    """
    # Kimina AST can represent declarations in multiple shapes.
    # Robust approach:
    # - locate the first declId node anywhere in this subtree
    # - extract the first non-empty `val` string within that declId
    decl_id = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Command.declId")
    if not decl_id:
        return None
    val_node = __find_first(decl_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if val_node is None:
        return None
    val = val_node.get("val")
    return str(val) if val is not None else None


def _extract_have_id_name(node: dict[str, Any]) -> str | None:
    """
    Extract the name from a Lean.Parser.Tactic.tacticHave_ node.

    Structure (based on Lean parser grammar):
    - tacticHave_: {"kind": "Lean.Parser.Tactic.tacticHave_", "args": [..., haveDecl, ...]}
    - haveDecl: {"kind": "Lean.Parser.Term.haveDecl", "args": [haveIdDecl, ...]}
    - haveIdDecl: {"kind": "Lean.Parser.Term.haveIdDecl", "args": [haveId, ...]}
    - haveId: {"kind": "Lean.Parser.Term.haveId", "args": [name_node, ...]}
    - name_node: {"val": "have_name", "info": {...}} (from Syntax.ident)

    Origin: Based on Lean's parser grammar where:
    - haveDecl is the second argument of tacticHave_ (args[1])
    - haveIdDecl is the first argument of haveDecl (args[0])
    - haveId is the first argument of haveIdDecl (args[0])
    - name is the first argument of haveId (args[0])

    This creates the path: args[1] -> args[0] -> args[0] -> args[0] -> val

    Parameters
    ----------
    node: dict[str, Any]
        A tacticHave_ node

    Returns
    -------
    Optional[str]
        The have statement name, or None if not found
    """
    # Structure: node["args"][1] = haveDecl
    #           haveDecl["args"][0] = haveIdDecl
    #           haveIdDecl["args"][0] = haveId
    #           haveId["args"][0] = name_node
    #           name_node["val"] = name
    # Based on Lean parser: have haveIdDecl : type := proof
    name_node = _extract_nested_value(node, ["args", 1, "args", 0, "args", 0, "args", 0])
    if isinstance(name_node, dict):
        val = name_node.get("val")
        if not isinstance(val, str):
            return None
        # Kimina sometimes represents anonymous haves as a placeholder identifier "[anonymous]".
        # Treat these as truly anonymous so we can assign stable synthetic names.
        #
        # We also treat `have _ : ...` as anonymous (non-referable) for decomposition purposes.
        val = val.strip()
        if not val or val in {"[anonymous]", "_"}:
            return None
        return val
    return None


def _context_after_decl(node: dict[str, Any], context: dict[str, str | None]) -> dict[str, str | None]:
    """
    Update context after encountering a theorem or lemma declaration.

    Structure documented in _extract_decl_id_name().
    """
    kind = __normalize_kind(node.get("kind"))
    if __is_theorem_or_lemma_kind(kind):
        name = _extract_decl_id_name(node)
        if name:
            return {"theorem": name, "have": None}
    return context


def _context_after_have(node: dict[str, Any], context: dict[str, str | None]) -> dict[str, str | None]:
    """
    Update context after encountering a have statement.

    Structure documented in _extract_have_id_name().
    """
    if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
        have_name = _extract_have_id_name(node)
        if have_name:
            return {**context, "have": have_name}
    return context


def _record_sorry(node: dict[str, Any], context: dict[str, str | None], results: dict[str | None, list[str]]) -> None:
    if node.get("kind") == "Lean.Parser.Tactic.tacticSorry":
        theorem = context.get("theorem")
        have = context.get("have")
        results.setdefault(theorem, []).append(have or "<main body>")


def _get_unproven_subgoal_names(
    node: Node,
    context: dict[str, str | None],
    results: dict[str | None, list[str]],
    anon_have_by_id: dict[int, str] | None = None,
) -> None:
    # Initialize anonymous-have mapping once at the root call and thread it through recursion.
    if anon_have_by_id is None:
        anon_have_by_id, _anon_by_name = __collect_anonymous_haves(node)

    if isinstance(node, dict):
        context = _context_after_decl(node, context)
        # Update context for have statements; if the have is anonymous, attach its synthetic name.
        if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
            have_name = _extract_have_id_name(node)
            if have_name:
                context = {**context, "have": have_name}
            else:
                synthetic = anon_have_by_id.get(id(node)) if anon_have_by_id is not None else None
                if synthetic:
                    context = {**context, "have": synthetic}
        else:
            context = _context_after_have(node, context)

        _record_sorry(node, context, results)
        for _key, val in node.items():
            _get_unproven_subgoal_names(val, dict(context), results, anon_have_by_id)
    elif isinstance(node, list):
        for item in node:
            _get_unproven_subgoal_names(item, dict(context), results, anon_have_by_id)


def _get_sorry_holes_by_name(ast: Node) -> dict[str, list[tuple[int, int]]]:  # noqa: C901
    """
    Return a mapping of subgoal-name -> (start, end) character offsets for each `sorry` token
    occurring inside a theorem/lemma proof.

    Names match the decomposition pipeline:
    - Named `have` statements use their have-id name (e.g. `hv'`)
    - Anonymous `have : ...` statements use synthetic names (`gp_anon_have__<decl>__<idx>`)
    - Standalone `sorry` in the main body uses the special name `"<main body>"`

    Notes
    -----
    The returned offsets are for the *full* source text that Kimina parsed (i.e. including any
    preamble that was passed to the server). Callers that want body-relative offsets should
    translate these offsets accordingly.
    """
    if not _validate_ast_structure(ast, raise_on_error=False):
        raise ValueError("Invalid AST structure: AST must be a dict or list")  # noqa: TRY003

    anon_have_by_id, _anon_by_name = __collect_anonymous_haves(ast)

    holes: dict[str, list[tuple[int, int]]] = {}

    def record(name: str, span: tuple[int, int]) -> None:
        holes.setdefault(name, []).append(span)

    def find_sorry_span(node: dict[str, Any]) -> tuple[int, int] | None:
        tok = __find_first(
            node,
            lambda n: n.get("val") == "sorry"
            and isinstance(n.get("info"), dict)
            and isinstance((n.get("info") or {}).get("pos"), list)
            and len(cast(list[Any], (n.get("info") or {}).get("pos"))) == 2,
        )
        if not tok:
            return None
        pos = (tok.get("info") or {}).get("pos")
        if not isinstance(pos, list) or len(pos) != 2:
            return None
        try:
            return int(pos[0]), int(pos[1])
        except Exception:
            return None

    def rec(node: Node, context: dict[str, str | None]) -> None:
        if isinstance(node, dict):
            # Update theorem context
            context = _context_after_decl(node, context)

            # Update have context (including synthetic anonymous have names)
            if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(node)
                if have_name:
                    context = {**context, "have": have_name}
                else:
                    synthetic = anon_have_by_id.get(id(node))
                    if synthetic:
                        context = {**context, "have": synthetic}
            else:
                context = _context_after_have(node, context)

            # Record sorry holes
            if node.get("kind") == "Lean.Parser.Tactic.tacticSorry":
                have = context.get("have")
                hole_name = have or "<main body>"
                span = find_sorry_span(node)
                if span is not None:
                    record(hole_name, span)

            for val in node.values():
                rec(val, dict(context))
        elif isinstance(node, list):
            for item in node:
                rec(item, dict(context))

    rec(ast, {"theorem": None, "have": None})
    return holes


def _get_named_subgoal_ast(node: Node, target_name: str) -> dict[str, Any] | None:  # noqa: C901
    """
    Find the sub-AST for a given theorem/lemma/have name.
    Returns the entire subtree rooted at that declaration.

    Parameters
    ----------
    node: Node
        The AST node to search
    target_name: str
        The name of the subgoal to find

    Returns
    -------
    Optional[dict[str, Any]]
        The AST of the named subgoal, or None if not found.
    """
    # Validate target_name
    if not isinstance(target_name, str) or not target_name:
        logging.warning(f"Invalid target_name: expected non-empty string, got {type(target_name).__name__}")
        return None

    if isinstance(node, dict):
        # Synthetic anonymous-have name support
        if isinstance(target_name, str) and target_name.startswith(__ANON_HAVE_NAME_PREFIX):
            _anon_by_id, anon_by_name = __collect_anonymous_haves(node)
            found = anon_by_name.get(target_name)
            if found is not None:
                return found

        kind = __normalize_kind(node.get("kind"))

        # Theorem or lemma
        # Structure documented in _extract_decl_id_name()
        if __is_theorem_or_lemma_kind(kind):
            name = _extract_decl_id_name(node)
            if name == target_name:
                return node

        # Have subgoal
        # Structure documented in _extract_have_id_name()
        if kind == "Lean.Parser.Tactic.tacticHave_":
            have_name = _extract_have_id_name(node)
            if have_name == target_name:
                return node

        # Recurse into children
        for val in node.values():
            result = _get_named_subgoal_ast(val, target_name)
            if result is not None:
                return result

    elif isinstance(node, list):
        for item in node:
            result = _get_named_subgoal_ast(item, target_name)
            if result is not None:
                return result

    return None


# ---------------------------
# AST -> Lean text renderer (keeps 'val' and info)
# ---------------------------
def _ast_to_code(node: Any) -> str:
    if isinstance(node, dict):
        kind = node.get("kind", "")
        # Handle custom containers
        if kind == "__value_container":
            # Just serialize the args directly
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))
        if kind == "__type_container":
            # Just serialize the args directly
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))
        if kind == "__equality_expr":
            # Serialize as "var = value"
            return "".join(_ast_to_code(arg) for arg in node.get("args", []))

        parts = []
        if "val" in node:
            info = node.get("info", {}) or {}
            leading = info.get("leading", "")
            trailing = info.get("trailing", "")
            parts.append(f"{leading}{node['val']}{trailing}")
        # prefer 'args' order first (parser uses args for ordered tokens)
        for arg in node.get("args", []):
            parts.append(_ast_to_code(arg))
        # then traverse other fields conservatively
        for k, v in node.items():
            if k in {"args", "val", "info", "kind"}:
                continue
            parts.append(_ast_to_code(v))
        return "".join(parts)
    elif isinstance(node, list):
        return "".join(_ast_to_code(x) for x in node)
    else:
        return ""


# ---------------------------
# Generic AST walkers
# ---------------------------
def __find_first(node: Node, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if predicate(node):
            return node
        for v in node.values():
            res = __find_first(v, predicate)
            if res is not None:
                return res
    elif isinstance(node, list):
        for it in node:
            res = __find_first(it, predicate)
            if res is not None:
                return res
    return None


def __find_all(
    node: Node, predicate: Callable[[dict[str, Any]], bool], acc: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if acc is None:
        acc = []
    if isinstance(node, dict):
        if predicate(node):
            acc.append(node)
        for v in node.values():
            __find_all(v, predicate, acc)
    elif isinstance(node, list):
        for it in node:
            __find_all(it, predicate, acc)
    return acc


# ---------------------------
# Collect named decls and haves
# ---------------------------
def __collect_named_decls(ast: Node) -> dict[str, dict]:  # noqa: C901
    name_map: dict[str, dict] = {}

    def rec(n: Any) -> None:  # noqa: C901
        if isinstance(n, dict):
            k = __normalize_kind(n.get("kind", ""))
            # Collect theorems, lemmas, and definitions
            if __is_decl_command_kind(k):
                decl_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Command.declId")
                if decl_id:
                    val_node = __find_first(decl_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
            # Collect have statements (only if referable by a real name).
            # Note: Kimina may emit placeholder "[anonymous]" for anonymous have statements; those
            # are handled separately via synthetic `gp_anon_have__...` names.
            if k == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(n)
                if have_name:
                    name_map[have_name] = n
            # Collect let bindings
            if k in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
                let_name = __extract_let_name(n)
                if let_name:
                    name_map[let_name] = n
            # Collect obtain statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticObtain_":
                obtained_names = __extract_obtain_names(n)
                for name in obtained_names:
                    if name:
                        name_map[name] = n
            # Collect set statements
            if k == "Lean.Parser.Tactic.tacticSet_":
                set_name = __extract_set_name(n)
                if set_name:
                    name_map[set_name] = n
            # Collect suffices statements
            if k == "Lean.Parser.Tactic.tacticSuffices_":
                suffices_name = __extract_suffices_name(n)
                if suffices_name:
                    name_map[suffices_name] = n
            # Collect choose statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticChoose_":
                chosen_names = __extract_choose_names(n)
                for name in chosen_names:
                    if name:
                        name_map[name] = n
            # Collect generalize statements (may introduce multiple names)
            if k == "Lean.Parser.Tactic.tacticGeneralize_":
                generalized_names = __extract_generalize_names(n)
                for name in generalized_names:
                    if name:
                        name_map[name] = n
            for v in n.values():
                rec(v)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(ast)
    return name_map


# ---------------------------
# Collect defined names inside a subtree
# ---------------------------
def __collect_defined_names(subtree: Node) -> set[str]:  # noqa: C901
    names: set[str] = set()

    def rec(n: Any) -> None:  # noqa: C901
        if isinstance(n, dict):
            k = n.get("kind", "")
            if k == "Lean.Parser.Term.haveId":
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            if k == "Lean.Parser.Command.declId":
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            if k in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"}:
                vn = __find_first(n, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                if vn:
                    names.add(vn["val"])
            for v in n.values():
                rec(v)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(subtree)
    return names


# ---------------------------
# Find cross-subtree dependencies
# ---------------------------
def __find_dependencies(subtree: Node, name_map: dict[str, dict]) -> set[str]:
    defined = __collect_defined_names(subtree)
    deps: set[str] = set()

    def rec(n: Any) -> None:
        if isinstance(n, dict):
            v = n.get("val")
            if isinstance(v, str) and v in name_map and v not in defined:  # noqa: SIM102
                if n.get("kind") not in {
                    "Lean.Parser.Term.haveId",
                    "Lean.Parser.Command.declId",
                    "Lean.binderIdent",
                    "Lean.Parser.Term.binderIdent",
                }:
                    deps.add(v)
            for val in n.values():
                rec(val)
        elif isinstance(n, list):
            for it in n:
                rec(it)

    rec(subtree)
    return deps


# ---------------------------
# Extract a best-effort type AST for a decl/have
# ---------------------------
__TYPE_KIND_CANDIDATES = {
    "Lean.Parser.Term.typeSpec",
    "Lean.Parser.Term.forall",
    "Lean.Parser.Term.typeAscription",
    "Lean.Parser.Term.app",
    "Lean.Parser.Term.bracketedBinderList",
    "Lean.Parser.Term.paren",
}


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
        if len(args) > 2 and isinstance(args[2], dict):
            return deepcopy(args[2])
        cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
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


# ---------------------------
# Strip a leading ":" token from a type AST (if present)
# ---------------------------
def __strip_leading_colon(type_ast: Any) -> Any:
    """If the AST begins with a ':' token (typeSpec style), return the inner type AST instead."""
    if not isinstance(type_ast, dict):
        return deepcopy(type_ast)
    args = type_ast.get("args", [])
    # Handle our custom __type_container - just return it as is
    if type_ast.get("kind") == "__type_container":
        return deepcopy(type_ast)
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
    # Nothing to strip: return a deepcopy of original
    return deepcopy(type_ast)


# ---------------------------
# Make an explicit binder AST for "(name : TYPE)"
# ---------------------------
def __make_binder(name: str, type_ast: dict | None) -> dict:
    if type_ast is None:
        type_ast = {"val": "Prop", "info": {"leading": " ", "trailing": " "}}
    inner_type = __strip_leading_colon(type_ast)
    binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": name, "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            inner_type,
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    return binder


# ---------------------------
# Make an equality binder AST for "(hname : name = value)"
# ---------------------------
def __make_equality_binder(hypothesis_name: str, var_name: str, value_ast: dict) -> dict:
    """
    Create a binder for an equality hypothesis like (hs : s = value).

    Parameters
    ----------
    hypothesis_name: str
        The name of the hypothesis (e.g., "hs")
    var_name: str
        The name of the variable being defined (e.g., "s")
    value_ast: dict
        The AST of the value expression (should be a __value_container or similar)
    """
    # Create the equality expression: var_name = value
    # We'll create a simple structure that serializes as "var_name = value"
    # The value_ast might be a __value_container, so we extract its args
    value_args = value_ast.get("args", []) if value_ast.get("kind") == "__value_container" else [value_ast]

    # Create nodes for the equality: var_name, "=", and the value
    var_node = {"val": var_name, "info": {"leading": "", "trailing": " "}}
    eq_node = {"val": "=", "info": {"leading": " ", "trailing": " "}}

    # Create a container that will serialize as "var_name = value"
    # We use a simple structure that _ast_to_code will handle correctly
    equality_expr = {
        "kind": "__equality_expr",
        "args": [var_node, eq_node, *value_args],
    }

    binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": hypothesis_name, "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            equality_expr,
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    return binder


# ---------------------------
# The main AST-level rewrite
# ---------------------------


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

    # Split at the last colon to separate name(s) from type
    # Using rsplit(":", 1) handles cases where type might contain colons
    # (though this is rare in Lean goal context, it's defensive)
    parts = line.rsplit(":", 1)
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


def __parse_goal_context(goal: str) -> dict[str, str]:
    r"""
    Parse the goal string to extract variable type declarations.

    Example goal string:
        "O A C B D : Complex
        hd₁ : ¬B = D
        hd₂ : ¬C = D
        OddProd : Nat := (Finset.filter ...).prod id
        hOddProd : OddProd = (Finset.filter ...).prod id
        ⊢ some_goal"

    Returns a dict mapping variable names to their types.

    Notes:
    - Uses rsplit(":", 1) to handle types that may contain colons (though rare in goal context)
    - Filters out empty and whitespace-only names
    - Validates that names are non-empty after processing
    - Handles both type declarations (name : type) and assignments (name : type := value)
    - For assignments, extracts the type part before ":="
    """
    var_types: dict[str, str] = {}
    if not isinstance(goal, str):
        return var_types

    lines = goal.split("\n")

    for line in lines:
        line = line.strip()
        # Stop at the turnstile (goal separator)
        if line.startswith("⊢"):
            break

        # Skip empty lines
        if not line:
            continue

        # Parse the line
        line_types = __parse_goal_context_line(line)
        if line_types:
            var_types.update(line_types)

    return var_types


def __make_binder_from_type_string(name: str, type_str: str) -> dict:
    """
    Create a binder AST node from a name and type string.
    """
    # Create a simple type AST node from the string
    type_ast = {"val": type_str, "info": {"leading": " ", "trailing": " "}}
    return __make_binder(name, type_ast)


def __is_referenced_in(subtree: Node, name: str) -> bool:
    """
    Check if a variable name is referenced in the given subtree.
    """
    if isinstance(subtree, dict):
        # Check if this node has a val that matches the name
        if subtree.get("val") == name:
            # Make sure it's not a binding occurrence
            kind = subtree.get("kind", "")
            if kind not in {
                "Lean.Parser.Term.haveId",
                "Lean.Parser.Command.declId",
                "Lean.binderIdent",
                "Lean.Parser.Term.binderIdent",
            }:
                return True
        # Recurse into children
        for v in subtree.values():
            if __is_referenced_in(v, name):
                return True
    elif isinstance(subtree, list):
        for item in subtree:
            if __is_referenced_in(item, name):
                return True
    return False


def __contains_target_name(node: Node, target_name: str, name_map: dict[str, dict]) -> bool:
    """
    Check if the given node contains the target by name.
    Uses name_map to check if target is defined within this node.
    """
    if isinstance(node, dict):
        # Check various node types that might contain the target
        kind = node.get("kind", "")
        if kind == "Lean.Parser.Tactic.tacticHave_":
            # Structure documented in _extract_have_id_name()
            have_name = _extract_have_id_name(node)
            if have_name == target_name:
                return True
        # Recurse into children
        for v in node.values():
            if __contains_target_name(v, target_name, name_map):
                return True
    elif isinstance(node, list):
        for item in node:
            if __contains_target_name(item, target_name, name_map):
                return True
    return False


def __find_enclosing_theorem(  # noqa: C901
    ast: Node, target_name: str, anon_have_by_id: dict[int, str] | None = None
) -> dict | None:
    """
    Find the theorem/lemma that encloses the given target (typically a have statement).
    Returns the theorem/lemma node if found, None otherwise.
    """

    def contains_target(node: Node) -> bool:  # noqa: C901
        """Check if the given node contains the target by name."""
        if isinstance(node, dict):
            # Check for theorem/lemma names
            kind = __normalize_kind(node.get("kind", ""))
            if __is_theorem_or_lemma_kind(kind):
                # Structure documented in _extract_decl_id_name()
                name = _extract_decl_id_name(node)
                if name == target_name:
                    return True
            # Check for have statement names
            # Structure documented in _extract_have_id_name()
            if kind == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(node)
                if have_name == target_name:
                    return True
                if (not have_name) and anon_have_by_id is not None:
                    synthetic = anon_have_by_id.get(id(node))
                    if synthetic == target_name:
                        return True
            # Recurse into children
            for v in node.values():
                if contains_target(v):
                    return True
        elif isinstance(node, list):
            for item in node:
                if contains_target(item):
                    return True
        return False

    if isinstance(ast, dict):
        kind = __normalize_kind(ast.get("kind", ""))
        # If this is a theorem/lemma and it contains the target, return it
        if __is_theorem_or_lemma_kind(kind) and contains_target(ast):
            return ast
        # Otherwise, recurse into children
        for v in ast.values():
            result = __find_enclosing_theorem(v, target_name, anon_have_by_id)
            if result is not None:
                return result
    elif isinstance(ast, list):
        for item in ast:
            result = __find_enclosing_theorem(item, target_name, anon_have_by_id)
            if result is not None:
                return result
    return None


def __extract_theorem_binders(theorem_node: dict, goal_var_types: dict[str, str]) -> list[dict]:  # noqa: C901
    """
    Extract all parameters and hypotheses from a theorem/lemma as binders.
    This includes both explicit binders like (x : T) and implicit ones.
    """
    binders: list[dict] = []

    # Look for bracketedBinderList or signature in the theorem
    def extract_from_node(node: Node) -> None:  # noqa: C901
        if isinstance(node, dict):
            kind = node.get("kind", "")

            # Handle explicit binder lists
            if kind == "Lean.Parser.Term.bracketedBinderList":
                for arg in node.get("args", []):
                    if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.explicitBinder":
                        binders.append(deepcopy(arg))
                    elif isinstance(arg, dict):
                        # Recurse to find nested binders
                        extract_from_node(arg)

            # Handle individual explicit binders
            elif kind == "Lean.Parser.Term.explicitBinder":
                binders.append(deepcopy(node))

            # Recurse into args (but stop at the proof body)
            elif kind not in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
                for arg in node.get("args", []):
                    extract_from_node(arg)
        elif isinstance(node, list):
            for item in node:
                extract_from_node(item)

    # Preferred path: many Kimina ASTs contain a declSig node that includes the binder list
    # (notably the unqualified `{"kind": "lemma", ...}` form observed in partial.log).
    decl_sig = __find_first(theorem_node, lambda n: n.get("kind") == "Lean.Parser.Command.declSig")
    if decl_sig is not None:
        extract_from_node(decl_sig)
        if binders:
            return binders

    # Fallback: Extract binders from the theorem signature (stop before the proof body)
    args = theorem_node.get("args", [])
    # Typically: [keyword, declId, signature, colonToken, type, :=, proof]
    # We want to process up to but not including the proof
    # The key fix: only stop when we've seen ":=" and then encounter a proof body
    # Don't stop at byTactic/tacticSeq that appear in the type expression or elsewhere
    seen_assign = False
    for _i, arg in enumerate(args):
        # Check if this is the ":=" token
        if isinstance(arg, dict) and arg.get("val") == ":=":
            seen_assign = True
            # Still process ":=" itself (though it won't contain binders)
            extract_from_node(arg)
            continue

        # Only stop at proof body nodes if we've already seen ":="
        # This prevents stopping at byTactic/tacticSeq that appear in type expressions
        if seen_assign and isinstance(arg, dict):
            kind = arg.get("kind", "")
            if kind in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
                # This is the actual proof body, stop here
                break

        # Process this argument
        extract_from_node(arg)

    return binders


def __find_earlier_bindings(  # noqa: C901
    theorem_node: dict, target_name: str, name_map: dict[str, dict], anon_have_by_id: dict[int, str] | None = None
) -> list[tuple[str, str, dict]]:
    """
    Find all bindings (have, let, obtain, set, suffices, choose, generalize, match, etc.) that appear textually before the target
    within the given theorem. Returns a list of (name, binding_type, node) tuples.

    Binding types: "have", "let", "obtain", "set", "suffices", "choose", "generalize", "match"

    Note: For match expressions, bindings are extracted from the pattern of the branch
    that contains the target, as match bindings are scoped to their branch.
    """
    earlier_bindings: list[tuple[str, str, dict]] = []
    target_found = False

    def traverse_for_bindings(node: Node) -> None:  # noqa: C901
        nonlocal target_found

        if target_found:
            return  # Stop searching once we've found the target

        if isinstance(node, dict):
            kind = node.get("kind", "")

            # Check if this is a have statement
            # Structure documented in _extract_have_id_name()
            if kind == "Lean.Parser.Tactic.tacticHave_":
                have_name = _extract_have_id_name(node)
                if have_name:
                    if have_name == target_name:
                        # Found the target, stop collecting
                        target_found = True
                        return
                    else:
                        # This is an earlier have, collect it
                        earlier_bindings.append((have_name, "have", node))
                else:
                    # Anonymous have: only use it as a stopping point if it's the target.
                    # Do NOT treat it as a named earlier binding (it isn't referable by a stable name
                    # in the original sketch without introducing additional rewriting).
                    if anon_have_by_id is not None:
                        synthetic = anon_have_by_id.get(id(node))
                        if synthetic == target_name:
                            target_found = True
                            return

            # Check if this is a let binding
            # Let can appear as: let name := value or let name : type := value
            elif kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
                try:
                    # Try to extract the let name
                    # Structure varies but usually: [let_keyword, letDecl, ...]
                    let_name = __extract_let_name(node)
                    if let_name:
                        if let_name == target_name:
                            target_found = True
                            return
                        else:
                            earlier_bindings.append((let_name, "let", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is an obtain statement
            # obtain ⟨x, hx⟩ := proof
            elif kind == "Lean.Parser.Tactic.tacticObtain_":
                try:
                    # Extract names from obtain pattern
                    obtained_names = __extract_obtain_names(node)
                    if target_name in obtained_names:
                        target_found = True
                        return
                    else:
                        # Add all obtained names as separate bindings
                        for name in obtained_names:
                            earlier_bindings.append((name, "obtain", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is a set statement
            # set x := value or set x : Type := value
            # Also handles: set x := value with h
            elif kind in {"Lean.Parser.Tactic.tacticSet_", "Mathlib.Tactic.setTactic"}:
                try:
                    set_name = __extract_set_name(node)
                    if set_name:
                        if set_name == target_name:
                            target_found = True
                            return
                        else:
                            earlier_bindings.append((set_name, "set", node))

                    # Also extract hypothesis name from "with" clause if present
                    # set x := value with h introduces both x and h
                    with_hypothesis_name = __extract_set_with_hypothesis_name(node)
                    if with_hypothesis_name:
                        if with_hypothesis_name == target_name:
                            target_found = True
                            return
                        else:
                            # Add the hypothesis as a separate binding
                            # Use "set_with_hypothesis" as the type to distinguish it
                            earlier_bindings.append((with_hypothesis_name, "set_with_hypothesis", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is a suffices statement
            # suffices h : P from Q or suffices h : P by ...
            elif kind == "Lean.Parser.Tactic.tacticSuffices_":
                try:
                    suffices_name = __extract_suffices_name(node)
                    if suffices_name:
                        if suffices_name == target_name:
                            target_found = True
                            return
                        else:
                            earlier_bindings.append((suffices_name, "suffices", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is a choose statement
            # choose x hx using h
            elif kind == "Lean.Parser.Tactic.tacticChoose_":
                try:
                    # Extract names from choose pattern
                    chosen_names = __extract_choose_names(node)
                    if target_name in chosen_names:
                        target_found = True
                        return
                    else:
                        # Add all chosen names as separate bindings
                        for name in chosen_names:
                            earlier_bindings.append((name, "choose", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is a generalize statement
            # generalize h : e = x or generalize e = x
            elif kind == "Lean.Parser.Tactic.tacticGeneralize_":
                try:
                    # Extract names from generalize pattern
                    generalized_names = __extract_generalize_names(node)
                    if target_name in generalized_names:
                        target_found = True
                        return
                    else:
                        # Add all generalized names as separate bindings
                        for name in generalized_names:
                            earlier_bindings.append((name, "generalize", node))
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    pass

            # Check if this is a match expression
            # match x with | pattern => body | pattern2 => body2 end
            elif kind in {"Lean.Parser.Term.match", "Lean.Parser.Tactic.tacticMatch_"}:
                try:
                    # For match expressions, we need to check each branch
                    # If the target is in a branch, include that branch's pattern bindings
                    args = node.get("args", [])
                    # Look for branches (matchAlt nodes)
                    for arg in args:
                        if isinstance(arg, dict):
                            branch_kind = arg.get("kind", "")
                            if branch_kind in {
                                "Lean.Parser.Term.matchAlt",
                                "Lean.Parser.Tactic.matchAlt",
                            } and __contains_target_name(arg, target_name, name_map):
                                # Extract pattern bindings from this branch
                                pattern_names = __extract_match_pattern_names(arg)
                                for name in pattern_names:
                                    if name:
                                        earlier_bindings.append((name, "match", arg))
                                # Continue traversal into this branch to collect earlier bindings
                                traverse_for_bindings(arg)
                                if target_found:
                                    return
                        elif isinstance(arg, list):
                            for item in arg:
                                if isinstance(item, dict):
                                    item_kind = item.get("kind", "")
                                    if item_kind in {
                                        "Lean.Parser.Term.matchAlt",
                                        "Lean.Parser.Tactic.matchAlt",
                                    } and __contains_target_name(item, target_name, name_map):
                                        pattern_names = __extract_match_pattern_names(item)
                                        for name in pattern_names:
                                            if name:
                                                earlier_bindings.append((name, "match", item))
                                        traverse_for_bindings(item)
                                        if target_found:
                                            return
                    # If target not found in any branch, continue normal traversal
                    # (to handle cases where match appears before target but target is outside)
                except (KeyError, IndexError, TypeError, AttributeError):
                    # Silently handle expected errors from malformed AST structures
                    # If match handling fails, continue with normal traversal
                    pass

            # Recurse into children in order (preserves textual order)
            for v in node.values():
                if target_found:
                    break
                traverse_for_bindings(v)

        elif isinstance(node, list):
            for item in node:
                if target_found:
                    break
                traverse_for_bindings(item)

    # Start traversal from the theorem node
    traverse_for_bindings(theorem_node)

    return earlier_bindings


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


def __extract_binder_name(binder: dict) -> str | None:
    """
    Extract the variable name from a binder AST node.

    Returns None if the name cannot be extracted, with debug logging for failures.
    """
    if not isinstance(binder, dict):
        logging.debug("__extract_binder_name: binder is not a dict")
        return None

    # Look for binderIdent node
    binder_ident = __find_first(binder, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"})
    if not binder_ident:
        logging.debug("__extract_binder_name: Could not find binderIdent in binder")
        return None

    name_node = __find_first(binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if not name_node:
        logging.debug("__extract_binder_name: Could not find val node with non-empty string in binderIdent")
        return None

    val = name_node.get("val")
    if val is None:
        logging.debug("__extract_binder_name: val node exists but val is None")
        return None

    return str(val)


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


def __construct_set_with_hypothesis_type(set_node: dict, hypothesis_name: str) -> dict | None:  # noqa: C901
    """
    Construct the type AST for a set_with_hypothesis binding from the set statement.

    For `set S := Finset.range 10000 with hS`, the hypothesis `hS` has type `S = Finset.range 10000`.
    This function constructs that equality type from the set statement AST.

    Parameters
    ----------
    set_node: dict
        The set statement AST node
    hypothesis_name: str
        The hypothesis name from the `with` clause (e.g., "hS")

    Returns
    -------
    Optional[dict]
        The equality type AST (e.g., representing "S = Finset.range 10000"), or None if construction fails.

    Notes
    -----
    The constructed type AST uses the `__equality_expr` kind, which serializes as "var_name = value".
    This can be used directly as a type_ast in `__make_binder`.

    This function handles two AST structures:
    1. Mathlib.Tactic.setTactic with Mathlib.Tactic.setArgsRest (extracts directly from setArgsRest)
    2. Lean.Parser.Tactic.tacticSet_ with Lean.Parser.Term.setDecl (uses __extract_set_name/__extract_set_value)
    """
    if not isinstance(set_node, dict):
        logging.debug("__construct_set_with_hypothesis_type: set_node is not a dict")
        return None

    # Verify the set node has a 'with' clause matching the hypothesis name
    extracted_hypothesis_name = __extract_set_with_hypothesis_name(set_node)
    if not extracted_hypothesis_name or extracted_hypothesis_name != hypothesis_name:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Hypothesis name mismatch or no 'with' clause. "
            f"Expected: {hypothesis_name}, Got: {extracted_hypothesis_name}"
        )
        return None

    var_name: str | None = None
    value_args: list = []

    # Check if this is a Mathlib.Tactic.setTactic structure (setArgsRest format)
    set_args_rest = __find_first(
        set_node,
        lambda n: n.get("kind")
        in {
            "Mathlib.Tactic.setArgsRest",
            "Lean.Parser.Tactic.setArgsRest",
            "Lean.Parser.Term.setArgsRest",
        },
    )

    if set_args_rest and isinstance(set_args_rest, dict):
        # Handle Mathlib.Tactic.setTactic structure: setArgsRest.args = [var_name, [], ":=", value, ["with", [], h]]
        sar_args = set_args_rest.get("args", [])
        if len(sar_args) >= 4:
            # Extract variable name (first arg)
            var_name_node = sar_args[0]
            if isinstance(var_name_node, dict) and var_name_node.get("val"):
                var_name = var_name_node.get("val")
            elif isinstance(var_name_node, str):
                var_name = var_name_node

            # Find ":=" token and extract value after it
            assign_idx = None
            for i, arg in enumerate(sar_args):
                if (isinstance(arg, dict) and arg.get("val") == ":=") or (isinstance(arg, str) and arg == ":="):
                    assign_idx = i
                    break

            if assign_idx is not None and assign_idx + 1 < len(sar_args):
                # Extract value tokens, stopping at "with" clause
                value_tokens = []
                for i in range(assign_idx + 1, len(sar_args)):
                    arg = sar_args[i]
                    # Stop at "with" clause (list starting with "with")
                    if (
                        isinstance(arg, list)
                        and len(arg) > 0
                        and (
                            (isinstance(arg[0], dict) and arg[0].get("val") == "with")
                            or (isinstance(arg[0], str) and arg[0] == "with")
                        )
                    ):
                        break
                    value_tokens.append(arg)
                if value_tokens:
                    value_args = value_tokens

    # If we didn't extract from setArgsRest, try using existing extraction functions
    if not var_name or not value_args:
        var_name = __extract_set_name(set_node)
        value_ast = __extract_set_value(set_node)
        if value_ast:
            value_args = value_ast.get("args", []) if value_ast.get("kind") == "__value_container" else [value_ast]

    if not var_name:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Could not extract variable name from set statement "
            f"for hypothesis '{hypothesis_name}'"
        )
        return None

    if not value_args:
        logging.debug(
            f"__construct_set_with_hypothesis_type: Could not extract value from set statement "
            f"for hypothesis '{hypothesis_name}' (variable: {var_name})"
        )
        return None

    # Construct the equality type AST: var_name = value
    # This uses the same structure as __equality_expr for consistency
    var_node = {"val": var_name, "info": {"leading": "", "trailing": " "}}
    eq_node = {"val": "=", "info": {"leading": " ", "trailing": " "}}

    equality_type_ast = {
        "kind": "__equality_expr",
        "args": [var_node, eq_node, *value_args],
    }

    return equality_type_ast


def __determine_general_binding_type(
    binding_name: str,
    binding_type: str,
    binding_node: dict,
    goal_var_types: dict[str, str],
) -> dict:
    """
    Determine the type for a general binding (have, obtain, choose, generalize, match, suffices).

    Uses a fallback chain appropriate for each binding type:
    - have/suffices: goal context → AST extraction → Prop
    - obtain/choose/generalize/match: goal context → Prop (types not in AST)

    Parameters
    ----------
    binding_name: str
        The name of the binding
    binding_type: str
        The type of binding ("have", "obtain", "choose", "generalize", "match", "suffices")
    binding_node: dict
        The AST node for the binding
    goal_var_types: dict[str, str]
        Dictionary mapping variable names to their types from goal context

    Returns
    -------
    dict
        A binder AST node with the determined type, or Prop if all methods fail
    """
    # Binding types that have types in AST (can extract from AST)
    ast_extractable_types = {"have", "suffices"}

    # Binding types that rely solely on goal context (types inferred, not in AST)
    goal_context_only_types = {"obtain", "choose", "generalize", "match"}

    # Try goal context first (most accurate for all binding types)
    if binding_name in goal_var_types:
        logging.debug(
            f"__determine_general_binding_type: Found type for {binding_type} '{binding_name}' in goal context"
        )
        return __make_binder_from_type_string(binding_name, goal_var_types[binding_name])

    # For have and suffices, try AST extraction as fallback
    if binding_type in ast_extractable_types:
        logging.debug(
            f"__determine_general_binding_type: Goal context unavailable for {binding_type} '{binding_name}', "
            "trying AST extraction"
        )
        binding_type_ast = __extract_type_ast(binding_node, binding_name=binding_name)
        if binding_type_ast is not None:
            logging.debug(
                f"__determine_general_binding_type: Successfully extracted type from AST for {binding_type} '{binding_name}'"
            )
            return __make_binder(binding_name, binding_type_ast)
        else:
            logging.warning(
                f"Could not determine type for {binding_type} binding '{binding_name}': "
                "goal context unavailable and AST extraction failed, using Prop"
            )
            return __make_binder(binding_name, None)

    # For obtain, choose, generalize, match: types must come from goal context
    if binding_type in goal_context_only_types:
        logging.warning(
            f"Could not determine type for {binding_type} binding '{binding_name}': "
            "types are inferred and not in AST, goal context unavailable, using Prop"
        )
        return __make_binder(binding_name, None)

    # Unknown binding type (shouldn't happen, but handle gracefully)
    logging.warning(
        f"Could not determine type for binding '{binding_name}' (unknown type '{binding_type}'): using Prop as fallback"
    )
    return __make_binder(binding_name, None)


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
        ld_args = let_decl.get("args", [])
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


def __get_binding_type_from_node(node: dict | None) -> str | None:
    """
    Determine if a node represents a set or let binding.
    Returns "set", "let", or None.
    """
    if not isinstance(node, dict):
        return None
    kind = node.get("kind", "")
    if kind == "Lean.Parser.Tactic.tacticSet_":
        return "set"
    if kind in {"Lean.Parser.Term.let", "Lean.Parser.Tactic.tacticLet_"}:
        return "let"
    return None


def __handle_set_let_binding_as_equality(
    var_name: str,
    binding_type: str,
    binding_node: dict,
    existing_names: set[str],
    variables_in_equality_hypotheses: set[str],
    goal_var_types: dict[str, str] | None = None,
) -> tuple[dict | None, bool]:
    """
    Handle a set or let binding by creating an equality hypothesis.

    Parameters
    ----------
    var_name: str
        The variable name from the binding (e.g., "l", "s")
    binding_type: str
        Either "set" or "let"
    binding_node: dict
        The AST node for the binding
    existing_names: set[str]
        Set of names that already exist (for conflict resolution)
    variables_in_equality_hypotheses: set[str]
        Set to track variables already handled as equality hypotheses
    goal_var_types: Optional[dict[str, str]]
        Optional dictionary mapping variable names to their types from goal context.
        Used as fallback when value extraction fails.

    Returns
    -------
    tuple[Optional[dict], bool]
        A tuple of (binder, was_handled):
        - binder: The equality hypothesis binder if successful, None if all fallbacks failed
        - was_handled: True if an equality hypothesis was created, False if all attempts failed
    """
    # Extract the value expression from the binding
    # Pass binding_name to ensure we extract from the correct binding if multiple exist
    value_ast = None
    if binding_type == "let":
        value_ast = __extract_let_value(binding_node, binding_name=var_name)
    elif binding_type == "set":
        value_ast = __extract_set_value(binding_node, binding_name=var_name)

    if value_ast is not None:
        # Generate hypothesis name (e.g., "hl" for "l"), avoiding conflicts
        hypothesis_name = __generate_equality_hypothesis_name(var_name, existing_names)
        # Add the generated hypothesis name to existing_names to avoid future conflicts
        existing_names.add(hypothesis_name)
        # Create equality binder: (hl : l = value)
        binder = __make_equality_binder(hypothesis_name, var_name, value_ast)
        # Track that this variable is included as an equality hypothesis
        variables_in_equality_hypotheses.add(var_name)
        return (binder, True)

    # Value extraction failed - try fallback strategies
    # Fallback 1: Try to extract type from AST and create a type annotation instead
    # This is better than nothing - at least we have the type information
    binding_type_ast = __extract_type_ast(binding_node, binding_name=var_name)
    if binding_type_ast is not None:
        # We have type information, create a type annotation binder
        # This is not ideal (we wanted an equality), but better than skipping
        logging.debug(
            f"__handle_set_let_binding_as_equality: Value extraction failed for {binding_type} '{var_name}', "
            "but type extraction succeeded, using type annotation as fallback"
        )
        binder = __make_binder(var_name, binding_type_ast)
        variables_in_equality_hypotheses.add(var_name)  # Still track it
        return (binder, True)

    # Fallback 2: Try to use goal context types if available
    if goal_var_types and var_name in goal_var_types:
        logging.debug(
            f"__handle_set_let_binding_as_equality: Value and type extraction failed for {binding_type} '{var_name}', "
            "but found type in goal context, using as fallback"
        )
        binder = __make_binder_from_type_string(var_name, goal_var_types[var_name])
        variables_in_equality_hypotheses.add(var_name)  # Still track it
        return (binder, True)

    # All fallbacks exhausted - return failure
    logging.debug(
        f"__handle_set_let_binding_as_equality: All extraction attempts failed for {binding_type} '{var_name}' "
        "(value extraction, type extraction, and goal context all failed)"
    )
    return (None, False)


def __generate_equality_hypothesis_name(var_name: str, existing_names: set[str]) -> str:
    """
    Generate a hypothesis name for an equality from a variable name, avoiding conflicts.
    Examples: s -> hs, sOdd -> hsOdd, sEven -> hsEven
    If the base name conflicts, tries h2{var_name}, h3{var_name}, etc.

    Parameters
    ----------
    var_name: str
        The variable name (e.g., "s")
    existing_names: set[str]
        Set of names that already exist (binders, hypotheses, etc.)

    Returns
    -------
    str
        A unique hypothesis name (e.g., "hs", "h2s", "h3s", etc.)
    """
    base_name = f"h{var_name}"
    if base_name not in existing_names:
        return base_name

    # Try numbered variants: h2s, h3s, h4s, etc.
    counter = 2
    while True:
        candidate = f"h{counter}{var_name}"
        if candidate not in existing_names:
            return candidate
        counter += 1
        # Safety limit to avoid infinite loops
        if counter > 1000:
            logging.warning(f"Could not generate unique hypothesis name for '{var_name}' after 1000 attempts")
            return f"h{counter}{var_name}"


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


def _get_named_subgoal_rewritten_ast(  # noqa: C901
    ast: Node, target_name: str, sorries: list[dict[str, Any]] | None = None
) -> dict:
    # Validate AST structure
    if not _validate_ast_structure(ast, raise_on_error=False):
        raise ValueError("Invalid AST structure: AST must be a dict or list")  # noqa: TRY003

    # Validate target_name
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("target_name must be a non-empty string")  # noqa: TRY003

    # Validate sorries if provided
    if sorries is not None:
        if not isinstance(sorries, list):
            raise ValueError("sorries must be a list or None")  # noqa: TRY003
        for i, sorry in enumerate(sorries):
            if not isinstance(sorry, dict):
                raise TypeError(f"sorries[{i}] must be a dict")  # noqa: TRY003

    MAIN_BODY_NAME = "<main body>"

    anon_have_by_id, anon_have_by_name = __collect_anonymous_haves(ast)
    name_map = __collect_named_decls(ast)
    # Add synthetic anonymous-have names so the rest of the pipeline can treat them like normal named subgoals.
    for k, v in anon_have_by_name.items():
        name_map.setdefault(k, v)

    # Special marker: main-body `sorry` (a standalone sorry in the theorem body, not inside a `have`).
    # For decomposition we still want to produce a child proof state for this hole, but it does not have
    # a stable decl-name in the AST. We treat it as "the enclosing theorem itself" and synthesize a
    # top-level lemma/theorem named `gp_main_body__<decl>`.
    is_main_body = target_name == MAIN_BODY_NAME
    lookup_name = target_name
    enclosing_theorem_for_main: dict | None = None
    if is_main_body:
        enclosing_theorem_for_main = __find_first(
            ast,
            lambda n: __is_theorem_or_lemma_kind(n.get("kind")),
        )
        if enclosing_theorem_for_main is None:
            raise KeyError("main body target found but no enclosing theorem/lemma in AST")  # noqa: TRY003
        decl = _extract_decl_id_name(enclosing_theorem_for_main) or "unknown_decl"
        lookup_name = decl

    if lookup_name not in name_map:
        raise KeyError(f"target '{target_name}' not found in AST")  # noqa: TRY003
    target = deepcopy(name_map[lookup_name])

    # Find the corresponding sorry entry with goal context
    # Collect types from all sorries to get the most complete picture
    # Single pass: collect types from all sorries, identifying target-specific sorry
    #
    # Strategy:
    # 1. Collect types from all non-target sorries into all_types (first occurrence wins)
    # 2. Identify target-specific sorry (first sorry containing lookup_name as a key)
    # 3. Merge: all_types (from non-target sorries) + target_sorry_types (with priority)
    #
    # This ensures:
    # - Types from earlier sorries are available (e.g., set_with_hypothesis bindings)
    # - Target-specific types take precedence when there are conflicts
    # - All relevant type information is collected from the complete proof context
    goal_var_types: dict[str, str] = {}
    if sorries:
        all_types: dict[str, str] = {}
        target_sorry_types: dict[str, str] = {}
        target_sorry_found = False

        # Single pass through all sorries
        for sorry in sorries:
            goal = sorry.get("goal", "")
            if not goal:
                continue

            parsed_types = __parse_goal_context(goal)

            # Check if this sorry mentions the target name
            # Use exact key matching in parsed_types instead of substring matching in goal
            # This avoids false positives (e.g., "h1" matching "h10")
            is_target_sorry = not target_sorry_found and lookup_name in parsed_types
            if is_target_sorry:
                target_sorry_types = parsed_types
                target_sorry_found = True

            # Merge types from this sorry into all_types (don't overwrite existing)
            # Skip adding target-specific sorry types here - we'll merge them with priority later
            # This ensures types from earlier sorries (including set_with_hypothesis) are collected
            if not is_target_sorry:
                for name, typ in parsed_types.items():
                    if name not in all_types:
                        all_types[name] = typ

        # Merge target-specific types with all types, giving priority to target-specific types
        # This ensures we have types from the target-specific sorry (most relevant) but also
        # includes types from other sorries (e.g., set_with_hypothesis bindings from earlier sorries)
        goal_var_types = all_types.copy()
        # Overwrite with target-specific types to give them priority
        goal_var_types.update(target_sorry_types)

    # Find enclosing theorem/lemma and extract its parameters/hypotheses
    enclosing_theorem = __find_enclosing_theorem(ast, lookup_name, anon_have_by_id)
    if enclosing_theorem is None and enclosing_theorem_for_main is not None:
        enclosing_theorem = enclosing_theorem_for_main
    theorem_binders: list[dict] = []
    if enclosing_theorem is not None:
        theorem_binders = __extract_theorem_binders(enclosing_theorem, goal_var_types)

    # Find earlier bindings (have, let, obtain) that appear textually before the target
    earlier_bindings: list[tuple[str, str, dict]] = []
    if enclosing_theorem is not None:
        earlier_bindings = __find_earlier_bindings(enclosing_theorem, lookup_name, name_map, anon_have_by_id)

    deps = __find_dependencies(target, name_map)
    binders: list[dict] = []

    # First, add theorem binders (parameters and hypotheses from enclosing theorem)
    binders.extend(theorem_binders)

    # Track variables that have been included as equality hypotheses
    # (so we don't add them again as type annotations)
    variables_in_equality_hypotheses: set[str] = set()

    # Collect all existing names to avoid hypothesis name conflicts
    # This includes theorem binders, earlier bindings, and dependencies
    existing_names: set[str] = set()
    # Add theorem binder names
    for binder in theorem_binders:
        binder_name = __extract_binder_name(binder)
        if binder_name:
            existing_names.add(binder_name)
    # Add earlier binding names (from have, obtain, choose, etc. that will be added)
    for binding_name, _binding_type, _binding_node in earlier_bindings:
        if binding_name != target_name:
            existing_names.add(binding_name)
    # Add dependency names
    existing_names.update(deps)
    # Add target name
    existing_names.add(lookup_name)

    # Next, add earlier bindings (have, let, obtain) as hypotheses
    for binding_name, binding_type, binding_node in earlier_bindings:
        # Skip if this is the target itself or already in theorem binders
        if binding_name == target_name:
            continue

        # Handle let and set bindings as equality hypotheses
        if binding_type in {"let", "set"}:
            set_let_binder, was_handled = __handle_set_let_binding_as_equality(
                binding_name,
                binding_type,
                binding_node,
                existing_names,
                variables_in_equality_hypotheses,
                goal_var_types=goal_var_types,
            )
            if was_handled and set_let_binder is not None:
                binders.append(set_let_binder)
            else:
                # Fallback: if we can't extract the value, log a warning and skip
                logging.warning(f"Could not extract value for {binding_type} binding '{binding_name}', skipping")
        elif binding_type == "set_with_hypothesis":
            # Hypothesis from "set ... with h" - treat like a have statement
            # The type is h : variable = value, which should be in goal context
            if binding_name in goal_var_types:
                # Prioritize goal context types as they're most accurate
                binder = __make_binder_from_type_string(binding_name, goal_var_types[binding_name])
            else:
                # Try to construct the type from the set statement AST
                # The type should be something like "S = Finset.range 10000"
                # We construct it from: variable name + "=" + value expression
                logging.debug(
                    f"Could not find type for set_with_hypothesis '{binding_name}' in goal context, "
                    "trying to construct from AST"
                )
                binding_type_ast = __construct_set_with_hypothesis_type(binding_node, binding_name)
                if binding_type_ast is not None:
                    binder = __make_binder(binding_name, binding_type_ast)
                else:
                    # Last resort: use Prop as placeholder
                    logging.warning(
                        f"Could not determine type for set_with_hypothesis '{binding_name}': "
                        "goal context unavailable and AST construction failed, using Prop"
                    )
                    binder = __make_binder(binding_name, None)
            binders.append(binder)
            existing_names.add(binding_name)
        else:
            # For have, obtain, choose, generalize, match, suffices: use improved type determination
            binder = __determine_general_binding_type(binding_name, binding_type, binding_node, goal_var_types)
            binders.append(binder)
            # Track the binding name in existing_names (it's already there, but this ensures consistency)
            existing_names.add(binding_name)

    # Finally, add any remaining dependencies not yet included
    existing_binder_names = {__extract_binder_name(b) for b in binders}
    for d in sorted(deps):
        # Skip if already included as a binder name or as an equality hypothesis variable
        if d in existing_binder_names or d in variables_in_equality_hypotheses:
            continue

        # Check if this dependency came from a set or let statement
        dep_node = name_map.get(d)
        dep_binding_type: str | None = None
        if dep_node is not None:
            dep_binding_type = __get_binding_type_from_node(dep_node)

        if dep_binding_type in {"set", "let"} and dep_node is not None:
            set_let_binder, was_handled = __handle_set_let_binding_as_equality(
                d,
                dep_binding_type,
                dep_node,
                existing_names,
                variables_in_equality_hypotheses,
                goal_var_types=goal_var_types,
            )
            if was_handled and set_let_binder is not None:
                binders.append(set_let_binder)
                continue  # Skip type annotation handling for set/let bindings
            else:
                # Fallback: if we can't extract the value, log a warning and use type annotation
                logging.warning(
                    f"Could not extract value for {dep_binding_type} dependency '{d}', falling back to type annotation"
                )
                # Fall through to type annotation handling below

        # For non-set/let dependencies, or set/let bindings where value extraction failed,
        # use type annotations. This code path handles:
        # - Regular variables (not from set/let statements)
        # - set/let bindings where the value expression couldn't be extracted from the AST
        # Prioritize goal context types (from sorries) as they're more specific and complete
        if d in goal_var_types:
            binder = __make_binder_from_type_string(d, goal_var_types[d])
        else:
            # Fall back to AST extraction if no goal context available
            dep_type_ast = __extract_type_ast(dep_node) if dep_node is not None else None
            binder = __make_binder(d, dep_type_ast)
        binders.append(binder)

    # Also add any variables from the goal context that aren't dependencies but are used
    # Skip this section if we already have theorem binders, as they should cover the variables
    if not theorem_binders:
        defined_in_target = __collect_defined_names(target)
        for var_name in sorted(goal_var_types.keys()):
            # Skip if already added as dependency, defined within target, or included as equality hypothesis
            if (
                var_name not in existing_binder_names
                and var_name not in defined_in_target
                and var_name != target_name
                and var_name not in variables_in_equality_hypotheses
            ):
                # Check if this variable is actually referenced in the target
                referenced = __is_referenced_in(target, var_name)
                if referenced:
                    binder = __make_binder_from_type_string(var_name, goal_var_types[var_name])
                    binders.append(binder)

    # find a proof node or fallback to minimal 'by ... sorry'
    proof_node = __find_first(
        target,
        lambda n: n.get("kind") == "Lean.Parser.Term.byTactic" or n.get("kind") == "Lean.Parser.Tactic.tacticSeq",
    )
    if proof_node is None:
        proof_node = {
            "kind": "Lean.Parser.Term.byTactic",
            "args": [
                {"val": "by", "info": {"leading": " ", "trailing": "\n  "}},
                {
                    "kind": "Lean.Parser.Tactic.tacticSeq",
                    "args": [
                        {
                            "kind": "Lean.Parser.Tactic.tacticSorry",
                            "args": [{"val": "sorry", "info": {"leading": "", "trailing": "\n"}}],
                        }
                    ],
                },
            ],
        }

    # Case: target is an in-proof 'have' -> produce a top-level lemma AST
    if target.get("kind") == "Lean.Parser.Tactic.tacticHave_":
        # Use the normalized extractor so placeholder "[anonymous]" / "_" are treated as anonymous.
        have_name = _extract_have_id_name(target) or target_name
        # extract declared type and strip leading colon
        type_ast_raw = __extract_type_ast(target)
        type_body = (
            __strip_leading_colon(type_ast_raw)
            if type_ast_raw is not None
            else {"val": "Prop", "info": {"leading": " ", "trailing": " "}}
        )
        # Build the new lemma node: "lemma NAME (binders) : TYPE := proof"
        have_args: list[dict[str, Any]] = []
        have_args.append({"val": "lemma", "info": {"leading": "", "trailing": " "}})
        have_args.append({"val": have_name, "info": {"leading": "", "trailing": " "}})
        if binders:
            have_args.append({"kind": "Lean.Parser.Term.bracketedBinderList", "args": binders})
        have_args.append({"val": ":", "info": {"leading": " ", "trailing": " "}})
        have_args.append(type_body)
        have_args.append({"val": ":=", "info": {"leading": " ", "trailing": " "}})
        have_args.append(proof_node)
        lemma_node = {"kind": "Lean.Parser.Command.lemma", "args": have_args}
        return lemma_node

    # Case: target is already top-level theorem/lemma -> insert binders after name and ensure single colon
    if __is_decl_command_kind(target.get("kind")):
        decl_id = __find_first(target, lambda n: n.get("kind") == "Lean.Parser.Command.declId")
        name_leaf = (
            __find_first(decl_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "") if decl_id else None
        )
        decl_name = name_leaf["val"] if name_leaf else lookup_name
        if is_main_body:
            decl_name = f"gp_main_body__{decl_name}"
        type_ast_raw = __extract_type_ast(target)
        type_body = (
            __strip_leading_colon(type_ast_raw)
            if type_ast_raw is not None
            else {"val": "Prop", "info": {"leading": " ", "trailing": " "}}
        )
        body = __find_first(
            target,
            lambda n: n.get("kind") == "Lean.Parser.Term.byTactic"
            or n.get("kind") == "Lean.Parser.Command.declValSimple"
            or n.get("kind") == "Lean.Parser.Tactic.tacticSeq",
        )
        if body is None:
            body = {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": " ", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": "\n"}}],
                            }
                        ],
                    },
                ],
            }
        top_args: list[dict[str, Any]] = []
        # keep same keyword (theorem/lemma/def)
        kw = (
            "theorem"
            if __normalize_kind(target.get("kind")) == "Lean.Parser.Command.theorem"
            else "lemma"
            if __normalize_kind(target.get("kind")) == "Lean.Parser.Command.lemma"
            else "def"
        )
        top_args.append({"val": kw, "info": {"leading": "", "trailing": " "}})
        top_args.append({"val": decl_name, "info": {"leading": "", "trailing": " "}})
        if binders:
            top_args.append({"kind": "Lean.Parser.Term.bracketedBinderList", "args": binders})
        top_args.append({"val": ":", "info": {"leading": " ", "trailing": " "}})
        top_args.append(type_body)
        top_args.append({"val": ":=", "info": {"leading": " ", "trailing": " "}})
        top_args.append(body)
        new_node = {"kind": target.get("kind"), "args": top_args}
        return new_node

    # fallback: return the target unchanged
    return deepcopy(target)
