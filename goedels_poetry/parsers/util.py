import contextlib
import logging
from copy import deepcopy
from typing import Any, Callable, Optional, Union

Node = Union[dict[str, Any], list[Any]]


def _context_after_decl(node: dict[str, Any], context: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    kind = node.get("kind")
    if kind in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}:
        name: Optional[str] = None
        for arg in node.get("args", []):
            if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Command.declId":
                with contextlib.suppress(Exception):
                    name = arg["args"][0]["val"]
        if name:
            return {"theorem": name, "have": None}
    return context


def _context_after_have(node: dict[str, Any], context: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    if node.get("kind") == "Lean.Parser.Tactic.tacticHave_":
        have_name: Optional[str] = None
        with contextlib.suppress(Exception):
            have_decl = node["args"][1]
            have_id_decl = have_decl["args"][0]
            have_id = have_id_decl["args"][0]["args"][0]["val"]
            have_name = have_id
        if have_name:
            return {**context, "have": have_name}
    return context


def _record_sorry(
    node: dict[str, Any], context: dict[str, Optional[str]], results: dict[Optional[str], list[str]]
) -> None:
    if node.get("kind") == "Lean.Parser.Tactic.tacticSorry":
        theorem = context.get("theorem")
        have = context.get("have")
        results.setdefault(theorem, []).append(have or "<main body>")


def _get_unproven_subgoal_names(
    node: Node, context: dict[str, Optional[str]], results: dict[Optional[str], list[str]]
) -> None:
    if isinstance(node, dict):
        context = _context_after_decl(node, context)
        context = _context_after_have(node, context)
        _record_sorry(node, context, results)
        for _key, val in node.items():
            _get_unproven_subgoal_names(val, dict(context), results)
    elif isinstance(node, list):
        for item in node:
            _get_unproven_subgoal_names(item, dict(context), results)


def _get_named_subgoal_ast(node: Node, target_name: str) -> Optional[dict[str, Any]]:  # noqa: C901
    """
    Find the sub-AST for a given theorem/lemma/have name.
    Returns the entire subtree rooted at that declaration.
    """
    if isinstance(node, dict):
        kind = node.get("kind")

        # Theorem or lemma
        if kind in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}:
            try:
                decl_id = node["args"][1]  # declId
                name = decl_id["args"][0]["val"]
                if name == target_name:
                    return node
            except Exception:
                logging.exception("Exception occurred")

        # Have subgoal
        if kind == "Lean.Parser.Tactic.tacticHave_":
            try:
                have_decl = node["args"][1]  # Term.haveDecl
                have_id_decl = have_decl["args"][0]
                have_id = have_id_decl["args"][0]["args"][0]["val"]
                if have_id == target_name:
                    return node
            except Exception:
                logging.exception("Exception occurred")

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
            if k in {"args", "val", "info"}:
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
def __find_first(node: Node, predicate: Callable[[dict[str, Any]], bool]) -> Optional[dict[str, Any]]:
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
    node: Node, predicate: Callable[[dict[str, Any]], bool], acc: Optional[list[dict[str, Any]]] = None
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
            k = n.get("kind", "")
            # Collect theorems, lemmas, and definitions
            if k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}:
                decl_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Command.declId")
                if decl_id:
                    val_node = __find_first(decl_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
            # Collect have statements
            if k == "Lean.Parser.Tactic.tacticHave_":
                have_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Term.haveId")
                if have_id:
                    val_node = __find_first(have_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
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


def __extract_type_ast(node: Any) -> Optional[dict]:  # noqa: C901
    if not isinstance(node, dict):
        return None
    k = node.get("kind", "")
    # top-level decl (common place: args[2] often contains the signature)
    if k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}:
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
            # Find ":" and extract type between ":" and ":="
            colon_idx = None
            assign_idx = None
            for i, arg in enumerate(ld_args):
                if isinstance(arg, dict):
                    if arg.get("val") == ":" and colon_idx is None:
                        colon_idx = i
                    elif arg.get("val") == ":=":
                        assign_idx = i
                        break

            # Extract type if explicitly provided
            if colon_idx is not None and assign_idx is not None and assign_idx > colon_idx + 1:
                type_tokens = ld_args[colon_idx + 1 : assign_idx]
                if type_tokens:
                    return {"kind": "__type_container", "args": type_tokens}
    # obtain: types are inferred from the source, not explicitly in the syntax
    # We rely on goal context for obtain types
    if k == "Lean.Parser.Tactic.tacticObtain_":
        # obtain doesn't have explicit type annotations in the syntax
        # Types must come from goal context
        return None
    # fallback: search anywhere under node
    cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
    return deepcopy(cand) if cand is not None else None


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
def __make_binder(name: str, type_ast: Optional[dict]) -> dict:
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
# The main AST-level rewrite
# ---------------------------


def __parse_goal_context(goal: str) -> dict[str, str]:
    """
    Parse the goal string to extract variable type declarations.

    Example goal string:
        "O A C B D : Complex
        hd₁ : ¬B = D
        hd₂ : ¬C = D
        ⊢ some_goal"

    Returns a dict mapping variable names to their types.
    """
    var_types: dict[str, str] = {}
    lines = goal.split("\n")

    for line in lines:
        line = line.strip()
        # Stop at the turnstile (goal separator)
        if line.startswith("⊢"):
            break

        # Check if line contains a type declaration (has colon)
        if ":" not in line:
            continue

        # Split at the last colon to separate name(s) from type
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue

        names_part = parts[0].strip()
        type_part = parts[1].strip()

        # Handle multiple variables with same type (e.g., "O A C B D : Complex")
        names = names_part.split()
        for name in names:
            var_types[name] = type_part

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


def __find_enclosing_theorem(ast: Node, target_name: str) -> Optional[dict]:  # noqa: C901
    """
    Find the theorem/lemma that encloses the given target (typically a have statement).
    Returns the theorem/lemma node if found, None otherwise.
    """

    def contains_target(node: Node) -> bool:  # noqa: C901
        """Check if the given node contains the target by name."""
        if isinstance(node, dict):
            # Check for theorem/lemma names
            kind = node.get("kind", "")
            if kind in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}:
                try:
                    decl_id = node["args"][1]
                    name = decl_id["args"][0]["val"]
                    if name == target_name:
                        return True
                except Exception:  # noqa: S110
                    pass
            # Check for have statement names
            if kind == "Lean.Parser.Tactic.tacticHave_":
                try:
                    have_decl = node["args"][1]
                    have_id_decl = have_decl["args"][0]
                    have_id = have_id_decl["args"][0]["args"][0]["val"]
                    if have_id == target_name:
                        return True
                except Exception:  # noqa: S110
                    pass
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
        kind = ast.get("kind", "")
        # If this is a theorem/lemma and it contains the target, return it
        if kind in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"} and contains_target(ast):
            return ast
        # Otherwise, recurse into children
        for v in ast.values():
            result = __find_enclosing_theorem(v, target_name)
            if result is not None:
                return result
    elif isinstance(ast, list):
        for item in ast:
            result = __find_enclosing_theorem(item, target_name)
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

    # Extract binders from the theorem signature (stop before the proof body)
    args = theorem_node.get("args", [])
    # Typically: [keyword, declId, signature, colonToken, type, :=, proof]
    # We want to process up to but not including the proof
    for _i, arg in enumerate(args):
        # Stop when we hit the proof body (by tactic)
        if isinstance(arg, dict) and arg.get("kind") in {"Lean.Parser.Term.byTactic", "Lean.Parser.Tactic.tacticSeq"}:
            break
        extract_from_node(arg)

    return binders


def __find_earlier_bindings(  # noqa: C901
    theorem_node: dict, target_name: str, name_map: dict[str, dict]
) -> list[tuple[str, str, dict]]:
    """
    Find all bindings (have, let, obtain, etc.) that appear textually before the target
    within the given theorem. Returns a list of (name, binding_type, node) tuples.

    Binding types: "have", "let", "obtain"
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
            if kind == "Lean.Parser.Tactic.tacticHave_":
                try:
                    have_decl = node["args"][1]
                    have_id_decl = have_decl["args"][0]
                    have_id = have_id_decl["args"][0]["args"][0]["val"]

                    if have_id == target_name:
                        # Found the target, stop collecting
                        target_found = True
                        return
                    else:
                        # This is an earlier have, collect it
                        earlier_bindings.append((have_id, "have", node))
                except Exception:  # noqa: S110
                    pass

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
                except Exception:  # noqa: S110
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
                except Exception:  # noqa: S110
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


def __extract_let_name(let_node: dict) -> Optional[str]:
    """
    Extract the variable name from a let binding node.
    """
    # Look for letIdDecl or letId patterns
    let_id = __find_first(
        let_node,
        lambda n: n.get("kind") in {"Lean.Parser.Term.letId", "Lean.Parser.Term.letIdDecl", "Lean.binderIdent"},
    )
    if let_id:
        val_node = __find_first(let_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
        if val_node:
            val = val_node.get("val")
            return str(val) if val is not None else None
    return None


def __extract_obtain_names(obtain_node: dict) -> list[str]:
    """
    Extract variable names from an obtain statement.
    obtain ⟨x, y, hz⟩ := proof extracts [x, y, hz]
    """
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
    return names


def __extract_binder_name(binder: dict) -> Optional[str]:
    """
    Extract the variable name from a binder AST node.
    """
    # Look for binderIdent node
    binder_ident = __find_first(binder, lambda n: n.get("kind") in {"Lean.binderIdent", "Lean.Parser.Term.binderIdent"})
    if binder_ident:
        name_node = __find_first(binder_ident, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
        if name_node:
            val = name_node.get("val")
            return str(val) if val is not None else None
    return None


def _get_named_subgoal_rewritten_ast(  # noqa: C901
    ast: Node, target_name: str, sorries: Optional[list[dict[str, Any]]] = None
) -> dict:
    name_map = __collect_named_decls(ast)
    if target_name not in name_map:
        raise KeyError(f"target '{target_name}' not found in AST")  # noqa: TRY003
    target = deepcopy(name_map[target_name])

    # Find the corresponding sorry entry with goal context
    # Collect types from all sorries to get the most complete picture
    goal_var_types: dict[str, str] = {}
    if sorries:
        # First pass: try to find a sorry that mentions the target name
        target_sorry_types: dict[str, str] = {}
        for sorry in sorries:
            goal = sorry.get("goal", "")
            if goal and target_name in goal:
                target_sorry_types = __parse_goal_context(goal)
                break

        # Second pass: collect types from all sorries, with target-specific types taking precedence
        all_types: dict[str, str] = {}
        for sorry in sorries:
            goal = sorry.get("goal", "")
            if goal:
                parsed_types = __parse_goal_context(goal)
                # Merge, but don't overwrite existing types
                for name, typ in parsed_types.items():
                    if name not in all_types:
                        all_types[name] = typ

        # Use target-specific types if available, otherwise use collected types
        goal_var_types = target_sorry_types if target_sorry_types else all_types

    # Find enclosing theorem/lemma and extract its parameters/hypotheses
    enclosing_theorem = __find_enclosing_theorem(ast, target_name)
    theorem_binders: list[dict] = []
    if enclosing_theorem is not None:
        theorem_binders = __extract_theorem_binders(enclosing_theorem, goal_var_types)

    # Find earlier bindings (have, let, obtain) that appear textually before the target
    earlier_bindings: list[tuple[str, str, dict]] = []
    if enclosing_theorem is not None:
        earlier_bindings = __find_earlier_bindings(enclosing_theorem, target_name, name_map)

    deps = __find_dependencies(target, name_map)
    binders: list[dict] = []

    # First, add theorem binders (parameters and hypotheses from enclosing theorem)
    binders.extend(theorem_binders)

    # Next, add earlier bindings (have, let, obtain) as hypotheses
    for binding_name, _binding_type, binding_node in earlier_bindings:
        # Skip if this is the target itself or already in theorem binders
        if binding_name == target_name:
            continue

        # Extract the type/conclusion of the binding
        if binding_name in goal_var_types:
            # Prioritize goal context types as they're most accurate
            binder = __make_binder_from_type_string(binding_name, goal_var_types[binding_name])
        else:
            # Try to extract type from AST
            binding_type_ast = __extract_type_ast(binding_node)
            if binding_type_ast is not None:
                binder = __make_binder(binding_name, binding_type_ast)
            else:
                # For obtain or untyped let, we need goal context
                # If not available, try to infer or skip
                if binding_name in goal_var_types:
                    binder = __make_binder_from_type_string(binding_name, goal_var_types[binding_name])
                else:
                    # Last resort: use Prop as placeholder (better than nothing)
                    logging.warning(f"Could not determine type for binding '{binding_name}', using Prop")
                    binder = __make_binder(binding_name, None)
        binders.append(binder)

    # Finally, add any remaining dependencies not yet included
    existing_binder_names = {__extract_binder_name(b) for b in binders}
    for d in sorted(deps):
        if d in existing_binder_names:
            continue
        # Prioritize goal context types (from sorries) as they're more specific and complete
        if d in goal_var_types:
            binder = __make_binder_from_type_string(d, goal_var_types[d])
        else:
            # Fall back to AST extraction if no goal context available
            dep_node = name_map.get(d)
            dep_type_ast = __extract_type_ast(dep_node) if dep_node is not None else None
            binder = __make_binder(d, dep_type_ast)
        binders.append(binder)

    # Also add any variables from the goal context that aren't dependencies but are used
    # Skip this section if we already have theorem binders, as they should cover the variables
    if not theorem_binders:
        defined_in_target = __collect_defined_names(target)
        for var_name in sorted(goal_var_types.keys()):
            # Skip if already added as dependency or defined within target
            if var_name not in existing_binder_names and var_name not in defined_in_target and var_name != target_name:
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
        have_id_node = __find_first(target, lambda n: n.get("kind") == "Lean.Parser.Term.haveId")
        have_name = None
        if have_id_node:
            name_leaf = __find_first(have_id_node, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
            if name_leaf:
                have_name = name_leaf["val"]
        if have_name is None:
            have_name = target_name
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
    if target.get("kind") in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}:
        decl_id = __find_first(target, lambda n: n.get("kind") == "Lean.Parser.Command.declId")
        name_leaf = (
            __find_first(decl_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "") if decl_id else None
        )
        decl_name = name_leaf["val"] if name_leaf else target_name
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
            if target.get("kind") == "Lean.Parser.Command.theorem"
            else "lemma"
            if target.get("kind") == "Lean.Parser.Command.lemma"
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
