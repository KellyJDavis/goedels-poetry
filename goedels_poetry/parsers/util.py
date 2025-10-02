import contextlib
import logging
from copy import deepcopy
from typing import Optional, Required, Union


def _get_unproven_subgoal_names(node: Required[Union[dict, list]], context: Required[dict], results: Required[dict]):  # noqa: C901
    if isinstance(node, dict):
        kind = node.get("kind")

        # If this is a theorem/lemma/def, update context
        if kind in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma"}:
            # Find declId name
            name = None
            for arg in node.get("args", []):
                if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Command.declId":
                    # the actual name is in arg["args"][0]["val"]
                    with contextlib.suppress(Exception):
                        name = arg["args"][0]["val"]
            if name:
                context = {"theorem": name, "have": None}

        # If this is a have declaration, update context
        if kind == "Lean.Parser.Tactic.tacticHave_":
            # descend into haveDecl → haveIdDecl → haveId
            have_name = None
            with contextlib.suppress(Exception):
                have_decl = node["args"][1]  # Term.haveDecl
                have_id_decl = have_decl["args"][0]
                have_id = have_id_decl["args"][0]["args"][0]["val"]
                have_name = have_id
            if have_name:
                context = {**context, "have": have_name}

        # If this is a sorry, record with current context
        if kind == "Lean.Parser.Tactic.tacticSorry":
            theorem = context.get("theorem")
            have = context.get("have")
            results.setdefault(theorem, []).append(have or "<main body>")

        # Recurse into children
        for _key, val in node.items():
            _get_unproven_subgoal_names(val, dict(context), results)

    elif isinstance(node, list):
        for item in node:
            _get_unproven_subgoal_names(item, dict(context), results)


def _get_named_subgoal_ast(node: Required[Union[dict, list]], target_name: Required[str]) -> dict:  # noqa: C901
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
def _ast_to_code(node):
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
def __find_first(node, predicate):
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


def __find_all(node, predicate, acc=None):
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
def __collect_named_decls(ast) -> dict[str, dict]:  # noqa: C901
    name_map = {}

    def rec(n):  # noqa: C901
        if isinstance(n, dict):
            k = n.get("kind", "")
            if k in {"Lean.Parser.Command.theorem", "Lean.Parser.Command.lemma", "Lean.Parser.Command.def"}:
                decl_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Command.declId")
                if decl_id:
                    val_node = __find_first(decl_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
            if k == "Lean.Parser.Tactic.tacticHave_":
                have_id = __find_first(n, lambda x: x.get("kind") == "Lean.Parser.Term.haveId")
                if have_id:
                    val_node = __find_first(have_id, lambda x: isinstance(x.get("val"), str) and x.get("val") != "")
                    if val_node:
                        name_map[val_node["val"]] = n
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
def __collect_defined_names(subtree) -> set[str]:  # noqa: C901
    names = set()

    def rec(n):  # noqa: C901
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
def __find_dependencies(subtree, name_map: dict[str, dict]) -> set[str]:
    defined = __collect_defined_names(subtree)
    deps = set()

    def rec(n):
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


def __extract_type_ast(node) -> Optional[dict]:
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
    # have: look for haveDecl then its type child
    if k == "Lean.Parser.Tactic.tacticHave_":
        have_decl = __find_first(node, lambda n: n.get("kind") == "Lean.Parser.Term.haveDecl")
        if have_decl and isinstance(have_decl, dict):
            hd_args = have_decl.get("args", [])
            if len(hd_args) > 1 and isinstance(hd_args[1], dict):
                return deepcopy(hd_args[1])
            cand = __find_first(have_decl, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
            return deepcopy(cand) if cand is not None else None
    # fallback: search anywhere under node
    cand = __find_first(node, lambda n: n.get("kind") in __TYPE_KIND_CANDIDATES)
    return deepcopy(cand) if cand is not None else None


# ---------------------------
# Strip a leading ":" token from a type AST (if present)
# ---------------------------
def __strip_leading_colon(type_ast):
    """If the AST begins with a ':' token (typeSpec style), return the inner type AST instead."""
    if not isinstance(type_ast, dict):
        return deepcopy(type_ast)
    args = type_ast.get("args", [])
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
def _get_named_subgoal_rewritten_ast(ast, target_name: str) -> dict:  # noqa: C901
    name_map = __collect_named_decls(ast)
    if target_name not in name_map:
        raise KeyError(f"target '{target_name}' not found in AST")  # noqa: TRY003
    target = deepcopy(name_map[target_name])
    deps = __find_dependencies(target, name_map)
    binders = []
    for d in sorted(deps):
        dep_node = name_map.get(d)
        dep_type_ast = __extract_type_ast(dep_node) if dep_node is not None else None
        binder = __make_binder(d, dep_type_ast)
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
        new_args = []
        new_args.append({"val": "lemma", "info": {"leading": "", "trailing": " "}})
        new_args.append({"val": have_name, "info": {"leading": "", "trailing": " "}})
        if binders:
            new_args.append({"kind": "Lean.Parser.Term.bracketedBinderList", "args": binders})
        new_args.append({"val": ":", "info": {"leading": " ", "trailing": " "}})
        new_args.append(type_body)
        new_args.append({"val": ":=", "info": {"leading": " ", "trailing": " "}})
        new_args.append(proof_node)
        lemma_node = {"kind": "Lean.Parser.Command.lemma", "args": new_args}
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
        new_args = []
        # keep same keyword (theorem/lemma/def)
        kw = (
            "theorem"
            if target.get("kind") == "Lean.Parser.Command.theorem"
            else "lemma"
            if target.get("kind") == "Lean.Parser.Command.lemma"
            else "def"
        )
        new_args.append({"val": kw, "info": {"leading": "", "trailing": " "}})
        new_args.append({"val": decl_name, "info": {"leading": "", "trailing": " "}})
        if binders:
            new_args.append({"kind": "Lean.Parser.Term.bracketedBinderList", "args": binders})
        new_args.append({"val": ":", "info": {"leading": " ", "trailing": " "}})
        new_args.append(type_body)
        new_args.append({"val": ":=", "info": {"leading": " ", "trailing": " "}})
        new_args.append(body)
        new_node = {"kind": target.get("kind"), "args": new_args}
        return new_node

    # fallback: return the target unchanged
    return deepcopy(target)
