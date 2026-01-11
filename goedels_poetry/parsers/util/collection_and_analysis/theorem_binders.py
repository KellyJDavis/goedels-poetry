"""Extract theorem binders and find earlier bindings."""

from copy import deepcopy
from typing import Any

from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node
from ..foundation.kind_utils import __normalize_kind
from ..names_and_bindings.binding_name_extraction import (
    __extract_choose_names,
    __extract_generalize_names,
    __extract_let_name,
    __extract_match_pattern_names,
    __extract_obtain_names,
    __extract_set_name,
    __extract_set_with_hypothesis_name,
    __extract_suffices_name,
)
from ..names_and_bindings.name_extraction import __extract_binder_name, _extract_have_id_name
from ..types_and_binders.binder_construction import __make_binder
from ..types_and_binders.type_extraction import __extract_exists_witness_binder, __extract_type_ast
from .reference_checking import __contains_target_name


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

            # Handle binder lists (explicit, implicit, instance, strict implicit)
            if kind == "Lean.Parser.Term.bracketedBinderList":
                for arg in node.get("args", []):
                    if isinstance(arg, dict) and arg.get("kind") in {
                        "Lean.Parser.Term.explicitBinder",
                        "Lean.Parser.Term.implicitBinder",
                        "Lean.Parser.Term.instBinder",
                        "Lean.Parser.Term.strictImplicitBinder",
                    }:
                        binders.append(deepcopy(arg))
                    elif isinstance(arg, dict):
                        extract_from_node(arg)

            # Handle individual binders of any flavor
            elif kind in {
                "Lean.Parser.Term.explicitBinder",
                "Lean.Parser.Term.implicitBinder",
                "Lean.Parser.Term.instBinder",
                "Lean.Parser.Term.strictImplicitBinder",
            }:
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

    # If the type head is an existential, also add a hypothesis binder (and witness when available).
    decl_type = __extract_type_ast(theorem_node)
    if isinstance(decl_type, dict) and __normalize_kind(decl_type.get("kind", "")) in {
        "Lean.Parser.Term.exists",
        "Lean.Parser.Term.existsContra",
    }:
        existing_names = {__extract_binder_name(b) for b in binders}
        # Try to preserve the witness binder if present in the existential.
        witness = __extract_exists_witness_binder(decl_type)
        if witness is not None:
            w_name = __extract_binder_name(witness)
            if w_name and w_name not in existing_names:
                binders.append(witness)
                existing_names.add(w_name)
        if "hExists" not in existing_names:
            binders.append(__make_binder("hExists", deepcopy(decl_type)))

    return binders


def __parse_pi_binders_from_type(type_node: Node) -> list[dict]:  # noqa: C901
    """
    Best-effort extraction of binder-like information from a type expression when no explicit
    binder list is present (e.g., arrow-only `P → Q`, top-level `∀` Pis, or `∃` in the head).
    Returns a list of binder-like AST fragments usable as parameters/hypotheses.
    """
    binders: list[dict] = []

    def rec(n: Any) -> None:  # noqa: C901
        if not isinstance(n, dict):
            return
        k = __normalize_kind(n.get("kind", ""))
        args = n.get("args", [])

        # Handle forall (Pi) binders directly
        if k in {"Lean.Parser.Term.forall", "Lean.Parser.Term.fun"}:
            # Expect a binder list in args[0] and body in args[1]
            if args:
                rec(args[0])
            if len(args) > 1:
                rec(args[1])
            return

        # Handle arrow types (anonymous hypotheses): P -> Q
        if k == "Lean.Parser.Term.arrow":
            # args[0] = domain, args[1] = codomain
            if len(args) >= 2:
                domain = args[0]
                # Create an anonymous hypothesis binder for the domain
                binder_name = "h"
                idx = 1
                existing = {__extract_binder_name(b) for b in binders}
                while binder_name in existing:
                    idx += 1
                    binder_name = f"h{idx}"
                binders.append(__make_binder(binder_name, deepcopy(domain)))
                rec(args[1])
            return

        # Handle explicit/implicit/instance/strict implicit binders nested in the type
        if k in {
            "Lean.Parser.Term.explicitBinder",
            "Lean.Parser.Term.implicitBinder",
            "Lean.Parser.Term.instBinder",
            "Lean.Parser.Term.strictImplicitBinder",
        }:
            binders.append(deepcopy(n))
            return

        # Handle existentials at the head: ∃ x : T, P  => add witness (if present) and hypothesis
        if k in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"}:
            witness = __extract_exists_witness_binder(n)
            if witness is not None:
                w_name = __extract_binder_name(witness)
                if w_name and w_name not in {__extract_binder_name(b) for b in binders}:
                    binders.append(witness)
            binders.append(__make_binder("hExists", deepcopy(n)))
            return

        # Recurse through children
        for v in args:
            rec(v)

    rec(type_node)
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
                    target_in_branch = False
                    # Look for branches (matchAlt nodes)
                    for arg in args:
                        if isinstance(arg, dict):
                            branch_kind = arg.get("kind", "")
                            if branch_kind in {
                                "Lean.Parser.Term.matchAlt",
                                "Lean.Parser.Tactic.matchAlt",
                            } and __contains_target_name(arg, target_name, name_map):
                                target_in_branch = True
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
                                        target_in_branch = True
                                        pattern_names = __extract_match_pattern_names(item)
                                        for name in pattern_names:
                                            if name:
                                                earlier_bindings.append((name, "match", item))
                                        traverse_for_bindings(item)
                                        if target_found:
                                            return
                    # If target is NOT in any branch but match appears before target,
                    # extract bindings from all branches (new behavior)
                    # This handles cases where match appears before target but target is outside
                    if not target_in_branch and not target_found:
                        # We're traversing in order, so if we haven't found the target yet,
                        # this match appears before it. Extract bindings from all branches.
                        # The post-processing step will filter out unused ones based on dependencies.
                        for arg in args:
                            if isinstance(arg, dict):
                                branch_kind = arg.get("kind", "")
                                if branch_kind in {
                                    "Lean.Parser.Term.matchAlt",
                                    "Lean.Parser.Tactic.matchAlt",
                                }:
                                    pattern_names = __extract_match_pattern_names(arg)
                                    for name in pattern_names:
                                        if name:
                                            earlier_bindings.append((name, "match", arg))
                            elif isinstance(arg, list):
                                for item in arg:
                                    if isinstance(item, dict):
                                        item_kind = item.get("kind", "")
                                        if item_kind in {
                                            "Lean.Parser.Term.matchAlt",
                                            "Lean.Parser.Tactic.matchAlt",
                                        }:
                                            pattern_names = __extract_match_pattern_names(item)
                                            for name in pattern_names:
                                                if name:
                                                    earlier_bindings.append((name, "match", item))
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
