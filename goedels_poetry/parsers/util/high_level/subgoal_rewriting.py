"""Main subgoal rewriting logic."""

import logging
from copy import deepcopy
from typing import Any

from ..collection_and_analysis.decl_collection import (
    __collect_defined_names,
    __collect_named_decls,
    __find_dependencies,
)
from ..collection_and_analysis.reference_checking import (
    __find_enclosing_theorem,
    __is_referenced_in,
)
from ..collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
    __find_earlier_bindings,
    __parse_pi_binders_from_type,
)
from ..foundation.ast_validation import _validate_ast_structure
from ..foundation.ast_walkers import __find_first
from ..foundation.constants import Node
from ..foundation.goal_context import __parse_goal_context
from ..foundation.kind_utils import (
    __is_decl_command_kind,
    __is_theorem_or_lemma_kind,
    __normalize_kind,
)
from ..names_and_bindings.anonymous_haves import __collect_anonymous_haves
from ..names_and_bindings.name_extraction import __extract_binder_name, _extract_decl_id_name, _extract_have_id_name
from ..types_and_binders.binder_construction import __make_binder, __make_binder_from_type_string
from ..types_and_binders.binding_types import (
    __construct_set_with_hypothesis_type,
    __determine_general_binding_type,
    __get_binding_type_from_node,
    __handle_set_let_binding_as_equality,
)
from ..types_and_binders.type_extraction import (
    __extract_all_exists_witness_binders,
    __extract_type_ast,
    __strip_leading_colon,
)


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
    target_sorry_full_types: dict[str, str] = {}
    if sorries:
        all_types: dict[str, str] = {}
        target_sorry_types: dict[str, str] = {}
        target_sorry_found = False

        # Single pass through all sorries
        for sorry in sorries:
            goal = sorry.get("goal", "")
            if not goal:
                continue

            # First parse without truncation
            parsed_types = __parse_goal_context(goal)

            # Check if this sorry mentions the target name
            # Use exact key matching in parsed_types instead of substring matching in goal
            # This avoids false positives (e.g., "h1" matching "h10")
            is_target_sorry = not target_sorry_found and lookup_name in parsed_types
            if is_target_sorry:
                # Reparse with truncation at the target to avoid pulling in later context
                target_sorry_types = __parse_goal_context(goal, stop_at_name=lookup_name)
                target_sorry_full_types = parsed_types
                target_sorry_found = True
            else:
                # Merge types from this sorry into all_types (don't overwrite existing)
                # This ensures types from earlier sorries (including set_with_hypothesis) are collected
                for name, typ in parsed_types.items():
                    if name not in all_types:
                        all_types[name] = typ

        merged_goal_var_types = all_types.copy()
        merged_goal_var_types.update(target_sorry_types)
        # Use merged map for type lookups; binder selection may filter later.
        goal_var_types = merged_goal_var_types
        target_goal_var_types = target_sorry_types
    else:
        merged_goal_var_types = {}
        target_goal_var_types = {}

    # If some variables appear only after the target in its goal context, include those
    # referenced in the target type so they can be bound.
    if target_sorry_full_types:
        for name, typ in target_sorry_full_types.items():
            if name in target_goal_var_types:
                continue
            if __is_referenced_in(target, name):
                target_goal_var_types[name] = typ
                if name not in goal_var_types:
                    goal_var_types[name] = typ

    # Find enclosing theorem/lemma and extract its parameters/hypotheses
    enclosing_theorem = __find_enclosing_theorem(ast, lookup_name, anon_have_by_id)
    if enclosing_theorem is None and enclosing_theorem_for_main is not None:
        enclosing_theorem = enclosing_theorem_for_main
    theorem_binders: list[dict] = []
    deps: set[str] = set()
    if enclosing_theorem is not None:
        # Compute dependencies early so fallback binder synthesis can use them.
        deps = __find_dependencies(target, name_map)
        theorem_binders = __extract_theorem_binders(enclosing_theorem, goal_var_types)
        if not theorem_binders:
            # As a fallback, try to parse binders from the declared type itself (Pi/arrow/exists).
            decl_type = __extract_type_ast(enclosing_theorem)
            if decl_type is not None:
                parsed_pi_binders = __parse_pi_binders_from_type(decl_type)
                if parsed_pi_binders:
                    theorem_binders.extend(parsed_pi_binders)
        # Fallback: if the enclosing theorem has no explicit binder list (e.g., only `∀`/`→`),
        # synthesize binders directly from the goal-context types collected from sorries.
        if not theorem_binders and goal_var_types:
            fallback_source = target_goal_var_types or goal_var_types
            seen_fallback: set[str] = set()
            added_any = False
            for name, typ in fallback_source.items():
                if not any(ch.isalnum() or ch == "_" for ch in name):
                    continue
                if name in seen_fallback:
                    continue
                # Require relevance: name is referenced in target or a dependency.
                if not (__is_referenced_in(target, name) or name in deps):
                    continue
                binder = __make_binder_from_type_string(name, typ)
                theorem_binders.append(binder)
                seen_fallback.add(name)
                added_any = True
            # If nothing qualified via relevance, fall back to the first goal-context name to avoid
            # dropping all theorem parameters in quantifier-only headers.
            if not added_any:
                kept: list[str] = []
                for name, typ in fallback_source.items():
                    if not any(ch.isalnum() or ch == "_" for ch in name):
                        continue
                    if name in seen_fallback:
                        continue
                    # Always keep the first name.
                    if not kept:
                        binder = __make_binder_from_type_string(name, typ)
                        theorem_binders.append(binder)
                        seen_fallback.add(name)
                        kept.append(name)
                        continue
                    # For subsequent names, keep only if their type references a kept name.
                    if any(k in str(typ) for k in kept):
                        binder = __make_binder_from_type_string(name, typ)
                        theorem_binders.append(binder)
                        seen_fallback.add(name)
                        kept.append(name)

    # Find earlier bindings (have, let, obtain) that appear textually before the target
    earlier_bindings: list[tuple[str, str, dict]] = []
    if enclosing_theorem is not None:
        earlier_bindings = __find_earlier_bindings(enclosing_theorem, lookup_name, name_map, anon_have_by_id)

    if not deps:
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
    # Initialize existing_binder_names set for duplicate prevention
    existing_binder_names = {__extract_binder_name(b) for b in binders if __extract_binder_name(b) is not None}
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
                sorries=sorries,
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
            # Special case: For have bindings with existential types, extract witness binders first
            if binding_type == "have":
                # Extract type AST (have bindings have types in AST)
                type_ast = __extract_type_ast(binding_node, binding_name=binding_name)
                if type_ast:
                    # Handle __type_container case: find exists node inside
                    exists_node = None
                    if type_ast.get("kind") == "__type_container":
                        exists_node = __find_first(
                            type_ast,
                            lambda n: __normalize_kind(n.get("kind", ""))
                            in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"},
                        )
                    elif __normalize_kind(type_ast.get("kind", "")) in {
                        "Lean.Parser.Term.exists",
                        "Lean.Parser.Term.existsContra",
                    }:
                        exists_node = type_ast

                    # If we found an existential type, extract witness binders
                    if exists_node:
                        witness_binders = __extract_all_exists_witness_binders(exists_node)
                        # Check if witnesses are referenced in target or dependencies
                        for witness_binder in witness_binders:
                            witness_name = __extract_binder_name(witness_binder)
                            if (
                                witness_name
                                and (witness_name in deps or __is_referenced_in(target, witness_name))
                                and witness_name not in existing_binder_names
                            ):
                                binders.append(witness_binder)
                                existing_binder_names.add(witness_name)

            # Continue with normal binder creation
            binder = __determine_general_binding_type(binding_name, binding_type, binding_node, goal_var_types)
            binders.append(binder)
            # Track the binding name in existing_names (it's already there, but this ensures consistency)
            existing_names.add(binding_name)

    # Post-process: Add variables from equality hypotheses that are also used elsewhere
    # This fixes the bug where equality hypotheses reference undefined variables
    # (e.g., "hx : x = value" but x itself is not defined as a parameter)
    # Note: existing_binder_names was initialized earlier, but refresh it here to include any binders added during the loop
    existing_binder_names = {__extract_binder_name(b) for b in binders if __extract_binder_name(b) is not None}
    for var_name in variables_in_equality_hypotheses:
        # Check if variable is used in subgoal (dependencies) or should be included (goal_var_types)
        if (
            var_name in deps or (goal_var_types and var_name in goal_var_types)
        ) and var_name not in existing_binder_names:
            # Variable is used but only has equality hypothesis - add as parameter too
            if goal_var_types and var_name in goal_var_types:
                binder = __make_binder_from_type_string(var_name, goal_var_types[var_name])
            else:
                # Try to extract type from AST
                var_node = name_map.get(var_name)
                var_type_ast = __extract_type_ast(var_node, binding_name=var_name) if var_node else None
                binder = __make_binder(var_name, var_type_ast)
            binders.append(binder)
            existing_binder_names.add(var_name)
            logging.debug(
                f"_get_named_subgoal_rewritten_ast: Added variable '{var_name}' as parameter "
                "because it's used in subgoal but only had equality hypothesis"
            )

    # Finally, add any remaining dependencies not yet included
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
                sorries=sorries,
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

    # Also add any variables from the goal context that aren't dependencies but are used.
    # Run this even if theorem binders exist, to pick up remaining goal-context variables.
    if goal_var_types:
        defined_in_target = __collect_defined_names(target)
        binder_source = goal_var_types
        referenced_names = {name for name in binder_source if __is_referenced_in(target, name)}
        target_goal_names = set(target_goal_var_types.keys())
        earlier_binding_names = {name for name, _, _ in earlier_bindings}
        target_full_referenced = {name for name in target_sorry_full_types if __is_referenced_in(target, name)}
        ascii_referenced = set()
        for name in target_sorry_full_types:
            if __is_referenced_in(target, name):
                normalized = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch == "_"))
                if normalized:
                    ascii_referenced.add(normalized)
        # Also allow ASCII versions of non-ASCII variables when both forms exist in the goal context.
        for name in goal_var_types:
            if not name.isascii():
                normalized = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch == "_"))
                if normalized and normalized in goal_var_types:
                    ascii_referenced.add(normalized)
        allowed_names = (
            set(deps)
            | referenced_names
            | earlier_binding_names
            | variables_in_equality_hypotheses
            | target_goal_names
            | target_full_referenced
            | ascii_referenced
        )

        for var_name in binder_source:  # preserves goal-context insertion order
            if var_name == target_name:
                continue
            if var_name not in allowed_names:
                continue
            # Skip if already present or locally defined
            if var_name in existing_binder_names or var_name in defined_in_target:
                continue
            # Don't skip variables in variables_in_equality_hypotheses - we need BOTH
            # the equality hypothesis (e.g., hN : N = value) AND the type annotation (e.g., N : Nat)
            # for standalone lemmas. The existing_binder_names check above prevents duplicates.
            binder = __make_binder_from_type_string(var_name, binder_source[var_name])
            binders.append(binder)
            existing_binder_names.add(var_name)

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
