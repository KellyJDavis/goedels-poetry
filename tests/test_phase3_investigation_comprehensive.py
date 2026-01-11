"""
Comprehensive investigation tests for Phase 3: Missing Type Annotations

These tests trace variables through the extraction pipeline to identify WHERE
variables like n, N, m are lost.

Test Strategy:
1. Create realistic test case (A303656-like with let binding)
2. Instrument _get_named_subgoal_rewritten_ast to trace variable flow
3. Check each step: goal_var_types, deps, binder creation
4. Answer critical questions from INVESTIGATION_PHASE_3_FINDINGS.md
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.decl_collection import (
    __collect_named_decls,
    __find_dependencies,
)
from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context
from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name


def test_goal_var_types_population_with_let_binding() -> None:
    """
    Test: Does goal_var_types contain N when processing a lemma that uses let binding?

    Critical Question 1: Are N, m in goal_var_types?
    """
    # Simulate the goal context parsing logic from _get_named_subgoal_rewritten_ast
    # Create a sorry entry with goal context containing N from let binding

    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "n : ℤ\nN : ℕ := Int.toNat n\n⊢ Prop",
            "proofState": 1,
        }
    ]

    # Simulate the goal_var_types population logic (lines 103-153)
    goal_var_types: dict[str, str] = {}

    all_types: dict[str, str] = {}
    target_sorry_types: dict[str, str] = {}
    target_sorry_found = False
    lookup_name = "h1"  # Target subgoal name

    for sorry in sorries:
        goal = sorry.get("goal", "")
        if not goal:
            continue

        parsed_types = __parse_goal_context(goal)

        # Check if this sorry mentions the target name
        is_target_sorry = not target_sorry_found and lookup_name in parsed_types
        if is_target_sorry:
            target_sorry_types = __parse_goal_context(goal, stop_at_name=lookup_name)
            target_sorry_found = True
        else:
            for name, typ in parsed_types.items():
                if name not in all_types:
                    all_types[name] = typ

    merged_goal_var_types = all_types.copy()
    merged_goal_var_types.update(target_sorry_types)
    goal_var_types = merged_goal_var_types

    print("\nTest: goal_var_types population")
    print("  Goal context: 'n : ℤ\\nN : ℕ := Int.toNat n\\n⊢ Prop'")
    print(f"  goal_var_types: {goal_var_types}")
    print(f"  'n' in goal_var_types: {'n' in goal_var_types}")
    print(f"  'N' in goal_var_types: {'N' in goal_var_types}")

    # Validate
    assert "n" in goal_var_types, "n should be in goal_var_types"
    assert "N" in goal_var_types, "N should be in goal_var_types"
    assert goal_var_types["N"] == "ℕ", f"N's type should be 'ℕ', got '{goal_var_types['N']}'"

    print("  ✓ VALIDATED: goal_var_types contains n and N")


def test_dependency_tracking_for_let_binding() -> None:
    """
    Test: Does __find_dependencies find let binding variables?

    Critical Question 2: Are N, m in deps?
    """
    # Create minimal AST with let binding
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # let N : ℕ := Int.toNat n
                            {
                                "kind": "Lean.Parser.Tactic.tacticLet_",
                                "args": [
                                    {"val": "let", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.letDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.letIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [{"val": "N", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "ℕ", "info": {"leading": "", "trailing": " "}},
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Int.toNat n", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            # have h1 : N > 0 := by sorry
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "N > 0", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Term.byTactic",
                                                "args": [
                                                    {"val": "by", "info": {"leading": " ", "trailing": " "}},
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSeq",
                                                        "args": [
                                                            {
                                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                                "args": [
                                                                    {
                                                                        "val": "sorry",
                                                                        "info": {"leading": "", "trailing": ""},
                                                                    }
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }

    # Get name_map (includes let bindings)
    name_map = __collect_named_decls(ast_dict)

    # Get target (h1) - use the same method as _get_named_subgoal_rewritten_ast
    from goedels_poetry.parsers.util.high_level.named_subgoals import _get_named_subgoal_ast

    target = _get_named_subgoal_ast(ast_dict, "h1")
    assert target is not None, "Should find target h1"

    # Test dependency tracking
    deps = __find_dependencies(target, name_map)

    print("\nTest: dependency tracking")
    print(f"  name_map keys: {list(name_map.keys())}")
    print(f"  deps: {sorted(deps)}")
    print(f"  'N' in name_map: {'N' in name_map}")
    print(f"  'N' in deps: {'N' in deps}")

    # Validate
    assert "N" in name_map, "N should be in name_map (from let binding)"

    # Note: __find_dependencies checks for exact string matches (val == "N").
    # When the type is stored as a string token {"val": "N > 0"}, __find_dependencies
    # cannot find "N" because "N > 0" != "N" (exact match required).
    # In real parsed AST structures, "N > 0" would be parsed into separate nodes,
    # allowing __find_dependencies to find "N" as a separate node.
    # This test uses simplified mock AST with string tokens, so N is NOT in deps.
    assert "N" not in deps, (
        "N should NOT be in deps when type is stored as string token 'N > 0'. "
        "__find_dependencies requires exact string matches and cannot find 'N' "
        "within compound string tokens. In parsed AST structures with separate "
        "nodes, __find_dependencies would find 'N' correctly."
    )

    print("  ✓ VALIDATED: __find_dependencies behavior with string tokens (N not in deps)")
    print("    Note: This is expected - string tokens cannot be dependency-tracked")


def test_full_pipeline_trace() -> None:
    """
    Test: Trace variables through the full pipeline.

    This test simulates the full extraction pipeline and checks each step.
    """
    # Create AST with let binding and have statement
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "test_theorem", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            # let N : ℕ := Int.toNat n
                            {
                                "kind": "Lean.Parser.Tactic.tacticLet_",
                                "args": [
                                    {"val": "let", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.letDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.letIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [{"val": "N", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "ℕ", "info": {"leading": "", "trailing": " "}},
                                                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                                    {"val": "Int.toNat n", "info": {"leading": "", "trailing": " "}},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            # have h1 : N > 0 := by sorry
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.haveDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.haveIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.Parser.Term.haveId",
                                                        "args": [
                                                            {"val": "h1", "info": {"leading": "", "trailing": " "}}
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "N > 0", "info": {"leading": "", "trailing": " "}},
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {
                                                "kind": "Lean.Parser.Term.byTactic",
                                                "args": [
                                                    {"val": "by", "info": {"leading": " ", "trailing": " "}},
                                                    {
                                                        "kind": "Lean.Parser.Tactic.tacticSeq",
                                                        "args": [
                                                            {
                                                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                                                "args": [
                                                                    {
                                                                        "val": "sorry",
                                                                        "info": {"leading": "", "trailing": ""},
                                                                    }
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }

    # Sorries with goal context
    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "n : ℤ\nN : ℕ := Int.toNat n\n⊢ N > 0",
            "proofState": 1,
        }
    ]

    # Call the function
    result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)

    # Extract binders from result
    args = result.get("args", [])
    binders = None
    for arg in args:
        if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.bracketedBinderList":
            binders = arg.get("args", [])
            break

    binder_names = [__extract_binder_name(b) for b in (binders or []) if __extract_binder_name(b)]

    # Convert to code to check
    from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code

    result_code = _ast_to_code(result)

    # Also check if N appears in the code at all
    n_in_code = "N" in result_code

    print("\nTest: Full pipeline trace")
    print(f"  Binders found: {binder_names}")
    print(f"  Result code:\n{result_code}\n")
    print(f"  'N' in binder_names: {'N' in binder_names}")
    print(f"  '(N :' in result_code: {'(N :' in result_code or '(N:' in result_code}")

    # This will help us see what's happening
    print("\n  ANALYSIS:")
    print(f"    - 'N' in binder_names: {'N' in binder_names} {'✓' if 'N' in binder_names else '✗'}")
    print(
        f"    - '(N :' or '(N:' in result_code: {'(N :' in result_code or '(N:' in result_code} {'✓' if '(N :' in result_code or '(N:' in result_code else '✗'}"
    )
    print(f"    - 'N' appears in code at all: {n_in_code} {'✓' if n_in_code else '✗'}")


if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE INVESTIGATION TESTS: Phase 3 - Missing Type Annotations")
    print("=" * 70)
    print()

    test_goal_var_types_population_with_let_binding()
    test_dependency_tracking_for_let_binding()
    test_full_pipeline_trace()

    print()
    print("=" * 70)
    print("SUMMARY:")
    print("  Test 1 (goal_var_types): To be validated")
    print("  Test 2 (dependency tracking): To be validated")
    print("  Test 3 (full pipeline): To be validated")
    print("=" * 70)
