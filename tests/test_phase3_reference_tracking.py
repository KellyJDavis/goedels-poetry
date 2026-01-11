"""
Investigation test for Phase 3: How variables are tracked through the pipeline.

This test focuses on understanding the difference between:
1. __find_dependencies (exact string match in val fields)
2. __is_referenced_in (recursive AST search)

And how they're used in binder creation.
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.decl_collection import (
    __find_dependencies,
)
from goedels_poetry.parsers.util.collection_and_analysis.reference_checking import (
    __is_referenced_in,
)


def test_reference_tracking_mechanisms() -> None:
    """
    Test: How do __find_dependencies and __is_referenced_in differ?

    This test investigates the key difference between these two mechanisms:
    - __find_dependencies: Looks for exact string matches in val fields
    - __is_referenced_in: Recursively searches AST tree
    """
    # Create a minimal AST where variable N appears in a compound expression "N > 0"
    # This represents: have h1 : N > 0 := by sorry
    target_ast = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "h1", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "N > 0", "info": {"leading": "", "trailing": " "}},  # N appears here
                                ],
                            },
                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                            {"kind": "Lean.Parser.Term.byTactic", "args": []},
                        ],
                    },
                ],
            },
        ],
    }

    # Create name_map with N (from let binding)
    name_map = {"N": {"kind": "Lean.Parser.Tactic.tacticLet_", "args": []}}

    # Test __find_dependencies
    deps = __find_dependencies(target_ast, name_map)

    # Test __is_referenced_in
    is_referenced = __is_referenced_in(target_ast, "N")

    print("\nTest: Reference tracking mechanisms")
    print("  Target AST type: 'N > 0' (stored as single string)")
    print(f"  'N' in name_map: {'N' in name_map}")
    print(f"  __find_dependencies result: {deps}")
    print(f"    'N' in deps: {'N' in deps}")
    print(f"  __is_referenced_in(target, 'N'): {is_referenced}")
    print()

    # Analysis
    print("  ANALYSIS:")
    print("    - When 'N > 0' is stored as single string: {'val': 'N > 0'}")
    print("      Both mechanisms check if val == 'N' (exact match)")
    print("      Since 'N > 0' != 'N', neither finds N")
    print("    - When 'N > 0' is parsed as tree with separate nodes:")
    print("      Both mechanisms recursively search and find {'val': 'N'} node")
    print()

    # Expected behavior for this test (single string case)
    if "N" not in deps:
        print("  ✓ VALIDATED: __find_dependencies does NOT find N when stored as single string")
    else:
        print("  ✗ UNEXPECTED: __find_dependencies found N")

    if not is_referenced:
        print("  ✓ VALIDATED: __is_referenced_in does NOT find N when stored as single string")
    else:
        print("  ✗ UNEXPECTED: __is_referenced_in found N")

    print()
    print("  KEY INSIGHT: Both mechanisms work IF AST structure is correct (tree with separate nodes)")
    print("  They fail IF expressions are stored as single strings")
    print()


def test_full_extraction_with_let_binding() -> None:
    """
    Test: Full extraction pipeline with let binding.

    Uses the actual _get_named_subgoal_rewritten_ast function to see what happens.
    """
    from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
    from goedels_poetry.parsers.util.high_level.subgoal_rewriting import _get_named_subgoal_rewritten_ast

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

    # Extract binders
    args = result.get("args", [])
    binders = None
    for arg in args:
        if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.bracketedBinderList":
            binders = arg.get("args", [])
            break

    # Convert to code
    result_code = _ast_to_code(result)

    # Extract binder names
    from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name

    binder_names = [__extract_binder_name(b) for b in (binders or []) if __extract_binder_name(b)]

    print("\nTest: Full extraction pipeline")
    print(f"  Result code:\n{result_code}\n")
    print(f"  Binder names: {binder_names}")
    print(f"  'N' in binder_names: {'N' in binder_names}")
    print(f"  '(N :' in result_code: {'(N :' in result_code or '(N:' in result_code}")
    print()

    # Analysis
    print("  ANALYSIS:")
    if "N" in binder_names:
        print("    ✓ N is in binders - extraction worked correctly")
    elif "(N :" in result_code or "(N:" in result_code:
        print("    ✓ N has type annotation - binder created")
    elif "N" in result_code:
        print("    ⚠ N appears in code but without type annotation")
    else:
        print("    ✗ N is missing from extracted lemma")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("INVESTIGATION: Reference Tracking Mechanisms")
    print("=" * 70)
    print()

    test_reference_tracking_mechanisms()
    test_full_extraction_with_let_binding()

    print("=" * 70)
    print("SUMMARY:")
    print("  Test 1: Reference tracking mechanisms - To be validated")
    print("  Test 2: Full extraction pipeline - To be validated")
    print("=" * 70)
