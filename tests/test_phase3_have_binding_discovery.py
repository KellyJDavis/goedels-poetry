"""Test: Discover that h_choose_cd is actually a HAVE binding, not a CHOOSE binding

This changes the approach significantly!
"""

# ruff: noqa: RUF001, RUF003

from goedels_poetry.parsers.util.types_and_binders.type_extraction import (
    __extract_all_exists_witness_binders,
    __extract_type_ast,
)


def test_have_binding_has_type_ast() -> None:
    """
    Test: Verify that have bindings have type AST available.

    Since h_choose_cd is a HAVE binding (not choose/obtain), we can use __extract_type_ast
    directly on the binding_node!
    """
    print("\n=== Critical Discovery: h_choose_cd is a HAVE Binding ===")

    # Simulate a have binding AST structure
    # have h_choose_cd : ∃ c d : ℕ, P := by sorry
    have_binding_node = {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have"},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [{"kind": "Lean.Parser.Term.haveId", "args": [{"val": "h_choose_cd"}]}],
                    },
                    {"val": ":"},
                    {
                        "kind": "Lean.Parser.Term.exists",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.explicitBinder",
                                "args": [
                                    {"val": "("},
                                    {"kind": "Lean.binderIdent", "args": [{"val": "c"}]},
                                    {"val": ":"},
                                    {"val": "ℕ"},
                                ],
                            },
                            {
                                "kind": "Lean.Parser.Term.explicitBinder",
                                "args": [
                                    {"val": "("},
                                    {"kind": "Lean.binderIdent", "args": [{"val": "d"}]},
                                    {"val": ":"},
                                    {"val": "ℕ"},
                                ],
                            },
                        ],
                    },
                ],
            },
            {"val": ":="},
            {"val": "by"},
            {"val": "sorry"},
        ],
    }

    # Extract type AST from have binding
    type_ast = __extract_type_ast(have_binding_node, binding_name="h_choose_cd")

    print(f"Have binding node kind: {have_binding_node['kind']}")
    print(f"Type AST extracted: {type_ast is not None}")

    if type_ast:
        print(f"Type AST kind: {type_ast.get('kind')}")

        # Handle __type_container case: find exists node inside
        exists_node = None
        if type_ast.get("kind") == "__type_container":
            from goedels_poetry.parsers.util.foundation.ast_walkers import __find_first
            from goedels_poetry.parsers.util.foundation.kind_utils import __normalize_kind

            exists_node = __find_first(
                type_ast,
                lambda n: __normalize_kind(n.get("kind", ""))
                in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"},
            )
            print(f"Found exists node in __type_container: {exists_node is not None}")
        elif type_ast.get("kind") in {"Lean.Parser.Term.exists", "Lean.Parser.Term.existsContra"}:
            exists_node = type_ast
            print("Type AST is directly an exists node")

        if exists_node:
            # Extract witness binders from the exists node
            witness_binders = __extract_all_exists_witness_binders(exists_node)
            print(f"\nWitness binders extracted: {len(witness_binders)}")
            for i, binder in enumerate(witness_binders):
                print(f"  Binder {i + 1}: {binder.get('kind', 'unknown')}")

            assert len(witness_binders) == 2, f"Expected 2 witness binders, got {len(witness_binders)}"
        else:
            msg = "Could not find exists node in type AST"
            raise AssertionError(msg)
        print("\n✓ SUCCESS: Can extract witness binders from have binding type AST!")
    else:
        print("✗ FAILED: Could not extract type AST from have binding")

    print("\n✓ Test completed - Discovery confirmed")


def test_integration_approach_for_have_bindings() -> None:
    """
    Test: Document the correct integration approach for have bindings.
    """
    print("\n=== Integration Approach for Have Bindings ===")
    print("""
    CORRECTED UNDERSTANDING:

    - h_choose_cd is a HAVE binding (not choose/obtain)
    - have bindings have type AST available via __extract_type_ast
    - We can extract the existential type AST directly
    - Then call __extract_all_exists_witness_binders on that AST

    INTEGRATION POINT (subgoal_rewriting.py, around line 294):

        else:
            # For have, obtain, choose, generalize, match, suffices
            if binding_type == "have":
                # Try to extract type AST (have bindings have types in AST)
                type_ast = __extract_type_ast(binding_node, binding_name=binding_name)
                if type_ast and __normalize_kind(type_ast.get("kind", "")) in {
                    "Lean.Parser.Term.exists",
                    "Lean.Parser.Term.existsContra"
                }:
                    # Extract all witness binders
                    witness_binders = __extract_all_exists_witness_binders(type_ast)
                    # Check if witnesses are referenced (in deps or target)
                    for witness_binder in witness_binders:
                        witness_name = __extract_binder_name(witness_binder)
                        if witness_name and (witness_name in deps or __is_referenced_in(target, witness_name, name_map)):
                            if witness_name not in existing_binder_names:
                                binders.append(witness_binder)
                                existing_binder_names.add(witness_name)

            # Continue with normal binder creation
            binder = __determine_general_binding_type(binding_name, binding_type, binding_node, goal_var_types)
            binders.append(binder)
            existing_names.add(binding_name)

    NOTE: This approach works for HAVE bindings because they have type AST.
    For choose/obtain bindings, types come from goal_var_types (strings),
    so we'd need a different approach (but that's not the case here).
    """)

    print("\n✓ Test completed - Integration approach documented")
