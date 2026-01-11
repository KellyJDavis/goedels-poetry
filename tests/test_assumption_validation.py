"""Focused tests to validate assumptions about bug causes.

These tests validate specific assumptions before including them in the bug fix plan.
Each test validates one specific assumption with minimal, focused test cases.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
    __parse_pi_binders_from_type,
)
from goedels_poetry.parsers.util.foundation.ast_to_code import _ast_to_code
from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name
from goedels_poetry.parsers.util.types_and_binders.binder_construction import (
    __make_binder_from_type_string,
)
from goedels_poetry.parsers.util.types_and_binders.type_extraction import (
    __extract_exists_witness_binder,
)


def test_assumption_1a_quantifier_only_signature_no_bracketed_binder_list() -> None:
    """
    ASSUMPTION: __extract_theorem_binders does not extract binders from quantifier-only
    theorem signatures when there's no explicit bracketedBinderList in the AST.

    VALIDATION: Test with a theorem that has a quantifier-only signature (type is "∀ n : ℤ, ...")
    and no bracketedBinderList node in the AST structure.
    """
    # Create a minimal theorem AST with quantifier-only signature (no bracketedBinderList)
    # This simulates what Kimina might produce for "theorem A : ∀ n : ℤ, n > 1 → True := by sorry"
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            # Type is just a string token (not parsed into AST structure with forall nodes)
            # This is a simplified representation - real Kimina ASTs might be more complex
            {"val": "∀ n : ℤ, n > 1 → True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [
                            {
                                "kind": "Lean.Parser.Tactic.tacticSorry",
                                "args": [{"val": "sorry", "info": {"leading": "", "trailing": ""}}],
                            }
                        ],
                    },
                ],
            },
        ],
    }

    goal_var_types: dict[str, str] = {}
    binders = __extract_theorem_binders(theorem_ast, goal_var_types)

    # If the assumption is TRUE: binders will be empty (no extraction happened)
    # If the assumption is FALSE: binders will contain (n : ℤ) or similar
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]
    print(f"\nAssumption 1a: Extracted binder names: {binder_names}")
    print(f"Assumption 1a: Number of binders: {len(binders)}")

    # This test documents what actually happens - we'll use the result to validate the assumption
    # Note: This simplified AST may not accurately represent real Kimina output, but it tests
    # the code path when there's no bracketedBinderList and the type is just a string token


def test_assumption_1b_parse_pi_binders_handles_forall() -> None:
    """
    ASSUMPTION: __parse_pi_binders_from_type can extract binders from forall types,
    but only if the type AST has proper forall node structure.

    VALIDATION: Test with a type AST that has a forall structure.
    """
    # Create a type AST with forall structure
    # This represents "∀ n : ℤ, n > 1"
    type_ast_with_forall = {
        "kind": "Lean.Parser.Term.forall",
        "args": [
            # args[0] should be a binder list or binder node
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    {"kind": "Lean.binderIdent", "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}]},
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "ℤ", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            # args[1] is the body
            {"val": "n > 1", "info": {"leading": " ", "trailing": ""}},
        ],
    }

    binders = __parse_pi_binders_from_type(type_ast_with_forall)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print(f"\nAssumption 1b: Extracted binder names from forall type: {binder_names}")
    print(f"Assumption 1b: Number of binders: {len(binders)}")

    # This validates whether __parse_pi_binders_from_type can handle forall structures
    # If it works, we have n in binder_names
    # If it doesn't work, binder_names is empty


def test_assumption_2a_goal_context_parses_let_binding_with_assignment() -> None:
    """
    ASSUMPTION: Goal context parsing may not capture variables correctly when they appear
    in assignments with ":=" syntax (e.g., "N : ℕ := Int.toNat n").

    VALIDATION: Test __parse_goal_context with a goal string containing let binding syntax.
    """
    # Goal context as it might appear in a sorry goal for a theorem with "let N : ℕ := Int.toNat n"
    goal_with_let = "n : ℤ\nhn : n > 1\nN : ℕ := Int.toNat n\n⊢ ∃ c d : ℕ, (3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N"

    parsed_types = __parse_goal_context(goal_with_let)

    print(f"\nAssumption 2a: Parsed types from goal with let binding: {parsed_types}")
    print(f"Assumption 2a: 'N' in parsed_types: {'N' in parsed_types}")
    if "N" in parsed_types:
        print(f"Assumption 2a: Type of 'N': {parsed_types['N']}")

    # If the assumption is TRUE: 'N' is NOT in parsed_types, or has wrong type
    # If the assumption is FALSE: 'N' IS in parsed_types with type "ℕ"


def test_assumption_2b_type_string_has_leading_colon_causes_double_colon() -> None:
    """
    ASSUMPTION: Type strings with leading colons (e.g., ": ℕ" instead of "ℕ") cause
    double colon syntax errors when creating binders.

    VALIDATION: Test __make_binder_from_type_string with type strings that have leading colons.
    """
    test_cases = [
        (": ℕ", "Single leading colon with space"),
        (":ℕ", "Single leading colon no space"),
        (" : ℕ", "Leading space and colon"),
        (": : ℕ", "Double colon"),
        ("ℕ", "No colon (control)"),
    ]

    for type_str, description in test_cases:
        binder = __make_binder_from_type_string("N", type_str)
        binder_code = _ast_to_code(binder)
        print(f"\nAssumption 2b ({description}):")
        print(f"  Input type string: {type_str!r}")
        print(f"  Output binder code: {binder_code}")
        # Check for double colon pattern "::" or ": :"
        has_double_colon = "::" in binder_code or ": :" in binder_code
        print(f"  Has double colon: {has_double_colon}")

        # If the assumption is TRUE: Some test cases produce double colons
        # If the assumption is FALSE: All test cases produce single colons


def test_assumption_3a_existential_witness_extraction() -> None:
    """
    ASSUMPTION: Witness variables from existential types (e.g., "c d" from "∃ c d : ℕ, ...")
    are not extracted and bound when they're referenced in later hypotheses.

    VALIDATION: Test if __extract_exists_witness_binder can extract witnesses from existential types.
    """
    # Create an existential type AST: "∃ c d : ℕ, (3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N"
    # This is simplified - real ASTs might have more complex structure
    existential_type_ast = {
        "kind": "Lean.Parser.Term.exists",
        "args": [
            # Witness binder(s) - could be a binder list or multiple binders
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [
                            {"val": "c", "info": {"leading": "", "trailing": " "}},
                            {"val": "d", "info": {"leading": " ", "trailing": ""}},
                        ],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "ℕ", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            # Body
            {"val": "(3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N", "info": {"leading": " ", "trailing": ""}},
        ],
    }

    witness_binder = __extract_exists_witness_binder(existential_type_ast)

    print(f"\nAssumption 3a: Witness binder extracted: {witness_binder is not None}")
    if witness_binder:
        witness_code = _ast_to_code(witness_binder)
        print(f"Assumption 3a: Witness binder code: {witness_code}")
        witness_name = __extract_binder_name(witness_binder)
        print(f"Assumption 3a: Witness name: {witness_name}")

    # If the assumption is TRUE: witness_binder is None, or only extracts one witness (c or d, not both)
    # If the assumption is FALSE: witness_binder contains both c and d (or properly handles multiple witnesses)


def test_assumption_3b_existential_with_multiple_witnesses() -> None:
    """
    ASSUMPTION: Existential types with multiple witnesses (e.g., "∃ c d : ℕ, ...") are not
    properly handled - only one witness is extracted or extraction fails.

    VALIDATION: Test extraction with multiple witnesses in the existential.
    """
    # Try a structure where witnesses are in a binderIdent with multiple names
    # This tests if the extraction handles "c d" as two separate witnesses
    existential_multi_witness = {
        "kind": "Lean.Parser.Term.exists",
        "args": [
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    # Multiple names in binderIdent
                    {
                        "kind": "Lean.binderIdent",
                        "args": [
                            {"val": "c", "info": {"leading": "", "trailing": " "}},
                            {"val": "d", "info": {"leading": " ", "trailing": ""}},
                        ],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "ℕ", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            {"val": "P c d", "info": {"leading": " ", "trailing": ""}},
        ],
    }

    witness_binder = __extract_exists_witness_binder(existential_multi_witness)

    print(f"\nAssumption 3b: Witness binder extracted (multi-witness): {witness_binder is not None}")
    if witness_binder:
        witness_code = _ast_to_code(witness_binder)
        print(f"Assumption 3b: Witness binder code: {witness_code}")

    # This validates whether multiple witnesses are properly extracted
    # Real validation would need to check if both "c" and "d" are accessible as separate binders
