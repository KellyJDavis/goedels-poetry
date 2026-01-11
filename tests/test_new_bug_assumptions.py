"""Validation tests for assumptions in the new bug analysis.

This file validates assumptions about missing theorem signature variables and hypotheses
when extracting standalone lemmas from decomposed theorems.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
    __parse_pi_binders_from_type,
)
from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name


def test_assumption_rc1_extract_theorem_binders_quantifier_only() -> None:
    """
    VALIDATION TEST: Does __extract_theorem_binders extract binders from quantifier-only signatures?

    Assumption: __extract_theorem_binders may not extract binders from quantifier-only signatures.

    Test: Create a theorem AST with signature ∀ n : ℤ, n > 1 → ... (quantifier-only, no bracketedBinderList)
    and check if it extracts n : ℤ.
    """
    # Create a realistic theorem AST structure with quantifier-only signature
    # Structure: theorem A303656 : ∀ n : ℤ, n > 1 → ... := by ...
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Command.declSig",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.forall",
                        "args": [
                            # args[0]: binder (n : ℤ)
                            {
                                "kind": "Lean.Parser.Term.explicitBinder",
                                "args": [
                                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "ℤ", "info": {"leading": "", "trailing": ""}},
                                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                                ],
                            },
                            # args[1]: body (n > 1 → ...)
                            {
                                "kind": "Lean.Parser.Term.arrow",
                                "args": [
                                    {"val": "n > 1", "info": {"leading": " ", "trailing": ""}},
                                    {"val": "...", "info": {"leading": " ", "trailing": ""}},
                                ],
                            },
                        ],
                    },
                ],
            },
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {"kind": "Lean.Parser.Term.byTactic", "args": []},
        ],
    }

    goal_var_types: dict[str, str] = {}  # Empty for this test

    binders = __extract_theorem_binders(theorem_ast, goal_var_types)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 1 - Quantifier-only signature extraction")
    print("=" * 70)
    print("Theorem AST: theorem A303656 : ∀ n : ℤ, n > 1 → ...")
    print(f"Extracted {len(binders)} binders")
    print(f"Binder names: {binder_names}")
    print()

    # VALIDATION RESULT
    if "n" in binder_names:
        print("✓ ASSUMPTION IS FALSE: __extract_theorem_binders DOES extract n from quantifier-only signature")
        print("  Root Cause 1 is NOT a problem for this case.")
    else:
        print("✗ ASSUMPTION IS TRUE: __extract_theorem_binders does NOT extract n from quantifier-only signature")
        print("  Root Cause 1 is a VALIDATED problem.")

    # Store result for later analysis
    assert isinstance(binders, list), "binders should be a list"


def test_assumption_rc2_arrow_domain_names() -> None:
    """
    VALIDATION TEST: Do arrow domain hypotheses get wrong names (auto-generated vs actual)?

    Assumption: Arrow domain hypotheses get auto-generated names (h, h1) instead of actual names from proof (hn).

    Test: Use __parse_pi_binders_from_type on a type AST with arrow domain and check names.
    """
    # Create type AST representing ∀ n : ℤ, n > 1 → ...
    type_ast = {
        "kind": "Lean.Parser.Term.forall",
        "args": [
            # args[0]: binder (n : ℤ)
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    {
                        "kind": "Lean.binderIdent",
                        "args": [{"val": "n", "info": {"leading": "", "trailing": ""}}],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "ℤ", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            # args[1]: body (n > 1 → ...)
            {
                "kind": "Lean.Parser.Term.arrow",
                "args": [
                    {"val": "n > 1", "info": {"leading": " ", "trailing": ""}},
                    {"val": "...", "info": {"leading": " ", "trailing": ""}},
                ],
            },
        ],
    }

    binders = __parse_pi_binders_from_type(type_ast, goal_var_types=None)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 2 - Arrow domain hypothesis names")
    print("=" * 70)
    print("Type AST: ∀ n : ℤ, n > 1 → ...")
    print(f"Extracted {len(binders)} binders")
    print(f"Binder names: {binder_names}")
    print()

    # Check if arrow domain hypothesis has auto-generated name
    # Expected: should have n (from forall) and h or h1 (from arrow domain)
    has_auto_generated_name = any(name.startswith("h") and name != "n" for name in binder_names if name)

    # VALIDATION RESULT
    if has_auto_generated_name:
        print("✓ ASSUMPTION IS TRUE: Arrow domain hypothesis gets auto-generated name (h, h1, etc.)")
        print("  Root Cause 2 is a VALIDATED problem.")
        print(f"  Auto-generated names found: {[n for n in binder_names if n and n.startswith('h') and n != 'n']}")
    else:
        print("✗ ASSUMPTION IS FALSE: Arrow domain hypothesis does NOT get auto-generated name")
        print("  Root Cause 2 is NOT a problem for this case.")

    assert isinstance(binders, list), "binders should be a list"


def test_assumption_rc4_goal_var_types_population() -> None:
    """
    VALIDATION TEST: Does goal_var_types contain theorem signature variables?

    Assumption: goal_var_types might not contain theorem signature variables like n : ℤ and hn : n > 1.

    Test: Parse goal context strings that should contain theorem signature variables.
    """
    # Simulate goal context from first sorry in theorem
    # Format: "n : ℤ\nhn : n > 1\n⊢ Prop"
    goal_context_str = "n : ℤ\nhn : n > 1\n⊢ Prop"

    parsed_types = __parse_goal_context(goal_context_str)

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 4 - Goal context population")
    print("=" * 70)
    print(f"Goal context string: {goal_context_str!r}")
    print(f"Parsed types: {parsed_types}")
    print()

    has_n = "n" in parsed_types
    has_hn = "hn" in parsed_types

    # VALIDATION RESULT
    if has_n and has_hn:
        print("✓ ASSUMPTION IS FALSE: goal_var_types DOES contain theorem signature variables (n, hn)")
        print("  Root Cause 4 is NOT a problem - parsing works correctly.")
        print(f"  n : {parsed_types.get('n')}")
        print(f"  hn : {parsed_types.get('hn')}")
    else:
        print("✗ ASSUMPTION IS TRUE: goal_var_types does NOT contain theorem signature variables")
        print("  Root Cause 4 is a VALIDATED problem.")
        if not has_n:
            print("  Missing: n")
        if not has_hn:
            print("  Missing: hn")

    assert isinstance(parsed_types, dict), "parsed_types should be a dict"


def test_assumption_rc3_fallback_trigger_condition() -> None:
    """
    VALIDATION TEST: Does goal context fallback include all theorem signature variables?

    Assumption: Goal context fallback may not include all theorem signature variables,
    especially when relevance check fails and only first name is kept.

    Test: This is more complex - requires full extraction pipeline.
    For now, we'll test if the fallback condition is correctly structured.
    """
    # This test requires a full theorem AST and extraction pipeline
    # For validation purposes, we'll check if the fallback code structure exists
    # The actual validation would require running _get_named_subgoal_rewritten_ast
    # with theorem_binders = [] and goal_var_types containing n and hn

    print("\n" + "=" * 70)
    print("VALIDATION TEST: Root Cause 3 - Goal context fallback")
    print("=" * 70)
    print("This test requires full extraction pipeline setup.")
    print("Manual inspection of code structure shows:")
    print("  - Fallback code exists at lines 178-216 in subgoal_rewriting.py")
    print("  - Condition: if not theorem_binders and goal_var_types:")
    print("  - Relevance check: __is_referenced_in(target, name) or name in deps")
    print("  - First-name-only fallback exists (lines 196-208)")
    print()
    print("NOTE: This assumption requires integration test to fully validate.")
    print("      Code inspection suggests the assumption COULD be true, but needs testing.")

    # This is a placeholder - actual validation would require more setup
    # We'll mark this as requiring further investigation


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VALIDATION TESTS FOR NEW BUG ASSUMPTIONS")
    print("=" * 70)
    print()

    test_assumption_rc1_extract_theorem_binders_quantifier_only()
    test_assumption_rc2_arrow_domain_names()
    test_assumption_rc4_goal_var_types_population()
    test_assumption_rc3_fallback_trigger_condition()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
