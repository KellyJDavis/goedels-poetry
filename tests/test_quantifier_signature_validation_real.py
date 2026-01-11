"""Validation test for quantifier-only signature using actual log data.

This test validates whether __extract_theorem_binders properly extracts binders
from quantifier-only theorem signatures by checking the actual behavior with
realistic AST structures based on the A303656 theorem from the log file.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util.collection_and_analysis.theorem_binders import (
    __extract_theorem_binders,
    __parse_pi_binders_from_type,
)
from goedels_poetry.parsers.util.names_and_bindings.name_extraction import __extract_binder_name


def test_parse_pi_binders_from_forall_type() -> None:
    """
    Test: Does __parse_pi_binders_from_type extract binders from forall types?

    Based on log analysis:
    - A303656 theorem has signature: "∀ n : ℤ, n > 1 → ..."
    - The AST has declSig with forall structure
    - Goal context shows "n : ℤ" (meaning n should be extracted)
    """
    # Create a forall type AST structure like what Kimina produces
    # This represents "∀ n : ℤ, n > 1 → ..."
    type_ast_with_forall = {
        "kind": "Lean.Parser.Term.forall",
        "args": [
            # args[0] should contain the binder
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
            # args[1] is the body (n > 1 → ...)
            {
                "kind": "Lean.Parser.Term.arrow",
                "args": [
                    {"val": "n > 1", "info": {"leading": " ", "trailing": ""}},
                    {"val": "...", "info": {"leading": " ", "trailing": ""}},
                ],
            },
        ],
    }

    binders = __parse_pi_binders_from_type(type_ast_with_forall)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\nTest parse_pi_binders_from_forall_type:")
    print(f"  Extracted {len(binders)} binders")
    print(f"  Binder names: {binder_names}")

    # VALIDATION RESULT
    assert "n" in binder_names, (
        "__parse_pi_binders_from_type should extract binders from forall types. "
        f"Expected 'n' in binder_names, but got: {binder_names}"
    )
    print("  ✓ VALIDATED: __parse_pi_binders_from_type CAN extract binders from forall types")


def test_extract_theorem_binders_with_declSig_forall() -> None:
    """
    Test: Does __extract_theorem_binders extract binders when declSig has forall structure?

    Based on log analysis (lines 16400-16500):
    - The AST has "kind": "Lean.Parser.Command.declSig"
    - Inside declSig, there's "kind": "Lean.Parser.Term.forall"
    - The code extracts from declSig first (line 69-71)
    """
    # Create a theorem AST with declSig containing forall structure
    # This simulates what Kimina produces for "theorem A303656 : ∀ n : ℤ, ..."
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
                    # declSig contains the forall type structure
                    {
                        "kind": "Lean.Parser.Term.forall",
                        "args": [
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
                            # Body would be here
                            {"val": "...", "info": {"leading": " ", "trailing": ""}},
                        ],
                    },
                ],
            },
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {"kind": "Lean.Parser.Term.byTactic", "args": []},
        ],
    }

    goal_var_types: dict[str, str] = {}  # Not used by __extract_theorem_binders
    binders = __extract_theorem_binders(theorem_ast, goal_var_types)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\nTest extract_theorem_binders_with_declSig_forall:")
    print(f"  Extracted {len(binders)} binders")
    print(f"  Binder names: {binder_names}")

    # VALIDATION RESULT
    assert "n" in binder_names, (
        "__extract_theorem_binders should extract binders from declSig with forall. "
        f"Expected 'n' in binder_names, but got: {binder_names}. "
        "Note: The code extracts from declSig (lines 69-71), so if this fails, "
        "it means the extraction logic doesn't handle forall in declSig properly"
    )
    print("  ✓ VALIDATED: __extract_theorem_binders CAN extract binders from declSig with forall")


def test_goal_var_types_not_used() -> None:
    """
    Test: Does __extract_theorem_binders use goal_var_types parameter?

    VALIDATION: The code inspection shows it does NOT use goal_var_types.
    This test confirms that.
    """
    # Create minimal theorem AST (no binders in AST)
    theorem_ast = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem"},
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "A303656"}]},
            {"val": ":"},
            {"val": "∀ n : ℤ, True"},  # Type as string (not parsed)
            {"val": ":="},
            {"kind": "Lean.Parser.Term.byTactic", "args": []},
        ],
    }

    goal_var_types = {"n": "ℤ"}  # Would have n from goal context

    binders = __extract_theorem_binders(theorem_ast, goal_var_types)
    binder_names = [__extract_binder_name(b) for b in binders if __extract_binder_name(b)]

    print("\nTest goal_var_types_not_used:")
    print(f"  goal_var_types provided: {goal_var_types}")
    print(f"  Extracted {len(binders)} binders")
    print(f"  Binder names: {binder_names}")

    # VALIDATION RESULT
    assert "n" not in binder_names, (
        "__extract_theorem_binders should NOT use goal_var_types parameter. "
        f"Expected 'n' not in binder_names, but got: {binder_names}. "
        "Conclusion: goal_var_types parameter should be ignored - only AST structure should be used"
    )
    print("  ✓ VALIDATED: __extract_theorem_binders does NOT use goal_var_types")
    print("  Conclusion: goal_var_types parameter is ignored - only AST structure is used")


if __name__ == "__main__":
    print("=" * 70)
    print("Validation Tests for Quantifier-Only Signature Assumption")
    print("=" * 70)

    test_parse_pi_binders_from_forall_type()
    test_extract_theorem_binders_with_declSig_forall()
    test_goal_var_types_not_used()

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  All tests passed (see output above for validation details)")
    print("=" * 70)
