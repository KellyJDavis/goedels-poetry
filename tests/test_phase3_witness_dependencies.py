"""Test 3: Dependency Tracking for Witnesses

Purpose: Determine when witness binders should be added (only if referenced?)
"""

# ruff: noqa: RUF001, RUF002

from goedels_poetry.parsers.util.collection_and_analysis.decl_collection import __find_dependencies


def test_witness_referenced_in_target() -> None:
    """
    Test: Check if witnesses (c, d) are found by dependency tracking.

    Simulate a target that references c and d:
    have hm : m = N - ((3 : ℕ) ^ c + (5 : ℕ) ^ d)
    """
    print("\n=== Test 3a: Witness Referenced in Target ===")

    # Simulate a target AST that references c and d
    # This is a simplified structure - in reality it would come from the actual AST
    target_with_references = {"val": "m", "info": {"leading": "", "trailing": " "}}

    # Create a name_map that includes c and d
    name_map = {
        "c": {"kind": "variable", "val": "c"},
        "d": {"kind": "variable", "val": "d"},
        "m": {"kind": "variable", "val": "m"},
    }

    # Note: __find_dependencies looks for references to names in name_map
    # If c and d are referenced in the target, they should be found
    deps = __find_dependencies(target_with_references, name_map)

    print(f"Target: {target_with_references}")
    print(f"Name map keys: {list(name_map.keys())}")
    print(f"Dependencies found: {deps}")

    # This test is more about understanding how dependency tracking works
    # than asserting specific behavior
    print("\n✓ Test 3a completed - Dependency tracking mechanism understood")


def test_witness_not_referenced() -> None:
    """
    Test: Check behavior when witnesses are NOT referenced in target.
    """
    print("\n=== Test 3b: Witness NOT Referenced ===")

    target_no_references = {"val": "x", "info": {"leading": "", "trailing": ""}}

    name_map = {
        "c": {"kind": "variable", "val": "c"},
        "d": {"kind": "variable", "val": "d"},
        "x": {"kind": "variable", "val": "x"},
    }

    deps = __find_dependencies(target_no_references, name_map)

    print(f"Target: {target_no_references}")
    print(f"Dependencies found: {deps}")

    print("\n✓ Test 3b completed")


def test_when_to_add_witnesses() -> None:
    """
    Test: Document decision on when to add witnesses.

    Based on the bug report:
    - h_choose_cd : ∃ c d : ℕ, P
    - Later hypothesis: hm : m = N - ((3 : ℕ) ^ c + (5 : ℕ) ^ d)
    - c and d are referenced in hm, so they need to be bound as parameters

    DECISION POINT:
    - Option A: Always add witnesses when type is existential
    - Option B: Only add witnesses if they're referenced in target or dependencies

    From the bug report, it seems witnesses SHOULD be added if referenced.
    But we need to determine if they should ALWAYS be added or conditionally.
    """
    print("\n=== Test 3c: When to Add Witnesses ===")
    print("""
    DECISION CRITERIA:

    From the bug report:
    - h_choose_cd : ∃ c d : ℕ, (3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N
    - Later: hm : m = N - ((3 : ℕ) ^ c + (5 : ℕ) ^ d)
    - c and d are referenced in hm (not just in h_choose_cd)

    RECOMMENDATION:
    - Add witnesses if they are referenced in target OR dependencies
    - This matches the pattern used for other variables
    - Use __find_dependencies(target, name_map) to check

    ALTERNATIVE:
    - Always add witnesses when type is existential (simpler, but might add unnecessary binders)
    """)

    print("✓ Test 3c completed - Decision criteria documented")
