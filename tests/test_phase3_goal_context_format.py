"""Test 1: Goal Context Type Format

Purpose: Understand what format the type string has in goal_var_types for choose/obtain bindings with existential types.
"""

# ruff: noqa: RUF001, RUF002, RUF003

from goedels_poetry.parsers.util.foundation.goal_context import __parse_goal_context


def test_goal_context_format_for_existential_type() -> None:
    """
    Test: Extract goal context for a binding with existential type.

    We'll use a simulated goal context string that represents what Kimina might return
    for a choose binding with existential type.
    """
    # Simulate a goal context string for: h_choose_cd : ∃ c d : ℕ, (3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N
    goal_context_str = "h_choose_cd : ∃ c d : ℕ, (3 : ℕ) ^ c + (5 : ℕ) ^ d ≤ N\n⊢ Prop"

    parsed_types = __parse_goal_context(goal_context_str)

    print("\n=== Test 1: Goal Context Type Format ===")
    print(f"Input goal context: {goal_context_str!r}")
    print(f"Parsed types: {parsed_types}")

    if "h_choose_cd" in parsed_types:
        type_str = parsed_types["h_choose_cd"]
        print(f"\nType string for 'h_choose_cd': {type_str!r}")
        print(f"Type string length: {len(type_str)}")
        print(f"Starts with '∃': {type_str.strip().startswith('∃')}")
        print(f"Contains 'exists': {'exists' in type_str.lower()}")

        # Check if it's parseable (basic sanity check)
        is_parseable_format = len(type_str) > 0 and ("∃" in type_str or "exists" in type_str.lower())
        print(f"Is parseable format (contains ∃ or exists): {is_parseable_format}")
    else:
        print("\n⚠️  'h_choose_cd' not found in parsed types")
        print(f"Available keys: {list(parsed_types.keys())}")

    assert "h_choose_cd" in parsed_types, "h_choose_cd should be in parsed types"
    type_str = parsed_types["h_choose_cd"]
    assert len(type_str) > 0, "Type string should not be empty"

    print("\n✓ Test 1 completed - Type string format determined")


def test_goal_context_format_simple_existential() -> None:
    """
    Test: Extract goal context for a simpler existential type.

    Simulate: h : ∃ x : ℕ, x > 0
    """
    goal_context_str = "h : ∃ x : ℕ, x > 0\n⊢ Prop"

    parsed_types = __parse_goal_context(goal_context_str)

    print("\n=== Test 1b: Simple Existential Type ===")
    print(f"Input goal context: {goal_context_str!r}")
    print(f"Parsed types: {parsed_types}")

    if "h" in parsed_types:
        type_str = parsed_types["h"]
        print(f"\nType string for 'h': {type_str!r}")
        print(f"Starts with '∃': {type_str.strip().startswith('∃')}")
    else:
        print("\n⚠️  'h' not found in parsed types")

    print("\n✓ Test 1b completed")
