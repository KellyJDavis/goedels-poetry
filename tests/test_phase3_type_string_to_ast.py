"""Test 2: Type String to AST Conversion

Purpose: Determine how to convert type string to AST that __extract_all_exists_witness_binders can process.
"""

# ruff: noqa: RUF001, RUF002

from goedels_poetry.parsers.util.types_and_binders.type_extraction import __extract_all_exists_witness_binders


def test_type_string_detection_existential() -> None:
    """
    Test: Check if we can detect existential types from type strings.
    """
    test_cases = [
        ("∃ c d : ℕ, P", True, "Multiple witnesses with ∃"),
        ("∃ x : ℕ, x > 0", True, "Single witness with ∃"),
        ("exists c d : ℕ, P", True, "Multiple witnesses with 'exists'"),
        ("ℕ → Prop", False, "Arrow type (not existential)"),
        ("Prop", False, "Simple type (not existential)"),
        ("  ∃ x : T, P  ", True, "Existential with whitespace"),
    ]

    print("\n=== Test 2a: Type String Detection ===")

    for type_str, expected_is_existential, description in test_cases:
        # Simple detection: check if starts with ∃ or contains "exists"
        is_existential = type_str.strip().startswith("∃") or "exists" in type_str.lower()

        print(f"\n{description}:")
        print(f"  Type string: {type_str!r}")
        print(f"  Detected as existential: {is_existential} (expected: {expected_is_existential})")

        assert is_existential == expected_is_existential, f"Detection failed for {description}"

    print("\n✓ Test 2a completed - Type string detection works")


def test_type_string_to_ast_manual_construction() -> None:
    """
    Test: Try to manually construct an AST from a type string to understand the structure.

    For "∃ c d : ℕ, P", the AST structure might be:
    {
        "kind": "Lean.Parser.Term.exists",
        "args": [
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "("},
                    {"kind": "Lean.binderIdent", "args": [{"val": "c"}]},
                    {"val": ":"},
                    {"val": "ℕ"}
                ]
            },
            # ... similar for d
        ]
    }
    """
    print("\n=== Test 2b: Manual AST Construction ===")

    # Try to construct a simple existential AST manually
    # Note: This is a simplified structure - real ASTs from Kimina might be different
    exists_ast_manual = {
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
    }

    try:
        result = __extract_all_exists_witness_binders(exists_ast_manual)
        print("✓ Manual AST construction works")
        print(f"  Extracted {len(result)} witness binders")
        for i, binder in enumerate(result):
            print(f"  Binder {i + 1}: {binder.get('kind', 'unknown')}")
    except Exception as e:
        print(f"✗ Manual AST construction failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n✓ Test 2b completed - Manual AST structure tested")


def test_note_about_parsing() -> None:
    """
    Test: Document the challenge of parsing type strings.

    Since we only have type strings from goal_var_types (not AST),
    we need to determine if we can:
    1. Parse the type string using Kimina
    2. Or if the type AST is available somewhere else
    3. Or if we need to parse it manually (not recommended)
    """
    print("\n=== Test 2c: Parsing Challenge Documentation ===")
    print("""
    CHALLENGE IDENTIFIED:

    - goal_var_types[binding_name] contains a STRING (e.g., "∃ c d : ℕ, P")
    - __extract_all_exists_witness_binders() expects an AST DICT

    OPTIONS:
    1. Parse type string using Kimina (if available)
    2. Check if type AST is available elsewhere (e.g., in binding_node)
    3. For 'have' bindings, types ARE in AST - check if choose/obtain can use similar approach

    NEXT STEP: Check if choose/obtain binding_node contains type AST information
    """)

    print("✓ Test 2c completed - Parsing challenge documented")
