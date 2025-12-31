"""Test to reproduce the bug where set bindings (odds/evens) are not included in extracted subgoals.

This test is based on the actual data from problem-1.log where:
- set odds and set evens are defined before h_partition
- h_partition uses odds and evens
- The sorries contain goal contexts with odds and evens
- But the extracted subgoal for h_partition doesn't include odds/evens as parameters
"""

from goedels_poetry.parsers.ast import AST


def test_set_binding_bug_h_partition_missing_odds_evens_from_log() -> None:
    """
    Test that reproduces the bug: h_partition subgoal doesn't include odds/evens.

    Based on problem-1.log:
    - set odds and set evens are defined before h_partition
    - h_partition uses odds and evens in its type
    - Sorries contain goal contexts with odds and evens types
    - But the extracted subgoal doesn't include odds/evens as parameters
    """
    # AST structure based on problem-1.log
    # This is a simplified version that captures the essential structure
    ast_dict = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "amc12_2001_p5", "info": {"leading": "", "trailing": " "}}],
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
                            # set odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)  # noqa: RUF003
                            {
                                "kind": "Lean.Parser.Tactic.tacticSet_",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.setDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.setIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [
                                                            {"val": "odds", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    },
                                                    {
                                                        "kind": "Lean.Parser.Term.typeSpec",
                                                        "args": [
                                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "Finset", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "ℕ", "info": {"leading": "", "trailing": ""}},  # noqa: RUF001
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            # Value would be here, but simplified for test
                                            {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # set evens : Finset ℕ := Finset.filter (fun x => Even x) (Finset.range 10000)  # noqa: RUF003
                            {
                                "kind": "Lean.Parser.Tactic.tacticSet_",
                                "args": [
                                    {"val": "set", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.setDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.setIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [
                                                            {"val": "evens", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    },
                                                    {
                                                        "kind": "Lean.Parser.Term.typeSpec",
                                                        "args": [
                                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                                            {"val": "Finset", "info": {"leading": "", "trailing": " "}},
                                                            {"val": "ℕ", "info": {"leading": "", "trailing": ""}},  # noqa: RUF001
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # have h_partition : evens ∪ odds = Finset.range 10000 ∧ Disjoint evens odds := by sorry  # noqa: RUF003
                            {
                                "kind": "Lean.Parser.Tactic.tacticHave_",
                                "args": [
                                    {"val": "have", "info": {"leading": "\n\n  ", "trailing": " "}},
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
                                                                "val": "h_partition",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            # Type: evens ∪ odds = Finset.range 10000 ∧ Disjoint evens odds  # noqa: RUF003
                                            {"val": "evens", "info": {"leading": "", "trailing": " "}},
                                            {"val": "∪", "info": {"leading": " ", "trailing": " "}},  # noqa: RUF001
                                            {"val": "odds", "info": {"leading": "", "trailing": " "}},
                                            {"val": "=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Finset.range", "info": {"leading": "", "trailing": " "}},
                                            {"val": "10000", "info": {"leading": "", "trailing": " "}},
                                            {"val": "∧", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Disjoint", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "evens", "info": {"leading": "", "trailing": " "}},
                                            {"val": "odds", "info": {"leading": " ", "trailing": " "}},
                                        ],
                                    },
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
                                                            {"val": "sorry", "info": {"leading": "", "trailing": ""}}
                                                        ],
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
    }

    # Sorries based on problem-1.log
    # The first sorry (for h_partition) has goal context with odds and evens
    sorries = [
        {
            "pos": {"line": 16, "column": 4},
            "endPos": {"line": 16, "column": 9},
            "goal": "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\nevens : Finset ℕ := Finset.filter (fun x => Even x) (Finset.range 10000)\n⊢ evens ∪ odds = Finset.range 10000 ∧ Disjoint evens odds",  # noqa: RUF001
            "proofState": 4,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h_partition")

    # The bug: odds and evens should be included but are not
    # They should appear as parameters or hypotheses in the extracted lemma
    print(f"\nExtracted subgoal code:\n{result}\n")

    # Verify the bug exists: odds and evens are NOT included as parameters
    # This is the bug we're trying to reproduce
    assert "lemma" in result
    assert "h_partition" in result
    assert "evens ∪ odds" in result or "odds" in result or "evens" in result, "Subgoal should reference odds/evens"  # noqa: RUF001

    # The bug: odds and evens are used in the type but not defined as parameters
    # Check if they appear as parameters (they should but don't due to the bug)
    has_odds_param = "(odds :" in result or "(odds:" in result
    has_evens_param = "(evens :" in result or "(evens:" in result

    # Verify goal context parsing works
    from goedels_poetry.parsers.util import __parse_goal_context

    goal = sorries[0]["goal"]
    parsed_types = __parse_goal_context(goal)
    print(f"\nGoal context parsing result: {parsed_types}")
    assert "odds" in parsed_types, "odds should be parsed from goal context"
    assert "evens" in parsed_types, "evens should be parsed from goal context"

    # Simulate goal_var_types population to verify it should work
    all_types_sim = {}
    target_sorry_types_sim = {}
    target_sorry_found_sim = False
    lookup_name = "h_partition"

    for sorry in sorries:
        goal_sim = sorry.get("goal", "")
        if not goal_sim:
            continue
        parsed_types_sim = __parse_goal_context(goal_sim)
        is_target_sorry_sim = not target_sorry_found_sim and lookup_name in parsed_types_sim
        if is_target_sorry_sim:
            target_sorry_types_sim = parsed_types_sim
            target_sorry_found_sim = True
        if not is_target_sorry_sim:
            for name, typ in parsed_types_sim.items():
                if name not in all_types_sim:
                    all_types_sim[name] = typ

    goal_var_types_sim = all_types_sim.copy()
    goal_var_types_sim.update(target_sorry_types_sim)

    print(f"\nSimulated goal_var_types: {goal_var_types_sim}")
    print(f"  Should contain odds: {'odds' in goal_var_types_sim}")
    print(f"  Should contain evens: {'evens' in goal_var_types_sim}")

    # This assertion will FAIL, reproducing the bug
    # The test documents that this is the expected behavior (odds/evens should be included)
    if not (has_odds_param or has_evens_param):
        print("\nBUG REPRODUCED: odds and evens are not included as parameters in the extracted subgoal")
        print("They are used in the type but not defined, making the subgoal invalid")
        print("\nRoot cause analysis:")
        print(f"  - Goal context parsing works: ✓ (parsed {parsed_types})")
        print(
            f"  - Simulated goal_var_types contains odds/evens: {'odds' in goal_var_types_sim and 'evens' in goal_var_types_sim}"
        )
        print("  - But extracted subgoal doesn't include them: ✗")
        print("\nThis suggests that goal_var_types is either:")
        print("  1. Empty when __handle_set_let_binding_as_equality is called")
        print("  2. Not being passed correctly to the function")
        print("  3. Or the check at line 2754 is failing for another reason")
        # After fix: odds and evens should be included
        assert has_odds_param or has_evens_param, (
            "FIX VERIFICATION: odds and evens should be included as parameters/hypotheses in the extracted subgoal.\n"
            f"Goal context parsing works correctly and populates goal_var_types with {goal_var_types_sim}.\n"
            "The fix ensures that variables from set/let bindings are included when they are used in the subgoal."
        )


def test_set_binding_bug_goal_context_parsing() -> None:
    """
    Test that goal context parsing correctly extracts odds and evens from sorries.

    This verifies that the goal context parsing works correctly,
    which is a prerequisite for the fix.
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # Sorries with goal context containing odds and evens (from problem-1.log)
    goal = "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\nevens : Finset ℕ := Finset.filter (fun x => Even x) (Finset.range 10000)\n⊢ Prop"  # noqa: RUF001

    parsed_types = __parse_goal_context(goal)
    print(f"\nParsed types from goal context: {parsed_types}")

    assert "odds" in parsed_types, "odds should be parsed from goal context"
    assert "evens" in parsed_types, "evens should be parsed from goal context"
    assert parsed_types["odds"] == "Finset ℕ", "odds type should be Finset ℕ"  # noqa: RUF001
    assert parsed_types["evens"] == "Finset ℕ", "evens type should be Finset ℕ"  # noqa: RUF001

    # This test verifies that goal context parsing works correctly
    # The bug is that these parsed types are not being used when creating the subgoal


def test_set_binding_bug_verify_goal_var_types_population() -> None:  # noqa: C901
    """
    Test to verify if goal_var_types is populated correctly in _get_named_subgoal_rewritten_ast.

    This test directly checks the internal logic to see if goal_var_types is populated
    when processing set bindings.
    """
    from goedels_poetry.parsers.util import __parse_goal_context, _get_named_subgoal_rewritten_ast

    # First, verify goal context parsing works
    goal = "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\n⊢ Prop"  # noqa: RUF001
    parsed_types = __parse_goal_context(goal)
    assert "odds" in parsed_types, "Goal context parsing should extract 'odds'"
    print(f"\nGoal context parsing result: {parsed_types}")

    # Now test with a minimal AST that has set odds before h_partition
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
                            # set odds
                            {
                                "kind": "Lean.Parser.Tactic.tacticSet_",
                                "args": [
                                    {"val": "set", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.setDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.setIdDecl",
                                                "args": [
                                                    {
                                                        "kind": "Lean.binderIdent",
                                                        "args": [
                                                            {"val": "odds", "info": {"leading": "", "trailing": ""}}
                                                        ],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "value", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # have h_partition
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
                                                            {
                                                                "val": "h_partition",
                                                                "info": {"leading": "", "trailing": " "},
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "Prop", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
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
                                                            {"val": "sorry", "info": {"leading": "", "trailing": ""}}
                                                        ],
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
    }

    # Sorries with goal context containing odds
    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    # Call the function and check the result
    result = _get_named_subgoal_rewritten_ast(ast_dict, "h_partition", sorries)

    # Extract binders from the result
    args = result.get("args", [])
    binders = None
    for arg in args:
        if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.bracketedBinderList":
            binders = arg.get("args", [])
            break

    print(f"\nBinders in extracted subgoal: {binders}")

    # Convert result to code to check if odds is included
    from goedels_poetry.parsers.util import _ast_to_code

    result_code = _ast_to_code(result)
    print(f"\nExtracted subgoal code:\n{result_code}\n")

    # Manually trace through the goal_var_types population logic to verify
    # This simulates what happens in _get_named_subgoal_rewritten_ast
    from goedels_poetry.parsers.util import __parse_goal_context

    goal_var_types_simulated = {}
    all_types_simulated = {}
    target_sorry_types_simulated = {}
    target_sorry_found_simulated = False
    lookup_name = "h_partition"

    for sorry in sorries:
        goal = sorry.get("goal", "")
        if not goal:
            continue
        parsed_types_sim = __parse_goal_context(goal)
        is_target_sorry_sim = not target_sorry_found_simulated and lookup_name in parsed_types_sim
        if is_target_sorry_sim:
            target_sorry_types_simulated = parsed_types_sim
            target_sorry_found_simulated = True
        if not is_target_sorry_sim:
            for name, typ in parsed_types_sim.items():
                if name not in all_types_simulated:
                    all_types_simulated[name] = typ

    goal_var_types_simulated = all_types_simulated.copy()
    goal_var_types_simulated.update(target_sorry_types_simulated)

    print("\nSimulated goal_var_types population:")
    print(f"  all_types: {all_types_simulated}")
    print(f"  target_sorry_types: {target_sorry_types_simulated}")
    print(f"  final goal_var_types: {goal_var_types_simulated}")
    print(f"  'odds' in goal_var_types: {'odds' in goal_var_types_simulated}")

    # Check if odds is included as a parameter
    has_odds_param = "(odds :" in result_code or "(odds:" in result_code or "odds : Finset" in result_code

    if not has_odds_param:
        print("\nBUG REPRODUCED: 'odds' is not included as a parameter in the extracted subgoal")
        print("This confirms the bug - odds should be included because:")
        print("1. It's defined as a set binding before h_partition")
        print("2. It's used in h_partition's type")
        print("3. It's in the goal context of the sorry")
        print("4. Goal context parsing works correctly (verified above)")
        print(f"5. Simulated goal_var_types contains odds: {'odds' in goal_var_types_simulated}")
        if goal_var_types_simulated:
            print(f"   goal_var_types should be: {goal_var_types_simulated}")
        else:
            print("   ⚠️  WARNING: Simulated goal_var_types is EMPTY!")
            print("   This suggests the bug is in goal_var_types population logic")
        print("\nThis suggests goal_var_types is either:")
        print("- Empty when __handle_set_let_binding_as_equality is called")
        print("- Not being passed correctly")
        print("- Or the check at line 2754 is failing for another reason")
    else:
        print("✓ 'odds' is included as a parameter (bug may be fixed or test setup is incorrect)")

    # Analyze the bug
    # The extracted code shows: "lemma h_partition  (hodds : odds  = value )  : Prop"
    # This means:
    # - An equality hypothesis was created: hodds : odds = value
    # - But odds itself is NOT defined as a parameter
    # - The equality hypothesis references odds, but odds is undefined!

    has_equality_hypothesis = "hodds" in result_code or "odds  = value" in result_code or "odds = value" in result_code

    print("\nAnalysis:")
    print(f"  Has equality hypothesis (hodds : odds = value): {has_equality_hypothesis}")
    print(f"  Has odds as parameter (odds : Finset ℕ): {has_odds_param}")  # noqa: RUF001

    if has_equality_hypothesis and not has_odds_param:
        print("\nBUG IDENTIFIED:")
        print("  The equality hypothesis 'hodds : odds = value' was created,")
        print("  but 'odds' itself is not defined as a parameter.")
        print("  This makes the subgoal invalid - it references 'odds' without defining it.")
        print("\n  The bug is that when __extract_set_value succeeds,")
        print("  we create an equality hypothesis but don't also include")
        print("  the variable itself as a parameter when it's needed.")
        print("\n  Expected: The subgoal should include 'odds : Finset ℕ' as a parameter")  # noqa: RUF001
        print("  (either instead of or in addition to the equality hypothesis)")

    # The bug: odds should be included as a parameter
    # Even if we create an equality hypothesis, odds must be defined
    assert has_odds_param, (
        "BUG: 'odds' should be included as a parameter in the extracted subgoal.\n"
        f"Extracted code: {result_code}\n"
        "The subgoal references 'odds' (in the equality hypothesis) but doesn't define it.\n"
        "This makes the subgoal invalid. 'odds' should be included as a parameter\n"
        "(odds : Finset ℕ) because:\n"  # noqa: RUF001
        "  1. It's defined as a set binding before h_partition\n"
        "  2. It's used in h_partition's type\n"
        "  3. It's in the goal context of the sorry\n"
        "  4. Goal context parsing works correctly\n"
        "  5. Simulated goal_var_types contains odds\n"
        "\n"
        "The bug is that when value extraction succeeds, we create an equality\n"
        "hypothesis but don't ensure the variable itself is also included as a parameter."
    )
