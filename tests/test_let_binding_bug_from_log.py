"""Test to reproduce the bug where let bindings are not included in extracted subgoals.

This test mirrors test_set_binding_bug_from_log.py but for let statements instead of set statements.
The same bug should occur because let and set use the same code path in __handle_set_let_binding_as_equality.
"""

from goedels_poetry.parsers.ast import AST


def test_let_binding_bug_goal_context_parsing() -> None:
    """
    Test that goal context parsing correctly extracts variables from let bindings.

    This verifies that the goal context parsing works correctly for let bindings,
    which is a prerequisite for the fix.
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # Goal context with let binding variable (similar to set binding)
    goal = "x : ℕ := 42\ny : ℕ := 100\n⊢ Prop"  # noqa: RUF001

    parsed_types = __parse_goal_context(goal)
    print(f"\nParsed types from goal context: {parsed_types}")

    assert "x" in parsed_types, "x should be parsed from goal context"
    assert "y" in parsed_types, "y should be parsed from goal context"
    assert parsed_types["x"] == "ℕ", "x type should be ℕ"  # noqa: RUF001
    assert parsed_types["y"] == "ℕ", "y type should be ℕ"  # noqa: RUF001

    # This test verifies that goal context parsing works correctly for let bindings
    # The bug is that these parsed types are not being used when creating the subgoal


def test_let_binding_bug_verify_goal_var_types_population() -> None:  # noqa: C901
    """
    Test to verify if goal_var_types is populated correctly for let bindings.

    This test directly checks the internal logic to see if goal_var_types is populated
    when processing let bindings, similar to the set binding test.
    """
    from goedels_poetry.parsers.util import __parse_goal_context, _get_named_subgoal_rewritten_ast

    # First, verify goal context parsing works
    goal = "x : ℕ := 42\n⊢ Prop"  # noqa: RUF001
    parsed_types = __parse_goal_context(goal)
    assert "x" in parsed_types, "Goal context parsing should extract 'x'"
    print(f"\nGoal context parsing result: {parsed_types}")

    # Now test with a minimal AST that has let x before h1
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
                            # let x := value
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
                                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                                    },
                                                ],
                                            },
                                            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "value", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # have h1
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

    # Sorries with goal context containing x
    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ := 42\n⊢ Prop",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    # Call the function and check the result
    result = _get_named_subgoal_rewritten_ast(ast_dict, "h1", sorries)

    # Extract binders from the result
    args = result.get("args", [])
    binders = None
    for arg in args:
        if isinstance(arg, dict) and arg.get("kind") == "Lean.Parser.Term.bracketedBinderList":
            binders = arg.get("args", [])
            break

    print(f"\nBinders in extracted subgoal: {binders}")

    # Convert result to code to check if x is included
    from goedels_poetry.parsers.util import _ast_to_code

    result_code = _ast_to_code(result)
    print(f"\nExtracted subgoal code:\n{result_code}\n")

    # Manually trace through the goal_var_types population logic to verify
    goal_var_types_simulated = {}
    all_types_simulated = {}
    target_sorry_types_simulated = {}
    target_sorry_found_simulated = False
    lookup_name = "h1"

    for sorry in sorries:
        goal_sim = sorry.get("goal", "")
        if not goal_sim:
            continue
        parsed_types_sim = __parse_goal_context(goal_sim)
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
    print(f"  'x' in goal_var_types: {'x' in goal_var_types_simulated}")

    # Check if x is included as a parameter
    has_x_param = "(x :" in result_code or "(x:" in result_code or "x : ℕ" in result_code  # noqa: RUF001

    # Analyze the bug
    has_equality_hypothesis = "hx" in result_code or "x  = value" in result_code or "x = value" in result_code

    print("\nAnalysis:")
    print(f"  Has equality hypothesis (hx : x = value): {has_equality_hypothesis}")
    print(f"  Has x as parameter (x : ℕ): {has_x_param}")  # noqa: RUF001

    if has_equality_hypothesis and not has_x_param:
        print("\nBUG IDENTIFIED (same as set):")
        print("  The equality hypothesis 'hx : x = value' was created,")
        print("  but 'x' itself is not defined as a parameter.")
        print("  This makes the subgoal invalid - it references 'x' without defining it.")
        print("\n  The bug is that when __extract_let_value succeeds,")
        print("  we create an equality hypothesis but don't also include")
        print("  the variable itself as a parameter when it's needed.")
        print("\n  Expected: The subgoal should include 'x : ℕ' as a parameter")  # noqa: RUF001
        print("  (either instead of or in addition to the equality hypothesis)")

    if not has_x_param:
        print("\nBUG REPRODUCED: 'x' is not included as a parameter in the extracted subgoal")
        print("This confirms the same bug exists for let bindings as for set bindings")
        print("Expected: x should be included because:")
        print("1. It's defined as a let binding before h1")
        print("2. It's in the goal context of the sorry")
        print("3. Goal context parsing works correctly (verified above)")
        print(f"4. Simulated goal_var_types contains x: {'x' in goal_var_types_simulated}")
        if goal_var_types_simulated:
            print(f"   goal_var_types should be: {goal_var_types_simulated}")
        else:
            print("   ⚠️  WARNING: Simulated goal_var_types is EMPTY!")
            print("   This suggests the bug is in goal_var_types population logic")
        print("\nThis suggests goal_var_types is either:")
        print("- Empty when __handle_set_let_binding_as_equality is called")
        print("- Not being passed correctly")
        print("- Or the check at line 2754 is failing for another reason")

    # After fix: x should be included as a parameter
    # The fix ensures that when we create an equality hypothesis, we also include
    # the variable itself as a parameter if it's used elsewhere
    assert has_x_param, (
        "FIX VERIFICATION: 'x' should be included as a parameter in the extracted subgoal.\n"
        f"Extracted code: {result_code}\n"
        "The fix ensures that variables from set/let bindings are included as parameters\n"
        "when they are used in the subgoal, even if an equality hypothesis is also created."
    )


def test_let_binding_bug_complex_case_matching_log() -> None:
    """
    Test let binding bug with a more complex case that matches the problem-1.log scenario.

    This test uses let bindings with type annotations and complex values,
    similar to how set bindings are used in problem-1.log.
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # AST with let bindings that have type annotations (similar to set odds : Finset ℕ := ...)  # noqa: RUF003
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
                            # let odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)  # noqa: RUF003
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
                                            # Complex value that might fail extraction
                                            {"val": "Finset.filter", "info": {"leading": "", "trailing": " "}},
                                        ],
                                    },
                                ],
                            },
                            # let evens : Finset ℕ := Finset.filter (fun x => Even x) (Finset.range 10000)  # noqa: RUF003
                            {
                                "kind": "Lean.Parser.Tactic.tacticLet_",
                                "args": [
                                    {"val": "let", "info": {"leading": "\n  ", "trailing": " "}},
                                    {
                                        "kind": "Lean.Parser.Term.letDecl",
                                        "args": [
                                            {
                                                "kind": "Lean.Parser.Term.letIdDecl",
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
                            # have h_partition : evens ∪ odds = Finset.range 10000 ∧ Disjoint evens odds  # noqa: RUF003
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
                                            # Type references odds and evens
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

    # Sorries with goal context containing odds and evens (from let bindings)
    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "odds : Finset ℕ := Finset.filter (fun x => ¬Even x) (Finset.range 10000)\nevens : Finset ℕ := Finset.filter (fun x => Even x) (Finset.range 10000)\n⊢ evens ∪ odds = Finset.range 10000 ∧ Disjoint evens odds",  # noqa: RUF001
            "proofState": 1,
        }
    ]

    ast = AST(ast_dict, sorries)
    result = ast.get_named_subgoal_code("h_partition")

    print(f"\nExtracted subgoal code:\n{result}\n")

    # Verify goal context parsing works
    goal = sorries[0]["goal"]
    parsed_types = __parse_goal_context(goal)
    print(f"\nGoal context parsing result: {parsed_types}")
    assert "odds" in parsed_types, "odds should be parsed from goal context"
    assert "evens" in parsed_types, "evens should be parsed from goal context"

    # Simulate goal_var_types population
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

    # Check if odds and evens are included as parameters
    has_odds_param = "(odds :" in result or "(odds:" in result
    has_evens_param = "(evens :" in result or "(evens:" in result

    # Verify the bug exists: odds and evens are NOT included as parameters
    assert "lemma" in result
    assert "h_partition" in result
    assert "evens ∪ odds" in result or "odds" in result or "evens" in result, "Subgoal should reference odds/evens"  # noqa: RUF001

    if not (has_odds_param or has_evens_param):
        print("\nBUG REPRODUCED: odds and evens are not included as parameters in the extracted subgoal")
        print("They are used in the type but not defined, making the subgoal invalid")
        print("\nRoot cause analysis:")
        print(f"  - Goal context parsing works: ✓ (parsed {parsed_types})")
        print(
            f"  - Simulated goal_var_types contains odds/evens: {'odds' in goal_var_types_sim and 'evens' in goal_var_types_sim}"
        )
        print("  - But extracted subgoal doesn't include them: ✗")
        print("\nThis confirms the SAME bug exists for let bindings as for set bindings.")
        print("The bug is that goal_var_types is either:")
        print("  1. Empty when __handle_set_let_binding_as_equality is called")
        print("  2. Not being passed correctly to the function")
        print("  3. Or the check at line 2754 is failing for another reason")
        # After fix: odds and evens should be included
        assert has_odds_param or has_evens_param, (
            "FIX VERIFICATION: odds and evens should be included as parameters/hypotheses in the extracted subgoal.\n"
            f"Goal context parsing works correctly and populates goal_var_types with {goal_var_types_sim}.\n"
            "The fix ensures that variables from set/let bindings are included when they are used in the subgoal."
        )
