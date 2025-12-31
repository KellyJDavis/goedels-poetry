"""Test to check if generalize bindings have similar issues to set/let bindings.

This test checks if variables introduced via generalize statements are properly
included in extracted subgoals when they are used in the subgoal's type.
"""

from goedels_poetry.parsers.ast import AST


def test_generalize_binding_bug_goal_context_parsing() -> None:
    """Test that goal context parsing correctly extracts variables from generalize bindings."""
    from goedels_poetry.parsers.util import __parse_goal_context

    # Goal context with generalize binding variable
    goal = "x : ℕ\n⊢ Prop"  # noqa: RUF001

    parsed_types = __parse_goal_context(goal)
    print(f"\nParsed types from goal context: {parsed_types}")

    assert "x" in parsed_types, "x should be parsed from goal context"


def test_generalize_binding_bug_variable_used_in_subgoal() -> None:
    """
    Test that generalize binding variables are included when used in subgoal type.

    Based on problem-1.log pattern:
    - generalize introduces variables before h_partition
    - h_partition uses those variables in its type
    - Sorries contain goal contexts with the variables
    - The extracted subgoal should include the variables as parameters
    """
    from goedels_poetry.parsers.util import __parse_goal_context

    # generalize x : ℕ  # noqa: RUF003
    # have h_partition : x > 0 := by sorry
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
                            {
                                "kind": "Lean.Parser.Tactic.tacticGeneralize_",
                                "args": [
                                    {"val": "generalize", "info": {"leading": "", "trailing": " "}},
                                    {
                                        "kind": "Lean.binderIdent",
                                        "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}],
                                    },
                                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                                    {"val": "ℕ", "info": {"leading": "", "trailing": " "}},  # noqa: RUF001
                                ],
                            },
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
                                            {"val": "x", "info": {"leading": "", "trailing": " "}},
                                            {"val": ">", "info": {"leading": " ", "trailing": " "}},
                                            {"val": "0", "info": {"leading": "", "trailing": " "}},
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

    sorries = [
        {
            "pos": {"line": 1, "column": 1},
            "endPos": {"line": 1, "column": 6},
            "goal": "x : ℕ\n⊢ x > 0",  # noqa: RUF001
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
    assert "x" in parsed_types, "x should be parsed from goal context"

    # Check if x is included as a parameter
    has_x_param = "(x :" in result or "(x:" in result

    # Verify the subgoal structure
    assert "lemma" in result
    assert "h_partition" in result
    assert "x > 0" in result or "x  > 0" in result, "Subgoal should reference x"

    # The bug: x should be included but might not be
    if not has_x_param:
        print("\nBUG DETECTED: x is not included as a parameter in the extracted subgoal")
        print("It is used in the type but not defined, making the subgoal invalid")
        print("\nRoot cause analysis:")
        print(f"  - Goal context parsing works: ✓ (parsed {parsed_types})")
        print("  - But extracted subgoal doesn't include it: ✗")
        print("\nThis suggests that goal_var_types may be empty when")
        print("__determine_general_binding_type is called for generalize bindings.")

    # Assert that it should be included
    assert has_x_param, "x should be included as a parameter because it's used in the subgoal type"
