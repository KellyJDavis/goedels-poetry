"""Regression tests for decomposition when a subgoal is inside an enclosing lemma.

These tests are designed to reproduce the bug observed in partial.log where Kimina emits
top-level declarations with unqualified `kind` values like `"lemma"`, causing the
decomposition machinery to miss enclosing binders.
"""

from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _explicit_binder(name: str, typ: str) -> dict:
    # Minimal explicit binder: (name : typ)
    return {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": "", "trailing": ""}},
            [{"val": name, "info": {"leading": "", "trailing": " "}}],
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": typ, "info": {"leading": "", "trailing": ""}},
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }


def _named_have(name: str, prop: str) -> dict:
    # Minimal named have: have name : prop := by sorry
    return {
        "kind": "Lean.Parser.Tactic.tacticHave_",
        "args": [
            {"val": "have", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.haveDecl",
                "args": [
                    {
                        "kind": "Lean.Parser.Term.haveIdDecl",
                        "args": [
                            {
                                "kind": "Lean.Parser.Term.haveId",
                                "args": [{"val": name, "info": {"leading": "", "trailing": " "}}],
                            }
                        ],
                    },
                    {"val": ":", "info": {"leading": "", "trailing": " "}},
                    {"val": prop, "info": {"leading": "", "trailing": " "}},
                ],
            },
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


def _qualified_outer_lemma_with_have(*, lemma_name: str = "Outer", have_name: str = "hx") -> dict:
    return {
        "kind": "Lean.Parser.Command.lemma",
        "args": [
            {"val": "lemma", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": lemma_name, "info": {"leading": "", "trailing": " "}}],
            },
            {
                "kind": "Lean.Parser.Term.bracketedBinderList",
                "args": [_explicit_binder("x", "Nat"), _explicit_binder("h", "x = x")],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {
                        "kind": "Lean.Parser.Tactic.tacticSeq",
                        "args": [_named_have(have_name, "x = x")],
                    },
                ],
            },
        ],
    }


def _unqualified_outer_lemma_with_have(*, lemma_name: str = "Outer", have_name: str = "hx") -> dict:
    # Mirrors the `partial.log` shape:
    # { "kind": "lemma", "args": [ declModifiers, group( "lemma", declId, declSig, ":=", byTactic ... ) ] }
    return {
        "kind": "lemma",
        "info": None,
        "args": [
            {"kind": "Lean.Parser.Command.declModifiers", "info": None, "args": [[], [], [], [], [], []]},
            {
                "kind": "group",
                "info": None,
                "args": [
                    {"val": "lemma", "info": {"leading": "", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Command.declId",
                        "info": None,
                        "args": [{"val": lemma_name, "info": {"leading": "", "trailing": " "}}, []],
                    },
                    {
                        "kind": "Lean.Parser.Command.declSig",
                        "info": None,
                        "args": [
                            [
                                _explicit_binder("x", "Nat"),
                                _explicit_binder("h", "x = x"),
                            ]
                        ],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "True", "info": {"leading": "", "trailing": " "}},
                    {"val": ":=", "info": {"leading": " ", "trailing": " "}},
                    {
                        "kind": "Lean.Parser.Term.byTactic",
                        "args": [
                            {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                            {
                                "kind": "Lean.Parser.Tactic.tacticSeq",
                                "args": [_named_have(have_name, "x = x")],
                            },
                        ],
                    },
                ],
            },
        ],
    }


def test_get_named_subgoal_code_carries_binders_from_enclosing_qualified_lemma() -> None:
    ast = AST(_qualified_outer_lemma_with_have())
    code = ast.get_named_subgoal_code("hx")
    # Must include enclosing binders x and h.
    assert "lemma hx" in code
    assert "(x" in code
    assert "(h" in code
    assert "x = x" in code


def test_get_named_subgoal_code_carries_binders_from_enclosing_unqualified_lemma() -> None:
    ast = AST(_unqualified_outer_lemma_with_have())
    code = ast.get_named_subgoal_code("hx")
    assert "lemma hx" in code
    assert "(x" in code
    assert "(h" in code
    assert "x = x" in code


def test_anonymous_have_in_unqualified_lemma_gets_stable_synthetic_name() -> None:
    # Anonymous have inside unqualified lemma should be named gp_anon_have__<decl>__1
    outer = _unqualified_outer_lemma_with_have(have_name="hx")
    # Replace the named have with an anonymous one
    outer["args"][1]["args"][-1]["args"][1]["args"] = [
        {
            "kind": "Lean.Parser.Tactic.tacticHave_",
            "args": [
                {"val": "have", "info": {"leading": "", "trailing": " "}},
                {"kind": "Lean.Parser.Term.haveDecl", "args": [{"val": ":"}, {"val": "False"}]},
                {
                    "kind": "Lean.Parser.Tactic.tacticSeq",
                    "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
                },
            ],
        }
    ]

    ast = AST(outer)
    names = ast.get_unproven_subgoal_names()
    assert "gp_anon_have__Outer__1" in names

    code = ast.get_named_subgoal_code("gp_anon_have__Outer__1")
    assert "lemma gp_anon_have__Outer__1" in code
    assert "(x" in code
    assert "(h" in code


def test_kimina_placeholder_anonymous_have_name_is_treated_as_anonymous() -> None:
    """
    Kimina sometimes emits anonymous `have : ...` as if it had the identifier `[anonymous]`.

    This must be treated as truly anonymous, assigned a stable synthetic name, and the generated
    lemma must carry enclosing binders.
    """
    outer = _unqualified_outer_lemma_with_have(have_name="[anonymous]")
    ast = AST(outer)

    names = ast.get_unproven_subgoal_names()
    assert "gp_anon_have__Outer__1" in names
    assert "[anonymous]" not in names

    code = ast.get_named_subgoal_code("gp_anon_have__Outer__1")
    assert "lemma gp_anon_have__Outer__1" in code
    assert "(x" in code
    assert "(h" in code


def test_main_body_sorry_in_unqualified_lemma_is_resolvable() -> None:
    # Ensure the "<main body>" synthetic extraction also works when the enclosing decl kind is unqualified.
    outer = _unqualified_outer_lemma_with_have()
    # Replace the proof body with just `sorry` in main body
    outer["args"][1]["args"][-1]["args"][1] = {
        "kind": "Lean.Parser.Tactic.tacticSeq",
        "args": [{"kind": "Lean.Parser.Tactic.tacticSorry", "args": [{"val": "sorry"}]}],
    }
    ast = AST(outer)
    names = ast.get_unproven_subgoal_names()
    assert "<main body>" in names
    code = ast.get_named_subgoal_code("<main body>")
    # This should synthesize gp_main_body__Outer
    assert "gp_main_body__Outer" in code
