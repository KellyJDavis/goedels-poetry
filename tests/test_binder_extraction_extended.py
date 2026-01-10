from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _implicit_instance_strict_theorem_ast() -> dict:
    # theorem foo {alpha : Type} ⦃beta : Type⦄ [inst : Group alpha] (x : alpha) : True := by
    #   have hgoal : True := by sorry
    #   sorry
    implicit_binder = {
        "kind": "Lean.Parser.Term.implicitBinder",
        "args": [
            {"val": "{", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "alpha", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Type", "info": {"leading": "", "trailing": ""}},
            {"val": "}", "info": {"leading": "", "trailing": " "}},
        ],
    }
    strict_implicit_binder = {
        "kind": "Lean.Parser.Term.strictImplicitBinder",
        "args": [
            {"val": "⦃", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "beta", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Type", "info": {"leading": "", "trailing": ""}},
            {"val": "⦄", "info": {"leading": "", "trailing": " "}},
        ],
    }
    inst_binder = {
        "kind": "Lean.Parser.Term.instBinder",
        "args": [
            {"val": "[", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "inst", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Group alpha", "info": {"leading": "", "trailing": ""}},
            {"val": "]", "info": {"leading": "", "trailing": " "}},
        ],
    }
    explicit_binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "alpha", "info": {"leading": "", "trailing": ""}},
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    bracketed = {
        "kind": "Lean.Parser.Term.bracketedBinderList",
        "args": [implicit_binder, strict_implicit_binder, inst_binder, explicit_binder],
    }
    have_goal = {
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
                                "args": [{"val": "hgoal", "info": {"leading": "", "trailing": " "}}],
                            }
                        ],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "True", "info": {"leading": "", "trailing": " "}},
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
    theorem = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "foo", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [have_goal]},
                ],
            },
        ],
    }
    # Insert binder list into a declSig-like position to exercise the extractor's fast path.
    theorem["args"].insert(
        2,
        {"kind": "Lean.Parser.Command.declSig", "args": [bracketed]},
    )
    return theorem


def _fallback_relevance_ast() -> tuple[dict, list[dict]]:
    # theorem bar : ∀ n : Int, True := by have hgoal : True := by sorry
    # goal context includes n and a spurious junk; only n should appear.
    have_goal = {
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
                                "args": [{"val": "hgoal", "info": {"leading": "", "trailing": " "}}],
                            }
                        ],
                    },
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "True", "info": {"leading": "", "trailing": " "}},
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
    theorem = {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "bar", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {"kind": "Lean.Parser.Tactic.tacticSeq", "args": [have_goal]},
                ],
            },
        ],
    }
    sorries = [{"goal": "n : Int\njunk : False\n⊢ True"}]
    return theorem, sorries


def test_implicit_instance_and_strict_binders_are_extracted() -> None:
    ast = AST(_implicit_instance_strict_theorem_ast(), sorries=[])

    code = ast.get_named_subgoal_code("hgoal")
    assert "{alpha : Type}" in code  # implicit
    assert "⦃beta : Type⦄" in code  # strict implicit
    assert "[inst : Group alpha]" in code  # instance binder
    assert "(x : alpha)" in code  # explicit remains


def test_goal_context_fallback_filters_unreferenced_names() -> None:
    theorem, sorries = _fallback_relevance_ast()
    ast = AST(theorem, sorries=sorries)

    code = ast.get_named_subgoal_code("hgoal")
    assert "(n : Int)" in code
    assert "junk" not in code
