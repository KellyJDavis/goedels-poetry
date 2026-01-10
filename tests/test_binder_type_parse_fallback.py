from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _arrow_only_theorem_ast() -> dict:
    # theorem arrow_only : P → Q → R := by have hgoal : R := by sorry; sorry
    arrow_type = {
        "kind": "Lean.Parser.Term.arrow",
        "args": [
            {"val": "P", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.arrow",
                "args": [
                    {"val": "Q", "info": {"leading": "", "trailing": " "}},
                    {"val": "R", "info": {"leading": "", "trailing": " "}},
                ],
            },
        ],
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
                    {"val": "R", "info": {"leading": "", "trailing": " "}},
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
                "args": [{"val": "arrow_only", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            arrow_type,
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
    return theorem


def _exists_head_theorem_ast() -> dict:
    # theorem ex_head : ∃ x : Nat, x > 0 := by have hgoal : True := by sorry; sorry
    exists_type = {
        "kind": "Lean.Parser.Term.exists",
        "args": [
            {"val": "∃", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.explicitBinder",
                "args": [
                    {"val": "(", "info": {"leading": " ", "trailing": ""}},
                    {"kind": "Lean.binderIdent", "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}]},
                    {"val": ":", "info": {"leading": " ", "trailing": " "}},
                    {"val": "Nat", "info": {"leading": "", "trailing": ""}},
                    {"val": ")", "info": {"leading": "", "trailing": " "}},
                ],
            },
            {"val": "x > 0", "info": {"leading": "", "trailing": " "}},
        ],
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
                "args": [{"val": "ex_head", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            exists_type,
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
    return theorem


def _mixed_implicit_instance_pi_ast() -> dict:
    # theorem mix {alpha} [Group alpha] (x : alpha) : True := by have hgoal : True := by sorry; sorry
    implicit_binder = {
        "kind": "Lean.Parser.Term.implicitBinder",
        "args": [
            {"val": "{", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "alpha", "info": {"leading": "", "trailing": ""}}]},
            {"val": "}", "info": {"leading": "", "trailing": " "}},
        ],
    }
    inst_binder = {
        "kind": "Lean.Parser.Term.instBinder",
        "args": [
            {"val": "[", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "hG", "info": {"leading": "", "trailing": ""}}]},
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
    pi_type = {
        "kind": "Lean.Parser.Term.forall",
        "args": [
            {"kind": "Lean.Parser.Term.bracketedBinderList", "args": [implicit_binder, inst_binder, explicit_binder]},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
        ],
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
                "args": [{"val": "mix", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            pi_type,
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
    return theorem


def test_type_parse_fallback_arrow_only_without_sorries() -> None:
    ast = AST(_arrow_only_theorem_ast(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    # Should synthesize anonymous hypotheses for arrow domains.
    assert "(h : P" in code or "(h1 : P" in code
    assert "(h2 : Q" in code or "(h1 : Q" in code


def test_type_parse_fallback_exists_head_without_sorries() -> None:
    ast = AST(_exists_head_theorem_ast(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    # Should include an existential hypothesis binder.
    assert "(hExists : ∃" in code or "∃ x : Nat" in code


def test_type_parse_handles_mixed_implicit_and_instance() -> None:
    ast = AST(_mixed_implicit_instance_pi_ast(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    assert "{alpha}" in code
    assert "[hG : Group alpha]" in code or "[hG : Group alpha]" in code  # instance binder present
    assert "(x : alpha)" in code
