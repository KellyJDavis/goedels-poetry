from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _theorem_with_default() -> dict:
    # theorem t (x : Nat := 5) : True := by have hgoal : True := by sorry; sorry
    binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "x", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "Nat := 5", "info": {"leading": "", "trailing": ""}},
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    binder_list = {"kind": "Lean.Parser.Term.bracketedBinderList", "args": [binder]}
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
            {"kind": "Lean.Parser.Command.declId", "args": [{"val": "t", "info": {"leading": "", "trailing": " "}}]},
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
    theorem["args"].insert(2, {"kind": "Lean.Parser.Command.declSig", "args": [binder_list]})
    return theorem


def _theorem_with_opt_auto() -> dict:
    # theorem t2 (y : optParam Nat 7) (z : autoParam Nat 3) : True := by have hgoal : True := by sorry; sorry
    opt_binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "y", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "optParam Nat 7", "info": {"leading": "", "trailing": ""}},
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    auto_binder = {
        "kind": "Lean.Parser.Term.explicitBinder",
        "args": [
            {"val": "(", "info": {"leading": " ", "trailing": ""}},
            {"kind": "Lean.binderIdent", "args": [{"val": "z", "info": {"leading": "", "trailing": ""}}]},
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "autoParam Nat 3", "info": {"leading": "", "trailing": ""}},
            {"val": ")", "info": {"leading": "", "trailing": " "}},
        ],
    }
    binder_list = {"kind": "Lean.Parser.Term.bracketedBinderList", "args": [opt_binder, auto_binder]}
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
                "args": [{"val": "t2", "info": {"leading": "", "trailing": " "}}],
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
    theorem["args"].insert(2, {"kind": "Lean.Parser.Command.declSig", "args": [binder_list]})
    return theorem


def _exists_head_with_witness() -> dict:
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


def test_default_param_is_preserved_in_subgoal() -> None:
    ast = AST(_theorem_with_default(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    assert "(x : Nat := 5)" in code


def test_opt_and_auto_params_are_preserved_in_subgoal() -> None:
    ast = AST(_theorem_with_opt_auto(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    assert "(y : optParam Nat 7)" in code
    assert "(z : autoParam Nat 3)" in code


def test_existential_witness_and_hypothesis_are_added() -> None:
    ast = AST(_exists_head_with_witness(), sorries=[])
    code = ast.get_named_subgoal_code("hgoal")
    assert "(x : Nat)" in code
    assert "hExists : ∃" in code
