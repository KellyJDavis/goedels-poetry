from __future__ import annotations

from goedels_poetry.parsers.ast import AST


def _named_have(name: str, prop: str) -> dict:
    """Minimal named `have` with a sorry proof."""
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


def _theorem_with_subgoals() -> dict:
    """A minimal theorem whose proof contains the subgoals we want to extract."""
    haves = [
        _named_have("hn_nonneg", "0 ≤ n"),
        _named_have("hm_coe", "m = n"),
        _named_have("hm_gt", "m > 1"),
        _named_have("nat_version", "True"),
    ]
    return {
        "kind": "Lean.Parser.Command.theorem",
        "args": [
            {"val": "theorem", "info": {"leading": "", "trailing": " "}},
            {
                "kind": "Lean.Parser.Command.declId",
                "args": [{"val": "A303656", "info": {"leading": "", "trailing": " "}}],
            },
            {"val": ":", "info": {"leading": " ", "trailing": " "}},
            {"val": "True", "info": {"leading": "", "trailing": " "}},
            {"val": ":=", "info": {"leading": " ", "trailing": " "}},
            {
                "kind": "Lean.Parser.Term.byTactic",
                "args": [
                    {"val": "by", "info": {"leading": "", "trailing": "\n  "}},
                    {"kind": "Lean.Parser.Tactic.tacticSeq", "args": haves},
                ],
            },
        ],
    }


def _sorries() -> list[dict]:
    # These goal contexts mirror the decomposition seen in goedels_poetry-v1.log.
    return [
        {"goal": "n : Int\nhn : n > 1\n⊢ 0 ≤ n"},
        {"goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm : Nat := n.toNat\n⊢ ↑m = n"},
        {
            "goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm : Nat := n.toNat\nhm_coe : ↑m = n\n⊢ m > 1",
        },
        {
            "goal": "n : Int\nhn : n > 1\nhn_nonneg : 0 ≤ n\nm✝ : Nat := n.toNat\nhm_coe : ↑m✝ = n\nhm_gt : m✝ > 1\nm : Nat\nhm : m > 1\n⊢ ∃ a b c d, ↑m = ↑a ^ 2 + ↑b ^ 2 + 3 ^ c + 5 ^ d",
        },
    ]


def test_goal_context_binders_are_carried_into_subgoals() -> None:
    ast = AST(_theorem_with_subgoals(), sorries=_sorries())

    code_hn = ast.get_named_subgoal_code("hn_nonneg")
    assert "lemma hn_nonneg" in code_hn
    assert "(n : Int)" in code_hn
    assert "(hn : n > 1)" in code_hn
    assert "::" not in code_hn

    code_hm_coe = ast.get_named_subgoal_code("hm_coe")
    assert "(n : Int)" in code_hm_coe
    assert "(hn : n > 1)" in code_hm_coe
    assert "(m : Nat)" in code_hm_coe
    assert "::" not in code_hm_coe

    code_hm_gt = ast.get_named_subgoal_code("hm_gt")
    assert "(n : Int)" in code_hm_gt
    assert "(hn : n > 1)" in code_hm_gt
    assert "(m : Nat)" in code_hm_gt
    assert "::" not in code_hm_gt

    code_nat_version = ast.get_named_subgoal_code("nat_version")
    assert "(n : Int)" in code_nat_version
    assert "(hn : n > 1)" in code_nat_version
    assert "(m : Nat)" in code_nat_version
    assert "::" not in code_nat_version
