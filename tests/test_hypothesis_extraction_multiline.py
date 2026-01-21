from __future__ import annotations

from goedels_poetry.parsers.util.hypothesis_extraction import (
    extract_hypotheses_from_unsolved_goals_data,
    parse_hypothesis_strings_to_binders,
)


def test_extract_hypotheses_merges_indented_continuations() -> None:
    # This reproduces the "h_id :" + newline + indented type formatting seen in partial.log.
    data = (
        "unsolved goals\n"
        "h_id :\n"
        "  (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id =\n"
        "    ∏ x ∈ Finset.filter (fun x => ¬Even x) (Finset.range 10000), x\n"
        "h_finset : Finset.filter (fun x => ¬Even x) (Finset.range 10000) = "
        "Finset.image (fun k => 2 * k + 1) (Finset.range 5000)\n"
        "h_prod_rewrite : ∏ x ∈ Finset.filter (fun x => ¬Even x) (Finset.range 10000), x = "
        "∏ k ∈ Finset.range 5000, (2 * k + 1)\n"
        "⊢ (Finset.filter (fun x => ¬Even x) (Finset.range 10000)).prod id = "
        "∏ x ∈ Finset.filter (fun x => ¬Even x) (Finset.range 10000), x"
    )

    hyps = extract_hypotheses_from_unsolved_goals_data(data)

    assert len(hyps) == 3
    assert all("\n" not in h for h in hyps)
    assert "h_id :" in hyps[0]
    assert "prod id =" in hyps[0]
    assert "∏ x ∈" in hyps[0]
    assert hyps[0] != "h_id :"

    binders = parse_hypothesis_strings_to_binders(hyps)
    assert "(h_id :)" not in " ".join(binders)
