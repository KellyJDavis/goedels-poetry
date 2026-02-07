from __future__ import annotations

from goedels_poetry.agents.prover_agent import _extract_code_block_for_decl, _extract_decl_name_from_decl


def test_extract_decl_name_from_decl_extracts_lemma_name() -> None:
    decl = "lemma h_eval (x : Nat) : True := by\n  trivial"
    assert _extract_decl_name_from_decl(decl) == "h_eval"


def test_extract_code_block_for_decl_picks_matching_decl_over_last_block() -> None:
    response = """Some analysis.

```lean4
lemma h_eval (x : Nat) : True := by
  trivial
```

Some more text.

```lean4
lemma hv_form (x : Nat) : True := by
  trivial
```
"""

    extracted = _extract_code_block_for_decl(response, "h_eval")
    assert extracted is not None
    assert extracted.lstrip().startswith("lemma h_eval")
