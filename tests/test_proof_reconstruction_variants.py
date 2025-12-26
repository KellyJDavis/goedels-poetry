"""Unit tests for Kimina-guided proof reconstruction variant generation."""

from contextlib import suppress

from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


def _with_default_preamble(body: str) -> str:
    return combine_preamble_and_body(DEFAULT_IMPORTS, body)


def test_reconstruction_variants_deterministic_and_capped() -> None:
    import uuid

    theorem_sig = f"theorem test_recon_variants_{uuid.uuid4().hex} : True"
    theorem = _with_default_preamble(theorem_sig)

    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(theorem)

    try:
        state = GoedelsPoetryState(formal_theorem=theorem)
        manager = GoedelsPoetryStateManager(state)

        variants_a = manager._get_reconstruction_variants(max_candidates=12)
        variants_b = manager._get_reconstruction_variants(max_candidates=12)

        assert variants_a == variants_b
        assert variants_a, "Expected at least one reconstruction variant"
        assert variants_a[0].variant_id == "baseline"
        assert len(variants_a) <= 12
        assert len({v.variant_id for v in variants_a}) == len(variants_a)

        variants_small = manager._get_reconstruction_variants(max_candidates=3)
        assert len(variants_small) == 3
        assert variants_small[0].variant_id == "baseline"

    finally:
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)
