"""
Runtime-generated Kimina-backed integration tests for proof reconstruction.

These tests are designed to be fast (tiny synthetic theorems) and broad (many structural
variations) so we can iterate on reconstruction correctness without proving real theorems.

Configuration (env vars)
------------------------
- RECONSTRUCTION_TEST_CASES: number of generated cases to run (default: 600). Set to 0 to skip.
- RECONSTRUCTION_TEST_SEED: seed used to shuffle the deterministic corpus (default: 0).

These tests match the conventions in `tests/test_kimina_agents.py`:
- Import-guard: skip module if required imports fail
- Require a running Kimina server and a local Lean install via fixtures
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, cast

import pytest

# Try to import the required modules - skip all tests if imports fail
# Note: These tests require a separate Kimina Lean Server installation
try:
    from kimina_client import KiminaClient

    from goedels_poetry.agents.sketch_decomposition_agent import SketchDecompositionAgentFactory
    from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
    from goedels_poetry.agents.util.common import (
        DEFAULT_IMPORTS,
        combine_preamble_and_body,
        remove_default_imports_from_ast,
    )
    from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response, parse_kimina_check_response
    from goedels_poetry.parsers.ast import AST
    from goedels_poetry.state import GoedelsPoetryStateManager

    IMPORTS_AVAILABLE = True
except Exception as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"Failed to import required modules: {e}"

# Skip entire module if imports not available
if not IMPORTS_AVAILABLE:
    pytestmark = pytest.mark.skip(reason=SKIP_REASON)
else:
    # Mark all tests in this module as requiring Lean
    pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ReconCase:
    case_id: str
    parent_body: str
    # Map hole-name -> proof-body (tactics under `by`)
    child_proofs: dict[str, str]


def _minimal_safe_closers() -> list[str]:
    # Keep the generated suite fast and robust by using only tactics that should be
    # universally available with `import Mathlib` and also close our tiny goals.
    #
    # NOTE: This list is independent from the reconstruction implementation's
    # "minimal safe set" matcher. The matcher can (and should) be broader; the tests
    # should be stable and avoid depending on optional tactics.
    return ["exact", "simpa", "rfl"]


def _mk_case_named_haves(case_id: str, *, closer: str, comment_before_close: bool, extra_blank_line: bool) -> ReconCase:
    """
    Two named holes: `h₁` and `h₂`. `h₂` depends on `h₁`.

    `h₂`'s child proof includes the formatting pattern seen in partial.log:
    a comment line followed by an over-indented closing tactic.
    """
    thm_name = f"recon_named_haves_{case_id}"
    parent_body = f"""theorem {thm_name} (n : Nat) : n = n := by
  have h₁ : n = n := by
    sorry
  have h₂ : n = n := by
    sorry
  exact h₂
"""

    # h₁ is straightforward.
    proof_h1 = "rfl"

    # h₂: inner have + comment + closing tactic. The closing line is intentionally indented by 2,
    # so the reconstructor must not create a new indentation level when it splices under the hole.
    if closer == "exact":
        close_line = "exact h_main"
    elif closer == "simpa":
        close_line = "simpa using h_main"
    else:
        close_line = "rfl"
    comment = "-- close the goal"
    proof_h2_lines: list[str] = [
        "have h_main : n = n := by",
        "  simpa using h₁",
    ]
    if extra_blank_line:
        proof_h2_lines.append("")
    if comment_before_close:
        proof_h2_lines.append(comment)
    proof_h2_lines.append(f"  {close_line}")
    proof_h2 = "\n".join(proof_h2_lines)

    return ReconCase(case_id=case_id, parent_body=parent_body, child_proofs={"h₁": proof_h1, "h₂": proof_h2})


def _mk_case_calc(case_id: str, *, closer: str, comment_before_close: bool) -> ReconCase:
    """
    A `calc` chain with a named have-hole used as a step.
    """
    thm_name = f"recon_calc_{case_id}"
    parent_body = f"""theorem {thm_name} (n : Nat) : n = n := by
  have h₁ : n = n := by
    sorry
  have h_step : n = n := by
    sorry
  calc
    n = n := h_step
    _ = n := by
      simpa using h₁
"""

    proof_h1 = "rfl"
    if closer == "exact":
        close_line = "exact h₁"
    elif closer == "simpa":
        close_line = "simpa using h₁"
    else:
        close_line = "rfl"

    proof_step_lines = ["have h_main : n = n := by", "  simpa using h₁"]
    if comment_before_close:
        proof_step_lines.append("-- close step")
    proof_step_lines.append(f"  {close_line if closer != 'rfl' else 'exact h_main'}")
    proof_h_step = "\n".join(proof_step_lines)
    return ReconCase(case_id=case_id, parent_body=parent_body, child_proofs={"h₁": proof_h1, "h_step": proof_h_step})


def _mk_case_anonymous_have(case_id: str, *, closer: str, comment_before_close: bool) -> ReconCase:
    """
    Anonymous have-hole:

      have : n = n := by
        sorry

    The decomposition pipeline should assign a synthetic name
    `gp_anon_have__<decl>__1` and record exact offsets for the `sorry`.
    """
    thm_name = f"recon_anon_have_{case_id}"
    proof_lines: list[str] = []
    if comment_before_close:
        proof_lines.append("-- close the anonymous goal")
    parent_body = f"""theorem {thm_name} (n : Nat) : n = n := by
  have : n = n := by
    sorry
  exact this
"""
    anon_name = f"gp_anon_have__{thm_name}__1"
    if closer == "exact":
        proof_lines.append("exact rfl")
    elif closer == "simpa":
        proof_lines.append("simpa")
    else:
        proof_lines.append("rfl")

    proof = "\n".join(proof_lines)
    return ReconCase(case_id=case_id, parent_body=parent_body, child_proofs={anon_name: proof})


def _mk_case_main_body(case_id: str, *, closer: str, comment_before_close: bool) -> ReconCase:
    """
    Main-body hole: a standalone `sorry` after a named have.

    The hole name should be the special marker `"<main body>"`.
    """
    thm_name = f"recon_main_body_{case_id}"
    parent_body = f"""theorem {thm_name} (n : Nat) : n = n := by
  have h₁ : n = n := by
    sorry
  sorry
"""

    proof_h1 = "rfl"
    if closer == "exact":
        main = "exact h₁"
    elif closer == "simpa":
        main = "simpa using h₁"
    else:
        main = "rfl"
    proof_main_lines: list[str] = []
    if comment_before_close:
        proof_main_lines.append("-- close main body")
    proof_main_lines.append(main)
    proof_main = "\n".join(proof_main_lines)

    return ReconCase(case_id=case_id, parent_body=parent_body, child_proofs={"h₁": proof_h1, "<main body>": proof_main})


def _mk_case_inline_by_sorry(case_id: str, *, closer: str, comment_before_close: bool) -> ReconCase:
    """
    Inline-hole form: `have h : ... := by sorry` (the `sorry` token is on the same line as `by`).

    This specifically exercises the reconstructor's "inline hole" insertion path that must turn
    `by sorry` into `by\\n  <proof>`.
    """
    thm_name = f"recon_inline_by_{case_id}"
    parent_body = f"""theorem {thm_name} (n : Nat) : n = n := by
  have h_inline : n = n := by sorry
  exact h_inline
"""

    if closer == "exact":
        close = "exact rfl"
    elif closer == "simpa":
        close = "simpa"
    else:
        close = "rfl"
    lines: list[str] = []
    if comment_before_close:
        lines.append("-- close inline")
    lines.append(close)
    return ReconCase(case_id=case_id, parent_body=parent_body, child_proofs={"h_inline": "\n".join(lines)})


def _generate_cases() -> list[ReconCase]:
    # Generate a deterministic corpus (no randomness here). We intentionally produce a corpus
    # larger than the default selection size (200) so the seed-shuffle has meaningful effect.
    closers = _minimal_safe_closers()
    cases: list[ReconCase] = []

    # Named have cases: 240 variants
    # - rotate closers
    # - toggle comment/blank lines
    # - vary whether we add an extra blank line before the closing tactic
    for idx in range(240):
        closer = closers[idx % len(closers)]
        comment_before_close = ((idx // len(closers)) % 2) == 1
        extra_blank_line = ((idx // (len(closers) * 2)) % 2) == 1
        cases.append(
            _mk_case_named_haves(
                f"{idx:04d}",
                closer=closer,
                comment_before_close=comment_before_close,
                extra_blank_line=extra_blank_line,
            )
        )

    # Calc cases: 120 variants
    base = len(cases)
    for j in range(120):
        closer = closers[j % len(closers)]
        comment_before_close = ((j // len(closers)) % 2) == 1
        cases.append(_mk_case_calc(f"{base + j:04d}", closer=closer, comment_before_close=comment_before_close))

    # Anonymous have cases: 120 variants
    base = len(cases)
    for k in range(120):
        closer = closers[k % len(closers)]
        comment_before_close = ((k // len(closers)) % 2) == 1
        cases.append(
            _mk_case_anonymous_have(f"{base + k:04d}", closer=closer, comment_before_close=comment_before_close)
        )

    # Main-body cases: 120 variants
    base = len(cases)
    for m in range(120):
        closer = closers[m % len(closers)]
        comment_before_close = ((m // len(closers)) % 2) == 1
        cases.append(_mk_case_main_body(f"{base + m:04d}", closer=closer, comment_before_close=comment_before_close))

    # Inline `:= by sorry` cases: 120 variants
    base = len(cases)
    for p in range(120):
        closer = closers[p % len(closers)]
        comment_before_close = ((p // len(closers)) % 2) == 1
        cases.append(
            _mk_case_inline_by_sorry(f"{base + p:04d}", closer=closer, comment_before_close=comment_before_close)
        )

    return cases


if IMPORTS_AVAILABLE:

    @pytest.fixture(scope="session")
    def _kimina_client(kimina_server_url: str) -> KiminaClient:
        return KiminaClient(api_url=kimina_server_url, http_timeout=36000, n_retries=3)

    def _parse_parent_ast(client: KiminaClient, parent_body: str) -> AST:
        normalized_preamble = DEFAULT_IMPORTS.strip()
        normalized_body = parent_body.strip()
        full = combine_preamble_and_body(normalized_preamble, normalized_body)

        # Compute body_start from the actual combined string.
        if normalized_preamble and normalized_body:
            body_start = full.find(normalized_body, len(normalized_preamble))
            body_start = body_start if body_start != -1 else len(normalized_preamble)
        else:
            body_start = 0

        ast_code_response = client.ast_code(full)
        parsed = parse_kimina_ast_code_response(ast_code_response)
        assert parsed["error"] is None, parsed["error"]
        ast_without_imports = remove_default_imports_from_ast(parsed["ast"], preamble=DEFAULT_IMPORTS)
        return AST(ast_without_imports, source_text=full, body_start=body_start)

    def _reconstruct_and_check(client: KiminaClient, case: ReconCase) -> None:
        # Build DecomposedFormalTheoremState and parse it with the real SketchParserAgent (mirrors pipeline).
        root: DecomposedFormalTheoremState = cast(
            DecomposedFormalTheoremState,
            {
                "parent": None,
                "children": [],
                "depth": 0,
                "formal_theorem": case.parent_body,
                "preamble": DEFAULT_IMPORTS,
                "proof_sketch": case.parent_body,
                "syntactic": True,
                "errors": "",
                "ast": None,
                "self_correction_attempts": 0,
                "decomposition_history": [],
                "search_queries": None,
                "search_results": None,
            },
        )

        # Attach AST for decomposition. We avoid invoking the agent graph here because we already
        # have a Kimina client and want stable, minimal overhead per case.
        root["ast"] = _parse_parent_ast(client, case.parent_body)

        # Decompose to create children states (with hole offsets).
        decomposer = SketchDecompositionAgentFactory.create_agent()
        out_states: DecomposedFormalTheoremStates = decomposer.invoke({"inputs": [root], "outputs": []})
        assert out_states["outputs"], "decomposer produced no outputs"
        decomposed = out_states["outputs"][0]

        # Fill child proofs based on hole_name.
        for child in decomposed["children"]:
            child_dict = cast(dict[str, Any], child)
            hole_name = cast(str | None, child_dict.get("hole_name"))
            assert hole_name is not None
            assert hole_name in case.child_proofs, f"Missing child proof for hole '{hole_name}' in case {case.case_id}"
            child_dict["formal_proof"] = case.child_proofs[hole_name]
            child_dict["proved"] = True
            child_dict["errors"] = ""

        # Reconstruct using state manager helper on the decomposed node.
        dummy_state = cast(Any, type("_S", (), {})())
        dummy_state._root_preamble = DEFAULT_IMPORTS
        manager = GoedelsPoetryStateManager(dummy_state)
        reconstructed_body = manager._reconstruct_decomposed_node_proof(decomposed)
        reconstructed_full = combine_preamble_and_body(DEFAULT_IMPORTS, reconstructed_body)

        # Sanity: no textual sorry should remain.
        assert "sorry" not in reconstructed_full

        # Kimina check must pass and be complete.
        check_resp = client.check(reconstructed_full)
        parsed_check = parse_kimina_check_response(check_resp)
        assert parsed_check["pass"] is True, parsed_check
        assert parsed_check["complete"] is True, parsed_check

    def _selected_cases() -> list[ReconCase]:
        n = _env_int("RECONSTRUCTION_TEST_CASES", 600)
        seed = _env_int("RECONSTRUCTION_TEST_SEED", 0)
        if n <= 0:
            return []
        corpus = _generate_cases()
        rng = random.Random(seed)  # noqa: S311
        rng.shuffle(corpus)
        return corpus[: min(n, len(corpus))]

    _CASES = _selected_cases()

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: cast(ReconCase, c).case_id)
    def test_reconstruction_generated_cases(_kimina_client: KiminaClient, case: ReconCase) -> None:
        _reconstruct_and_check(_kimina_client, case)
