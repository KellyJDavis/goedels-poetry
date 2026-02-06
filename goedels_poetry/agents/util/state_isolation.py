"""
Helpers for isolating per-item state in LangGraph map/reduce fan-out.

The LangGraph Send API and state merging operate on Python objects. If we pass live
TreeNode dicts that participate in a cyclic object graph (e.g. via `parent` pointers
and `children` dicts), then parallel execution can observe shared mutable references
and can also trigger implementation-dependent copying/merging behaviour.

To keep map/reduce robust, we fan out *detached, acyclic* per-item payloads and
later reattach them into the canonical proof tree by id.
"""

from __future__ import annotations

from typing import cast

from goedels_poetry.agents.state import (
    APISearchResponseTypedDict,
    DecomposedFormalTheoremState,
    FormalTheoremProofState,
)


def detach_formal_proof_state(state: FormalTheoremProofState) -> FormalTheoremProofState:
    """
    Return an acyclic copy of a FormalTheoremProofState for parallel processing.

    - Clears the `parent` pointer to break cycles.
    - Copies `proof_history` so worker appends are isolated from any shared list.

    The returned dict is intended to be merged back into the canonical tree by id
    (see GoedelsPoetryStateManager._reconstruct_tree()).
    """
    detached = dict(state)
    detached["parent"] = None
    # Defensive: ensure history list is not shared by reference.
    detached["proof_history"] = list(state.get("proof_history", []))
    return cast(FormalTheoremProofState, detached)


def detach_decomposed_theorem_state(state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremState:
    """
    Return an acyclic copy of a DecomposedFormalTheoremState for parallel processing.

    - Clears the `parent` pointer to break cycles.
    - Copies list fields that are commonly appended to by workers (e.g. message history).
    - Copies `children` mapping (shallow) so workers can't mutate the canonical mapping.

    NOTE: Decomposition-stage workers that *add* children will still populate the copied
    `children` mapping on the detached state; the resulting node will be merged back
    into the canonical proof tree by id.
    """
    detached = dict(state)
    detached["parent"] = None
    detached["children"] = dict(state.get("children", {}))
    detached["decomposition_history"] = list(state.get("decomposition_history", []))
    if state.get("search_queries") is not None:
        detached["search_queries"] = list(cast(list[str], state["search_queries"]))
    search_results = state.get("search_results")
    if search_results is not None:
        detached["search_results"] = list(cast(list[APISearchResponseTypedDict], search_results))
    return cast(DecomposedFormalTheoremState, detached)
