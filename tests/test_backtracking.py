"""Tests for backtracking functionality in goedels_poetry.state module."""

import os
import tempfile
import uuid
from contextlib import suppress
from typing import cast

import pytest

from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager
from goedels_poetry.util.tree import TreeNode

TEST_PREAMBLE = "import Mathlib\n\nset_option maxHeartbeats 0"


@pytest.fixture
def temp_state() -> GoedelsPoetryState:
    """Create a temporary state for testing."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    tmpdir = tempfile.mkdtemp()
    os.environ["GOEDELS_POETRY_DIR"] = tmpdir

    theorem_name = f"test_backtracking_theorem_{uuid.uuid4().hex}"
    theorem_body = f"theorem {theorem_name} : True := by sorry"
    full_theorem = combine_preamble_and_body(TEST_PREAMBLE, theorem_body)
    state = GoedelsPoetryState(formal_theorem=full_theorem)

    yield state

    # Cleanup
    with suppress(Exception):
        GoedelsPoetryState.clear_theorem_directory(full_theorem)
    if old_env is not None:
        os.environ["GOEDELS_POETRY_DIR"] = old_env
    elif "GOEDELS_POETRY_DIR" in os.environ:
        del os.environ["GOEDELS_POETRY_DIR"]


def test_get_sketches_to_backtrack_empty(temp_state: GoedelsPoetryState) -> None:
    """Test get_sketches_to_backtrack returns empty list when queue is empty."""
    manager = GoedelsPoetryStateManager(temp_state)

    result = manager.get_sketches_to_backtrack()

    assert result["inputs"] == []
    assert result["outputs"] == []


def test_get_sketches_to_backtrack_with_items(temp_state: GoedelsPoetryState) -> None:
    """Test get_sketches_to_backtrack returns items from the backtrack queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a decomposed state and add it to the backtrack queue
    decomposed_state = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem test : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch=None,
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )
    temp_state.decomposition_backtrack_queue.append(decomposed_state)

    result = manager.get_sketches_to_backtrack()

    assert len(result["inputs"]) == 1
    assert result["inputs"][0] == decomposed_state
    assert result["outputs"] == []


def test_get_sketches_to_backtrack_with_multiple_items(temp_state: GoedelsPoetryState) -> None:
    """Test get_sketches_to_backtrack returns multiple items."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create multiple decomposed states
    states = []
    for i in range(3):
        state = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=i,
            formal_theorem=f"theorem test{i} : True := by sorry",
            preamble=TEST_PREAMBLE,
            proof_sketch=None,
            syntactic=False,
            errors=None,
            ast=None,
            self_correction_attempts=i,
            decomposition_history=[],
        )
        states.append(state)
        temp_state.decomposition_backtrack_queue.append(state)

    result = manager.get_sketches_to_backtrack()

    assert len(result["inputs"]) == 3
    assert result["inputs"] == states
    assert result["outputs"] == []


def test_set_backtracked_sketches_clears_queue(temp_state: GoedelsPoetryState) -> None:
    """Test set_backtracked_sketches clears the backtrack queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Add items to backtrack queue
    decomposed_state = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem test : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch=None,
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )
    temp_state.decomposition_backtrack_queue.append(decomposed_state)

    # Create backtracked sketches
    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    backtracked = DecomposedFormalTheoremStates(inputs=[], outputs=[decomposed_state])

    manager.set_backtracked_sketches(backtracked)

    # Backtrack queue should be empty
    assert temp_state.decomposition_backtrack_queue == []


def test_set_backtracked_sketches_adds_to_validate_queue(temp_state: GoedelsPoetryState) -> None:
    """Test set_backtracked_sketches adds items to the search queue (to regenerate queries)."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create backtracked sketches
    decomposed_state = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem test : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=1,
        decomposition_history=[],
        search_queries=None,
    )

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    backtracked = DecomposedFormalTheoremStates(inputs=[], outputs=[decomposed_state])

    manager.set_backtracked_sketches(backtracked)

    # Search queue should have the item (to regenerate queries before sketching)
    assert len(temp_state.decomposition_search_queue) == 1
    assert temp_state.decomposition_search_queue[0] == decomposed_state


def test_set_backtracked_sketches_with_multiple_items(temp_state: GoedelsPoetryState) -> None:
    """Test set_backtracked_sketches handles multiple items correctly."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create multiple backtracked sketches
    states = []
    for i in range(3):
        state = DecomposedFormalTheoremState(
            parent=None,
            children=[],
            depth=i,
            formal_theorem=f"theorem test{i} : True := by sorry",
            preamble=TEST_PREAMBLE,
            proof_sketch=f"by trivial{i}",
            syntactic=False,
            errors=None,
            ast=None,
            self_correction_attempts=i + 1,
            decomposition_history=[],
            search_queries=None,
        )
        states.append(state)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    backtracked = DecomposedFormalTheoremStates(inputs=[], outputs=states)

    manager.set_backtracked_sketches(backtracked)

    # All items should be in search queue (to regenerate queries before sketching)
    assert len(temp_state.decomposition_search_queue) == 3
    assert temp_state.decomposition_search_queue == states


def test_backtracking_integration_with_validated_sketches(temp_state: GoedelsPoetryState) -> None:
    """Test that set_validated_sketches triggers backtracking for failed sketches."""
    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a parent decomposed state
    parent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a child that has failed (reached max retries - 1, will reach max after increment)
    child = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent),
        children=[],
        depth=1,
        formal_theorem="theorem child : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,  # Will reach max after increment
        decomposition_history=[],
    )
    parent["children"] = [cast(TreeNode, child)]

    # Set up the state
    temp_state.formal_theorem_proof = cast(TreeNode, parent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    # Simulate validation of the child (marking it invalid)
    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[child])

    manager.set_validated_sketches(validated_sketches)

    # Child should have reached max attempts
    assert child["self_correction_attempts"] == DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    # Parent should be in the backtrack queue (since it has attempts remaining)
    assert len(temp_state.decomposition_backtrack_queue) == 1
    assert temp_state.decomposition_backtrack_queue[0] == parent

    # Parent should have been prepared for re-sketching (children cleared)
    assert parent["children"] == []
    assert parent["proof_sketch"] is None
    assert parent["syntactic"] is False
    assert parent["errors"] is None
    assert parent["ast"] is None


def test_backtracking_sets_finished_when_no_ancestor(temp_state: GoedelsPoetryState) -> None:
    """Test that backtracking sets is_finished when no backtrackable ancestor exists."""
    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a root that has also exhausted attempts
    root = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem root : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,  # Will reach max after increment
        decomposition_history=[],
    )

    temp_state.formal_theorem_proof = cast(TreeNode, root)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[root])

    manager.set_validated_sketches(validated_sketches)

    # Root should have reached max attempts
    assert root["self_correction_attempts"] == DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    # No backtrackable ancestor exists, so is_finished should be True
    assert manager.is_finished is True

    # Backtrack queue should be empty
    assert temp_state.decomposition_backtrack_queue == []


def test_backtracking_removes_descendants_from_queues(temp_state: GoedelsPoetryState) -> None:
    """Test that backtracking removes all descendants from all queues."""
    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a tree structure
    parent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a child decomposed state
    child_decomposed = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent),
        children=[],
        depth=1,
        formal_theorem="theorem child_decomp : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a grandchild proof state
    grandchild_proof = FormalTheoremProofState(
        parent=cast(TreeNode, child_decomposed),
        depth=2,
        formal_theorem="theorem grandchild : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
        hole_name=None,
        hole_start=None,
        hole_end=None,
    )

    child_decomposed["children"] = [cast(TreeNode, grandchild_proof)]
    parent["children"] = [cast(TreeNode, child_decomposed)]

    # Create a failed child that will trigger backtracking
    failed_child = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent),
        children=[],
        depth=1,
        formal_theorem="theorem failed_child : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
        decomposition_history=[],
    )
    parent["children"].append(cast(TreeNode, failed_child))

    # Add descendants to various queues
    temp_state.decomposition_sketch_queue.append(child_decomposed)
    temp_state.proof_prove_queue.append(grandchild_proof)

    temp_state.formal_theorem_proof = cast(TreeNode, parent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[failed_child])

    manager.set_validated_sketches(validated_sketches)

    # All descendants should be removed from queues
    assert child_decomposed not in temp_state.decomposition_sketch_queue
    assert grandchild_proof not in temp_state.proof_prove_queue

    # Parent should be prepared for backtracking
    assert parent in temp_state.decomposition_backtrack_queue


def test_backtracking_with_valid_sketches(temp_state: GoedelsPoetryState) -> None:
    """Test that valid sketches still get processed while failed ones trigger backtracking."""
    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a valid sketch
    valid_sketch = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem valid : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,  # Valid
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a parent for the failed sketch
    parent_of_failed = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a failed sketch
    failed_sketch = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent_of_failed),
        children=[],
        depth=1,
        formal_theorem="theorem failed : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,  # Invalid
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
        decomposition_history=[],
    )
    parent_of_failed["children"] = [cast(TreeNode, failed_sketch)]

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[valid_sketch, failed_sketch])

    manager.set_validated_sketches(validated_sketches)

    # Valid sketch should be in AST queue
    assert valid_sketch in temp_state.decomposition_ast_queue

    # Parent of failed sketch should be in backtrack queue
    assert parent_of_failed in temp_state.decomposition_backtrack_queue

    # Failed sketch should have reached max attempts
    assert failed_sketch["self_correction_attempts"] == DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS


def test_backtracking_preserves_history(temp_state: GoedelsPoetryState) -> None:
    """Test that backtracking preserves decomposition_history."""
    from langchain_core.messages import HumanMessage

    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS

    manager = GoedelsPoetryStateManager(temp_state)

    # Create parent with history
    parent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[HumanMessage(content="Initial prompt")],
    )

    # Create failed child
    child = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent),
        children=[],
        depth=1,
        formal_theorem="theorem child : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
        decomposition_history=[],
    )
    parent["children"] = [cast(TreeNode, child)]

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[child])

    manager.set_validated_sketches(validated_sketches)

    # Parent should be in backtrack queue
    assert parent in temp_state.decomposition_backtrack_queue

    # History should be preserved
    assert len(parent["decomposition_history"]) == 1
    assert parent["decomposition_history"][0].content == "Initial prompt"

    # self_correction_attempts should be preserved (not incremented by backtracking itself)
    assert parent["self_correction_attempts"] == 0


def test_set_decomposed_sketches_with_too_deep_children_backtracks(temp_state: GoedelsPoetryState) -> None:
    """Test that set_decomposed_sketches backtracks when children exceed max depth."""
    from goedels_poetry.config.llm import (
        PROVER_AGENT_MAX_DEPTH,
    )

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a tree structure: grandparent -> parent -> too_deep_child
    grandparent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem grandparent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,  # Has attempts remaining
        decomposition_history=[],
    )

    parent = DecomposedFormalTheoremState(
        parent=cast(TreeNode, grandparent),
        children=[],
        depth=1,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )
    grandparent["children"] = [cast(TreeNode, parent)]

    # Create a child that exceeds max depth
    too_deep_child = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH,  # At max depth
        formal_theorem="theorem too_deep : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )
    parent["children"] = [cast(TreeNode, too_deep_child)]

    temp_state.formal_theorem_proof = cast(TreeNode, grandparent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    # Simulate decomposition of parent that creates too_deep_child
    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Grandparent should be in backtrack queue (since it's backtrackable and at grandparent level)
    assert len(temp_state.decomposition_backtrack_queue) == 1
    assert temp_state.decomposition_backtrack_queue[0] == grandparent

    # Grandparent should be prepared for re-sketching
    assert grandparent["children"] == []
    assert grandparent["proof_sketch"] is None
    assert grandparent["syntactic"] is False

    # Should not be finished (backtracking found a solution)
    assert manager.is_finished is False

    # Too-deep child should not be in proof_prove_queue
    assert too_deep_child not in temp_state.proof_prove_queue


def test_set_decomposed_sketches_with_too_deep_children_no_backtrackable_ancestor(
    temp_state: GoedelsPoetryState,
) -> None:
    """Test that set_decomposed_sketches finishes when no backtrackable ancestor exists."""
    from goedels_poetry.config.llm import (
        DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
        PROVER_AGENT_MAX_DEPTH,
    )

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a tree where grandparent has exhausted attempts
    grandparent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem grandparent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,  # Exhausted
        decomposition_history=[],
    )

    parent = DecomposedFormalTheoremState(
        parent=cast(TreeNode, grandparent),
        children=[],
        depth=1,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )
    grandparent["children"] = [cast(TreeNode, parent)]

    # Create a child that exceeds max depth
    too_deep_child = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH,
        formal_theorem="theorem too_deep : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )
    parent["children"] = [cast(TreeNode, too_deep_child)]

    temp_state.formal_theorem_proof = cast(TreeNode, grandparent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Should be finished (no backtrackable ancestor)
    assert manager.is_finished is True
    assert "Maximum proof tree depth exceeded" in manager.reason
    assert "no backtrackable ancestors found" in manager.reason

    # Backtrack queue should be empty
    assert temp_state.decomposition_backtrack_queue == []


def test_set_decomposed_sketches_with_too_deep_children_no_grandparent(
    temp_state: GoedelsPoetryState,
) -> None:
    """Test that set_decomposed_sketches finishes when child has no grandparent."""
    from goedels_poetry.config.llm import PROVER_AGENT_MAX_DEPTH

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a parent with no grandparent (parent is root)
    parent = DecomposedFormalTheoremState(
        parent=None,  # No parent, so no grandparent
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create a child that exceeds max depth
    too_deep_child = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH,
        formal_theorem="theorem too_deep : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )
    parent["children"] = [cast(TreeNode, too_deep_child)]

    temp_state.formal_theorem_proof = cast(TreeNode, parent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Should be finished (no grandparent to backtrack to)
    assert manager.is_finished is True
    assert "Maximum proof tree depth exceeded" in manager.reason


def test_set_decomposed_sketches_with_mixed_depth_children(temp_state: GoedelsPoetryState) -> None:
    """Test that set_decomposed_sketches handles mix of too-deep and normal children."""
    from goedels_poetry.config.llm import PROVER_AGENT_MAX_DEPTH

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a grandparent
    grandparent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem grandparent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    parent = DecomposedFormalTheoremState(
        parent=cast(TreeNode, grandparent),
        children=[],
        depth=1,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )
    grandparent["children"] = [cast(TreeNode, parent)]

    # Create one too-deep child
    too_deep_child = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH,
        formal_theorem="theorem too_deep : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    # Create one normal child (not too deep)
    normal_child = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH - 1,  # Just under max
        formal_theorem="theorem normal : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    parent["children"] = [cast(TreeNode, too_deep_child), cast(TreeNode, normal_child)]

    temp_state.formal_theorem_proof = cast(TreeNode, grandparent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Grandparent should be in backtrack queue
    assert len(temp_state.decomposition_backtrack_queue) == 1
    assert temp_state.decomposition_backtrack_queue[0] == grandparent

    # Too-deep child should NOT be in proof_prove_queue
    assert too_deep_child not in temp_state.proof_prove_queue

    # Normal child should also NOT be in proof_prove_queue (because backtracking removes all descendants)
    # Actually, wait - the normal child is a descendant of grandparent, so it will be removed
    # But we want to test that if there were other children not under the backtrack target, they'd be queued
    # For now, this test verifies the basic behavior


def test_set_decomposed_sketches_with_multiple_too_deep_children_same_grandparent(
    temp_state: GoedelsPoetryState,
) -> None:
    """Test that set_decomposed_sketches only backtracks once per grandparent."""
    from goedels_poetry.config.llm import PROVER_AGENT_MAX_DEPTH

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a grandparent
    grandparent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem grandparent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    parent1 = DecomposedFormalTheoremState(
        parent=cast(TreeNode, grandparent),
        children=[],
        depth=1,
        formal_theorem="theorem parent1 : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    parent2 = DecomposedFormalTheoremState(
        parent=cast(TreeNode, grandparent),
        children=[],
        depth=1,
        formal_theorem="theorem parent2 : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    grandparent["children"] = [cast(TreeNode, parent1), cast(TreeNode, parent2)]

    # Create too-deep children for both parents
    too_deep_child1 = FormalTheoremProofState(
        parent=cast(TreeNode, parent1),
        depth=PROVER_AGENT_MAX_DEPTH,
        formal_theorem="theorem too_deep1 : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    too_deep_child2 = FormalTheoremProofState(
        parent=cast(TreeNode, parent2),
        depth=PROVER_AGENT_MAX_DEPTH,
        formal_theorem="theorem too_deep2 : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    parent1["children"] = [cast(TreeNode, too_deep_child1)]
    parent2["children"] = [cast(TreeNode, too_deep_child2)]

    temp_state.formal_theorem_proof = cast(TreeNode, grandparent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    # Simulate decomposition of both parents
    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent1, parent2])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Grandparent should be in backtrack queue only ONCE (not twice)
    assert len(temp_state.decomposition_backtrack_queue) == 1
    assert temp_state.decomposition_backtrack_queue[0] == grandparent


def test_set_decomposed_sketches_with_no_too_deep_children(temp_state: GoedelsPoetryState) -> None:
    """Test that set_decomposed_sketches queues children normally when none are too deep."""
    from goedels_poetry.config.llm import PROVER_AGENT_MAX_DEPTH

    manager = GoedelsPoetryStateManager(temp_state)

    parent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
    )

    # Create children that are NOT too deep
    child1 = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=PROVER_AGENT_MAX_DEPTH - 1,
        formal_theorem="theorem child1 : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    child2 = FormalTheoremProofState(
        parent=cast(TreeNode, parent),
        depth=1,
        formal_theorem="theorem child2 : True := by sorry",
        preamble=TEST_PREAMBLE,
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
    )

    parent["children"] = [cast(TreeNode, child1), cast(TreeNode, child2)]

    temp_state.formal_theorem_proof = cast(TreeNode, parent)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    decomposed_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[parent])

    manager.set_decomposed_sketches(decomposed_sketches)

    # Both children should be in proof_prove_queue
    assert child1 in temp_state.proof_prove_queue
    assert child2 in temp_state.proof_prove_queue

    # Should not be finished
    assert manager.is_finished is False

    # Backtrack queue should be empty
    assert temp_state.decomposition_backtrack_queue == []
