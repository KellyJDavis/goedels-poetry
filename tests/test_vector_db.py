"""Tests for vector database query functionality."""

import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

from goedels_poetry.agents.state import (
    APISearchResponseTypedDict,
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
)
from goedels_poetry.agents.util.common import combine_preamble_and_body
from goedels_poetry.agents.vector_db_agent import VectorDBAgentFactory, _query_vectordb
from goedels_poetry.config.lean_explore_server import LEAN_EXPLORE_SERVER, _parse_package_filters
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

TEST_PREAMBLE = "import Mathlib\n\nset_option maxHeartbeats 0"


@pytest.fixture
def temp_state() -> Generator[GoedelsPoetryState, None, None]:
    """Create a temporary state for testing."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    tmpdir = tempfile.mkdtemp()
    os.environ["GOEDELS_POETRY_DIR"] = tmpdir

    theorem_name = f"test_vector_db_{uuid.uuid4().hex}"
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


def test_parse_package_filters() -> None:
    """Test parsing comma-separated package filters."""
    assert _parse_package_filters("Mathlib,Batteries,Std") == ["Mathlib", "Batteries", "Std"]
    assert _parse_package_filters("Mathlib, Batteries, Std") == ["Mathlib", "Batteries", "Std"]
    assert _parse_package_filters("Mathlib") == ["Mathlib"]
    assert _parse_package_filters("") == []
    assert _parse_package_filters("   ") == []


def test_lean_explore_server_config() -> None:
    """Test that LEAN_EXPLORE_SERVER config is loaded correctly."""
    assert "url" in LEAN_EXPLORE_SERVER
    assert "package_filters" in LEAN_EXPLORE_SERVER
    assert isinstance(LEAN_EXPLORE_SERVER["url"], str)
    assert isinstance(LEAN_EXPLORE_SERVER["package_filters"], list)
    assert len(LEAN_EXPLORE_SERVER["package_filters"]) > 0


def test_get_theorems_with_search_queries_for_vectordb_empty(temp_state: GoedelsPoetryState) -> None:
    """Test get_theorems_with_search_queries_for_vectordb returns empty list when queue is empty."""
    manager = GoedelsPoetryStateManager(temp_state)

    result = manager.get_theorems_with_search_queries_for_vectordb()

    assert result["inputs"] == []
    assert result["outputs"] == []


def test_get_theorems_with_search_queries_for_vectordb_with_items(temp_state: GoedelsPoetryState) -> None:
    """Test get_theorems_with_search_queries_for_vectordb returns items from the query queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a decomposed state with search queries and add it to the query queue
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
        search_queries=["query1", "query2"],
        search_results=None,
    )
    temp_state.decomposition_query_queue.append(decomposed_state)

    result = manager.get_theorems_with_search_queries_for_vectordb()

    assert len(result["inputs"]) == 1
    assert result["inputs"][0] == decomposed_state
    assert result["outputs"] == []


def test_set_theorems_with_vectordb_results(temp_state: GoedelsPoetryState) -> None:
    """Test set_theorems_with_vectordb_results moves states to sketch queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create states with search results
    states = []
    for i in range(3):
        search_result: APISearchResponseTypedDict = {
            "query": f"query{i}",
            "packages_applied": ["Mathlib"],
            "results": [{"name": f"theorem{i}", "type": "theorem"}],
            "count": 1,
            "total_candidates_considered": 10,
            "processing_time_ms": 100,
        }
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
            self_correction_attempts=0,
            decomposition_history=[],
            search_queries=[f"query{i}"],
            search_results=[search_result],
        )
        states.append(state)

    states_with_results = DecomposedFormalTheoremStates(inputs=[], outputs=states)

    # Add one state to query queue
    temp_state.decomposition_query_queue.append(states[0])

    manager.set_theorems_with_vectordb_results(states_with_results)

    # Query queue should be cleared
    assert len(temp_state.decomposition_query_queue) == 0

    # All states should be in sketch queue
    assert len(temp_state.decomposition_sketch_queue) == 3
    assert all(state in temp_state.decomposition_sketch_queue for state in states)


def test_set_theorems_with_search_queries_generated_routes_to_query_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that set_theorems_with_search_queries_generated routes to query queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create states with queries
    states = []
    for i in range(2):
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
            self_correction_attempts=0,
            decomposition_history=[],
            search_queries=[f"query{i}", f"query{i}_alt"],
            search_results=None,
        )
        states.append(state)

    states_with_queries = DecomposedFormalTheoremStates(inputs=[], outputs=states)

    # Add one state to search queue
    temp_state.decomposition_search_queue.append(states[0])

    manager.set_theorems_with_search_queries_generated(states_with_queries)

    # Search queue should be cleared
    assert len(temp_state.decomposition_search_queue) == 0

    # All states should be in query queue (not sketch queue)
    assert len(temp_state.decomposition_query_queue) == 2
    assert all(state in temp_state.decomposition_query_queue for state in states)
    assert len(temp_state.decomposition_sketch_queue) == 0


def test_remove_decomposition_node_from_queues_includes_query_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that _remove_decomposition_node_from_queues removes nodes from query queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a node
    node = DecomposedFormalTheoremState(
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
        search_queries=["query1"],
        search_results=None,
    )

    # Add to query queue
    temp_state.decomposition_query_queue.append(node)

    manager._remove_decomposition_node_from_queues(node)

    # Node should be removed from query queue
    assert node not in temp_state.decomposition_query_queue


def test_prepare_node_for_resketching_clears_search_results(temp_state: GoedelsPoetryState) -> None:
    """Test that _prepare_node_for_resketching clears search_results."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a node with search results
    search_result: APISearchResponseTypedDict = {
        "query": "test query",
        "packages_applied": ["Mathlib"],
        "results": [{"name": "theorem1", "type": "theorem"}],
        "count": 1,
        "total_candidates_considered": 10,
        "processing_time_ms": 100,
    }
    node = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem test : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by trivial",
        syntactic=True,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        decomposition_history=[],
        search_queries=["query1"],
        search_results=[search_result],
    )

    manager._prepare_node_for_resketching(node)

    assert node["search_queries"] is None
    assert node["search_results"] is None
    assert node["proof_sketch"] is None
    assert node["syntactic"] is False
    assert node["errors"] is None
    assert node["ast"] is None
    assert node["children"] == []


def test_query_vectordb_with_none_search_queries() -> None:
    """Test _query_vectordb handles None search_queries."""
    state = DecomposedFormalTheoremState(
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
        search_queries=None,
        search_results=None,
    )

    result = _query_vectordb("http://localhost:8001/api/v1", ["Mathlib"], state)

    assert result["outputs"][0]["search_results"] is None
    assert result["outputs"][0]["search_queries"] is None


def test_query_vectordb_with_empty_search_queries() -> None:
    """Test _query_vectordb handles empty search_queries."""
    state = DecomposedFormalTheoremState(
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
        search_queries=[],
        search_results=None,
    )

    result = _query_vectordb("http://localhost:8001/api/v1", ["Mathlib"], state)

    assert result["outputs"][0]["search_results"] == []
    assert result["outputs"][0]["search_queries"] == []


@patch("goedels_poetry.agents.vector_db_agent.Client")
@patch("goedels_poetry.agents.vector_db_agent.asyncio.run")
def test_query_vectordb_with_single_query(mock_asyncio_run: MagicMock, mock_client_class: MagicMock) -> None:
    """Test _query_vectordb with a single search query."""
    # Create mock API response - use a regular object (not Pydantic model)
    # The code checks for model_dump first, then falls back to attribute access
    mock_api_response = MagicMock()
    mock_api_response.query = "test query"
    mock_api_response.packages_applied = ["Mathlib", "Batteries"]
    mock_api_response.results = [
        {"name": "theorem1", "type": "theorem", "code": "theorem theorem1 : True := by trivial"}
    ]
    mock_api_response.count = 1
    mock_api_response.total_candidates_considered = 10
    mock_api_response.processing_time_ms = 100
    # Ensure model_dump doesn't exist (so it uses attribute access)
    del mock_api_response.model_dump

    # Mock asyncio.run to return the API response
    mock_asyncio_run.return_value = mock_api_response

    # Mock client
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    state = DecomposedFormalTheoremState(
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
        search_queries=["test query"],
        search_results=None,
    )

    result = _query_vectordb("http://localhost:8001/api/v1", ["Mathlib", "Batteries"], state)

    # Verify asyncio.run was called
    assert mock_asyncio_run.call_count == 1

    # Verify the search result
    assert result["outputs"][0]["search_results"] is not None
    assert len(result["outputs"][0]["search_results"]) == 1
    search_result = result["outputs"][0]["search_results"][0]
    assert search_result["query"] == "test query"
    assert search_result["packages_applied"] == ["Mathlib", "Batteries"]
    assert search_result["count"] == 1
    assert search_result["total_candidates_considered"] == 10
    assert search_result["processing_time_ms"] == 100


@patch("goedels_poetry.agents.vector_db_agent.Client")
@patch("goedels_poetry.agents.vector_db_agent.asyncio.run")
def test_query_vectordb_with_multiple_queries(mock_asyncio_run: MagicMock, mock_client_class: MagicMock) -> None:
    """Test _query_vectordb with multiple search queries (sequential processing)."""
    # Create mock API responses for each query - use regular objects (not Pydantic models)
    mock_responses = []
    for i in range(3):
        mock_response = MagicMock()
        mock_response.query = f"query{i}"
        mock_response.packages_applied = ["Mathlib"]
        mock_response.results = [{"name": f"theorem{i}", "type": "theorem"}]
        mock_response.count = 1
        mock_response.total_candidates_considered = 10
        mock_response.processing_time_ms = 100 + i
        # Ensure model_dump doesn't exist (so it uses attribute access)
        del mock_response.model_dump
        mock_responses.append(mock_response)

    # Mock asyncio.run to return different responses for each call
    mock_asyncio_run.side_effect = mock_responses

    # Mock client
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    state = DecomposedFormalTheoremState(
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
        search_queries=["query0", "query1", "query2"],
        search_results=None,
    )

    result = _query_vectordb("http://localhost:8001/api/v1", ["Mathlib"], state)

    # Verify asyncio.run was called 3 times (once per query)
    assert mock_asyncio_run.call_count == 3

    # Verify the search results
    assert result["outputs"][0]["search_results"] is not None
    assert len(result["outputs"][0]["search_results"]) == 3

    # Verify each result corresponds to the correct query
    for i, search_result in enumerate(result["outputs"][0]["search_results"]):
        assert search_result["query"] == f"query{i}"
        assert search_result["packages_applied"] == ["Mathlib"]
        assert search_result["count"] == 1
        assert search_result["processing_time_ms"] == 100 + i


@patch("goedels_poetry.agents.vector_db_agent.Client")
@patch("goedels_poetry.agents.vector_db_agent.asyncio.run")
def test_query_vectordb_propagates_exceptions(mock_asyncio_run: MagicMock, mock_client_class: MagicMock) -> None:
    """Test that _query_vectordb propagates exceptions from the API."""
    import httpx

    # Mock asyncio.run to raise an exception
    mock_asyncio_run.side_effect = httpx.HTTPStatusError(
        "Server error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    # Mock client
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    state = DecomposedFormalTheoremState(
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
        search_queries=["test query"],
        search_results=None,
    )

    # Exception should propagate
    with pytest.raises(httpx.HTTPStatusError):
        _query_vectordb("http://localhost:8001/api/v1", ["Mathlib"], state)


def test_vector_db_agent_factory() -> None:
    """Test that VectorDBAgentFactory creates an agent."""
    agent = VectorDBAgentFactory.create_agent(
        server_url="http://localhost:8001/api/v1", package_filters=["Mathlib", "Batteries"]
    )

    assert agent is not None


def test_decomposed_formal_theorem_state_initialization_with_search_results() -> None:
    """Test that DecomposedFormalTheoremState can be initialized with search_results=None."""
    state = DecomposedFormalTheoremState(
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
        search_queries=None,
        search_results=None,
    )

    assert state["search_results"] is None
    assert state["search_queries"] is None


def test_decomposed_formal_theorem_state_with_search_results() -> None:
    """Test that DecomposedFormalTheoremState can store search results."""
    search_result: APISearchResponseTypedDict = {
        "query": "test query",
        "packages_applied": ["Mathlib"],
        "results": [{"name": "theorem1", "type": "theorem"}],
        "count": 1,
        "total_candidates_considered": 10,
        "processing_time_ms": 100,
    }

    state = DecomposedFormalTheoremState(
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
        search_queries=["test query"],
        search_results=[search_result],
    )

    assert state["search_results"] is not None
    assert len(state["search_results"]) == 1
    assert state["search_results"][0]["query"] == "test query"
    assert state["search_results"][0]["count"] == 1


def test_backtracking_removes_node_from_query_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that backtracking removes the backtrack target from decomposition_query_queue."""
    from typing import cast

    from goedels_poetry.agents.state import FormalTheoremProofState
    from goedels_poetry.config.llm import DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS
    from goedels_poetry.util.tree import TreeNode

    manager = GoedelsPoetryStateManager(temp_state)

    # Create a parent node that will be the backtrack target
    parent = DecomposedFormalTheoremState(
        parent=None,
        children=[],
        depth=0,
        formal_theorem="theorem parent : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch=None,
        syntactic=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,  # Has remaining attempts
        decomposition_history=[],
        search_queries=["query1", "query2"],
        search_results=None,  # In query queue waiting for vector DB
    )

    # Create a failed child that will trigger backtracking
    failed_child = DecomposedFormalTheoremState(
        parent=cast(TreeNode, parent),
        children=[],
        depth=1,
        formal_theorem="theorem failed : True := by sorry",
        preamble=TEST_PREAMBLE,
        proof_sketch="by invalid",
        syntactic=False,
        errors="Compilation error",
        ast=None,
        self_correction_attempts=DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
        decomposition_history=[],
        search_queries=None,
        search_results=None,
    )

    parent["children"] = [cast(TreeNode, failed_child)]
    temp_state.formal_theorem_proof = cast(TreeNode, parent)

    # Add parent to query queue (simulating it waiting for vector DB queries)
    temp_state.decomposition_query_queue.append(parent)

    # Create a proof state that will fail validation and trigger backtracking
    proof_state = FormalTheoremProofState(
        parent=cast(TreeNode, failed_child),
        depth=2,
        formal_theorem="theorem proof : True := by sorry",
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
    failed_child["children"] = [cast(TreeNode, proof_state)]

    # Simulate validation that triggers backtracking
    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    validated_sketches = DecomposedFormalTheoremStates(inputs=[], outputs=[failed_child])
    manager.set_validated_sketches(validated_sketches)

    # Parent should be removed from query queue
    assert parent not in temp_state.decomposition_query_queue

    # Parent should be in backtrack queue
    assert parent in temp_state.decomposition_backtrack_queue

    # Parent's search_queries and search_results should be cleared
    assert parent["search_queries"] is None
    assert parent["search_results"] is None
