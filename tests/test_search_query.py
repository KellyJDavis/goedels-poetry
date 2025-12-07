"""Tests for search query generation functionality."""

import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import suppress

import pytest

from goedels_poetry.agents.search_query_agent import (
    SearchQueryAgentFactory,
    _is_backtracking,
    _parse_search_queries_response,
)
from goedels_poetry.agents.state import DecomposedFormalTheoremState, FormalTheoremProofState
from goedels_poetry.agents.util.common import LLMParsingError, combine_preamble_and_body
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

TEST_PREAMBLE = "import Mathlib\n\nset_option maxHeartbeats 0"


@pytest.fixture
def temp_state() -> Generator[GoedelsPoetryState, None, None]:
    """Create a temporary state for testing."""
    old_env = os.environ.get("GOEDELS_POETRY_DIR")
    tmpdir = tempfile.mkdtemp()
    os.environ["GOEDELS_POETRY_DIR"] = tmpdir

    theorem_name = f"test_search_query_{uuid.uuid4().hex}"
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


def test_get_theorems_for_search_query_generation_empty(temp_state: GoedelsPoetryState) -> None:
    """Test get_theorems_for_search_query_generation returns empty list when queue is empty."""
    manager = GoedelsPoetryStateManager(temp_state)

    result = manager.get_theorems_for_search_query_generation()

    assert result["inputs"] == []
    assert result["outputs"] == []


def test_get_theorems_for_search_query_generation_with_items(temp_state: GoedelsPoetryState) -> None:
    """Test get_theorems_for_search_query_generation returns items from the search queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a decomposed state and add it to the search queue
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
        search_queries=None,
    )
    temp_state.decomposition_search_queue.append(decomposed_state)

    result = manager.get_theorems_for_search_query_generation()

    assert len(result["inputs"]) == 1
    assert result["inputs"][0] == decomposed_state
    assert result["outputs"] == []


def test_set_theorems_with_search_queries_generated(temp_state: GoedelsPoetryState) -> None:
    """Test set_theorems_with_search_queries_generated moves states to query queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create states with queries
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
            self_correction_attempts=0,
            decomposition_history=[],
            search_queries=[f"query{i}", f"query{i}_alt"],
            search_results=None,
        )
        states.append(state)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    states_with_queries = DecomposedFormalTheoremStates(inputs=[], outputs=states)

    # Add one state to search queue
    temp_state.decomposition_search_queue.append(states[0])

    manager.set_theorems_with_search_queries_generated(states_with_queries)

    # Search queue should be cleared
    assert len(temp_state.decomposition_search_queue) == 0

    # All states should be in query queue (not sketch queue)
    assert len(temp_state.decomposition_query_queue) == 3
    assert all(state in temp_state.decomposition_query_queue for state in states)
    assert len(temp_state.decomposition_sketch_queue) == 0


def test_queue_proofs_for_decomposition_adds_to_search_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that _queue_proofs_for_decomposition adds states to search queue with search_queries=None."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a proof state that's too difficult
    proof_state = FormalTheoremProofState(
        parent=None,
        depth=0,
        formal_theorem="theorem test : True := by sorry",
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

    from goedels_poetry.config.llm import PROVER_AGENT_MAX_PASS

    # Set pass_attempts to max to trigger decomposition
    proof_state["pass_attempts"] = PROVER_AGENT_MAX_PASS

    manager._queue_proofs_for_decomposition([proof_state])

    # Should have one state in search queue
    assert len(temp_state.decomposition_search_queue) == 1
    decomposed_state = temp_state.decomposition_search_queue[0]
    assert decomposed_state["search_queries"] is None
    assert decomposed_state["formal_theorem"] == "theorem test : True := by sorry"


def test_prepare_node_for_resketching_clears_search_queries(temp_state: GoedelsPoetryState) -> None:
    """Test that _prepare_node_for_resketching clears search_queries."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a node with search queries
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
        search_queries=["query1", "query2"],
    )

    manager._prepare_node_for_resketching(node)

    assert node["search_queries"] is None
    assert node["proof_sketch"] is None
    assert node["syntactic"] is False
    assert node["errors"] is None
    assert node["ast"] is None
    assert node["children"] == []


def test_set_backtracked_sketches_routes_to_search_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that set_backtracked_sketches routes to search queue instead of validate queue."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create backtracked states (search_queries should be None after _prepare_node_for_resketching)
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
            self_correction_attempts=i,
            decomposition_history=[],
            search_queries=None,  # Cleared by _prepare_node_for_resketching
        )
        states.append(state)

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    backtracked_states = DecomposedFormalTheoremStates(inputs=[], outputs=states)

    # Add states to backtrack queue
    temp_state.decomposition_backtrack_queue.extend(states)

    manager.set_backtracked_sketches(backtracked_states)

    # Backtrack queue should be cleared
    assert len(temp_state.decomposition_backtrack_queue) == 0

    # States should be in search queue (not validate queue)
    assert len(temp_state.decomposition_search_queue) == 2
    assert all(state in temp_state.decomposition_search_queue for state in states)
    assert len(temp_state.decomposition_validate_queue) == 0


def test_remove_decomposition_node_from_queues_includes_search_queue(temp_state: GoedelsPoetryState) -> None:
    """Test that _remove_decomposition_node_from_queues removes nodes from search queue."""
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
        search_queries=None,
    )

    # Add to search queue
    temp_state.decomposition_search_queue.append(node)

    manager._remove_decomposition_node_from_queues(node)

    # Node should be removed from search queue
    assert node not in temp_state.decomposition_search_queue


def test_parse_search_queries_response_with_tags() -> None:
    """Test parsing search queries from <search> tag format."""
    response = """<search>divisibility properties</search>
<search>greatest common divisor</search>
<search>Euclidean algorithm</search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 3
    assert queries[0] == "divisibility properties"
    assert queries[1] == "greatest common divisor"
    assert queries[2] == "Euclidean algorithm"


def test_parse_search_queries_response_with_whitespace() -> None:
    """Test parsing search queries with whitespace in tags."""
    response = """<search> prime numbers </search>
<search>factorization</search>
<search>  number theory  </search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 3
    assert queries[0] == "prime numbers"
    assert queries[1] == "factorization"
    assert queries[2] == "number theory"


def test_parse_search_queries_response_multiline_queries() -> None:
    """Test parsing search queries that span multiple lines."""
    response = """<search>induction
proof technique</search>
<search>contradiction</search>
<search>case analysis</search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 3
    assert "induction\nproof technique" in queries
    assert "contradiction" in queries
    assert "case analysis" in queries


def test_parse_search_queries_response_with_extra_text() -> None:
    """Test parsing search queries when response contains extra text."""
    response = """Here are the search queries:
<search>first query</search>
<search>second query</search>
Some additional text here.
<search>third query</search>
More text after."""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 3
    assert queries[0] == "first query"
    assert queries[1] == "second query"
    assert queries[2] == "third query"


def test_parse_search_queries_response_filters_empty() -> None:
    """Test that empty queries are filtered out."""
    response = """<search>valid query</search>
<search></search>
<search>another valid query</search>
<search>   </search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 2
    assert "valid query" in queries
    assert "another valid query" in queries


def test_parse_search_queries_response_deduplicates() -> None:
    """Test that duplicate queries are removed."""
    response = """<search>duplicate query</search>
<search>duplicate query</search>
<search>unique query</search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 2
    assert queries.count("duplicate query") == 1
    assert "unique query" in queries


def test_parse_search_queries_response_no_queries_raises_error() -> None:
    """Test that parsing raises error when no queries found."""
    response = """Here are some instructions:
Generate queries below.
But no actual queries."""
    with pytest.raises(LLMParsingError):
        _parse_search_queries_response(response)


def test_parse_search_queries_response_case_insensitive_tags() -> None:
    """Test that parsing works with case-insensitive tags."""
    response = """<SEARCH>first query</SEARCH>
<Search>second query</Search>
<search>third query</search>"""
    queries = _parse_search_queries_response(response)
    assert len(queries) == 3
    assert "first query" in queries
    assert "second query" in queries
    assert "third query" in queries


def test_is_backtracking_detects_backtrack() -> None:
    """Test that _is_backtracking detects backtrack messages."""
    from langchain_core.messages import HumanMessage

    from goedels_poetry.agents.util.common import load_prompt

    # Load the actual backtrack prompt template
    backtrack_prompt = load_prompt("decomposer-backtrack", prev_round_num="1")

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
        self_correction_attempts=1,
        decomposition_history=[
            HumanMessage(content=backtrack_prompt),
        ],
        search_queries=None,
    )

    assert _is_backtracking(state) is True


def test_is_backtracking_no_backtrack() -> None:
    """Test that _is_backtracking returns False when no backtrack detected."""
    from langchain_core.messages import HumanMessage

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
        decomposition_history=[
            HumanMessage(content="Please sketch a proof for this theorem."),
        ],
        search_queries=None,
    )

    assert _is_backtracking(state) is False


def test_search_query_agent_factory() -> None:
    """Test that SearchQueryAgentFactory creates an agent."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="test-model",
    )
    agent = SearchQueryAgentFactory.create_agent(llm)

    assert agent is not None


def test_search_query_generation_integration(temp_state: GoedelsPoetryState) -> None:
    """Test integration of search query generation."""
    manager = GoedelsPoetryStateManager(temp_state)

    # Create a state needing queries
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
    )

    temp_state.decomposition_search_queue.append(state)

    # Get states to search
    states_to_search = manager.get_theorems_for_search_query_generation()
    assert len(states_to_search["inputs"]) == 1

    # Generate queries (would normally use agent, but we'll test the flow)
    # In real usage, the agent would be invoked here
    state["search_queries"] = ["divisibility", "gcd", "Euclidean algorithm"]

    from goedels_poetry.agents.state import DecomposedFormalTheoremStates

    states_with_queries = DecomposedFormalTheoremStates(inputs=[], outputs=[state])
    manager.set_theorems_with_search_queries_generated(states_with_queries)

    # State should now be in query queue with queries (not sketch queue)
    assert len(temp_state.decomposition_query_queue) == 1
    assert temp_state.decomposition_query_queue[0]["search_queries"] == [
        "divisibility",
        "gcd",
        "Euclidean algorithm",
    ]
    assert len(temp_state.decomposition_sketch_queue) == 0
