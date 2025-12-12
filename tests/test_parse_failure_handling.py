"""Tests for parse failure handling in agents and state manager.

These tests verify that LLMParsingError exceptions are properly caught and handled,
with appropriate requeueing and attempt tracking in the state manager.
"""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import MagicMock

from goedels_poetry.agents.formalizer_agent import FormalizerAgentFactory
from goedels_poetry.agents.informal_theorem_semantics_agent import InformalTheoremSemanticsAgentFactory
from goedels_poetry.agents.proof_sketcher_agent import ProofSketcherAgentFactory
from goedels_poetry.agents.prover_agent import ProverAgentFactory
from goedels_poetry.agents.state import (
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
    FormalTheoremProofState,
    FormalTheoremProofStates,
    InformalTheoremState,
)
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
from goedels_poetry.config.llm import (
    DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
    FORMALIZER_AGENT_MAX_RETRIES,
    PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
)
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


class TestFormalizerParseFailure:
    """Tests for formalizer agent parse failure handling."""

    def test_formalizer_returns_none_on_parse_failure(self) -> None:
        """Test that formalizer returns formal_theorem=None on LLMParsingError."""
        # Create a mock LLM that returns a response without code blocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is just text, no code block"

        # Create agent
        agent = FormalizerAgentFactory.create_agent(llm=mock_llm)

        # Create input state
        input_state: InformalTheoremState = {
            "informal_theorem": "For all x, x = x",
            "formalization_attempts": 0,
            "formal_theorem": None,
            "syntactic": False,
            "semantic": False,
        }

        # Run agent
        result = agent.invoke(input_state)

        # Verify parse failure marker
        assert result["formal_theorem"] is None

    def test_state_manager_requeues_on_formalizer_parse_failure(self) -> None:
        """Test that state manager requeues formalizer parse failures."""
        informal_theorem = "For all x, x = x"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(informal_theorem)

        try:
            # Create state with informal theorem
            state = GoedelsPoetryState(informal_theorem=informal_theorem)
            manager = GoedelsPoetryStateManager(state)

            # Get the initial state
            initial_state = manager.get_informal_theorem_to_formalize()
            assert initial_state is not None

            # Simulate parse failure (formal_theorem=None)
            parse_failure_state: InformalTheoremState = {
                "informal_theorem": initial_state["informal_theorem"],
                "formalization_attempts": initial_state["formalization_attempts"],
                "formal_theorem": None,  # Parse failure marker
                "syntactic": False,
                "semantic": False,
            }

            # Set the parse failure state
            manager.set_formalized_informal_theorem(parse_failure_state)

            # Verify it was requeued
            requeued_state = manager.get_informal_theorem_to_formalize()
            assert requeued_state is not None
            assert requeued_state["formal_theorem"] is None
            assert requeued_state["formalization_attempts"] == 1  # Should be incremented
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(informal_theorem)

    def test_state_manager_finishes_on_max_formalizer_attempts(self) -> None:
        """Test that state manager finishes when max formalizer attempts exceeded."""
        informal_theorem = "For all x, x = x"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(informal_theorem)

        try:
            # Create state with informal theorem
            state = GoedelsPoetryState(informal_theorem=informal_theorem)
            manager = GoedelsPoetryStateManager(state)

            # Create state at max attempts
            max_attempts_state: InformalTheoremState = {
                "informal_theorem": "For all x, x = x",
                "formalization_attempts": FORMALIZER_AGENT_MAX_RETRIES - 1,  # One less than max
                "formal_theorem": None,  # Parse failure marker
                "syntactic": False,
                "semantic": False,
            }

            # Set the parse failure state (will increment to max)
            manager.set_formalized_informal_theorem(max_attempts_state)

            # Verify finished
            assert manager.is_finished is True
            assert manager.reason is not None
            assert "maximum formalization attempts exceeded" in manager.reason.lower()

            # Verify not requeued
            assert manager.get_informal_theorem_to_formalize() is None
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(informal_theorem)


class TestProverParseFailure:
    """Tests for prover agent parse failure handling."""

    def test_prover_sets_markers_on_parse_failure(self) -> None:
        """Test that prover sets error markers on LLMParsingError."""
        # Create a mock LLM that returns a response without code blocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is just text, no code block"

        # Create agent
        agent = ProverAgentFactory.create_agent(llm=mock_llm)

        # Create input state
        input_state: FormalTheoremProofState = {
            "parent": None,
            "depth": 0,
            "formal_theorem": "theorem test : True := by sorry",
            "preamble": DEFAULT_IMPORTS,
            "syntactic": True,
            "formal_proof": None,
            "proved": False,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 0,
            "proof_history": [],
            "pass_attempts": 0,
        }

        input_states: FormalTheoremProofStates = {"inputs": [input_state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify parse failure markers
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["formal_proof"] is None
        assert output_state["errors"] is not None
        assert "Malformed LLM response" in output_state["errors"]
        assert "unable to parse proof body" in output_state["errors"]

    def test_state_manager_requeues_on_prover_parse_failure(self) -> None:
        """Test that state manager requeues prover parse failures."""
        # Create state with formal theorem
        theorem = f"{DEFAULT_IMPORTS}theorem test : True := by sorry"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)

        try:
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Get the initial state from syntax queue and move it to prove queue
            syntax_states = manager.get_theorems_to_validate()
            assert len(syntax_states["inputs"]) == 1
            syntax_state = syntax_states["inputs"][0]
            # Mark as syntactically valid and move to prove queue
            validated_state: FormalTheoremProofState = {
                **syntax_state,
                "syntactic": True,
            }
            manager.set_validated_theorems(FormalTheoremProofStates(outputs=[validated_state]))

            # Now get from prove queue
            initial_states = manager.get_theorems_to_prove()
            assert len(initial_states["inputs"]) == 1
            initial_state = initial_states["inputs"][0]

            # Simulate parse failure
            parse_failure_message = (
                "Malformed LLM response: unable to parse proof body from LLM output. "
                "The response did not contain a valid Lean4 code block or the code block could not be extracted."
            )
            parse_failure_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": None,  # Parse failure marker
                "proved": False,
                "errors": parse_failure_message,  # Parse failure error
                "ast": None,
                "self_correction_attempts": initial_state["self_correction_attempts"],
                "proof_history": initial_state["proof_history"],
                "pass_attempts": initial_state["pass_attempts"],
            }

            parse_failure_states: FormalTheoremProofStates = {"outputs": [parse_failure_state]}

            # Set the parse failure state
            manager.set_proven_theorems(parse_failure_states)

            # Verify it was requeued
            requeued_states = manager.get_theorems_to_prove()
            assert len(requeued_states["inputs"]) == 1
            requeued_state = requeued_states["inputs"][0]
            assert requeued_state["formal_proof"] is None
            assert requeued_state["self_correction_attempts"] == 1  # Should be incremented
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_state_manager_handles_max_prover_attempts(self) -> None:
        """Test that state manager handles max prover attempts correctly."""
        # Create state with formal theorem
        theorem = f"{DEFAULT_IMPORTS}theorem test : True := by sorry"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)

        try:
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Get the initial state from syntax queue and move it to prove queue
            syntax_states = manager.get_theorems_to_validate()
            assert len(syntax_states["inputs"]) == 1
            syntax_state = syntax_states["inputs"][0]
            # Mark as syntactically valid and move to prove queue
            validated_state: FormalTheoremProofState = {
                **syntax_state,
                "syntactic": True,
            }
            manager.set_validated_theorems(FormalTheoremProofStates(outputs=[validated_state]))

            # Now get from prove queue
            initial_states = manager.get_theorems_to_prove()
            assert len(initial_states["inputs"]) == 1
            initial_state = initial_states["inputs"][0]

            # Create state at max attempts
            parse_failure_message = (
                "Malformed LLM response: unable to parse proof body from LLM output. "
                "The response did not contain a valid Lean4 code block or the code block could not be extracted."
            )
            max_attempts_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": None,
                "proved": False,
                "errors": parse_failure_message,
                "ast": None,
                "self_correction_attempts": PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,  # One less than max
                "proof_history": [],
                "pass_attempts": 0,
            }

            max_attempts_states: FormalTheoremProofStates = {"outputs": [max_attempts_state]}

            # Set the parse failure state (will increment to max)
            manager.set_proven_theorems(max_attempts_states)

            # Verify it was handled (either requeued for pass restart or queued for decomposition)
            # The state should have been modified in place - attempts incremented
            # After incrementing, it should be >= max, so it will either:
            # 1. Be requeued for pass restart (if pass_attempts < max_pass) - attempts reset to 0
            # 2. Be queued for decomposition (if pass_attempts >= max_pass)
            requeued_states = manager.get_theorems_to_prove()
            decomposition_queue = state.decomposition_search_queue

            # Verify the state was processed
            # Note: If it was requeued for pass restart, attempts will be reset to 0
            # If it was queued for decomposition, attempts will still be >= max
            # Either way, the state was processed correctly

            # Verify it's either in prove queue (requeued for pass restart) or decomposition queue
            assert len(requeued_states["inputs"]) > 0 or len(decomposition_queue) > 0, (
                "State should have been requeued for pass restart or queued for decomposition"
            )
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)


class TestProofSketcherParseFailure:
    """Tests for proof sketcher agent parse failure handling."""

    def test_proof_sketcher_sets_markers_on_parse_failure(self) -> None:
        """Test that proof sketcher sets error markers on LLMParsingError."""
        # Create a mock LLM that returns a response without code blocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is just text, no code block"

        # Create agent
        agent = ProofSketcherAgentFactory.create_agent(llm=mock_llm)

        # Create input state
        input_state: DecomposedFormalTheoremState = {
            "parent": None,
            "children": [],
            "depth": 0,
            "formal_theorem": "theorem test : True := by sorry",
            "preamble": DEFAULT_IMPORTS,
            "proof_sketch": None,
            "syntactic": False,
            "errors": None,
            "ast": None,
            "self_correction_attempts": 0,
            "decomposition_history": [],
            "search_queries": None,
            "search_results": None,
        }

        input_states: DecomposedFormalTheoremStates = {"inputs": [input_state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify parse failure markers
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["proof_sketch"] is None
        assert output_state["errors"] is not None
        assert "Malformed LLM response" in output_state["errors"]
        assert "unable to parse proof sketch" in output_state["errors"]

    def test_state_manager_requeues_on_sketcher_parse_failure(self) -> None:
        """Test that state manager requeues sketcher parse failures."""
        # Create state with formal theorem (will be decomposed)
        theorem = f"{DEFAULT_IMPORTS}theorem test : True := by sorry"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)

        try:
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Manually create a decomposed state in the sketch queue
            decomposed_state: DecomposedFormalTheoremState = {
                "parent": None,
                "children": [],
                "depth": 0,
                "formal_theorem": "theorem test : True := by sorry",
                "preamble": DEFAULT_IMPORTS,
                "proof_sketch": None,
                "syntactic": False,
                "errors": None,
                "ast": None,
                "self_correction_attempts": 0,
                "decomposition_history": [],
                "search_queries": None,
                "search_results": None,
            }
            state.decomposition_sketch_queue.append(decomposed_state)

            # Get the initial state
            initial_states = manager.get_theorems_to_sketch()
            assert len(initial_states["inputs"]) == 1
            initial_state = initial_states["inputs"][0]

            # Simulate parse failure
            parse_failure_message = (
                "Malformed LLM response: unable to parse proof sketch from LLM output. "
                "The response did not contain a valid Lean4 code block or the code block could not be extracted."
            )
            parse_failure_state: DecomposedFormalTheoremState = {
                "parent": initial_state["parent"],
                "children": initial_state["children"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "proof_sketch": None,  # Parse failure marker
                "syntactic": False,
                "errors": parse_failure_message,  # Parse failure error
                "ast": None,
                "self_correction_attempts": initial_state["self_correction_attempts"],
                "decomposition_history": initial_state["decomposition_history"],
                "search_queries": initial_state["search_queries"],
                "search_results": initial_state["search_results"],
            }

            parse_failure_states: DecomposedFormalTheoremStates = {"outputs": [parse_failure_state]}

            # Set the parse failure state
            manager.set_sketched_theorems(parse_failure_states)

            # Verify it was requeued
            requeued_states = manager.get_theorems_to_sketch()
            assert len(requeued_states["inputs"]) == 1
            requeued_state = requeued_states["inputs"][0]
            assert requeued_state["proof_sketch"] is None
            assert requeued_state["self_correction_attempts"] == 1  # Should be incremented
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_state_manager_handles_max_sketcher_attempts(self) -> None:
        """Test that state manager handles max sketcher attempts correctly."""
        # Create state with formal theorem
        theorem = f"{DEFAULT_IMPORTS}theorem test : True := by sorry"
        # Clear directory if it exists
        with suppress(Exception):
            GoedelsPoetryState.clear_theorem_directory(theorem)

        try:
            state = GoedelsPoetryState(formal_theorem=theorem)
            manager = GoedelsPoetryStateManager(state)

            # Manually create a decomposed state in the sketch queue
            decomposed_state: DecomposedFormalTheoremState = {
                "parent": None,
                "children": [],
                "depth": 0,
                "formal_theorem": "theorem test : True := by sorry",
                "preamble": DEFAULT_IMPORTS,
                "proof_sketch": None,
                "syntactic": False,
                "errors": None,
                "ast": None,
                "self_correction_attempts": DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,  # One less than max
                "decomposition_history": [],
                "search_queries": None,
                "search_results": None,
            }
            state.decomposition_sketch_queue.append(decomposed_state)

            # Get the initial state
            initial_states = manager.get_theorems_to_sketch()
            initial_state = initial_states["inputs"][0]

            # Create state at max attempts
            parse_failure_message = (
                "Malformed LLM response: unable to parse proof sketch from LLM output. "
                "The response did not contain a valid Lean4 code block or the code block could not be extracted."
            )
            max_attempts_state: DecomposedFormalTheoremState = {
                "parent": initial_state["parent"],
                "children": initial_state["children"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "proof_sketch": None,
                "syntactic": False,
                "errors": parse_failure_message,
                "ast": None,
                "self_correction_attempts": DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
                "decomposition_history": [],
                "search_queries": None,
                "search_results": None,
            }

            max_attempts_states: DecomposedFormalTheoremStates = {"outputs": [max_attempts_state]}

            # Set the parse failure state (will increment to max and trigger backtrack/finish)
            manager.set_sketched_theorems(max_attempts_states)

            # Verify it was handled (either finished or backtracked)
            # The exact behavior depends on whether backtrackable ancestor exists
            # But it should not be in sketch_queue anymore
            # The attempts should have been incremented
            assert max_attempts_state["self_correction_attempts"] >= DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)


class TestSemanticsParseFailure:
    """Tests for semantics agent parse failure handling."""

    def test_semantics_returns_false_on_parse_failure(self) -> None:
        """Test that semantics agent returns semantic=False on LLMParsingError."""
        # Create a mock LLM that returns a response without judgement
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is just text, no judgement"

        # Create agent
        agent = InformalTheoremSemanticsAgentFactory.create_agent(llm=mock_llm)

        # Create input state
        input_state: InformalTheoremState = {
            "informal_theorem": "For all x, x = x",
            "formalization_attempts": 0,
            "formal_theorem": "theorem test : ∀ x, x = x := by sorry",
            "syntactic": True,
            "semantic": False,
        }

        # Run agent
        result = agent.invoke(input_state)

        # Verify parse failure returns False
        assert result["semantic"] is False
