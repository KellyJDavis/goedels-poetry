"""Tests for proof reset routing behavior.

These tests verify that proofs that reach max_self_correction_attempts
are correctly routed to the prover queue (for fresh attempts) rather than
the corrector queue.
"""

from __future__ import annotations

from contextlib import suppress

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS
from goedels_poetry.config.llm import (
    PROVER_AGENT_MAX_PASS,
    PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
)
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager


class TestProofResetRouting:
    """Tests for proof reset routing after max_self_correction_attempts."""

    def test_reset_proof_goes_to_prover_queue_not_corrector(self) -> None:
        """Test that reset proofs go to prover queue, not corrector queue."""
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

            # Create state at max attempts (will be max after increment in set_validated_proofs)
            reset_proof_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": None,
                "proved": False,
                "errors": "Compilation error",
                "ast": None,
                "self_correction_attempts": PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS
                - 1,  # Will be max after increment
                "proof_history": [],  # Will be reset to empty
                "pass_attempts": 0,  # Less than max_pass
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            reset_proof_states: FormalTheoremProofStates = {"outputs": [reset_proof_state]}

            # Clear the prove queue to avoid interference from the initial state
            state.proof_prove_queue.clear()

            # Set the validated proof state (will increment attempts and reset if needed)
            manager.set_validated_proofs(reset_proof_states)

            # Verify reset proof is in prover queue, not corrector queue
            prover_queue_states = manager.get_theorems_to_prove()
            corrector_queue_states = manager.get_proofs_to_correct()

            # Find the reset proof in the prover queue (there may be other items from set_proven_theorems)
            reset_states = [
                s
                for s in prover_queue_states["inputs"]
                if s["self_correction_attempts"] == 0 and s["errors"] is None and s["pass_attempts"] == 1
            ]
            assert len(reset_states) == 1, "Reset proof should be in prover queue"
            assert len(corrector_queue_states["inputs"]) == 0, "Reset proof should NOT be in corrector queue"

            # Verify the state was reset
            reset_state = reset_states[0]
            assert reset_state["self_correction_attempts"] == 0, "Attempts should be reset to 0"
            assert reset_state["errors"] is None, "Errors should be reset to None"
            assert reset_state["proof_history"] == [], "Proof history should be reset to empty list"
            assert reset_state["pass_attempts"] == 1, "Pass attempts should be incremented to 1"
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_proof_with_errors_still_goes_to_corrector(self) -> None:
        """Test that proofs needing correction still go to corrector queue."""
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

            # Clear the prove queue to avoid interference from the initial state
            state.proof_prove_queue.clear()

            # Create state with errors but below max attempts
            correction_proof_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": "some proof",
                "proved": False,
                "errors": "Compilation error",
                "ast": None,
                "self_correction_attempts": 0,  # Less than max (will be 1 after increment)
                "proof_history": [],  # Should not be reset
                "pass_attempts": 0,
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            correction_proof_states: FormalTheoremProofStates = {"outputs": [correction_proof_state]}

            # Set the validated proof state
            manager.set_validated_proofs(correction_proof_states)

            # Verify proof is in corrector queue, not prover queue
            prover_queue_states = manager.get_theorems_to_prove()
            corrector_queue_states = manager.get_proofs_to_correct()

            assert len(corrector_queue_states["inputs"]) == 1, "Proof with errors should be in corrector queue"
            # Note: prover queue may have items from set_proven_theorems, so we check it's not the correction proof
            correction_in_prover = any(
                s.get("errors") == "Compilation error" and s.get("self_correction_attempts") == 1
                for s in prover_queue_states["inputs"]
            )
            assert not correction_in_prover, "Proof with errors should NOT be in prover queue"

            # Verify the state was NOT reset
            correction_state = corrector_queue_states["inputs"][0]
            assert correction_state["self_correction_attempts"] == 1, "Attempts should be incremented to 1 (was 0)"
            assert correction_state["errors"] == "Compilation error", "Errors should still be set"
            # Note: proof_history is not modified by set_validated_proofs, so it remains as initialized
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_reset_proof_with_max_pass_goes_to_decomposition(self) -> None:
        """Test that proofs at max_pass go to decomposition."""
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

            # Clear the prove queue to avoid interference from the initial state
            state.proof_prove_queue.clear()

            # Create state at max attempts and max_pass
            max_pass_proof_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": None,
                "proved": False,
                "errors": "Compilation error",
                "ast": None,
                "self_correction_attempts": PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS
                - 1,  # Will be max after increment
                "proof_history": [],
                "pass_attempts": PROVER_AGENT_MAX_PASS,  # At max
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            max_pass_proof_states: FormalTheoremProofStates = {"outputs": [max_pass_proof_state]}

            # Set the validated proof state
            manager.set_validated_proofs(max_pass_proof_states)

            # Verify proof is queued for decomposition, not in prover or corrector queues
            prover_queue_states = manager.get_theorems_to_prove()
            corrector_queue_states = manager.get_proofs_to_correct()
            decomposition_queue = state.decomposition_search_queue

            assert len(decomposition_queue) == 1, "Proof at max_pass should be queued for decomposition"
            assert len(prover_queue_states["inputs"]) == 0, "Proof at max_pass should NOT be in prover queue"
            assert len(corrector_queue_states["inputs"]) == 0, "Proof at max_pass should NOT be in corrector queue"
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_successful_proof_goes_to_ast_queue(self) -> None:
        """Test that successful proofs go to AST queue."""
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

            # Clear the prove queue to avoid interference from the initial state
            state.proof_prove_queue.clear()

            # Create successful proof state
            successful_proof_state: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": "by trivial",
                "proved": True,  # Success!
                "errors": None,
                "ast": None,
                "self_correction_attempts": 1,
                "proof_history": [],
                "pass_attempts": 0,
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            successful_proof_states: FormalTheoremProofStates = {"outputs": [successful_proof_state]}

            # Set the validated proof state
            manager.set_validated_proofs(successful_proof_states)

            # Verify proof is in AST queue, not in prover or corrector queues
            prover_queue_states = manager.get_theorems_to_prove()
            corrector_queue_states = manager.get_proofs_to_correct()
            ast_queue = state.proof_ast_queue

            assert len(ast_queue) == 1, "Successful proof should be in AST queue"
            assert len(prover_queue_states["inputs"]) == 0, "Successful proof should NOT be in prover queue"
            assert len(corrector_queue_states["inputs"]) == 0, "Successful proof should NOT be in corrector queue"
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)

    def test_multiple_proofs_routed_correctly(self) -> None:
        """Test that mixed proofs are routed correctly."""
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

            # Clear the prove queue to avoid interference from the initial state
            state.proof_prove_queue.clear()

            # Create three different proof states
            # 1. Reset proof (max attempts, pass_attempts < max_pass)
            reset_proof: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": None,
                "proved": False,
                "errors": "Error 1",
                "ast": None,
                "self_correction_attempts": PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS - 1,
                "proof_history": [],
                "pass_attempts": 0,
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            # 2. Correction proof (below max attempts, has errors)
            # Use 0 so after increment it becomes 1, which is still < max (assuming max >= 2)
            correction_proof: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": "some proof",
                "proved": False,
                "errors": "Error 2",
                "ast": None,
                "self_correction_attempts": 0,  # Will be 1 after increment, still < max
                "proof_history": [],
                "pass_attempts": 0,
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            # 3. Successful proof
            successful_proof: FormalTheoremProofState = {
                "parent": initial_state["parent"],
                "depth": initial_state["depth"],
                "formal_theorem": initial_state["formal_theorem"],
                "preamble": initial_state["preamble"],
                "syntactic": initial_state["syntactic"],
                "formal_proof": "by trivial",
                "proved": True,
                "errors": None,
                "ast": None,
                "self_correction_attempts": 1,
                "proof_history": [],
                "pass_attempts": 0,
                "hole_name": None,
                "hole_start": None,
                "hole_end": None,
            }

            mixed_proof_states: FormalTheoremProofStates = {
                "outputs": [reset_proof, correction_proof, successful_proof]
            }

            # Set the validated proof states
            manager.set_validated_proofs(mixed_proof_states)

            # Verify routing
            prover_queue_states = manager.get_theorems_to_prove()
            corrector_queue_states = manager.get_proofs_to_correct()
            ast_queue = state.proof_ast_queue

            # Reset proof should be in prover queue only
            reset_states = [
                s
                for s in prover_queue_states["inputs"]
                if s["self_correction_attempts"] == 0 and s["errors"] is None and s["pass_attempts"] == 1
            ]
            assert len(reset_states) == 1, "Reset proof should be in prover queue"
            assert reset_states[0]["self_correction_attempts"] == 0, "Reset proof should have attempts=0"
            assert reset_states[0]["errors"] is None, "Reset proof should have errors=None"

            # Correction proof should be in corrector queue only
            assert len(corrector_queue_states["inputs"]) == 1, "Correction proof should be in corrector queue"
            assert corrector_queue_states["inputs"][0]["self_correction_attempts"] == 1, (
                "Correction proof should have attempts=1 (was 0, incremented to 1)"
            )
            assert corrector_queue_states["inputs"][0]["errors"] == "Error 2", (
                "Correction proof should still have errors"
            )

            # Successful proof should be in AST queue only
            assert len(ast_queue) == 1, "Successful proof should be in AST queue"
            assert ast_queue[0]["proved"] is True, "Successful proof should be marked as proved"
        finally:
            with suppress(Exception):
                GoedelsPoetryState.clear_theorem_directory(theorem)
