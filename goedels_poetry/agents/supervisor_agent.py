from __future__ import annotations

from goedels_poetry.state import GoedelsPoetryStateManager


class SupervisorAgentFactory:
    """
    Factory class for creating instances of the SupervisorAgent.
    """

    @staticmethod
    def create_agent(state_manager: GoedelsPoetryStateManager) -> SupervisorAgent:
        """
        Creates a SupervisorAgent instance with the passed state manager.

        Parameters
        ----------
        state_manager: GoedelsPoetryStateManager
            The state manager to be used by the agent.

        Returns
        -------
        SupervisorAgent
            An instance of the SupervisorAgent.
        """
        return SupervisorAgent(state_manager=state_manager)


class SupervisorAgent:
    """
    The supervisor that determines which action to take next based upon the current state.
    This action is returned as a string from the get_action() method and this string is
    the name of of the method on GoedelsPoetryFramework that should be called to carry out
    this action.
    """

    def __init__(self, state_manager: GoedelsPoetryStateManager):
        self._state_manager = state_manager

    def get_action(self) -> str:  # noqa: C901
        """
        Gets the next action to perform based upon the contained GoedelsPoetryStateManager.
        This action is the name of of the method on GoedelsPoetryFramework that should be
        called to carry out the action.

        Returns
        --------------
        str
            The name of the action to perform.
        """
        # Return the name of the proper action
        if self._state_manager.get_informal_theorem_to_formalize():
            return "formalize_informal_theorem"
        if self._state_manager.get_informal_theorem_to_validate():
            return "check_informal_theorem_syntax"
        if self._state_manager.get_informal_theorem_to_check_semantics_of():
            return "check_informal_theorem_semantics"
        if self._state_manager.get_theorems_to_validate()["inputs"]:
            return "check_theorems_syntax"
        if self._state_manager.get_theorems_to_prove()["inputs"]:
            return "prove_theorems"
        if self._state_manager.get_proofs_to_validate()["inputs"]:
            return "check_theorems_proofs"
        if self._state_manager.get_proofs_to_correct()["inputs"]:
            return "request_proofs_corrections"
        if self._state_manager.get_proofs_to_parse()["inputs"]:
            return "parse_proofs"
        if self._state_manager.get_theorems_for_search_query_generation()["inputs"]:
            return "generate_search_queries"
        if self._state_manager.get_theorems_with_search_queries_for_vectordb()["inputs"]:
            return "query_vectordb"
        if self._state_manager.get_theorems_to_sketch()["inputs"]:
            return "sketch_proofs"
        if self._state_manager.get_sketches_to_validate()["inputs"]:
            return "check_proof_sketches_syntax"
        if self._state_manager.get_sketches_to_correct()["inputs"]:
            return "request_proof_sketches_corrections"
        if self._state_manager.get_sketches_to_backtrack()["inputs"]:
            return "request_proof_sketches_backtrack"
        if self._state_manager.get_sketches_to_parse()["inputs"]:
            return "parse_proof_sketches"
        if self._state_manager.get_sketches_to_decompose()["inputs"]:
            return "decompose_proof_sketches"

        # No action is pending, we are finished
        self._state_manager.is_finished = True
        self._state_manager.reason = "Proof completed successfully."

        # Return the final action "finish"
        return "finish"
