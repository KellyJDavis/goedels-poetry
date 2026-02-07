from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofStates
from goedels_poetry.agents.util.common import load_prompt
from goedels_poetry.agents.util.debug import log_llm_prompt
from goedels_poetry.agents.util.state_isolation import detach_formal_proof_state


class ProofCorrectorAgentFactory:
    """
    Factory class for creating instances of the ProofCorrectorAgent.
    """

    @staticmethod
    def create_agent() -> CompiledStateGraph:
        """
        Creates a ProofCorrectorAgent instance.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the proof corrector agent.
        """
        return _build_agent()


def _build_agent() -> CompiledStateGraph:
    """
    Builds a compiled state graph for the proof corrector agent.

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the proof corrector agent.
    """
    # Create the proof corrector agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Add the nodes
    graph_builder.add_node("corrector_agent", _corrector)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["corrector_agent"])
    graph_builder.add_edge("corrector_agent", END)

    return graph_builder.compile()


def _map_edge(states: FormalTheoremProofStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    corrector_agent nodes.

    IMPORTANT: We send detached, acyclic per-item payloads to prevent shared mutable references
    from cross-talking across parallel tasks.
    """
    return [
        Send(
            "corrector_agent",
            {"inputs": [], "outputs": [], "item": detach_formal_proof_state(input_state)},
        )
        for input_state in states["inputs"]
    ]


def _corrector(state: FormalTheoremProofStates) -> FormalTheoremProofStates:
    """
    Adds a HumanMessage to the proof_history of the passed FormalTheoremProofState indicating
    a request for a correction of the previous formal proof and indicating the errors in the
    last formal proof. This FormalTheoremProofState is then added to the outputs of the returned
    FormalTheoremProofStates.

    Parameters
    ----------
    state: FormalTheoremProofState
        The FormalTheoremProofState containing an error string indicating the error in the previous
        attempt at proving its formal theorem.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates containing in its outputs the modified FormalTheoremProofState
    """
    proof_state = detach_formal_proof_state(state["item"])

    # Construct the prompt
    prompt = load_prompt(
        "goedel-prover-v2-subsequent",
        prev_round_num=str(proof_state["self_correction_attempts"] - 1),
        error_message_for_prev_round=str(proof_state["errors"]),
    )

    # Log debug prompt
    log_llm_prompt(
        "PROOF_CORRECTOR_AGENT",
        prompt,
        "goedel-prover-v2-subsequent",
        attempt_num=proof_state["self_correction_attempts"],
        pass_num=proof_state["pass_attempts"],
    )

    # Add correction request to the state's proof_history
    proof_state["proof_history"] += [HumanMessage(content=prompt)]

    # Reset llm_lean_output as it is now invalid for this new round
    proof_state["llm_lean_output"] = None

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [proof_state]}  # type: ignore[typeddict-item]
