from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import load_prompt
from goedels_poetry.agents.util.debug import log_llm_prompt


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

    Parameters
    ----------
    states: FormalTheoremProofStates
        The FormalTheoremProofStates containing in the "inputs" member the FormalTheoremProofState
        instances to create the proof corrections for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("corrector_agent", state) for state in states["inputs"]]


def _corrector(state: FormalTheoremProofState) -> FormalTheoremProofStates:
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
    # Construct the prompt
    prompt = load_prompt(
        "goedel-prover-v2-subsequent",
        prev_round_num=str(state["self_correction_attempts"] - 1),
        error_message_for_prev_round=str(state["errors"]),
    )

    # Log debug prompt
    log_llm_prompt("PROOF_CORRECTOR_AGENT", prompt, "goedel-prover-v2-subsequent")

    # Add correction request to the state's proof_history
    state["proof_history"] += [HumanMessage(content=prompt)]

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
