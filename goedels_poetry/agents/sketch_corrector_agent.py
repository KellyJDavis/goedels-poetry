from typing import cast

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import load_prompt
from goedels_poetry.agents.util.debug import log_llm_prompt


class SketchCorrectorAgentFactory:
    """
    Factory class for creating instances of the SketchCorrectorAgent.
    """

    @staticmethod
    def create_agent() -> CompiledStateGraph:
        """
        Creates a SketchCorrectorAgent instance.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch corrector agent.
        """
        return _build_agent()


def _build_agent() -> CompiledStateGraph:
    """
    Builds a compiled state graph for the sketch corrector agent.

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the sketch corrector agent.
    """
    # Create the sketch corrector agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Add the nodes
    graph_builder.add_node("corrector_agent", _corrector)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["corrector_agent"])
    graph_builder.add_edge("corrector_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    corrector_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to create the sketch corrections for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("corrector_agent", {"item": state}) for state in states["inputs"]]


def _corrector(state: DecomposedFormalTheoremStates) -> DecomposedFormalTheoremStates:
    """
    Adds a HumanMessage to the decomposition_history of the passed DecomposedFormalTheoremState
    indicating a request for a correction of the previous formal sketch and indicating the errors
    in the last formal sketch. This DecomposedFormalTheoremState is then added to the outputs of
    the returned DecomposedFormalTheoremStates.

    Parameters
    ----------
    state: DecomposedFormalTheoremState
        The DecomposedFormalTheoremState containing an error string indicating the error in the
        previous attempt at sketching a proof to its formal theorem.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates containing in its outputs the modified
        DecomposedFormalTheoremState
    """
    theorem_state = cast(DecomposedFormalTheoremState, state["item"])

    # Construct the prompt
    prompt = load_prompt(
        "decomposer-subsequent",
        prev_round_num=str(theorem_state["self_correction_attempts"] - 1),
        error_message_for_prev_round=str(theorem_state["errors"]),
    )

    # Log debug prompt
    log_llm_prompt(
        "SKETCH_CORRECTOR_AGENT",
        prompt,
        "decomposer-subsequent",
        attempt_num=theorem_state["self_correction_attempts"],
    )

    # Add correction request to the state's decomposition_history
    theorem_state["decomposition_history"] += [HumanMessage(content=prompt)]

    # Reset llm_lean_output as it is now invalid for this new round
    theorem_state["llm_lean_output"] = None

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]
