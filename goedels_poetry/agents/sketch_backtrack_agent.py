from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import _format_theorem_hints_section, load_prompt
from goedels_poetry.agents.util.debug import log_llm_prompt


class SketchBacktrackAgentFactory:
    """
    Factory class for creating instances of the SketchBacktrackAgent.
    """

    @staticmethod
    def create_agent() -> CompiledStateGraph:
        """
        Creates a SketchBacktrackAgent instance.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch backtrack agent.
        """
        return _build_agent()


def _build_agent() -> CompiledStateGraph:
    """
    Builds a compiled state graph for the sketch backtrack agent.

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the sketch backtrack agent.
    """
    # Create the sketch backtrack agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Add the nodes
    graph_builder.add_node("backtrack_agent", _backtrack)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["backtrack_agent"])
    graph_builder.add_edge("backtrack_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    backtrack_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to create the backtrack sketches for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("backtrack_agent", state) for state in states["inputs"]]


def _backtrack(state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremStates:
    """
    Adds a HumanMessage to the decomposition_history of the passed DecomposedFormalTheoremState
    requesting a completely different decomposition strategy because the previous decomposition's
    children could not be proven. This DecomposedFormalTheoremState is then added to the outputs
    of the returned DecomposedFormalTheoremStates.

    Parameters
    ----------
    state: DecomposedFormalTheoremState
        The DecomposedFormalTheoremState whose previous decomposition failed to be proven.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates containing in its outputs the modified
        DecomposedFormalTheoremState
    """
    # Format theorem hints section from search results
    theorem_hints_section = _format_theorem_hints_section(state["search_results"])
    # Construct the prompt for backtracking
    prompt = load_prompt(
        "decomposer-backtrack",
        prev_round_num=str(state["self_correction_attempts"]),
        theorem_hints_section=theorem_hints_section,
    )

    # Log debug prompt
    log_llm_prompt("SKETCH_BACKTRACK_AGENT", prompt, "decomposer-backtrack")

    # Add backtrack request to the state's decomposition_history
    state["decomposition_history"] += [HumanMessage(content=prompt)]

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
