from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import combine_preamble_and_body, get_error_str
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response


class SketchCheckerAgentFactory:
    """
    Factory class for creating instances of the SketchCheckerAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
        """
        Creates a SketchCheckerAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch checker agent.
        """
        return _build_agent(server_url=server_url, server_max_retries=server_max_retries)


def _build_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the specified Kimina server.

    Parameters
    ----------
    server_url: str
        The URL of the Kimina server.
    server_max_retries: int
        The maximum number of retries for the Kimina server.

    Returns
    -------
    CompiledStateGraph
        The compiled state graph for the sketch checker agent.
    """
    # Create the sketch checker agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the server related arguments of _check_sketch
    bound_check_sketch = partial(_check_sketch, server_url, server_max_retries)

    # Add the nodes
    graph_builder.add_node("check_sketch_agent", bound_check_sketch)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["check_sketch_agent"])
    graph_builder.add_edge("check_sketch_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    check_sketch_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances whose sketches' to check the syntax of.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("check_sketch_agent", state) for state in states["inputs"]]


def _check_sketch(
    server_url: str, server_max_retries: int, state: DecomposedFormalTheoremState
) -> DecomposedFormalTheoremStates:
    """
    Checks syntax of the proof sketch in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state  with the proof sketch whose syntax is to be checked.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the sketch
        checked added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Check the proof sketch with the stored preamble prefix
    sketch_with_imports = combine_preamble_and_body(state["preamble"], str(state["proof_sketch"]))
    check_response = kimina_client.check(sketch_with_imports, timeout=36000)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Update the state with the sketch check result
    state["syntactic"] = parsed_response["pass"]

    # Update the state with the formatted error string
    # Note: get_error_str expects the code with DEFAULT_IMPORTS for proper line number handling
    state["errors"] = get_error_str(sketch_with_imports, parsed_response.get("errors", []), False)

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
