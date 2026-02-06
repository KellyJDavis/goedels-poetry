from functools import partial
from typing import cast

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import combine_preamble_and_body, get_error_str
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response
from goedels_poetry.agents.util.state_isolation import detach_decomposed_theorem_state


class SketchCheckerAgentFactory:
    """
    Factory class for creating instances of the SketchCheckerAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
        """
        Creates a SketchCheckerAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.
        server_timeout: int
            The timeout in seconds for requests to the Kimina server.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch checker agent.
        """
        return _build_agent(server_url=server_url, server_max_retries=server_max_retries, server_timeout=server_timeout)


def _build_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the specified Kimina server.

    Parameters
    ----------
    server_url: str
        The URL of the Kimina server.
    server_max_retries: int
        The maximum number of retries for the Kimina server.
    server_timeout: int
        The timeout in seconds for requests to the Kimina server.

    Returns
    -------
    CompiledStateGraph
        The compiled state graph for the sketch checker agent.
    """
    # Create the sketch checker agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the server related arguments of _check_sketch
    bound_check_sketch = partial(_check_sketch, server_url, server_max_retries, server_timeout)

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
    # Fan out detached per-item payloads to avoid sharing cyclic proof-tree references.
    return [Send("check_sketch_agent", {"item": detach_decomposed_theorem_state(state)}) for state in states["inputs"]]


def _check_sketch(
    server_url: str, server_max_retries: int, server_timeout: int, state: DecomposedFormalTheoremStates
) -> DecomposedFormalTheoremStates:
    """
    Checks syntax of the proof sketch in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    server_timeout: int
        The timeout in seconds for requests to the server.
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state  with the proof sketch whose syntax is to be checked.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the sketch
        checked added to the DecomposedFormalTheoremStates "outputs" member.
    """
    theorem_state = cast(DecomposedFormalTheoremState, state["item"])

    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=server_timeout, n_retries=server_max_retries)

    # Use the raw LLM output directly for validation
    # new_state["llm_lean_output"] contains the complete declaration from the LLM
    raw_output = str(theorem_state["llm_lean_output"]) if theorem_state["llm_lean_output"] else ""

    # Check the proof sketch with the stored preamble prefix
    sketch_with_imports = combine_preamble_and_body(str(theorem_state["preamble"]), raw_output)
    check_response = kimina_client.check(sketch_with_imports, timeout=server_timeout)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Update the state with the sketch check result
    theorem_state["syntactic"] = parsed_response["pass"]

    # Update the state with the formatted error string
    # Note: get_error_str expects the code with DEFAULT_IMPORTS for proper line number handling
    theorem_state["errors"] = get_error_str(sketch_with_imports, parsed_response.get("errors", []), False)

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]
