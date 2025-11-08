from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import combine_preamble_and_body
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response


class FormalTheoremSyntaxAgentFactory:
    """
    Factory class for creating instances of the FormalTheoremSyntaxAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
        """
        Creates a FormalTheoremSyntaxAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the formal theorem syntax agent.
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
        The compiled state graph for the formal theorem syntax agent.
    """
    # Create the formalizer agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Bind the server related arguments of check_syntax
    bound_check_syntax = partial(_check_syntax, server_url, server_max_retries)

    # Add the nodes
    graph_builder.add_node("syntax_agent", bound_check_syntax)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["syntax_agent"])
    graph_builder.add_edge("syntax_agent", END)

    return graph_builder.compile()


def _map_edge(states: FormalTheoremProofStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    syntax_agent nodes.

    Parameters
    ----------
    states: FormalTheoremProofStates
        The FormalTheoremProofStates containing in the "inputs" member the FormalTheoremProofState
        instances to check the syntax of.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("syntax_agent", state) for state in states["inputs"]]


def _check_syntax(server_url: str, server_max_retries: int, state: FormalTheoremProofState) -> FormalTheoremProofStates:
    """
    Checks syntax of the formal theorem in the passed FormalTheoremProofState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    state: FormalTheoremProofState
        The formal theorem state  with the formal theorem to be checked.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates with the FormalTheoremProofState with the syntax checked added
        to the FormalTheoremProofStates "outputs" member.
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Check syntax of state["formal_theorem"] with the stored preamble prefix
    code_with_imports = combine_preamble_and_body(state["preamble"], str(state["formal_theorem"]))
    check_response = kimina_client.check(code_with_imports, timeout=36000)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Get the response from the server
    syntactic = parsed_response["pass"]

    # Update the state with the syntax check result
    state["syntactic"] = syntactic

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
