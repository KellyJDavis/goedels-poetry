from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from goedels_poetry.agents.state import InformalTheoremState
from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response


class InformalTheoremSyntaxAgentFactory:
    """
    Factory class for creating instances of the InformalTheoremSyntaxAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
        """
        Creates a InformalTheoremSyntaxAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the informal theorem syntax agent.
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
        The compiled state graph for the informal theorem syntax agent.
    """
    # Create the formalizer agent state graph
    graph_builder = StateGraph(InformalTheoremState)

    # Bind the server related arguments of check_syntax
    bound_check_syntax = partial(_check_syntax, server_url, server_max_retries)

    # Add the nodes
    graph_builder.add_node("syntax_agent", bound_check_syntax)

    # Add the edges
    graph_builder.add_edge(START, "syntax_agent")
    graph_builder.add_edge("syntax_agent", END)

    return graph_builder.compile()


def _check_syntax(server_url: str, server_max_retries: int, state: InformalTheoremState) -> InformalTheoremState:
    """
    Checks the syntax of a formal theorem of the passed informal theorem state.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    state : InformalTheoremState
        The informal theorem state with the formal theorem to be checked.

    Returns
    -------
    InformalTheoremState
        A InformalTheoremState indicating if the formal statement is syntactic
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Check syntax of state["formal_theorem"] with the default preamble prefix
    code_with_imports = combine_preamble_and_body(DEFAULT_IMPORTS, str(state["formal_theorem"]))
    check_response = kimina_client.check(code_with_imports, timeout=36000)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Get the response from the server
    syntactic = parsed_response["pass"]

    # Return a InformalTheoremState with the response from Kimina Server
    return {"syntactic": syntactic}  # type: ignore[typeddict-item]
