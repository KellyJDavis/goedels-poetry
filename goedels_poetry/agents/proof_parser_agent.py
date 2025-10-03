from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response
from goedels_poetry.parsers.ast import AST


class ProofParserAgentFactory:
    """
    Factory class for creating instances of the ProofParserAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> StateGraph:
        """
        Creates a ProofParserAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.

        Returns
        -------
        StateGraph
            An StateGraph instance of the proof parser agent.
        """
        return _build_agent(server_url=server_url, server_max_retries=server_max_retries)


def _build_agent(server_url: str, server_max_retries: int) -> StateGraph:
    """
    Builds a state graph for the specified Kimina server.

    Parameters
    ----------
    server_url: str
        The URL of the Kimina server.
    server_max_retries: int
        The maximum number of retries for the Kimina server.

    Returns
    -------
    StateGraph
        The state graph for the proof parser agent.
    """
    # Create the proof parser agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Bind the server related arguments of _parse_proof
    bound_parse_proof = partial(_parse_proof, server_url, server_max_retries)

    # Add the nodes
    graph_builder.add_node("parser_agent", bound_parse_proof)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["parser_agent"])
    graph_builder.add_edge("parser_agent", END)

    return graph_builder.compile()  # type: ignore[return-value]


def _map_edge(states: FormalTheoremProofStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    parser_agent nodes.

    Parameters
    ----------
    states: FormalTheoremProofStates
        The FormalTheoremProofStates containing in the "inputs" member the FormalTheoremProofState
        instances to parse the proofs of.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("parser_agent", state) for state in states["inputs"]]


def _parse_proof(server_url: str, server_max_retries: int, state: FormalTheoremProofState) -> FormalTheoremProofStates:
    """
    Parses the proof of the formal proof in the passed FormalTheoremProofState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    state: FormalTheoremProofState
        The formal theorem proof state  with the formal proof to be parsed.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates with the FormalTheoremProofState with the parsed proof added
        to the FormalTheoremProofStates "outputs" member.
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Parse formal proof of the passed state
    proof_code = state["formal_proof"] or ""
    ast_code_response = kimina_client.ast_code(proof_code)

    # Parse ast_code_response
    parsed_response = parse_kimina_ast_code_response(ast_code_response)

    # Set state["ast"] with the parsed_response
    if parsed_response["ast"] is not None:
        state["ast"] = AST(parsed_response["ast"])

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"inputs": [], "outputs": [state]}
