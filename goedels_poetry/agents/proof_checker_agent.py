from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import add_default_imports, get_error_str
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response


class ProofCheckerAgentFactory:
    """
    Factory class for creating instances of the ProofCheckerAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
        """
        Creates a ProofCheckerAgent instance that employs the server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Kimina server.
        server_max_retries: int
            The maximum number of retries for the Kimina server.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the proof checker agent.
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
        The compiled state graph for the proof checker agent.
    """
    # Create the proof checker agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Bind the server related arguments of _check_proof
    bound_check_proof = partial(_check_proof, server_url, server_max_retries)

    # Add the nodes
    graph_builder.add_node("check_proof_agent", bound_check_proof)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["check_proof_agent"])
    graph_builder.add_edge("check_proof_agent", END)

    return graph_builder.compile()


def _map_edge(states: FormalTheoremProofStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    check_proof_agent nodes.

    Parameters
    ----------
    states: FormalTheoremProofStates
        The FormalTheoremProofStates containing in the "inputs" member the FormalTheoremProofState
        instances to check the proofs of.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("check_proof_agent", state) for state in states["inputs"]]


def _check_proof(server_url: str, server_max_retries: int, state: FormalTheoremProofState) -> FormalTheoremProofStates:
    """
    Checks proof of the formal proof in the passed FormalTheoremProofState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    state: FormalTheoremProofState
        The formal theorem state  with the formal proof to be checked.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates with the FormalTheoremProofState with the proof checked added
        to the FormalTheoremProofStates "outputs" member.
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Check the formal proof state["formal_proof"] with DEFAULT_IMPORTS prefix
    proof_with_imports = add_default_imports(str(state["formal_proof"]))
    check_response = kimina_client.check(proof_with_imports, timeout=36000)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Update the state with the proof check result
    state["proved"] = parsed_response["pass"]

    # Update the state with the error string formatted for Goedel-Prover-V2 use
    # Note: get_error_str expects the code with DEFAULT_IMPORTS for proper line number handling
    state["errors"] = get_error_str(proof_with_imports, parsed_response.get("errors", []), False)

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
