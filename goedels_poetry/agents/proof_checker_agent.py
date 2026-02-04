from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import (
    combine_preamble_and_body,
    get_error_str,
)
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response


class ProofCheckerAgentFactory:
    """
    Factory class for creating instances of the ProofCheckerAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
        """
        Creates a ProofCheckerAgent instance that employs the server at the passed URL.

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
            An CompiledStateGraph instance of the proof checker agent.
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
        The compiled state graph for the proof checker agent.
    """
    # Create the proof checker agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Bind the server related arguments of _check_proof
    bound_check_proof = partial(_check_proof, server_url, server_max_retries, server_timeout)

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


def _check_proof(
    server_url: str, server_max_retries: int, server_timeout: int, state: FormalTheoremProofState
) -> FormalTheoremProofStates:
    """
    Checks proof of the formal proof in the passed FormalTheoremProofState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    server_timeout: int
        The timeout in seconds for requests to the server.
    state: FormalTheoremProofState
        The formal theorem state  with the formal proof to be checked.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates with the FormalTheoremProofState with the proof checked added
        to the FormalTheoremProofStates "outputs" member.
    """
    # Copy state to prevent issues with LangGraph's mapreduce implementation
    new_state = {
        **state,  # shallow copy is OK if you also copy mutables
        "proof_history": list(state["proof_history"]),
    }

    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=server_timeout, n_retries=server_max_retries)

    # Use the raw LLM output directly for validation
    # new_state["llm_lean_output"] contains the complete declaration from the LLM
    raw_output = str(new_state["llm_lean_output"]) if new_state["llm_lean_output"] else ""

    # Check the formal proof with the stored preamble prefix
    proof_with_imports = combine_preamble_and_body(str(new_state["preamble"]), raw_output)
    check_response = kimina_client.check(proof_with_imports, timeout=server_timeout)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Update the state with the proof check result
    # Note: We use "complete" instead of "pass" to ensure proofs with sorries are marked as unsuccessful
    new_state["proved"] = parsed_response["complete"]

    # Update the state with the error string formatted for Goedel-Prover-V2 use
    # Note: get_error_str expects the code with DEFAULT_IMPORTS for proper line number handling
    new_state["errors"] = get_error_str(proof_with_imports, parsed_response.get("errors", []), False)

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [new_state]}  # type: ignore[typeddict-item]


def check_complete_proof(
    complete_proof: str, server_url: str, server_max_retries: int, server_timeout: int
) -> tuple[bool, str]:
    """
    Checks a complete proof (assembled from subgoals) to verify it proves the desired theorem.

    This function is designed to be called after a proof has been successfully completed
    and assembled from multiple subgoals, before it is printed or written to a file.

    The complete_proof is already a valid Lean file with preamble and theorem with proof,
    so we can pass it directly to the Kimina server for verification without any parsing.

    Parameters
    ----------
    complete_proof: str
        The complete proof string including preamble and theorem with proof.
    server_url: str
        The URL of the Kimina server.
    server_max_retries: int
        The maximum number of retries for the Kimina server.
    server_timeout: int
        The timeout in seconds for requests to the Kimina server.

    Returns
    -------
    tuple[bool, str]
        A tuple containing:
        - bool: True if the proof is valid (complete and no errors), False otherwise
        - str: Error message string if proof is invalid, empty string if valid
    """
    # Create a client to access the Kimina Server
    kimina_client = KiminaClient(api_url=server_url, http_timeout=server_timeout, n_retries=server_max_retries)

    # Ensure trailing newline to prevent Kimina server hangs
    # This follows POSIX standard that text files should end with a newline
    normalized_proof = complete_proof if complete_proof.endswith("\n") else complete_proof + "\n"

    # The complete_proof is already a valid Lean file, so we can check it directly
    check_response = kimina_client.check(normalized_proof, timeout=server_timeout)

    # Parse check_response
    parsed_response = parse_kimina_check_response(check_response)

    # Log debug response
    log_kimina_response("check", parsed_response)

    # Extract the result
    is_valid = parsed_response["complete"]
    # Use normalized_proof (with trailing newline) for consistency with what was sent to Kimina
    error_msg = get_error_str(normalized_proof, parsed_response.get("errors", []), False) if not is_valid else ""

    return is_valid, error_msg
