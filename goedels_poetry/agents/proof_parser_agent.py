from functools import partial

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import (
    combine_preamble_and_body,
    combine_theorem_with_proof,
    remove_default_imports_from_ast,
)
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response
from goedels_poetry.parsers.ast import AST


class ProofParserAgentFactory:
    """
    Factory class for creating instances of the ProofParserAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int) -> CompiledStateGraph:
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
        CompiledStateGraph
            An CompiledStateGraph instance of the proof parser agent.
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
        The compiled state graph for the proof parser agent.
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

    return graph_builder.compile()


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

    # Combine the original theorem statement with the proof body
    # state["formal_theorem"] contains the theorem with `:= by sorry`
    # state["formal_proof"] contains only the proof body (tactics after `:= by`)
    theorem_with_proof = combine_theorem_with_proof(
        str(state["formal_theorem"]), str(state["formal_proof"]) if state["formal_proof"] else ""
    )

    # Parse formal proof of the passed state with the stored preamble prefix
    normalized_preamble = state["preamble"].strip()
    normalized_body = theorem_with_proof.strip()
    proof_with_imports = combine_preamble_and_body(normalized_preamble, normalized_body)
    # Compute the body start offset based on the actual combined string (avoid assuming "\n\n").
    if normalized_preamble and normalized_body:
        body_start = proof_with_imports.find(normalized_body, len(normalized_preamble))
        body_start = body_start if body_start != -1 else len(normalized_preamble)
    else:
        body_start = 0
    ast_code_response = kimina_client.ast_code(proof_with_imports)

    # Parse ast_code_response
    parsed_response = parse_kimina_ast_code_response(ast_code_response)

    # Log debug response
    log_kimina_response("ast_code", parsed_response)

    # Remove the preamble-specific commands from the parsed AST when applicable
    ast_without_imports = remove_default_imports_from_ast(parsed_response["ast"], preamble=state["preamble"])

    # Set state["ast"] with the parsed_response (without DEFAULT_IMPORTS)
    state["ast"] = AST(
        ast_without_imports,
        sorries=parsed_response.get("sorries"),
        source_text=proof_with_imports,
        body_start=body_start,
    )

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
