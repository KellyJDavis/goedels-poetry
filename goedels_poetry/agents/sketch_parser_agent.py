from functools import partial
from typing import cast
from uuid import uuid4

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import combine_preamble_and_body, remove_default_imports_from_ast
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import is_no_usable_ast, parse_kimina_ast_code_response
from goedels_poetry.agents.util.state_isolation import detach_decomposed_theorem_state
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.decl_extraction import (
    extract_preamble_from_ast,
    extract_proof_body_from_ast,
    extract_signature_from_ast,
)


class SketchParserAgentFactory:
    """
    Factory class for creating instances of the SketchParserAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
        """
        Creates a SketchParserAgent instance that employs the server at the passed URL.

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
            An CompiledStateGraph instance of the sketch parser agent.
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
        The compiled state graph for the sketch parser agent.
    """
    # Create the sketch parser agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the server related arguments of _parse_sketch
    bound_parse_sketch = partial(_parse_sketch, server_url, server_max_retries, server_timeout)

    # Add the nodes
    graph_builder.add_node("parser_agent", bound_parse_sketch)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["parser_agent"])
    graph_builder.add_edge("parser_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    parser_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to parse the proof sketches of.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    # Fan out detached per-item payloads to avoid sharing cyclic proof-tree references.
    return [Send("parser_agent", {"item": detach_decomposed_theorem_state(state)}) for state in states["inputs"]]


def _actionable_suffix(parsed: dict, code_preview: str) -> str:
    """Always include something actionable: error when present, else short preview (plan 2.3, 8.7)."""
    err = parsed.get("error")
    if err:
        return f"; error: {err}"
    return f"; preview: {code_preview[:200]!r}"


def _parse_sketch(
    server_url: str, server_max_retries: int, server_timeout: int, state: DecomposedFormalTheoremStates
) -> DecomposedFormalTheoremStates:
    """
    Parses the proof sketch in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    server_url: str
        The URL of the server.
    server_max_retries: int
        The maximum number of retries for the server.
    server_timeout: int
        The timeout in seconds for requests to the server.
    state: DecomposedFormalTheoremState
        The decomposed formal theorem proof state with the formal proof sketch to be parsed.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the parsed proof
        sketch added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Create transaction id
    transaction_id = uuid4().hex

    theorem_state = cast(DecomposedFormalTheoremState, state["item"])

    kimina_client = KiminaClient(api_url=server_url, http_timeout=server_timeout, n_retries=server_max_retries)
    normalized_preamble = theorem_state["preamble"].strip()
    normalized_formal = str(theorem_state["formal_theorem"]).strip()
    formal_with_preamble = combine_preamble_and_body(normalized_preamble, normalized_formal)

    if normalized_preamble and normalized_formal:
        idx = formal_with_preamble.find(normalized_formal, len(normalized_preamble))
        body_start_formal = idx if idx != -1 else len(normalized_preamble)
    else:
        body_start_formal = 0

    ast_code_response_formal = kimina_client.ast_code(formal_with_preamble)
    parsed_formal = parse_kimina_ast_code_response(ast_code_response_formal)
    log_kimina_response(f"ast_code_formal ({transaction_id})", parsed_formal)

    if is_no_usable_ast(parsed_formal):
        raise ValueError(
            "Kimina failed to parse formal theorem" + _actionable_suffix(parsed_formal, formal_with_preamble)
        )

    ast_formal = AST(
        parsed_formal["ast"],
        sorries=parsed_formal.get("sorries"),
        source_text=formal_with_preamble,
        body_start=body_start_formal,
    )
    target_sig = extract_signature_from_ast(ast_formal)
    if target_sig is None:
        raise ValueError(
            "Could not extract signature from formal theorem AST"
            + _actionable_suffix(parsed_formal, formal_with_preamble)
        )

    raw_output = str(theorem_state["llm_lean_output"]) if theorem_state["llm_lean_output"] else ""
    normalized_body = raw_output.strip()
    sketch_with_imports = combine_preamble_and_body(normalized_preamble, normalized_body)
    if normalized_preamble and normalized_body:
        idx = sketch_with_imports.find(normalized_body, len(normalized_preamble))
        body_start = idx if idx != -1 else len(normalized_preamble)
    else:
        body_start = 0

    ast_code_response = kimina_client.ast_code(sketch_with_imports)
    parsed_response = parse_kimina_ast_code_response(ast_code_response)
    log_kimina_response(f"ast_code ({transaction_id})", parsed_response)

    if is_no_usable_ast(parsed_response):
        raise ValueError("Kimina failed to parse proof" + _actionable_suffix(parsed_response, sketch_with_imports))

    sketch_ast_without_imports = remove_default_imports_from_ast(parsed_response["ast"], preamble=normalized_preamble)
    ast = AST(
        sketch_ast_without_imports,
        sorries=parsed_response.get("sorries"),
        source_text=sketch_with_imports,
        body_start=body_start,
    )
    extracted_sketch = extract_proof_body_from_ast(ast, target_sig)
    if extracted_sketch is None:
        msg = f"Structural extraction failed for target signature: {target_sig}; preview: {sketch_with_imports[:200]!r}"
        raise ValueError(msg)

    # Store the full sketch body (the complete declaration without the preamble), not tactics-only.
    #
    # Reconstruction and hole-replacement logic assumes `proof_sketch` matches the AST body's
    # source-text slice: `ast.get_source_text()[ast.get_body_start():]` (modulo stripping and a
    # trailing newline). Storing tactics-only here breaks that invariant and causes reconstruction
    # validation failures.
    theorem_state["proof_sketch"] = normalized_body

    ast_with_imports = AST(
        parsed_response["ast"],
        sorries=parsed_response.get("sorries"),
        source_text=sketch_with_imports,
        body_start=body_start,
    )
    extracted_preamble = extract_preamble_from_ast(ast_with_imports)
    if extracted_preamble:
        theorem_state["preamble"] = extracted_preamble

    ast_without_imports = remove_default_imports_from_ast(parsed_response["ast"], preamble=theorem_state["preamble"])
    theorem_state["ast"] = AST(
        ast_without_imports,
        sorries=parsed_response.get("sorries"),
        source_text=sketch_with_imports,
        body_start=body_start,
    )
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]
