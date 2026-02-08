from functools import partial
from typing import cast

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import (
    combine_preamble_and_body,
    get_error_str,
    remove_default_imports_from_ast,
)
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_ast_utils import (
    actionable_suffix,
    ast_code_parsed,
    compute_body_start,
)
from goedels_poetry.agents.util.kimina_server import (
    is_no_usable_ast,
    parse_kimina_check_response,
)
from goedels_poetry.agents.util.state_isolation import detach_decomposed_theorem_state
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.foundation.decl_extraction import (
    extract_proof_body_from_ast,
    extract_signature_from_ast,
)


def _mark_unsyntactic_with_error(theorem_state: DecomposedFormalTheoremState, message: str) -> None:
    """
    Mark a sketch-state as syntactically invalid and clear derived parse artifacts.

    This is used when the sketch *typechecks* but is malformed for downstream structural parsing.
    """
    theorem_state["syntactic"] = False
    theorem_state["proof_sketch"] = None
    theorem_state["ast"] = None
    theorem_state["errors"] = message


def _try_extract_target_signature(
    *,
    kimina_client: KiminaClient,
    server_timeout: int,
    normalized_preamble: str,
    formal_theorem: str,
) -> str | None:
    normalized_formal = str(formal_theorem).strip()
    formal_with_preamble = combine_preamble_and_body(normalized_preamble, normalized_formal)
    body_start_formal = compute_body_start(normalized_preamble, normalized_formal, formal_with_preamble)

    parsed_formal, err = ast_code_parsed(
        kimina_client,
        formal_with_preamble,
        server_timeout=server_timeout,
        log_label="ast_code_formal",
        log_fn=log_kimina_response,
    )
    if err is not None or parsed_formal is None or is_no_usable_ast(parsed_formal):
        return None

    try:
        ast_formal = AST(
            parsed_formal["ast"],
            sorries=parsed_formal.get("sorries"),
            source_text=formal_with_preamble,
            body_start=body_start_formal,
        )
    except Exception:
        return None

    return extract_signature_from_ast(ast_formal)


def _maybe_flag_downstream_parser_errors(
    *,
    theorem_state: DecomposedFormalTheoremState,
    kimina_client: KiminaClient,
    server_timeout: int,
    raw_output: str,
) -> None:
    """
    If the sketch otherwise passes Kimina (`syntactic=True` and no compilation errors), ensure it
    is parseable and structurally extractable by the downstream parser.

    On failure, mutate `theorem_state` to `syntactic=False` with an error string to allow correction.
    Never raises.
    """
    if not (theorem_state["syntactic"] and theorem_state["errors"] == ""):
        return

    normalized_preamble = str(theorem_state["preamble"]).strip()
    normalized_body = str(raw_output).strip()
    normalized_sketch_with_imports = combine_preamble_and_body(normalized_preamble, normalized_body)
    body_start = compute_body_start(normalized_preamble, normalized_body, normalized_sketch_with_imports)

    # Parser exceptional condition (line ~190): Kimina failed to parse proof AST.
    parsed_ast, err = ast_code_parsed(
        kimina_client,
        normalized_sketch_with_imports,
        server_timeout=server_timeout,
        log_label="ast_code",
        log_fn=log_kimina_response,
    )
    if err is not None:
        _mark_unsyntactic_with_error(theorem_state, f"Kimina failed to parse proof; error: {err}")
        return

    if parsed_ast is None or is_no_usable_ast(parsed_ast):
        _mark_unsyntactic_with_error(
            theorem_state,
            "Kimina failed to parse proof" + actionable_suffix(parsed_ast or {}, normalized_sketch_with_imports),
        )
        return

    # Parser exceptional condition (line ~202): structural extraction failed.
    target_sig = _try_extract_target_signature(
        kimina_client=kimina_client,
        server_timeout=server_timeout,
        normalized_preamble=normalized_preamble,
        formal_theorem=str(theorem_state["formal_theorem"]),
    )
    if target_sig is None:
        return

    try:
        sketch_ast_without_imports = remove_default_imports_from_ast(parsed_ast["ast"], preamble=normalized_preamble)
        ast = AST(
            sketch_ast_without_imports,
            sorries=parsed_ast.get("sorries"),
            source_text=normalized_sketch_with_imports,
            body_start=body_start,
        )
        extracted_sketch = extract_proof_body_from_ast(ast, target_sig)
    except Exception as e:
        _mark_unsyntactic_with_error(
            theorem_state,
            (
                f"Structural extraction failed for target signature: {target_sig}; error: {e!r}; "
                f"preview: {normalized_sketch_with_imports[:200]!r}"
            ),
        )
        return

    if extracted_sketch is None:
        _mark_unsyntactic_with_error(
            theorem_state,
            (
                f"Structural extraction failed for target signature: {target_sig}; "
                f"preview: {normalized_sketch_with_imports[:200]!r}"
            ),
        )


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

    _maybe_flag_downstream_parser_errors(
        theorem_state=theorem_state,
        kimina_client=kimina_client,
        server_timeout=server_timeout,
        raw_output=raw_output,
    )

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]
