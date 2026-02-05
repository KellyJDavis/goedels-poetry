import bisect
import uuid
from functools import partial
from typing import cast

from kimina_client import KiminaClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import (
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
    FormalTheoremProofState,
)
from goedels_poetry.agents.util.debug import log_kimina_response
from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response
from goedels_poetry.parsers.ast import AST
from goedels_poetry.parsers.util.high_level.subgoal_extraction_v2 import (
    extract_subgoal_with_check_responses,
)
from goedels_poetry.util.tree import InternalTreeNode, TreeNode, add_child


class SketchDecompositionAgentFactory:
    """
    Factory class for creating instances of the SketchDecompositionAgent.
    """

    @staticmethod
    def create_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
        """
        Creates a SketchDecompositionAgent instance.

        Parameters
        ----------
        server_url: str
            URL of the Kimina server
        server_max_retries: int
            Maximum number of retries for HTTP requests
        server_timeout: int
            Timeout in seconds for HTTP requests

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch decomposition agent.
        """
        return _build_agent(server_url, server_max_retries, server_timeout)


def _build_agent(server_url: str, server_max_retries: int, server_timeout: int) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the sketch decomposition agent.

    Parameters
    ----------
    server_url: str
        URL of the Kimina server
    server_max_retries: int
        Maximum number of retries for HTTP requests
    server_timeout: int
        Timeout in seconds for HTTP requests

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the sketch decomposition agent.
    """
    # Create the sketch decomposition agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind server parameters to _sketch_decomposer using partial
    bound_sketch_decomposer = partial(_sketch_decomposer, server_url, server_max_retries, server_timeout)

    # Add the nodes
    graph_builder.add_node("sketch_decomposition_agent", bound_sketch_decomposer)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["sketch_decomposition_agent"])
    graph_builder.add_edge("sketch_decomposition_agent", END)

    return graph_builder.compile()


def _byte_to_char_index(byte_off: int, source_text: str) -> int:
    """
    Convert a byte offset to a character index in the source text.

    Parameters
    ----------
    byte_off: int
        Byte offset in UTF-8 encoding
    source_text: str
        Source text string

    Returns
    -------
    int
        Character index (clamped to valid range)
    """
    byte_prefix: list[int] = [0]
    for ch in source_text:
        byte_prefix.append(byte_prefix[-1] + len(ch.encode("utf-8")))

    i = bisect.bisect_right(byte_prefix, byte_off) - 1
    if i < 0:
        return 0
    if i >= len(source_text):
        return len(source_text)
    return i


def _generate_theorem_portions(ast: AST, source_text: str, body_start: int) -> list[tuple[str, str]]:  # noqa: C901
    """
    Generate portions of the theorem text ending just before each sorry-proven have statement.

    Each portion includes the preamble and all text up to (but not including) the have statement
    or sorry. This ensures the check response will contain hypotheses that are true before
    the have statement, not including the have statement's own hypotheses.

    Parameters
    ----------
    ast: AST
        The AST of the decomposed theorem
    source_text: str
        Full text (preamble + body) for portion generation
    body_start: int
        Character offset where the body starts (for converting body-relative to full-text positions)

    Returns
    -------
    list[tuple[str, str]]
        List of tuples (subgoal_identifier, portion_text) in textual order.
        Multiple sorries with the same name will have unique identifiers like "{name}##sorry##{i}".

    Raises
    ------
    ValueError
        If a subgoal is not found in the AST, position info is missing, or a portion is empty.
    """
    unproven_subgoal_names = ast.get_unproven_subgoal_names()
    sorry_holes_by_name = ast.get_sorry_holes_by_name()

    # Collect all subgoals with their positions for sorting
    subgoal_info: list[tuple[int, str, bool]] = []  # (position, name, is_main_body)
    have_positions: dict[str, int] = {}  # Map subgoal_name -> have_start_char

    for subgoal_name in unproven_subgoal_names:
        if subgoal_name == "<main body>":
            # Get sorry positions for main body
            main_body_sorries = sorry_holes_by_name.get("<main body>", [])
            if not main_body_sorries:
                continue
            # Use first sorry position for sorting
            first_sorry_pos = main_body_sorries[0][0]
            full_text_pos = first_sorry_pos + body_start
            subgoal_info.append((full_text_pos, subgoal_name, True))
        else:
            # Find have statement node
            have_node = ast.get_named_subgoal_ast(subgoal_name)
            if have_node is None:
                raise ValueError(f"Subgoal '{subgoal_name}' not found in AST")  # noqa: TRY003

            # Extract start position (byte offset)
            args = have_node.get("args", [])
            if not args:
                raise ValueError(f"Position info missing for have statement '{subgoal_name}': no args")  # noqa: TRY003
            have_token = args[0]
            if not isinstance(have_token, dict):
                raise ValueError(f"Position info missing for have statement '{subgoal_name}': first arg is not dict")  # noqa: TRY003
            info = have_token.get("info")
            if not isinstance(info, dict):
                raise ValueError(f"Position info missing for have statement '{subgoal_name}': no info")  # noqa: TRY003
            pos = info.get("pos")
            if not isinstance(pos, list) or len(pos) < 1:
                raise ValueError(f"Position info missing for have statement '{subgoal_name}': invalid pos")  # noqa: TRY003
            have_start_byte = int(pos[0])

            # Convert byte offset to character index
            have_start_char = _byte_to_char_index(have_start_byte, source_text)
            have_positions[subgoal_name] = have_start_char
            subgoal_info.append((have_start_char, subgoal_name, False))

    # Sort subgoals by position (textual order)
    subgoal_info.sort(key=lambda x: (x[0], x[1]))

    # Generate portions
    portions: list[tuple[str, str]] = []

    for _, subgoal_name, is_main_body in subgoal_info:
        if is_main_body:
            # Main body: create portion for each sorry
            main_body_sorries = sorry_holes_by_name.get("<main body>", [])
            for i, (sorry_pos, _) in enumerate(main_body_sorries):
                full_text_sorry_pos = sorry_pos + body_start
                portion_text = source_text[0 : max(0, full_text_sorry_pos - 1)]
                if not portion_text.strip():
                    raise ValueError("Portion for <main body> is empty (processing error)")  # noqa: TRY003
                # Use identifier: "<main body>##sorry##{i}" for multiple sorries, or "<main body>" for single
                identifier = f"<main body>##sorry##{i}" if len(main_body_sorries) > 1 else "<main body>"
                portions.append((identifier, portion_text))
        else:
            # Have statement: create portion for each sorry (all ending before the have statement)
            have_sorries = sorry_holes_by_name.get(subgoal_name, [])
            have_start_char = have_positions[subgoal_name]
            for i, _ in enumerate(have_sorries):
                portion_text = source_text[0:have_start_char]
                if not portion_text.strip():
                    raise ValueError(f"Portion for '{subgoal_name}' is empty (processing error)")  # noqa: TRY003
                # Use identifier: "{name}##sorry##{i}" for multiple sorries, or "{name}" for single
                identifier = f"{subgoal_name}##sorry##{i}" if len(have_sorries) > 1 else subgoal_name
                portions.append((identifier, portion_text))

    return portions


def _check_theorem_portions(
    server_url: str,
    server_max_retries: int,
    server_timeout: int,
    portions: list[tuple[str, str]],
) -> tuple[dict[str, dict], KiminaClient]:
    """
    Check all theorem portions using the Kimina server and return parsed responses.

    Parameters
    ----------
    server_url: str
        URL of the Kimina server
    server_max_retries: int
        Maximum number of retries for HTTP requests
    server_timeout: int
        Timeout in seconds for HTTP requests
    portions: list[tuple[str, str]]
        List of (subgoal_identifier, portion_text) tuples to check

    Returns
    -------
    tuple[dict[str, dict], KiminaClient]
        Tuple containing:
        - Dictionary mapping subgoal identifiers to their parsed check responses
        - KiminaClient instance (returned for potential reuse)

    Raises
    ------
    ValueError
        If a check response doesn't contain "unsolved goals" in any error message.
    """
    kimina_client = KiminaClient(api_url=server_url, http_timeout=server_timeout, n_retries=server_max_retries)

    check_responses: dict[str, dict] = {}

    for subgoal_identifier, portion_text in portions:
        # Normalize trailing newline
        normalized_portion = portion_text if portion_text.endswith("\n") else portion_text + "\n"

        # Call check - let errors propagate (no try-catch)
        check_response = kimina_client.check(normalized_portion, timeout=server_timeout)

        # Parse response
        parsed_response = parse_kimina_check_response(check_response)

        # Log response
        log_kimina_response("check", parsed_response)

        # Verify "unsolved goals" exists
        errors = parsed_response.get("errors", [])
        has_unsolved_goals = any(
            isinstance(err, dict)
            and isinstance(err.get("data"), str)
            and err.get("data", "").startswith("unsolved goals")
            for err in errors
        )
        if not has_unsolved_goals:
            raise ValueError(f'Check response for "{subgoal_identifier}" does not contain "unsolved goals" message')  # noqa: TRY003

        # Store in mapping
        check_responses[subgoal_identifier] = parsed_response

    return (check_responses, kimina_client)


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    sketch_decomposition_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to create the sketch decompositions for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("sketch_decomposition_agent", {"item": state}) for state in states["inputs"]]


def _sketch_decomposer(
    server_url: str,
    server_max_retries: int,
    server_timeout: int,
    state: DecomposedFormalTheoremStates,
) -> DecomposedFormalTheoremStates:
    """
    Queries the AST of the passed DecomposedFormalTheoremState for all unproved subgoals. For
    each such subgoal, extracts a standalone lemma using check responses from theorem portions,
    then creates a FormalTheoremProofState instance corresponding to the subgoal and adds it as
    a child of the passed DecomposedFormalTheoremState and adds that DecomposedFormalTheoremState
    to the outputs of the returned DecomposedFormalTheoremStates.

    Parameters
    ----------
    server_url: str
        URL of the Kimina server
    server_max_retries: int
        Maximum number of retries for HTTP requests
    server_timeout: int
        Timeout in seconds for HTTP requests
    state: DecomposedFormalTheoremState
        The DecomposedFormalTheoremState containing a proof sketch to be decomposed.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates containing in its outputs the modified DecomposedFormalTheoremState

    Raises
    ------
    ValueError
        If source_text is None (should always be set by sketch_parser_agent.py)
    """
    theorem_state = cast(DecomposedFormalTheoremState, state["item"])

    ast = cast(AST, theorem_state["ast"])

    # Get source text - should always be available
    source_text = ast.get_source_text()
    if source_text is None:
        raise ValueError(  # noqa: TRY003
            "AST source_text is None. This indicates sketch_parser_agent.py failed to set source_text "
            "(should always be set at line 143)."
        )

    # Get body_start for converting body-relative to full-text positions
    body_start = ast.get_body_start()

    # Generate portions
    portions = _generate_theorem_portions(ast, source_text, body_start)

    # Check portions using Kimina server
    check_responses, _kimina_client = _check_theorem_portions(server_url, server_max_retries, server_timeout, portions)

    # Create bidirectional mapping between subgoal_name and subgoal_identifier
    # Extract base names from identifiers in portions (portions are already sorted)
    identifier_to_name: dict[str, str] = {}
    name_to_identifiers: dict[str, list[str]] = {}

    for identifier, _portion_text in portions:
        # Extract base subgoal name: if identifier contains ##sorry##, split and take first part
        base_name = identifier.split("##sorry##")[0] if "##sorry##" in identifier else identifier

        # Add to mappings
        identifier_to_name[identifier] = base_name
        name_to_identifiers.setdefault(base_name, []).append(identifier)

    # Get sorry positions for hole metadata
    holes_by_name = ast.get_sorry_holes_by_name()

    # Process all portions in order (portions are already sorted by textual position)
    # Group by base name for hole metadata, but process in portion order
    for identifier, _portion_text in portions:
        # Get base name for this identifier
        base_name = identifier_to_name[identifier]

        # Extract standalone lemma using new method
        standalone_lemma_code = extract_subgoal_with_check_responses(ast, check_responses, identifier, base_name)

        # Get hole spans for this base name
        hole_spans = holes_by_name.get(base_name, [])

        # Find which sorry index this identifier corresponds to
        # If identifier has ##sorry##{i}, use that index; otherwise use 0
        sorry_index = 0
        if "##sorry##" in identifier:
            try:
                index_str = identifier.split("##sorry##")[1]
                sorry_index = int(index_str)
            except (ValueError, IndexError):
                sorry_index = 0

        # Get hole span for this specific sorry
        hole_span = hole_spans[sorry_index] if sorry_index < len(hole_spans) else None
        hole_start = hole_span[0] if hole_span is not None else None
        hole_end = hole_span[1] if hole_span is not None else None

        # Create FormalTheoremProofState with the extracted standalone lemma code
        new_child = cast(
            TreeNode,
            _create_formal_theorem_proof_state(
                standalone_lemma_code,
                theorem_state,
                hole_name=base_name,
                hole_start=hole_start,
                hole_end=hole_end,
            ),
        )
        add_child(cast(InternalTreeNode, theorem_state), new_child)

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]


def _create_formal_theorem_proof_state(
    formal_theorem: str,
    state: DecomposedFormalTheoremState,
    *,
    hole_name: str | None = None,
    hole_start: int | None = None,
    hole_end: int | None = None,
) -> FormalTheoremProofState:
    """
    Creates a FormalTheoremProofState with the passed formal_theorem and with state as its parent.

    Parameters
    ----------
    formal_theorem: str
        The formal theorm of the FormalTheoremProofState returned
    state: DecomposedFormalTheoremState
        The parent of the returned FormalTheoremProofState
    """
    return FormalTheoremProofState(
        id=uuid.uuid4().hex,
        parent=cast(TreeNode | None, state),
        depth=(state["depth"] + 1),
        formal_theorem=formal_theorem,
        preamble=state["preamble"],
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
        hole_name=hole_name,
        hole_start=hole_start,
        hole_end=hole_end,
        llm_lean_output=None,
    )
