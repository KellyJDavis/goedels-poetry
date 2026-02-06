import re
from functools import partial
from typing import cast
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import (
    LLMParsingError,
    _format_theorem_hints_section,
    combine_preamble_and_body,
    load_prompt,
)
from goedels_poetry.agents.util.debug import log_llm_prompt, log_llm_response
from goedels_poetry.agents.util.state_isolation import detach_decomposed_theorem_state


class ProofSketcherAgentFactory:
    """
    Factory class for creating instances of the ProofSketcherAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> CompiledStateGraph:
        """
        Creates a ProofSketcherAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the proof sketcher agent

        Returns
        -------
        CompiledStateGraph
            A CompiledStateGraph instance of the proof sketcher agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the proof sketcher agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the proof sketcher agent

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the proof sketcher agent.
    """
    # Create the proof sketcher agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the llm argument of _proof_sketcher
    bound_proof_sketcher = partial(_proof_sketcher, llm)

    # Add the nodes
    graph_builder.add_node("proof_sketcher", bound_proof_sketcher)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["proof_sketcher"])
    graph_builder.add_edge("proof_sketcher", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    proof_sketcher nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to sketch proofs for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    # Fan out detached per-item payloads to avoid sharing cyclic proof-tree references.
    return [Send("proof_sketcher", {"item": detach_decomposed_theorem_state(state)}) for state in states["inputs"]]


def _extract_responses_api_content(response_content: str | list) -> str:
    """
    Extract text content from Responses API format or return string as-is.

    When using OpenAI's Responses API (with use_responses_api=True), the response content
    is returned as a list of dictionaries, each with a 'type' key and corresponding content.
    This function extracts all text segments from such responses and concatenates them.

    Parameters
    ----------
    response_content: str | list
        The response content, either a string (standard format) or a list of dictionaries
        (Responses API format like [{'type': 'text', 'text': '...'}]).

    Returns
    -------
    str
        The extracted text content as a single string. If the input is already a string,
        it is returned as-is. If it's a list (Responses API format), all text segments
        are extracted and concatenated in order.
    """
    # If it's already a string, return as-is (non-Responses API format)
    if isinstance(response_content, str):
        return response_content

    # Handle Responses API format: list of dictionaries
    if isinstance(response_content, list):
        text_parts = []
        for item in response_content:
            # Check if this is a dictionary with type='text'
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                if text_value:
                    text_parts.append(str(text_value))
        # Concatenate all text parts in order
        return "".join(text_parts)

    # Fallback: convert to string if it's neither string nor list
    return str(response_content)


def _proof_sketcher(llm: BaseChatModel, state: DecomposedFormalTheoremStates) -> DecomposedFormalTheoremStates:
    """
    Sketch the proof of the formal theorem in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the proof sketcher agent
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state with the formal theorem to have its proof sketched.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the formal proof
        sketch added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Create transaction id
    transaction_id = uuid4().hex

    theorem_state = cast(DecomposedFormalTheoremState, state["item"])

    # Check if errors is None
    if theorem_state["errors"] is None:
        # If it is, load the prompt used when not correcting a previous proof sketch
        # Combine the stored preamble with the formal theorem for the prompt
        formal_theorem_with_imports = combine_preamble_and_body(
            str(theorem_state["preamble"]), str(theorem_state["formal_theorem"])
        )
        # Format theorem hints section from search results
        theorem_hints_section = _format_theorem_hints_section(theorem_state["search_results"])
        prompt = load_prompt(
            "decomposer-initial",
            formal_theorem=formal_theorem_with_imports,
            theorem_hints_section=theorem_hints_section,
        )

        # Log debug prompt
        log_llm_prompt(
            f"PROOF_SKETCHER_AGENT ({transaction_id})",
            prompt,
            "decomposer-initial",
            attempt_num=theorem_state["self_correction_attempts"],
        )

        # Put the prompt in the final message
        theorem_state["decomposition_history"] += [HumanMessage(content=prompt)]

    # Sketch the proof of the formal theorem
    response_content = llm.invoke(theorem_state["decomposition_history"]).content

    # Extract text content from Responses API format (list of dicts) or use string as-is
    normalized_content = _extract_responses_api_content(response_content)

    # Log debug response
    log_llm_response(
        f"PROOF_SKETCHER_AGENT ({transaction_id})",
        normalized_content,
        attempt_num=theorem_state["self_correction_attempts"],
    )

    # Parse sketcher response
    try:
        raw_code = _parse_proof_sketcher_response(normalized_content, theorem_state["preamble"])

        # Store the raw LLM output in the new field
        theorem_state["llm_lean_output"] = raw_code

        # Clear proof_sketch initially - it will be populated by the sketch_parser_agent
        theorem_state["proof_sketch"] = None

        # Add the raw code to the state's decomposition_history
        theorem_state["decomposition_history"] += [AIMessage(content=raw_code)]
    except LLMParsingError:
        # Set parse failure markers - state manager will handle requeueing and attempt increments
        theorem_state["proof_sketch"] = None
        theorem_state["errors"] = (
            "Malformed LLM response: unable to parse proof sketch from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        # Do not add to decomposition_history on parse failure

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [theorem_state]}  # type: ignore[typeddict-item]


def _parse_proof_sketcher_response(response: str, expected_preamble: str) -> str:
    """
    Extract the final lean code snippet from the passed string.

    Parameters
    ----------
    response: str
        The string to extract the final lean code snippet from
    expected_preamble: str
        DEPRECATED: No longer used for stripping preamble here.

    Returns
    -------
    str
        A string containing the raw lean code snippet if found.

    Raises
    ------
    LLMParsingError
        If no code block is found in the response.
    """
    # TODO: Figure out if this algorithm works for the non-Goedel LLM
    pattern = r"```lean4?\n(.*?)\n?```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise LLMParsingError("Failed to extract code block from LLM response", response)  # noqa: TRY003
    proof_sketch = cast(str, matches[-1]).strip()
    return proof_sketch
