import os
import re
from functools import partial
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent  # type: ignore[import-untyped]
from deepagents.backends import FilesystemBackend  # type: ignore[import-untyped]
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
    strip_known_preamble,
)
from goedels_poetry.agents.util.debug import log_llm_prompt, log_llm_response
from goedels_poetry.config.paths import OUTPUT_DIR


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
    # Prepare a dedicated filesystem backend for the Deep Agent
    root_dir = os.path.join(OUTPUT_DIR, "proof_sketcher_agent")
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    filesystem_backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)

    # Create the Deep Agent runnable bound to the provided llm
    deep_agent = create_deep_agent(
        model=llm,
        context_schema=DecomposedFormalTheoremStates,
        backend=filesystem_backend,
    )

    # Create the proof sketcher agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the Deep Agent into _proof_sketcher
    bound_proof_sketcher = partial(_proof_sketcher, deep_agent)

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
    return [Send("proof_sketcher", state) for state in states["inputs"]]


def _proof_sketcher(
    deep_agent: CompiledStateGraph, state: DecomposedFormalTheoremState
) -> DecomposedFormalTheoremStates:
    """
    Sketch the proof of the formal theorem in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    deep_agent: CompiledStateGraph
        The Deep Agent runnable that generates the proof sketch content
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state with the formal theorem to have its proof sketched.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the formal proof
        sketch added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Check if errors is None
    if state["errors"] is None:
        # If it is, load the prompt used when not correcting a previous proof sketch
        # Combine the stored preamble with the formal theorem for the prompt
        formal_theorem_with_imports = combine_preamble_and_body(state["preamble"], state["formal_theorem"])
        # Format theorem hints section from search results
        theorem_hints_section = _format_theorem_hints_section(state["search_results"])
        prompt = load_prompt(
            "decomposer-initial",
            formal_theorem=formal_theorem_with_imports,
            theorem_hints_section=theorem_hints_section,
        )

        # Log debug prompt
        log_llm_prompt("PROOF_SKETCHER_AGENT", prompt, "decomposer-initial")

        # Put the prompt in the final message
        state["decomposition_history"] += [HumanMessage(content=prompt)]

    # Sketch the proof of the formal theorem
    deep_agent_result: Any = deep_agent.invoke({"messages": state["decomposition_history"]})
    response_content = _extract_deep_agent_content(deep_agent_result)

    # Log debug response
    log_llm_response("DECOMPOSER_AGENT_LLM", str(response_content))

    # Parse sketcher response
    try:
        proof_sketch = _parse_proof_sketcher_response(str(response_content), state["preamble"])

        # Add the proof sketch to the state
        state["proof_sketch"] = proof_sketch

        # Add the proof sketch to the state's decomposition_history
        state["decomposition_history"] += [AIMessage(content=proof_sketch)]
    except LLMParsingError:
        # Set parse failure markers - state manager will handle requeueing and attempt increments
        state["proof_sketch"] = None
        state["errors"] = (
            "Malformed LLM response: unable to parse proof sketch from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        # Do not add to decomposition_history on parse failure

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


def _parse_proof_sketcher_response(response: str, expected_preamble: str) -> str:
    """
    Extract the final lean code snippet from the passed string and remove DEFAULT_IMPORTS.

    Parameters
    ----------
    response: str
        The string to extract the final lean code snippet from

    Returns
    -------
    str
        A string containing the lean code snippet if found.

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
    if not proof_sketch:
        return proof_sketch

    stripped, matched = strip_known_preamble(proof_sketch, expected_preamble)
    return stripped if matched else proof_sketch


def _extract_deep_agent_content(result: Any) -> str:
    """
    Normalize Deep Agent outputs to a string content payload.
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if messages:
            return getattr(messages[-1], "content", str(messages[-1]))
        return str(result)
    return getattr(result, "content", str(result))
