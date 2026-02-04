import re
from functools import partial
from typing import cast
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import (
    LLMParsingError,
    combine_preamble_and_body,
    load_prompt,
)
from goedels_poetry.agents.util.debug import log_llm_prompt, log_llm_response


class ProverAgentFactory:
    """
    Factory class for creating instances of the ProverAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> CompiledStateGraph:
        """
        Creates a ProverAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the prover agent

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the prover agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the prover agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the prover agent

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the prover agent.
    """
    # Create the prover agent state graph
    graph_builder = StateGraph(FormalTheoremProofStates)

    # Bind the llm argument of prover
    bound_prover = partial(_prover, llm)

    # Add the nodes
    graph_builder.add_node("prover_agent", bound_prover)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["prover_agent"])
    graph_builder.add_edge("prover_agent", END)

    return graph_builder.compile()


def _map_edge(states: FormalTheoremProofStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    prover_agent nodes.

    Parameters
    ----------
    states: FormalTheoremProofStates
        The FormalTheoremProofStates containing in the "inputs" member the FormalTheoremProofState
        instances to create the proofs for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("prover_agent", state) for state in states["inputs"]]


def _prover(llm: BaseChatModel, state: FormalTheoremProofState) -> FormalTheoremProofStates:
    """
    Proves the formal theorem in the passed FormalTheoremProofState.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the prover agent
    state: FormalTheoremProofState
        The formal theorem state  with the formal theorem to be proven.

    Returns
    -------
    FormalTheoremProofStates
        A FormalTheoremProofStates with the FormalTheoremProofState with the formal proof added
        to the FormalTheoremProofStates "outputs" member.
    """
    # Create transaction id
    transaction_id = uuid4().hex

    # Copy state to prevent issues with LangGraph's mapreduce implementation
    new_state: FormalTheoremProofState = {
        **state,  # shallow copy is OK if you also copy mutables
        "proof_history": list(state["proof_history"]),
    }

    # Check if errors is None
    if new_state["errors"] is None:
        # If it is, load the prompt in use when not correcting a previous proof
        # Combine the stored preamble with the formal theorem for the prompt
        formal_statement_with_imports = combine_preamble_and_body(
            str(new_state["preamble"]), str(new_state["formal_theorem"])
        )
        prompt = load_prompt("goedel-prover-v2-initial", formal_statement=formal_statement_with_imports)

        # Log debug prompt
        log_llm_prompt(
            f"PROVER_AGENT ({transaction_id})",
            prompt,
            "goedel-prover-v2-initial",
            attempt_num=new_state["self_correction_attempts"],
            pass_num=new_state["pass_attempts"],
        )

        # Put the prompt in the final message
        new_state["proof_history"] += [HumanMessage(content=prompt)]

    # Prove the formal statement
    response_content = llm.invoke(new_state["proof_history"]).content

    # Log debug response
    log_llm_response(
        f"PROVER_AGENT ({transaction_id})",
        str(response_content),
        attempt_num=new_state["self_correction_attempts"],
        pass_num=new_state["pass_attempts"],
    )

    # Parse prover response
    try:
        raw_code = _extract_code_block(str(response_content))

        # Store the raw LLM output in the new field
        new_state["llm_lean_output"] = raw_code

        # Clear formal_proof initially - it will be populated by the proof_parser_agent
        new_state["formal_proof"] = None

        # Add the raw code to the state's proof_history
        new_state["proof_history"] += [AIMessage(content=raw_code)]
    except LLMParsingError:
        # Set parse failure markers - state manager will handle requeueing and attempt increments
        new_state["formal_proof"] = None
        new_state["errors"] = (
            "Malformed LLM response: unable to parse proof body from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        # Do not add to proof_history on parse failure

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [new_state]}  # type: ignore[typeddict-item]


def _extract_code_block_fallback(response: str) -> str:
    """
    Fallback method to extract the last code block, even if it's missing closing ticks.

    Parameters
    ----------
    response: str
        The LLM response containing a code block

    Returns
    -------
    str
        The extracted code block content

    Raises
    ------
    LLMParsingError
        If no code block is found in the response.
    """
    pattern_start = r"```lean4?\s*\n"
    matches = list(re.finditer(pattern_start, response, re.DOTALL))
    if not matches:
        raise LLMParsingError("Failed to extract code block from LLM response", response)  # noqa: TRY003

    code_start = matches[-1].end()
    closing_index = response.rfind("\n```")
    if closing_index == -1 or closing_index < code_start:
        closing_index = response.rfind("```")

    if closing_index == -1 or closing_index < code_start:
        return response[code_start:].strip()

    return response[code_start:closing_index].strip()


def _extract_code_block(response: str) -> str:
    """
    Extract the code block from an LLM response, handling nested code blocks in doc comments.

    Parameters
    ----------
    response: str
        The LLM response containing a code block

    Returns
    -------
    str
        The extracted code block content

    Raises
    ------
    LLMParsingError
        If no code block is found in the response.
    """
    pattern = r"```lean4?\n(.*?)\n?```"
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if not matches:
        return _extract_code_block_fallback(response)

    # Check if there are more ``` after the last match (nested block issue)
    last_match = matches[-1]
    remaining = response[last_match.end() :]
    if "```" in remaining:
        # Likely nested blocks in doc comments - use fallback
        return _extract_code_block_fallback(response)

    return cast(str, matches[-1].group(1)).strip()


def _parse_prover_response(response: str, expected_preamble: str) -> str:
    """
    DEPRECATED: No longer used. Replaced by direct extraction in _prover.
    Originally extracted the final lean code snippet, removed preamble,
    and extracted only the proof body (tactics after `:= by`).
    """
    # This remains for potential internal references, but the agent now calls _extract_code_block directly
    formal_proof = _extract_code_block(response)
    return formal_proof
