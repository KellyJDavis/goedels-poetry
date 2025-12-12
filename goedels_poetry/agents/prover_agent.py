import re
from functools import partial
from typing import cast

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
    strip_known_preamble,
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
    # Check if errors is None
    if state["errors"] is None:
        # If it is, load the prompt in use when not correcting a previous proof
        # Combine the stored preamble with the formal theorem for the prompt
        formal_statement_with_imports = combine_preamble_and_body(state["preamble"], state["formal_theorem"])
        prompt = load_prompt("goedel-prover-v2-initial", formal_statement=formal_statement_with_imports)

        # Log debug prompt
        log_llm_prompt("PROVER_AGENT", prompt, "goedel-prover-v2-initial")

        # Put the prompt in the final message
        state["proof_history"] += [HumanMessage(content=prompt)]

    # Prove the formal statement
    response_content = llm.invoke(state["proof_history"]).content

    # Log debug response
    log_llm_response("PROVER_AGENT_LLM", str(response_content))

    # Parse prover response
    try:
        formal_proof = _parse_prover_response(str(response_content), state["preamble"])

        # Add the formal proof to the state
        state["formal_proof"] = formal_proof

        # Add the formal proof to the state's proof_history
        state["proof_history"] += [AIMessage(content=formal_proof)]
    except LLMParsingError:
        # Set parse failure markers - state manager will handle requeueing and attempt increments
        state["formal_proof"] = None
        state["errors"] = (
            "Malformed LLM response: unable to parse proof body from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        # Do not add to proof_history on parse failure

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


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

    # Standard extraction worked fine
    return cast(str, matches[-1].group(1)).strip()


def _extract_proof_body(code_without_preamble: str, prefer_theorem: bool = True) -> str | None:
    """
    Extract proof body from code, optionally preferring theorem/example declarations.

    Parameters
    ----------
    code_without_preamble: str
        Code without preamble
    prefer_theorem: bool
        If True, try to find theorem/example declarations first. If False, find any := by pattern.

    Returns
    -------
    Optional[str]
        The proof body if found, None if prefer_theorem=True and no theorem/example found,
        empty string if prefer_theorem=False and no := by pattern found
    """
    # Find := by pattern, optionally requiring it to be in a theorem/example
    if prefer_theorem:
        # Try to find := by within a theorem/example declaration
        theorem_pattern = r"(theorem|example)\s+[a-zA-Z0-9_']+.*?:=\s*by"
        match = re.search(theorem_pattern, code_without_preamble, re.DOTALL)
        if not match:
            return None
        # The match ends at "by", so proof starts right after
        proof_start = match.end()
        proof_body_raw = code_without_preamble[proof_start:]
    else:
        # Find any := by pattern
        by_match = re.search(r":=\s*by", code_without_preamble, re.DOTALL)
        if not by_match:
            return code_without_preamble.strip()
        proof_start = by_match.end()
        proof_body_raw = code_without_preamble[proof_start:]

    # Stop at next declaration
    next_decl_match = re.search(
        r"\n\s*(?:/-.*?-\/\s*)?(theorem|lemma|def|abbrev|example|end|namespace)\s+",
        proof_body_raw,
        re.DOTALL,
    )
    if next_decl_match:
        proof_body_raw = proof_body_raw[: next_decl_match.start()]

    # Find first non-empty line (preserving leading empty lines for indentation)
    lines = proof_body_raw.split("\n")
    first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_idx is not None:
        return "\n".join(lines[first_idx:]).rstrip()

    return None if prefer_theorem else ""


def _parse_prover_response(response: str, expected_preamble: str) -> str:
    """
    Extract the final lean code snippet from the passed string, remove DEFAULT_IMPORTS,
    and extract only the proof body (the tactics after `:= by`).

    Parameters
    ----------
    response: str
        The string to extract the final lean code snippet from
    expected_preamble: str
        The expected preamble to strip from the response

    Returns
    -------
    str
        A string containing only the proof body (tactics after `:= by`).

    Raises
    ------
    LLMParsingError
        If no code block is found in the response.
    """
    formal_proof = _extract_code_block(response)
    if not formal_proof:
        return formal_proof

    # Strip the preamble if it matches
    stripped, matched = strip_known_preamble(formal_proof, expected_preamble)
    code_without_preamble = stripped if matched else formal_proof

    # Try to extract proof from theorem/example first (preferred)
    proof_body = _extract_proof_body(code_without_preamble, prefer_theorem=True)
    if proof_body is not None:
        return proof_body

    # Fallback: extract from any := by pattern
    fallback_result = _extract_proof_body(code_without_preamble, prefer_theorem=False)
    return fallback_result if fallback_result is not None else ""
