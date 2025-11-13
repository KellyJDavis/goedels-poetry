import re
from functools import partial
from typing import Optional, cast

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
from goedels_poetry.agents.util.debug import log_llm_response


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

        # Put the prompt in the final message
        state["proof_history"] += [HumanMessage(content=prompt)]

    # Prove the formal statement
    response_content = llm.invoke(state["proof_history"]).content

    # Log debug response
    log_llm_response("PROVER_AGENT_LLM", str(response_content))

    # Parse prover response
    formal_proof = _parse_prover_response(str(response_content), state["preamble"])

    # Add the formal proof to the state
    state["formal_proof"] = formal_proof

    # Add the formal proof to the state's proof_history
    state["proof_history"] += [AIMessage(content=formal_proof)]

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


def _extract_code_block_fallback(response: str) -> str:
    """
    Fallback method to extract code block by finding the last ``` in the response.

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
    match_start = re.search(pattern_start, response, re.DOTALL)
    if not match_start:
        raise LLMParsingError("Failed to extract code block from LLM response", response)  # noqa: TRY003

    code_start = match_start.end()
    last_backtick = response.rfind("\n```")
    if last_backtick == -1:
        last_backtick = response.rfind("```")
    if last_backtick == -1 or last_backtick < code_start:
        raise LLMParsingError("Failed to find closing ``` in LLM response", response)  # noqa: TRY003

    return response[code_start:last_backtick].strip()


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
    # First try the standard pattern
    pattern = r"```lean4?\n(.*?)\n?```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return _extract_code_block_fallback(response)

    formal_proof = cast(str, matches[-1]).strip()
    # Check if there are more ``` after the LAST match (indicating nested blocks)
    # Find all matches to get the position of the last one
    all_matches = list(re.finditer(pattern, response, re.DOTALL))
    if not all_matches:
        return formal_proof

    # Get the last match's end position
    last_match = all_matches[-1]
    match_end_pos = last_match.end()
    remaining = response[match_end_pos:]
    if "```" not in remaining:
        return formal_proof

    # There are more ```, likely a nested block issue - use fallback
    fallback_proof = _extract_code_block_fallback(response)
    # Only use fallback if it gives us significantly more content
    if len(fallback_proof) > len(formal_proof) * 1.5:
        return fallback_proof
    return formal_proof


def _extract_proof_from_theorem(code_without_preamble: str) -> Optional[str]:
    """
    Extract proof body from a theorem/example declaration.

    Parameters
    ----------
    code_without_preamble: str
        Code without preamble

    Returns
    -------
    Optional[str]
        The proof body if found, None otherwise
    """
    theorem_pattern = r"(theorem|example)\s+[a-zA-Z0-9_']+.*?:=\s*by"
    theorem_match = re.search(theorem_pattern, code_without_preamble, re.DOTALL)
    if not theorem_match:
        return None

    by_match = re.search(r":=\s*by", code_without_preamble[theorem_match.start() :], re.DOTALL)
    if not by_match:
        return None

    proof_start = theorem_match.start() + by_match.end()
    proof_body_raw = code_without_preamble[proof_start:]
    # Stop at next declaration
    next_decl_match = re.search(
        r"\n\s*(?:/-.*?-\/\s*)?(theorem|lemma|def|abbrev|example|end|namespace)\s+", proof_body_raw, re.DOTALL
    )
    if next_decl_match:
        proof_body_raw = proof_body_raw[: next_decl_match.start()]

    # Extract just the tactics, preserving indentation
    lines = proof_body_raw.split("\n")
    first_content_line_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_content_line_idx = i
            break

    if first_content_line_idx is not None:
        return "\n".join(lines[first_content_line_idx:]).rstrip()
    return None


def _extract_proof_fallback(code_without_preamble: str) -> str:
    """
    Fallback method to extract proof body from any := by pattern.

    Parameters
    ----------
    code_without_preamble: str
        Code without preamble

    Returns
    -------
    str
        The proof body
    """
    match = re.search(r":=\s*by", code_without_preamble)
    if match is None:
        return code_without_preamble.strip()

    proof_body_raw = code_without_preamble[match.end() :]
    next_decl_match = re.search(
        r"\n\s*(?:/-.*?-\/\s*)?(theorem|lemma|def|abbrev|example|end|namespace)\s+", proof_body_raw, re.DOTALL
    )
    if next_decl_match:
        proof_body_raw = proof_body_raw[: next_decl_match.start()]

    lines = proof_body_raw.split("\n")
    first_content_line_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_content_line_idx = i
            break

    if first_content_line_idx is None:
        return ""

    return "\n".join(lines[first_content_line_idx:]).rstrip()


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
    proof_body = _extract_proof_from_theorem(code_without_preamble)
    if proof_body is not None:
        return proof_body

    # Fallback: extract from any := by pattern
    return _extract_proof_fallback(code_without_preamble)
