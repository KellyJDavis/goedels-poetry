import re
from functools import partial
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import FormalTheoremProofState, FormalTheoremProofStates
from goedels_poetry.agents.util.common import LLMParsingError, add_default_imports, load_prompt, remove_default_imports
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
        # Add DEFAULT_IMPORTS prefix to the formal_theorem for the prompt
        formal_statement_with_imports = add_default_imports(state["formal_theorem"])
        prompt = load_prompt("goedel-prover-v2-initial", formal_statement=formal_statement_with_imports)

        # Put the prompt in the final message
        state["proof_history"] += [HumanMessage(content=prompt)]

    # Prove the formal statement
    response_content = llm.invoke(state["proof_history"]).content

    # Log debug response
    log_llm_response("PROVER_AGENT_LLM", str(response_content))

    # Parse prover response
    formal_proof = _parse_prover_response(str(response_content))

    # Add the formal proof to the state
    state["formal_proof"] = formal_proof

    # Add the formal proof to the state's proof_history
    state["proof_history"] += [AIMessage(content=formal_proof)]

    # Return a FormalTheoremProofStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


def _parse_prover_response(response: str) -> str:
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
    pattern = r"```lean4?\n(.*?)\n?```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise LLMParsingError("Failed to extract code block from LLM response", response)  # noqa: TRY003
    formal_proof = cast(str, matches[-1]).strip()
    # Remove DEFAULT_IMPORTS if present
    if formal_proof:
        formal_proof = remove_default_imports(formal_proof)
    return formal_proof
