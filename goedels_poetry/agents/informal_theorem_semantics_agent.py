from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from goedels_poetry.agents.state import InformalTheoremState
from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    LLMParsingError,
    combine_preamble_and_body,
    load_prompt,
    parse_semantic_check_response,
)
from goedels_poetry.agents.util.debug import log_llm_prompt, log_llm_response


class InformalTheoremSemanticsAgentFactory:
    """
    Factory class for creating instances of the InformalTheoremSemanticsAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> CompiledStateGraph:
        """
        Creates a InformalTheoremSemanticsAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the informal theorem sementics agent

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the informal theorem sementics agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the informal theorem sementics agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the informal theorem sementics agent

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the informal theorem sementics agent.
    """
    # Create the formalizer agent state graph
    graph_builder = StateGraph(InformalTheoremState)

    # Bind the llm argument of check_semantics
    bound_check_semantics = partial(_check_semantics, llm)

    # Add the nodes
    graph_builder.add_node("semantics_agent", bound_check_semantics)

    # Add the edges
    graph_builder.add_edge(START, "semantics_agent")
    graph_builder.add_edge("semantics_agent", END)

    # Return the agent
    return graph_builder.compile()


def _check_semantics(llm: BaseChatModel, state: InformalTheoremState) -> InformalTheoremState:
    """
    Checks if the semantics of the informal theorem in the passed state is the same as the
    semantics of the formal theorem in the passed state.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM used to check the semantics of the informal and formal theorems
    state : InformalTheoremState
        The state with the informal and formal theorems to be compared.

    Returns
    -------
    InformalTheoremState
        A InformalTheoremState containing a bool semantic indicating if the semantics of the
        informal and formal statements are the same.
    """
    # Construct prompt with the default preamble prefix added to formal_statement
    formal_statement_with_imports = combine_preamble_and_body(DEFAULT_IMPORTS, str(state["formal_theorem"]))
    prompt = load_prompt(
        "goedel-semiotician-v2",
        formal_statement=formal_statement_with_imports,
        informal_statement=str(state["informal_theorem"]),
    )

    # Log debug prompt
    log_llm_prompt("SEMANTICS_AGENT", prompt, "goedel-semiotician-v2")

    # Determine if the semantics of the informal and formal theorems are the same
    response_content = llm.invoke(prompt).content

    # Log debug response
    log_llm_response("SEMANTICS_AGENT_LLM", str(response_content))

    # Parse semantics checker response
    try:
        judgement = parse_semantic_check_response(str(response_content))
    except LLMParsingError:
        # On parse failure, return semantic=False - existing code will handle retry
        return {"semantic": False}  # type: ignore[typeddict-item]
    else:
        # Return InformalTheoremState with semantic set appropriately
        return {"semantic": (judgement == "Appropriate")}  # type: ignore[typeddict-item]
