from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from goedels_poetry.agents.state import InformalTheoremState
from goedels_poetry.agents.util.common import load_prompt
from goedels_poetry.agents.util.kimina_server import parse_semantic_check_response


class InformalTheoremSemanticsAgentFactory:
    """
    Factory class for creating instances of the InformalTheoremSemanticsAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> StateGraph:
        """
        Creates a InformalTheoremSemanticsAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the informal theorem sementics agent

        Returns
        -------
        StateGraph
            An StateGraph instance of the informal theorem sementics agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds a state graph for the informal theorem sementics agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the informal theorem sementics agent

    Returns
    ----------
    StateGraph
        The state graph for the informal theorem sementics agent.
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
    return graph_builder.compile()  # type: ignore[return-value]


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
    # Construct prompt
    prompt = load_prompt(
        "goedel-semiotician-v2", formal_statement=state["formal_theorem"], informal_statement=state["informal_theorem"]
    )

    # Determine if the semantics of the informal and formal theorems are the same
    response_content = llm.invoke(prompt).content

    # Parse semantics checker response
    judgement = parse_semantic_check_response(str(response_content))

    # Return InformalTheoremState with semantic set appropriately
    return {**state, "semantic": (judgement == "Appropriate")}
