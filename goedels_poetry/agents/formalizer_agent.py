import re
from functools import partial
from hashlib import sha256

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from goedels_poetry.agents.state import InformalTheoremState
from goedels_poetry.agents.util.common import load_prompt


class FormalizerAgentFactory:
    """
    Factory class for creating instances of the FormalizerAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> StateGraph:
        """
        Creates a FormalizerAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the formalizer agent

        Returns
        -------
        StateGraph
            An StateGraph instance of the formalizer agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds a state graph for the formalizer agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the formalizer agent

    Returns
    ----------
    StateGraph
        The state graph for the formalizer agent.
    """
    # Create the formalizer agent state graph
    graph_builder = StateGraph(InformalTheoremState)

    # Bind the llm argument of formalizer
    bound_formalizer = partial(_formalizer, llm)

    # Add the nodes
    graph_builder.add_node("formalizer_agent", bound_formalizer)

    # Add the edges
    graph_builder.add_edge(START, "formalizer_agent")
    graph_builder.add_edge("formalizer_agent", END)

    # Return the agent
    return graph_builder.compile()


def _formalizer(llm: BaseChatModel, state: InformalTheoremState) -> InformalTheoremState:
    """
    Formalizes the informal tmeorem in the passed state and returns the formalized theorem in
    the returned InformalTheoremState

    Parameters
    ----------
    llm: BaseChatModel
        The LLM used to formalize the informal theorem
    state : InformalTheoremStat]
        The state of the agent with the informal theorem to be formalized.

    Returns
    -------
    InformalTheoremState
        A InformalTheoremState containing the formalized version of the passed informal theorem.
    """
    # Construct a hash used to name the formal theorem
    hashed_informal_statement = _hash_informal_statement(state["informal_theorem"])

    # Construct prompt
    prompt = load_prompt(
        "goedel-formalizer-v2",
        formal_statement_name=f"theorem_{hashed_informal_statement}",
        informal_statement=state["informal_theorem"],
    )

    # Formalize informal statement
    response_content = llm.invoke(prompt).content

    # Parse formalizer response
    formal_statement = _parser_formalizer_response(response_content)

    # Return InformalTheoremState with the formal theorem
    return {"formal_theorem": formal_statement}


def _hash_informal_statement(informal_statement: str) -> str:
    """
    Generate a hash string from the informal statement for formal theorem naming.

    Parameters
    ----------
    informal_statement : str
        The informal statement string

    Returns
    -------
    str
        First 12 characters of SHA256 hash of informal_statement
    """
    normalized_informal_statement = _normalize_informal_statement(informal_statement)
    return sha256(normalized_informal_statement.encode("utf-8")).hexdigest()[:12]


def _normalize_informal_statement(informal_statement: str) -> str:
    """
    Normalize the informal_statement string for consistent hashing.

    Parameters
    ----------
    informal_statement : str
        The informal statement string

    Returns
    -------
    str
        Normalized informal statement string (stripped and lowercased)
    """
    return informal_statement.strip().lower()


def _parser_formalizer_response(response: str) -> str:
    """
    Extract the final lean code snippet from the passed string.

    Parameters
    ----------
    response: str
        The string to extract the final lean code snippet from

    Returns
    -------
    str
        A string containing the lean code snippet if found, otherwise None.
    """
    pattern = r"```lean4?\n(.*?)\n?```"
    matches = re.findall(pattern, response, re.DOTALL)
    formal_statement = matches[-1].strip() if matches else None
    return formal_statement
