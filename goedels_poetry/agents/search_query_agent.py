import re
from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import DecomposedFormalTheoremState, DecomposedFormalTheoremStates
from goedels_poetry.agents.util.common import LLMParsingError, combine_preamble_and_body, load_prompt
from goedels_poetry.agents.util.debug import log_llm_prompt, log_llm_response


class SearchQueryAgentFactory:
    """
    Factory class for creating instances of the SearchQueryAgent.
    """

    @staticmethod
    def create_agent(llm: BaseChatModel) -> CompiledStateGraph:
        """
        Creates a SearchQueryAgent instance with the passed llm.

        Parameters
        ----------
        llm: BaseChatModel
            The LLM to use for the search query agent

        Returns
        -------
        CompiledStateGraph
            A CompiledStateGraph instance of the search query agent.
        """
        return _build_agent(llm=llm)


def _build_agent(llm: BaseChatModel) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the search query agent.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the search query agent

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the search query agent.
    """
    # Create the search query agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the llm argument of _search_query_generator
    bound_search_query_generator = partial(_search_query_generator, llm)

    # Add the nodes
    graph_builder.add_node("search_query_generator", bound_search_query_generator)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["search_query_generator"])
    graph_builder.add_edge("search_query_generator", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    search_query_generator nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to generate search queries for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("search_query_generator", state) for state in states["inputs"]]


def _is_backtracking(state: DecomposedFormalTheoremState) -> bool:
    """
    Detect if this state is in a backtrack scenario by checking if the backtrack
    prompt template appears in decomposition_history.

    This is done by checking if any HumanMessage matches the backtrack prompt template
    for any round number from 1 to the current self_correction_attempts.

    Parameters
    ----------
    state: DecomposedFormalTheoremState
        The state to check for backtrack context

    Returns
    -------
    bool
        True if backtracking is detected, False otherwise
    """
    # Check if any HumanMessage in history matches the backtrack prompt template
    # We check for round numbers from 1 to current self_correction_attempts
    for round_num in range(1, state["self_correction_attempts"] + 1):
        backtrack_prompt = load_prompt(
            "decomposer-backtrack",
            prev_round_num=str(round_num),
        )

        for message in state["decomposition_history"]:
            if isinstance(message, HumanMessage):
                message_content = str(message.content).strip()
                if message_content == backtrack_prompt.strip():
                    return True

    return False


def _search_query_generator(llm: BaseChatModel, state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremStates:
    """
    Generate search queries for the formal theorem in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the search query agent
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state with the formal theorem to generate queries for.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with the search queries
        added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Combine the stored preamble with the formal theorem for the prompt
    formal_theorem_with_imports = combine_preamble_and_body(state["preamble"], state["formal_theorem"])

    # Detect if this is a backtrack scenario
    is_backtracking = _is_backtracking(state)

    if is_backtracking:
        # Use backtrack prompt - the full history is already in decomposition_history
        prompt = load_prompt("search-query-backtrack", formal_theorem=formal_theorem_with_imports)
        # Log debug prompt
        log_llm_prompt("SEARCH_QUERY_AGENT", prompt, "search-query-backtrack")
    else:
        prompt = load_prompt("search-query-initial", formal_theorem=formal_theorem_with_imports)
        # Log debug prompt
        log_llm_prompt("SEARCH_QUERY_AGENT", prompt, "search-query-initial")

    # Add the prompt to decomposition_history
    state["decomposition_history"] += [HumanMessage(content=prompt)]

    # Generate search queries - LLM receives full history including the new prompt
    response_content = llm.invoke(state["decomposition_history"]).content

    # Log debug response
    log_llm_response("SEARCH_QUERY_AGENT_LLM", str(response_content))

    # Parse search query response
    search_queries = _parse_search_queries_response(str(response_content))

    # Add the search queries to the state
    state["search_queries"] = search_queries

    # Add the LLM response to decomposition_history
    state["decomposition_history"] += [AIMessage(content=str(response_content))]

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


def _parse_search_queries_response(response: str) -> list[str]:
    """
    Extract search queries from the LLM response using <search> tags.

    Parameters
    ----------
    response: str
        The string to extract search queries from

    Returns
    -------
    list[str]
        A list of search query strings.

    Raises
    ------
    LLMParsingError
        If no queries are found in the response.
    """
    # Pattern to match <search>query text</search> tags
    # Uses non-greedy matching to handle multiple tags
    pattern = r"<search>(.*?)</search>"

    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    if not matches:
        msg = "Failed to extract search queries from LLM response. Expected queries in <search>query</search> format."
        raise LLMParsingError(msg, response)

    # Clean and filter queries
    queries = [match.strip() for match in matches if match.strip()]

    # Remove duplicates while preserving order
    queries = list(dict.fromkeys(queries))

    if not queries:
        raise LLMParsingError("Found <search> tags but all were empty", response)  # noqa: TRY003

    return queries
