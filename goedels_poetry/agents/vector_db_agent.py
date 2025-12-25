import asyncio
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import (
    APISearchResponseTypedDict,
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
)
from goedels_poetry.agents.util.debug import log_vectordb_response

try:
    # Optional dependency: `lean_explore` is only required when actually querying a LeanExplore server.
    # Tests patch/mock this client, so we keep the import lazy-friendly.
    from lean_explore.api.client import Client as _LeanExploreClient  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional import
    _LeanExploreClient = None

Client: Any = _LeanExploreClient


class LeanExploreDependencyMissing(ModuleNotFoundError):
    """Raised when vector DB querying is attempted without the optional `lean_explore` dependency installed."""

    def __init__(self) -> None:
        super().__init__(
            "Optional dependency `lean_explore` is not installed. "
            "Install it to enable vector DB queries against a LeanExplore server."
        )


class VectorDBAgentFactory:
    """
    Factory class for creating instances of the VectorDBAgent.
    """

    @staticmethod
    def create_agent(server_url: str, package_filters: list[str]) -> CompiledStateGraph:
        """
        Creates a VectorDBAgent instance that queries the Lean Explore server at the passed URL.

        Parameters
        ----------
        server_url: str
            The URL of the Lean Explore server.
        package_filters: list[str]
            List of package names to filter search results.

        Returns
        -------
        CompiledStateGraph
            A CompiledStateGraph instance of the vector DB agent.
        """
        return _build_agent(server_url=server_url, package_filters=package_filters)


def _build_agent(server_url: str, package_filters: list[str]) -> CompiledStateGraph:
    """
    Builds a compiled state graph for the vector DB agent.

    Parameters
    ----------
    server_url: str
        The URL of the Lean Explore server.
    package_filters: list[str]
        List of package names to filter search results.

    Returns
    -------
    CompiledStateGraph
        The compiled state graph for the vector DB agent.
    """
    # Create the vector DB agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Bind the server related arguments of _query_vectordb
    bound_query_vectordb = partial(_query_vectordb, server_url, package_filters)

    # Add the nodes
    graph_builder.add_node("vector_db_agent", bound_query_vectordb)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["vector_db_agent"])
    graph_builder.add_edge("vector_db_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and disperses them to the
    vector_db_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to query the vector database for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating their target node and its input, singular.
    """
    return [Send("vector_db_agent", state) for state in states["inputs"]]


def _query_vectordb(
    server_url: str, package_filters: list[str], state: DecomposedFormalTheoremState
) -> DecomposedFormalTheoremStates:
    """
    Queries the vector database for each search query in the passed DecomposedFormalTheoremState.

    Parameters
    ----------
    server_url: str
        The URL of the Lean Explore server.
    package_filters: list[str]
        List of package names to filter search results.
    state: DecomposedFormalTheoremState
        The decomposed formal theorem state with search_queries to query the vector database for.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates with the DecomposedFormalTheoremState with search_results
        added to the DecomposedFormalTheoremStates "outputs" member.
    """
    # Handle None or empty search_queries
    if state["search_queries"] is None:
        state["search_results"] = None
        return {"outputs": [state]}  # type: ignore[typeddict-item]

    if not state["search_queries"]:
        state["search_results"] = []
        return {"outputs": [state]}  # type: ignore[typeddict-item]

    if Client is None:
        raise LeanExploreDependencyMissing()

    # Create a client to access the Lean Explore Server
    client = Client(base_url=server_url)

    # Query the vector database for each search query sequentially
    search_results: list[APISearchResponseTypedDict] = []
    for search_query in state["search_queries"]:
        # Use asyncio.run() to wrap the async client.search() call
        # Exceptions are allowed to propagate (as per specification, similar to KiminaClient)
        api_response = asyncio.run(client.search(search_query, package_filters=package_filters))

        # Convert APISearchResponse to APISearchResponseTypedDict
        # Handle both Pydantic models (with model_dump) and regular objects
        if hasattr(api_response, "model_dump"):
            # Pydantic model - use model_dump for conversion
            api_dict = api_response.model_dump()
            query = api_dict.get("query", "")
            packages_applied = list(api_dict.get("packages_applied", []))
            results = api_dict.get("results", [])
            count = api_dict.get("count", 0)
            total_candidates_considered = api_dict.get("total_candidates_considered", 0)
            processing_time_ms = api_dict.get("processing_time_ms", 0)
        else:
            # Regular object - access attributes directly
            query = api_response.query if hasattr(api_response, "query") else ""
            packages_applied = (
                list(api_response.packages_applied)
                if hasattr(api_response, "packages_applied") and api_response.packages_applied
                else []
            )
            results = api_response.results if hasattr(api_response, "results") else []
            count = api_response.count if hasattr(api_response, "count") else 0
            total_candidates_considered = (
                api_response.total_candidates_considered if hasattr(api_response, "total_candidates_considered") else 0
            )
            processing_time_ms = api_response.processing_time_ms if hasattr(api_response, "processing_time_ms") else 0

        # Convert results to list of dicts
        converted_results = []
        for result in results:
            if hasattr(result, "model_dump"):
                converted_results.append(result.model_dump())
            elif hasattr(result, "__dict__"):
                converted_results.append(result.__dict__)
            elif isinstance(result, dict):
                converted_results.append(result)
            else:
                # Try to convert to dict if possible
                converted_results.append(
                    dict(result) if hasattr(result, "__iter__") and not isinstance(result, str) else {}
                )

        search_result: APISearchResponseTypedDict = {
            "query": query,
            "packages_applied": packages_applied,
            "results": converted_results,
            "count": count,
            "total_candidates_considered": total_candidates_considered,
            "processing_time_ms": processing_time_ms,
        }
        search_results.append(search_result)

    # Log debug response
    log_vectordb_response("search", search_results)

    # Update the state with the search results
    state["search_results"] = search_results

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]
