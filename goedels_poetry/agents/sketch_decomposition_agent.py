from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goedels_poetry.agents.state import (
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
    FormalTheoremProofState,
)
from goedels_poetry.parsers.ast import AST
from goedels_poetry.util.tree import TreeNode


class SketchDecompositionAgentFactory:
    """
    Factory class for creating instances of the SketchDecompositionAgent.
    """

    @staticmethod
    def create_agent() -> CompiledStateGraph:
        """
        Creates a SketchDecompositionAgent instance.

        Returns
        -------
        CompiledStateGraph
            An CompiledStateGraph instance of the sketch decomposition agent.
        """
        return _build_agent()


def _build_agent() -> CompiledStateGraph:
    """
    Builds a compiled state graph for the sketch decomposition agent.

    Returns
    ----------
    CompiledStateGraph
        The compiled state graph for the sketch decomposition agent.
    """
    # Create the sketch decomposition agent state graph
    graph_builder = StateGraph(DecomposedFormalTheoremStates)

    # Add the nodes
    graph_builder.add_node("sketch_decomposition_agent", _sketch_decomposer)

    # Add the edges
    graph_builder.add_conditional_edges(START, _map_edge, ["sketch_decomposition_agent"])
    graph_builder.add_edge("sketch_decomposition_agent", END)

    return graph_builder.compile()


def _map_edge(states: DecomposedFormalTheoremStates) -> list[Send]:
    """
    Map edge that takes the members of the states["inputs"] list and dispers them to the
    sketch_decomposition_agent nodes.

    Parameters
    ----------
    states: DecomposedFormalTheoremStates
        The DecomposedFormalTheoremStates containing in the "inputs" member the
        DecomposedFormalTheoremState instances to create the sketch decompositions for.

    Returns
    -------
    list[Send]
        List of Send objects each indicating the their target node and its input, singular.
    """
    return [Send("sketch_decomposition_agent", state) for state in states["inputs"]]


def _sketch_decomposer(state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremStates:
    """
    Queries the AST of the passed DecomposedFormalTheoremState for all unproved subgoals. For
    each such subgoal, queries the AST for the code of that subgoal, then creates a
    FormalTheoremProofState instance corresponding to the subgoal and adds it as a child of
    the passed DecomposedFormalTheoremState and adds that DecomposedFormalTheoremState to
    the outputs of the returned DecomposedFormalTheoremStates.

    Parameters
    ----------
    state: DecomposedFormalTheoremState
        The DecomposedFormalTheoremState containing a proof sketch to be decomposed.

    Returns
    -------
    DecomposedFormalTheoremStates
        A DecomposedFormalTheoremStates containing in its outputs the modified DecomposedFormalTheoremState
    """
    # Obtain the names of all unproven subgoals
    unproven_subgoal_names = cast(AST, state["ast"]).get_unproven_subgoal_names()
    # Map subgoal-name -> (start, end) offsets of the corresponding `sorry` token in the sketch (body-relative).
    holes_by_name = cast(AST, state["ast"]).get_sorry_holes_by_name()

    # Loop over named unproven subgoals
    for unproven_subgoal_name in unproven_subgoal_names:
        # Obtain code of named unproven subgoal
        unproven_subgoal_code = cast(AST, state["ast"]).get_named_subgoal_code(unproven_subgoal_name)
        # Multiple sorries can share the same have-name (rare but possible in LLM sketches).
        # We match by traversal order: consume the next span for this name.
        hole_spans = holes_by_name.get(unproven_subgoal_name) or []
        hole_span = hole_spans.pop(0) if hole_spans else None
        hole_start = hole_span[0] if hole_span is not None else None
        hole_end = hole_span[1] if hole_span is not None else None

        # Append a FormalTheoremProofState corresponding to the unproven subgoal as a child of state
        state["children"].append(
            cast(
                TreeNode,
                _create_formal_theorem_proof_state(
                    unproven_subgoal_code,
                    state,
                    hole_name=unproven_subgoal_name,
                    hole_start=hole_start,
                    hole_end=hole_end,
                ),
            )
        )

    # Return a DecomposedFormalTheoremStates with state added to its outputs
    return {"outputs": [state]}  # type: ignore[typeddict-item]


def _create_formal_theorem_proof_state(
    formal_theorem: str,
    state: DecomposedFormalTheoremState,
    *,
    hole_name: str | None = None,
    hole_start: int | None = None,
    hole_end: int | None = None,
) -> FormalTheoremProofState:
    """
    Creates a FormalTheoremProofState with the passed formal_theorem and with state as its parent.

    Parameters
    ----------
    formal_theorem: str
        The formal theorm of the FormalTheoremProofState returned
    state: DecomposedFormalTheoremState
        The parent of the returned FormalTheoremProofState
    """
    return FormalTheoremProofState(
        parent=cast(TreeNode | None, state),
        depth=(state["depth"] + 1),
        formal_theorem=formal_theorem,
        preamble=state["preamble"],
        syntactic=True,
        formal_proof=None,
        proved=False,
        errors=None,
        ast=None,
        self_correction_attempts=0,
        proof_history=[],
        pass_attempts=0,
        hole_name=hole_name,
        hole_start=hole_start,
        hole_end=hole_end,
    )
