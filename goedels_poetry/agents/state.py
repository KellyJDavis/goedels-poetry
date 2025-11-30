from __future__ import annotations

from operator import add
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from typing_extensions import Required, TypedDict

from goedels_poetry.parsers.ast import AST
from goedels_poetry.util.tree import TreeNode


class InformalTheoremState(TypedDict):
    """
    State for an informal theorem

    Attributes
    ----------
    informal_theorem: Required[str]
        The text of the informal theorem
    formalization_attempts: Required[int]
        The number of attempts to formalize the informal theorem
    formal_theorem: Required[str | None]
        The text of the formalization of the informal theorem
    syntactic: Required[bool]
        A bool indicating if formal_theorem is syntactically valid
    semantic: Required[bool]
        A bool indicating if the informal_theorem and formal_theorem is semantically equivalent
    """

    informal_theorem: Required[str]
    formalization_attempts: Required[int]
    formal_theorem: Required[str | None]
    syntactic: Required[bool]
    semantic: Required[bool]


class FormalTheoremProofState(TypedDict):
    """
    State for an formal theorem proof

    Attributes
    ----------
    parent: Required[TreeNode | None]
        The parent in the proof tree
    depth: Required[int]
        The depth of this node in the proof tree
    formal_theorem: Required[str]
        The text of the formalization of the informal theorem
    syntactic: Required[bool]
        A bool indicating if formal_theorem is syntactically valid
    preamble: Required[str]
        A string containing the theorem's preamble
    formal_proof: Required[str | None]
        Formal proof of the formal_theorem
    proved: Required[bool]
        A bool indicating if the formal_proof is valid
    errors: Required[str | None]
        A string indicating errors in formal_proof
    ast: Required[AST | None]
        The AST of the formal_proof
    self_correction_attempts: Required[int]
        The number of self-correction attempts to prove formal_theorem (was proof_attempts)
    proof_history: Required[Annotated[list[AnyMessage],add]]
        The history of messages sent and received from the LLMs
    pass_attempts: Required[int]  # The number of times the self-correction loop has been executed
    """

    parent: Required[TreeNode | None]
    depth: Required[int]
    formal_theorem: Required[str]
    preamble: Required[str]
    syntactic: Required[bool]
    formal_proof: Required[str | None]
    proved: Required[bool]
    errors: Required[str | None]
    ast: Required[AST | None]
    self_correction_attempts: Required[int]
    proof_history: Required[Annotated[list[AnyMessage], add]]  # TODO: Correct annotation?
    pass_attempts: Required[int]  # The number of times the self-correction loop has been executed


class FormalTheoremProofStates(TypedDict):
    """
    A list, inputs, of FormalTheoremProofState to process using map reduce and a list, outputs,
    of FormalTheoremProofState to contain the outputs.

    inputs: Required[list[FormalTheoremProofState]]
       List of FormalTheoremProofState to process using map reduce.
    outputs: Required[Annotated[list[FormalTheoremProofState], add]
       List of FormalTheoremProofState that are the results of the map reduce
    """

    inputs: Required[list[FormalTheoremProofState]]
    outputs: Required[Annotated[list[FormalTheoremProofState], add]]  # TODO: Correct annotation?


class APISearchResponseTypedDict(TypedDict):
    """
    TypedDict representation of APISearchResponse from lean_explore.shared.models.api.

    This TypedDict matches the structure of APISearchResponse to enable type checking
    while maintaining compatibility with the lean_explore API.

    Attributes
    ----------
    query: Required[str]
        The search query that was executed
    packages_applied: Required[list[str]]
        List of package filters that were applied to the search
    results: Required[list[dict[str, Any]]]
        List of search results, where each result is a dictionary containing
        theorem information (name, type, code, etc.)
    count: Required[int]
        Number of results returned
    total_candidates_considered: Required[int]
        Total number of candidates that were considered during the search
    processing_time_ms: Required[int]
        Time taken to process the search query in milliseconds
    """

    query: Required[str]
    packages_applied: Required[list[str]]
    results: Required[list[dict[str, Any]]]
    count: Required[int]
    total_candidates_considered: Required[int]
    processing_time_ms: Required[int]


class DecomposedFormalTheoremState(TypedDict):
    """
    State for decomposition of a formal theorem

    Attributes
    ----------
    parent: Required[TreeNode | None]
        The parent in the proof tree
    children: Required[list[TreeNode]]
        The children of this node in the proof tree
    depth: Required[int]
        The depth of this node in the proof tree
    formal_theorem: Required[str]
        The text of the formalization of the informal theorem
    preamble: Required[str]
        A string containing the theorem's preamble
    proof_sketch: Required[str | None]
        The formal sketch of the proof of formal_theorem
    syntactic: Required[bool]
        A bool indicating if proof_sketch is syntactically valid
    errors: Required[str | None]
        A string indicating errors in proof_sketch
    ast: Required[AST | None]
        The AST of the proof_sketch
    self_correction_attempts: Required[int]
        The number of self-correction attempts to decompose formal_theorem (was decomposition_attempts)
    decomposition_history: Required[Annotated[list[AnyMessage],add]]
        The history of messages sent and received from the LLMs
    search_queries: Required[list[str] | None]
        List of search queries generated for retrieving relevant theorems from a vector database.
        None indicates queries have not been generated yet.
    search_results: Required[list[APISearchResponseTypedDict] | None]
        List of search results from the vector database, where search_results[i] corresponds to
        the results from search_queries[i]. None indicates results have not been retrieved yet,
        and an empty list indicates no queries were provided.
    """

    # InternalTreeNode specific properties
    parent: Required[TreeNode | None]
    children: Required[list[TreeNode]]
    depth: Required[int]
    # FormalTheorem specific properties
    formal_theorem: Required[str]
    preamble: Required[str]
    # DecomposedFormalTheoremState specific properties
    proof_sketch: Required[str | None]
    syntactic: Required[bool]
    errors: Required[str | None]
    ast: Required[AST | None]
    self_correction_attempts: Required[int]
    decomposition_history: Required[Annotated[list[AnyMessage], add]]
    search_queries: Required[list[str] | None]
    search_results: Required[list[APISearchResponseTypedDict] | None]


class DecomposedFormalTheoremStates(TypedDict):
    """
    A list, inputs, of DecomposedFormalTheoremState to process using map reduce and a list, outputs,
    of DecomposedFormalTheoremState to contain the outputs.

    inputs: Required[list[DecomposedFormalTheoremState]]
       List of DecomposedFormalTheoremState to process using map reduce.
    outputs: Required[Annotated[list[DecomposedFormalTheoremState], add]
       List of DecomposedFormalTheoremState that are the results of the map reduce
    """

    inputs: Required[list[DecomposedFormalTheoremState]]
    outputs: Required[Annotated[list[DecomposedFormalTheoremState], add]]  # TODO: Correct annotation?
