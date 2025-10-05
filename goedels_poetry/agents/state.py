from __future__ import annotations

from operator import add
from typing import Annotated

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
    formal_proof: Required[str | None]
        Formal proof of the formal_theorem
    proved: Required[bool]
        A bool indicating if the formal_proof is valid
    errors: Required[str | None]
        A string indicating errors in formal_proof
    ast: Required[AST | None]
        The AST of the formal_proof
    proof_attempts: Required[int]
        The number of attempts to prove formal_theorem
    proof_history: Required[Annotated[list[AnyMessage],add]]
        The history of messages sent and received from the LLMs
    """

    parent: Required[TreeNode | None]
    depth: Required[int]
    formal_theorem: Required[str]
    syntactic: Required[bool]
    formal_proof: Required[str | None]
    proved: Required[bool]
    errors: Required[str | None]
    ast: Required[AST | None]
    proof_attempts: Required[int]
    proof_history: Required[Annotated[list[AnyMessage], add]]  # TODO: Correct annotation?


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
    proof_sketch: Required[str | None]
        The formal sketch of the proof of formal_theorem
    syntactic: Required[bool]
        A bool indicating if proof_sketch is syntactically valid
    errors: Required[str | None]
        A string indicating errors in proof_sketch
    ast: Required[AST | None]
        The AST of the proof_sketch
    decomposition_attempts: Required[int]
        The number of decomposition attempts of formal_theorem
    decomposition_history: Required[Annotated[list[AnyMessage],add]]
        The history of messages sent and received from the LLMs
    """

    # InternalTreeNode specific properties
    parent: Required[TreeNode | None]
    children: Required[list[TreeNode]]
    depth: Required[int]
    # FormalTheorem specific properties
    formal_theorem: Required[str]
    # DecomposedFormalTheoremState specific properties
    proof_sketch: Required[str | None]
    syntactic: Required[bool]
    errors: Required[str | None]
    ast: Required[AST | None]
    decomposition_attempts: Required[int]
    decomposition_history: Required[Annotated[list[AnyMessage], add]]


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
