from __future__ import annotations

import logging
import os
import pickle
import uuid
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from kimina_client import KiminaClient

    from goedels_poetry.parsers.ast import AST

from goedels_poetry.agents.state import (
    DecomposedFormalTheoremState,
    DecomposedFormalTheoremStates,
    FormalTheoremProofState,
    FormalTheoremProofStates,
    InformalTheoremState,
)
from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    combine_preamble_and_body,
    ensure_mandatory_preamble,
    split_preamble_and_body,
)
from goedels_poetry.config.llm import (
    DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
    FORMALIZER_AGENT_MAX_RETRIES,
    PROVER_AGENT_MAX_DEPTH,
    PROVER_AGENT_MAX_PASS,
    PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
)

# Note: All LLM instances are imported from goedels_poetry.config.llm
from goedels_poetry.functools import maybe_save
from goedels_poetry.util.tree import InternalTreeNode, TreeNode, add_child, remove_child

logger = logging.getLogger(__name__)

# Global configuration for output directory
_OUTPUT_DIR = os.environ.get("GOEDELS_POETRY_DIR", os.path.expanduser("~/.goedels_poetry"))

# Configuration constants for proof reconstruction
# PROOF_BODY_INDENT_SPACES: Number of spaces to indent proof bodies in Lean4 code.
# Set to 2 to follow Lean4's standard indentation convention, where tactics inside
# a 'by' block are indented 2 spaces relative to the containing statement.
# Example:
#   theorem foo : P := by
#     have h : Q := by  -- indented 2 spaces
#       constructor     -- indented 4 spaces (2 from 'have', 2 more from 'by')
#     exact h
PROOF_BODY_INDENT_SPACES = 2

MISSING_FORMAL_PREAMBLE_MSG = "Formal theorems must include a Lean preamble/header (imports, options, etc.)."


class ProofReconstructionError(Exception):
    """Raised when proof reconstruction fails validation."""

    pass


class GoedelsPoetryState:
    def __init__(
        self,
        formal_theorem: str | None = None,
        informal_theorem: str | None = None,
        *,
        start_with_decomposition: bool = False,
    ):
        # Check that the proper number of arguments has been provided
        if (formal_theorem is None) and (informal_theorem is None):
            raise ValueError("Either 'formal_theorem' xor 'informal_theorem' must be provided")  # noqa: TRY003
        if (formal_theorem is not None) and (informal_theorem is not None):
            raise ValueError("Only one of 'formal_theorem' or 'informal_theorem' can be provided")  # noqa: TRY003

        # Debug mode: start formal theorem processing directly in decomposition (skips initial root syntax checking)
        self.start_with_decomposition: bool = start_with_decomposition

        # Introduce a bool to indicate if the proof is finished unable to be finished
        self.is_finished: bool = False

        # Introduce a string to hold the reason for finishing
        self.reason: str | None = None

        # Introduce a bool | None to hold the final proof validation result
        # True = validation passed, False = validation failed, None = validation not run or exception occurred
        self.proof_validation_result: bool | None = None

        # Kimina-guided reconstruction metadata (persisted in checkpoints)
        self.reconstruction_attempts: int = 0
        self.reconstruction_strategy_used: str | None = None

        # If set, this is the final complete proof text (including preamble) selected at finish-time.
        # This allows CLI output to write the same proof that passed final verification.
        self.final_complete_proof: str | None = None

        # Introduce a list of strings to hold the action history
        self.action_history: list[str] = []

        self._root_preamble: str | None = None

        # Initialize state with provided arguments
        self.formal_theorem_proof: TreeNode | None = None
        if formal_theorem is not None:
            preamble, body = split_preamble_and_body(formal_theorem)
            if not preamble.strip():
                if not start_with_decomposition:
                    raise ValueError(MISSING_FORMAL_PREAMBLE_MSG)
                # Debug-only path: allow missing preamble by synthesizing a default.
                preamble = ensure_mandatory_preamble(DEFAULT_IMPORTS)
                body = formal_theorem.strip()
            else:
                preamble = ensure_mandatory_preamble(preamble)

            self._root_preamble = preamble
            initial_formal_state = FormalTheoremProofState(
                id=uuid.uuid4().hex,
                parent=None,
                depth=0,
                formal_theorem=body,
                preamble=preamble,
                syntactic=False,
                formal_proof=None,
                proved=False,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name=None,
                hole_start=None,
                hole_end=None,
                llm_lean_output=None,
            )
            self.formal_theorem_proof = cast(TreeNode, initial_formal_state)
            theorem_for_metadata = combine_preamble_and_body(preamble, body)
        else:
            theorem_for_metadata = str(informal_theorem)

        # Initialize InformalTheoremState queues
        self.informal_formalizer_queue: InformalTheoremState | None = (
            None
            if informal_theorem is None
            else InformalTheoremState(
                informal_theorem=informal_theorem,
                formalization_attempts=0,
                formal_theorem=None,
                syntactic=False,
                semantic=False,
            )
        )
        self.informal_syntax_queue: InformalTheoremState | None = None
        self.informal_semantics_queue: InformalTheoremState | None = None

        # Initialize FormalTheoremProofState lists
        self.proof_syntax_queue: list[FormalTheoremProofState] = (
            [] if self.formal_theorem_proof is None else [cast(FormalTheoremProofState, self.formal_theorem_proof)]
        )
        self.proof_prove_queue: list[FormalTheoremProofState] = []
        self.proof_validate_queue: list[FormalTheoremProofState] = []
        self.proof_correct_queue: list[FormalTheoremProofState] = []
        self.proof_ast_queue: list[FormalTheoremProofState] = []

        # Initialize DecomposedFormalTheoremState lists
        self.decomposition_search_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_query_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_sketch_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_validate_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_correct_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_backtrack_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_ast_queue: list[DecomposedFormalTheoremState] = []
        self.decomposition_decompose_queue: list[
            DecomposedFormalTheoremState
        ] = []  # Calls AST.get_named_subgoal_code to get child postulates of sketch, creates a FormalTheoremProofState for each, and puts the FormalTheoremProofState in self.proof_syntax_queue

        # Initialize hidden parameter for tracking saves
        self._iteration = 0

        # Create theorem specific output directory
        theorem = theorem_for_metadata
        theorem_hash = self._hash_theorem(theorem)
        self._output_dir = os.path.join(_OUTPUT_DIR, theorem_hash)

        # Check if directory already exists
        if os.path.exists(self._output_dir):
            raise FileExistsError(  # noqa: TRY003
                f"Directory for theorem already exists: {self._output_dir}\n"
                f"Please use GoedelsPoetryState.load_latest(theorem='{theorem}') "
                f"to resume, or call GoedelsPoetryState.clear_theorem_directory('{theorem}') "
                f"to start fresh."
            )

        # Create the directory
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        # Store theorem metadata for discoverability
        theorem_file = os.path.join(self._output_dir, "theorem.txt")
        with open(theorem_file, "w", encoding="utf-8") as f:
            f.write(theorem)

    def __setstate__(self, state: dict) -> None:
        """
        Backward-compatible unpickling for older checkpoints.
        """
        self.__dict__.update(state)

        # Fields added after earlier releases: set defaults if missing.
        if not hasattr(self, "proof_validation_result"):
            self.proof_validation_result = None
        if not hasattr(self, "reconstruction_attempts"):
            self.reconstruction_attempts = 0
        if not hasattr(self, "reconstruction_strategy_used"):
            self.reconstruction_strategy_used = None
        if not hasattr(self, "final_complete_proof"):
            self.final_complete_proof = None

    @staticmethod
    def _hash_theorem(theorem: str) -> str:
        """
        Generate a hash string from the theorem for directory naming.

        Parameters
        ----------
        theorem : str
            The theorem string

        Returns
        -------
        str
            First 12 characters of SHA256 hash of the normalized theorem
        """
        normalized_theorem = GoedelsPoetryState._normalize_theorem(theorem)
        return sha256(normalized_theorem.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _normalize_theorem(theorem: str) -> str:
        """
        Normalize the theorem string for consistent hashing.

        Parameters
        ----------
        theorem : str
            The theorem string

        Returns
        -------
        str
            Normalized theorem string (stripped and lowercased)
        """
        return theorem.strip().lower()

    @classmethod
    def load_latest(cls, directory: str | None = None, theorem: str | None = None) -> GoedelsPoetryState | None:
        """
        Load the most recent checkpoint from the directory.

        Parameters
        ----------
        directory : Optional[str]
            Directory to search for checkpoints. Cannot be used with theorem parameter.
        theorem : Optional[str]
            Theorem to search checkpoints for. Cannot be used with directory parameter.

        Returns
        -------
        GoedelsPoetryState | None
            The loaded state object, or None if no checkpoints found

        Raises
        ------
        ValueError
            If both directory and theorem are provided, or if neither is provided
        """
        checkpoints = cls.list_checkpoints(directory=directory, theorem=theorem)
        if not checkpoints:
            return None

        return cls.load(checkpoints[0])  # Load the newest checkpoint

    @staticmethod
    def list_checkpoints(directory: str | None = None, theorem: str | None = None) -> list[str]:
        """
        List all available checkpoint files in the directory.

        Parameters
        ----------
        directory : Optional[str]
            Directory to search for checkpoints. Cannot be used with theorem parameter.
        theorem : Optional[str]
            Theorem to search checkpoints for. Cannot be used with directory parameter.

        Returns
        -------
        list[str]
            List of checkpoint filepaths, sorted by modification time (newest first)

        Raises
        ------
        ValueError
            If both directory and theorem are provided, or if neither is provided
        """
        if (directory is not None) and (theorem is not None):
            raise ValueError("Cannot specify both directory and theorem parameters")  # noqa: TRY003
        if (directory is None) and (theorem is None):
            raise ValueError("Must specify either directory or theorem parameter")  # noqa: TRY003

        if theorem is not None:
            theorem_hash = GoedelsPoetryState._hash_theorem(theorem)
            search_directory = os.path.join(_OUTPUT_DIR, theorem_hash)
        else:
            search_directory = str(directory)

        if not os.path.exists(search_directory):
            return []

        # Find all pickle files matching our naming pattern
        checkpoint_files = []
        for filename in os.listdir(search_directory):
            if filename.startswith("goedels_poetry_state_") and filename.endswith(".pkl"):
                filepath = os.path.join(search_directory, filename)
                checkpoint_files.append(filepath)

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        return checkpoint_files

    @classmethod
    def load(cls, filepath: str) -> GoedelsPoetryState:
        """
        Load a GoedelsPoetryState from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file to load

        Returns
        -------
        GoedelsPoetryState
            The loaded state object
        """
        with open(filepath, "rb") as f:
            return cast(GoedelsPoetryState, pickle.load(f))  # noqa: S301

    @classmethod
    def clear_theorem_directory(cls, theorem: str) -> str:
        """
        Clear the directory for a specific theorem.

        Parameters
        ----------
        theorem : str
            The research theorem whose directory should be cleared

        Returns
        -------
        str
            Confirmation message with the path that was cleared
        """
        theorem_hash = cls._hash_theorem(theorem)
        theorem_dir = os.path.join(_OUTPUT_DIR, theorem_hash)

        if os.path.exists(theorem_dir):
            rmtree(theorem_dir)
            return f"Successfully cleared directory: {theorem_dir}"
        else:
            return f"Directory does not exist: {theorem_dir}"

    def save(self) -> str:
        """
        Save the current state to a pickle file.

        Returns
        -------
        str
            Path to the saved checkpoint file
        """
        # Generate filename with datetime and iteration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"goedels_poetry_state_{timestamp}_iter_{self._iteration:04d}.pkl"
        filepath = os.path.join(self._output_dir, filename)

        # Save state to pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        # Increment iteration counter
        self._iteration += 1

        return filepath


class GoedelsPoetryStateManager:
    """
    Manager class for coordinating operations on GoedelsPoetryState.

    This class provides higher-level operations for managing the flow of the multi-agent pipeline.
    """

    def __init__(self, state: GoedelsPoetryState):
        """
        Initialize the manager with a GoedelsPoetryState.

        Parameters
        ----------
        state : GoedelsPoetryState
            The state object to manage
        """
        # This state should not be accessed directly. All the methods
        # that update the state have logic to save checkpoints.
        self._state = state

        # Debug-only behavior: allow starting from decomposition without initial syntax checking.
        if getattr(self._state, "start_with_decomposition", False):
            self.enqueue_for_decomposition_first()

    def enqueue_for_decomposition_first(self) -> None:
        """
        DEBUG: Convert the root FormalTheoremProofState into a DecomposedFormalTheoremState and
        seed the decomposition pipeline immediately.

        This intentionally skips initial root theorem syntax checking and proving.
        """
        if self._state.formal_theorem_proof is None:
            return

        root = cast(FormalTheoremProofState, self._state.formal_theorem_proof)

        # Ensure no syntax-validation work is pending for the root.
        self._state.proof_syntax_queue.clear()

        # Convert the root theorem to a decomposition node and enqueue it.
        self._queue_proofs_for_decomposition([root])

    @property
    def is_finished(self) -> bool:
        """
        A bool indicating if the proof process is finished
        """
        return self._state.is_finished

    @is_finished.setter
    def is_finished(self, is_finished: bool) -> None:
        """
        Setter for the bool is_finished

        Parameters
        ----------
        is_finished: bool
            New is_finished value
        """
        self._state.is_finished = is_finished

    @property
    def reason(self) -> str | None:
        """
        A string indicating the reason for finishing

        Returns
        -------
        str | None
            The reason for finishing, or None if not finished
        """
        return self._state.reason

    @reason.setter
    def reason(self, reason: str | None) -> None:
        """
        Setter for the reason string

        Parameters
        ----------
        reason: str | None
            The reason for finishing
        """
        self._state.reason = reason

    def add_action(self, action: str) -> None:
        """
        Adds the passed action to the action history

        Parameters
        ----------
        action: str
            The action to add to the action history
        """
        self._state.action_history.append(action)

    def get_informal_theorem_to_formalize(self) -> InformalTheoremState | None:
        """
        Gets the InformalTheoremState that needs to be formalized. This may be None if there is no
        InformalTheoremState that needs to be formalized.

        Returns
        -------
        InformalTheoremState
            The InformalTheoremState that needs to be formalized, may be None.
        """
        return self._state.informal_formalizer_queue

    @maybe_save(n=1)
    def set_formalized_informal_theorem(self, formalized_informal_theorem: InformalTheoremState) -> None:
        """
        Sets the InformalTheoremState that has been formalized. This InformalTheoremState may have
        a syntactically valid formalization or it may not be syntactically valid.

        Parameters
        ----------
        formalized_informal_theorem: InformalTheoremState
            The InformalTheoremState that has been formalized, may or may not be syntactic.
        """
        # Remove all elements from the formalizer queue
        self._state.informal_formalizer_queue = None

        # Check if this is a parse failure (formal_theorem is None indicates LLMParsingError)
        if formalized_informal_theorem["formal_theorem"] is None:
            # Increment formalization attempts
            formalized_informal_theorem["formalization_attempts"] += 1

            # Check if we've exceeded max attempts
            if formalized_informal_theorem["formalization_attempts"] >= FORMALIZER_AGENT_MAX_RETRIES:
                # Exceeded max attempts - finish with error
                self._state.is_finished = True
                self._state.reason = (
                    "Proof failed: Unable to formalize informal theorem - maximum formalization attempts exceeded."
                )
                return

            # Still within retry limit - requeue for retry
            self._state.informal_formalizer_queue = formalized_informal_theorem
            return

        # Successful parse - place formalized_informal_theorem on the queue to be syntactically validated
        self._state.informal_syntax_queue = formalized_informal_theorem

    def get_informal_theorem_to_validate(self) -> InformalTheoremState | None:
        """
        Gets the InformalTheoremState that needs to be validated syntactically. This may be None if
        there is no InformalTheoremState that needs to be validated syntactically.

        Returns
        -------
        InformalTheoremState
            The InformalTheoremState that needs to be validated syntactically, may be None.
        """
        return self._state.informal_syntax_queue

    @maybe_save(n=1)
    def set_validated_informal_theorem(self, validated_informal_theorem: InformalTheoremState) -> None:
        """
        Sets the InformalTheoremState that has been validated syntactically. This
        InformalTheoremState may be valid syntactically or invalid syntactically.

        Parameters
        ----------
        validated_informal_theorem: InformalTheoremState
            The InformalTheoremState that has been validated syntactically. It may be valid
            syntactically or invalid syntactically.
        """
        # Remove all elements from the syntax queue
        self._state.informal_syntax_queue = None

        # Check if validated_informal_theorem is syntactically valid
        if validated_informal_theorem["syntactic"]:
            # If it is, queue it for semantic validation
            self._state.informal_semantics_queue = validated_informal_theorem
        else:
            # If it isn't, queue it for re-formalization
            self._state.informal_formalizer_queue = validated_informal_theorem

        # In both cases increment the formalization attempts count
        validated_informal_theorem["formalization_attempts"] += 1

        # Set is_finished appropriately
        self._state.is_finished = validated_informal_theorem["formalization_attempts"] >= FORMALIZER_AGENT_MAX_RETRIES
        if self._state.is_finished:
            self._state.reason = (
                "Proof failed: Unable to formalize informal theorem - maximum formalization attempts exceeded."
            )

    def get_informal_theorem_to_check_semantics_of(self) -> InformalTheoremState | None:
        """
        Gets the InformalTheoremState that needs to have its semantics checked, making sure that
        the semantics of the informal statement matches that of the formal statement.

        Returns
        -------
        InformalTheoremState
           The InformalTheoremState to check the semantics of.
        """
        return self._state.informal_semantics_queue

    @maybe_save(n=1)
    def set_semantically_checked_informal_theorem(
        self, semantically_checked_informal_theorem: InformalTheoremState
    ) -> None:
        """
        Sets the InformalTheoremState that has been check semantically. This InformalTheoremState
        may be valid or invalid semantically.

        Parameters
        ----------
        semantically_checked_informal_theorem: InformalTheoremState
            The InformalTheoremState that has been check semantically, may be semantically invalid.
        """
        # Remove all elements from the semantics queue
        self._state.informal_semantics_queue = None

        # Check if semantically_checked_informal_theorem is semantically valid
        if semantically_checked_informal_theorem["semantic"]:
            # If it is semantically valid, create an associated FormalTheoremProofState
            default_preamble = ensure_mandatory_preamble(DEFAULT_IMPORTS)
            theorem_to_prove = FormalTheoremProofState(
                id=uuid.uuid4().hex,
                parent=None,
                depth=0,
                formal_theorem=str(semantically_checked_informal_theorem["formal_theorem"]),
                preamble=default_preamble,
                syntactic=semantically_checked_informal_theorem["syntactic"],
                formal_proof=None,
                proved=False,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                proof_history=[],
                pass_attempts=0,
                hole_name=None,
                hole_start=None,
                hole_end=None,
                llm_lean_output=None,
            )
            # Queue theorem_to_prove to be proven
            self._state.proof_prove_queue += [theorem_to_prove]
            # Set this FormalTheoremProofState as the root theorem to prove.
            self._state.formal_theorem_proof = cast(TreeNode, theorem_to_prove)
            if self._state._root_preamble is None:
                self._state._root_preamble = default_preamble
        else:
            # If it isn't semantically valid, queue it to be re-formalized
            self._state.informal_formalizer_queue = semantically_checked_informal_theorem

    def _reconstruct_tree(self, tree_nodes: list[TreeNode]) -> None:
        """
        Insert modified TreeNode instances back into the tree rooted at formal_theorem_proof.

        LangGraph Send API returns modified copies of TreeNodes; they share the same id as nodes
        in the current tree but are different objects. This method replaces nodes in the tree with
        their modified versions when present in tree_nodes, and fixes parent pointers so the tree
        remains consistent. The passed list need not contain the root (a node with parent None).

        Parameters
        ----------
        tree_nodes : list[TreeNode]
            Modified TreeNode instances (e.g. from mapreduce outputs) keyed by the same id as
            nodes in the current tree.
        """
        if not tree_nodes:
            return
        current_root = self._state.formal_theorem_proof
        if current_root is None:
            return
        id_to_modified = {cast(dict, n)["id"]: n for n in tree_nodes}

        # Replace root with modified version if present
        root_id = cast(dict, current_root)["id"]
        new_root = id_to_modified.get(root_id, current_root)
        self._state.formal_theorem_proof = new_root

        def replace_with_modified(node: TreeNode) -> None:
            if not isinstance(node, dict) or "children" not in node:
                return
            internal = cast(DecomposedFormalTheoremState, node)
            for cid in list(internal["children"].keys()):
                if cid in id_to_modified:
                    internal["children"][cid] = id_to_modified[cid]
                    cast(dict, id_to_modified[cid])["parent"] = node
                replace_with_modified(internal["children"][cid])

        replace_with_modified(new_root)

    def get_theorems_to_validate(self) -> FormalTheoremProofStates:
        """
        Gets a FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
        FormalTheoremProofState that need to have the syntax of their root theorem validated. This
        list may be empty.

        Returns
        -------
        FormalTheoremProofStates
            The FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
            FormalTheoremProofState that need their root theorems validated, may be empty.
        """
        return FormalTheoremProofStates(inputs=self._state.proof_syntax_queue, outputs=[])

    @maybe_save(n=1)
    def set_validated_theorems(self, validated_theorems: FormalTheoremProofStates) -> None:
        """
        Sets the FormalTheoremProofStates containing validated_theorems["outputs"] the list
        of root theorem validated FormalTheoremProofState's. Each list item's root theorem may have
        been sucessfully or unsuccessfully validated.

        Parameters
        ---------
        validated_theorems: FormalTheoremProofStates
            FormalTheoremProofStates containing validated_theorems["outputs"] the list of
            FormalTheoremProofState each of which has been validated sucessfully or unsuccessfully.
        """
        # Remove all elements from the syntax queue
        self._state.proof_syntax_queue.clear()

        # Reset proof tree root to modified FormalTheoremProofState
        self._reconstruct_tree(cast(list[TreeNode], validated_theorems["outputs"]))

        # Get FormalTheoremProofStates outputs
        validated_theorems_outputs = validated_theorems["outputs"]

        # For each sucessfully validated element queue it to be proven
        sucessfully_validated_theorems = [vt for vt in validated_theorems_outputs if vt["syntactic"]]
        self._state.proof_prove_queue += sucessfully_validated_theorems

        # Unsucessfully validated theorems are user supplied; we can't fix them. So finish
        self._state.is_finished = any((not vt["syntactic"]) for vt in validated_theorems_outputs)
        if self._state.is_finished:
            self._state.reason = "Proof failed: User-supplied formal theorem is syntactically invalid."

    def get_theorems_to_prove(self) -> FormalTheoremProofStates:
        """
        Gets a FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
        FormalTheoremProofState that need to be proven. This list man be empty.

        Returns
        -------
        FormalTheoremProofStates
            FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
            FormalTheoremProofState that need to be proven, may be empty.
        """
        return FormalTheoremProofStates(inputs=self._state.proof_prove_queue, outputs=[])

    @maybe_save(n=1)
    def set_proven_theorems(self, proven_theorems: FormalTheoremProofStates) -> None:
        """
        Sets the FormalTheoremProofStates containing proven_theorems["outputs"] the list
        of proven FormalTheoremProofState. The proof of each list item has yet to be validated or
        invalidated.

        Parameters
        ---------
        proven_theorems: FormalTheoremProofStates
            FormalTheoremProofStates containing proven_theorems["outputs"] the list of
            FormalTheoremProofState seach of which has been attempted to be proven.
        """
        # Remove all attempted proofs elements from the queue to be proven
        self._state.proof_prove_queue.clear()

        # Reset proof tree root to modified FormalTheoremProofState
        self._reconstruct_tree(cast(list[TreeNode], proven_theorems["outputs"]))

        # Partition outputs into parse failures and successful parses
        parse_failure_message = (
            "Malformed LLM response: unable to parse proof body from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        parse_failures = [
            pt
            for pt in proven_theorems["outputs"]
            if pt["formal_proof"] is None and pt["errors"] == parse_failure_message
        ]
        successful_parses = [
            pt
            for pt in proven_theorems["outputs"]
            if not (pt["formal_proof"] is None and pt["errors"] == parse_failure_message)
        ]

        # Handle parse failures: increment attempts, requeue or handle exhaustion
        for parse_failure in parse_failures:
            parse_failure["self_correction_attempts"] += 1

            # Check if we've exceeded max self-correction attempts
            if parse_failure["self_correction_attempts"] >= PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                # Exceeded max attempts - handle like a too-difficult proof
                parse_failure["pass_attempts"] += 1
                if parse_failure["pass_attempts"] < PROVER_AGENT_MAX_PASS:
                    # Restart self-correction loop: reset state, requeue for correction
                    self._reset_self_correction_state(parse_failure)
                    self._state.proof_prove_queue.append(parse_failure)
                else:
                    # Hit max_pass: queue for decomposition
                    self._queue_proofs_for_decomposition([parse_failure])
            else:
                # Still within retry limit - requeue for retry
                self._state.proof_prove_queue.append(parse_failure)

        # Handle successful parses - place attempted proofs in the queue of proofs to be validated
        self._state.proof_validate_queue += successful_parses

    def get_proofs_to_validate(self) -> FormalTheoremProofStates:
        """
        Gets a FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
        FormalTheoremProofState that have proofs that need to be validated. This list may be empty.

        Returns
        -------
        FormalTheoremProofStates
            FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
            FormalTheoremProofState that have proofs that need to be validated, may be an empty
            list.
        """
        return FormalTheoremProofStates(inputs=self._state.proof_validate_queue, outputs=[])

    @maybe_save(n=1)
    def set_validated_proofs(self, validated_proofs: FormalTheoremProofStates) -> None:
        """
        Sets the FormalTheoremProofStates containing validated_proofs["outputs"] the list of
        validated FormalTheoremProofState. Each list item's proof is marked as being valid or
        invalid.

        When a proof reaches max_self_correction_attempts and pass_attempts < max_pass,
        it is reset and routed to the prover queue (not corrector queue) to start a fresh
        proof attempt with the initial prompt.

        Parameters
        ---------
        validated_proofs: FormalTheoremProofStates
            FormalTheoremProofStates containing validated_proofs["outputs"] the list of
            FormalTheoremProofState each of which has its proof been validated or invalided.
        """
        # Remove all elements from the queue of proofs to validate
        self._state.proof_validate_queue.clear()

        # Reset proof tree root to modified FormalTheoremProofState
        self._reconstruct_tree(cast(list[TreeNode], validated_proofs["outputs"]))

        # Get validated_proofs outputs
        validated_proofs_outputs = validated_proofs["outputs"]

        # Increment the proof attempt count for all validated proofs
        for validated_proof in validated_proofs_outputs:
            validated_proof["self_correction_attempts"] += 1

        # Gather all unsuccessful proofs
        unsuccessful_proofs = [vp for vp in validated_proofs_outputs if (not vp["proved"])]

        proofs_too_difficult = []
        proofs_to_correct = []
        proofs_to_restart = []  # Proofs that have been reset and should bypass corrector

        for up in unsuccessful_proofs:
            # Note: We use >= because self_correction_attempts was incremented above
            # before this check. When attempts == max, we've exhausted the allowed attempts
            # (e.g., with max=2: 0->1 allows correction 1, 1->2 allows correction 2, 2->3 exhausts).
            if up["self_correction_attempts"] >= PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                up["pass_attempts"] += 1
                if up["pass_attempts"] < PROVER_AGENT_MAX_PASS:
                    # Restart self-correction loop: reset state, requeue for fresh proof attempt
                    self._reset_self_correction_state(up)
                    proofs_to_restart.append(up)  # Route to prover, not corrector
                else:
                    # Hit max_pass: queue for decomposition
                    proofs_too_difficult.append(up)
            else:
                # Still within a self-correction attempt cycle
                proofs_to_correct.append(up)

        # Queue proofs too difficult for decomposition
        self._queue_proofs_for_decomposition(proofs_too_difficult)
        # Queue proofs to correct for correction
        self._state.proof_correct_queue += proofs_to_correct
        # Queue reset proofs for fresh proof attempt (bypass corrector)
        self._state.proof_prove_queue += proofs_to_restart

        # Queue all successful proofs to have their ASTs generated
        successful_proofs = [vp for vp in validated_proofs_outputs if vp["proved"]]
        self._state.proof_ast_queue += successful_proofs

    def _reset_self_correction_state(self, proof: FormalTheoremProofState) -> None:
        """
        Resets the self-correction state for a proof so that a new self-correction pass starts cleanly.

        After resetting, the proof will be routed to the prover queue (not corrector queue)
        to start a fresh proof attempt with the initial prompt.
        """
        proof["self_correction_attempts"] = 0
        # Clear any derived artifacts from a previous attempt/pass so stale outputs can't leak
        # into later validation/reconstruction.
        proof["proved"] = False
        proof["errors"] = None
        proof["proof_history"] = []
        proof["llm_lean_output"] = None
        proof["formal_proof"] = None
        proof["ast"] = None

    def _queue_proofs_for_decomposition(self, proofs_too_difficult: list[FormalTheoremProofState]) -> None:
        """
        Queues the list of FormalTheoremProofState containing proofs too difficult to be decomposed.

        Parameters
        ----------
        proofs_too_difficult: list[FormalTheoremProofState]
            The lisr of FormalTheoremProofState containing proofs too difficult to be decomposed.
        """
        for proof_too_difficult in proofs_too_difficult:
            # Create a new DecomposedFormalTheoremState and add it to the search queue
            formal_theorem_to_decompose = DecomposedFormalTheoremState(
                id=uuid.uuid4().hex,
                parent=proof_too_difficult["parent"],
                children={},
                depth=proof_too_difficult["depth"],
                formal_theorem=proof_too_difficult["formal_theorem"],
                preamble=proof_too_difficult["preamble"],
                proof_sketch=None,
                syntactic=False,
                errors=None,
                ast=None,
                self_correction_attempts=0,
                decomposition_history=[],
                search_queries=None,
                search_results=None,
                # Preserve the parent's hole metadata so reconstruction can remain offset-based,
                # even after converting a leaf proof into a decomposed (internal) node.
                hole_name=proof_too_difficult.get("hole_name"),
                hole_start=proof_too_difficult.get("hole_start"),
                hole_end=proof_too_difficult.get("hole_end"),
                llm_lean_output=proof_too_difficult["llm_lean_output"],
            )
            self._state.decomposition_search_queue.append(formal_theorem_to_decompose)

            # Remove proof_too_difficult from the proof tree
            if proof_too_difficult["parent"] is not None:
                remove_child(
                    cast(InternalTreeNode, proof_too_difficult["parent"]),
                    cast(TreeNode, proof_too_difficult),
                )
                proof_too_difficult["parent"] = None

            # Check to see if formal_theorem_to_decompose is the root theorem
            if formal_theorem_to_decompose["parent"] is None:
                # If so, set the root to formal_theorem_to_decompose
                self._state.formal_theorem_proof = cast(TreeNode, formal_theorem_to_decompose)
            else:
                # If not, add formal_theorem_to_decompose as its parent's child
                add_child(
                    cast(InternalTreeNode, formal_theorem_to_decompose["parent"]),
                    cast(TreeNode, formal_theorem_to_decompose),
                )

    def get_proofs_to_correct(self) -> FormalTheoremProofStates:
        """
        Gets FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
        FormalTheoremProofState that have proofs that need to be corrected, may be and empty list.

        Returns
        -------
        FormalTheoremProofStates
            FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
            FormalTheoremProofState that have proofs that need to be corrected, may be and empty
            list.
        """
        return FormalTheoremProofStates(inputs=self._state.proof_correct_queue, outputs=[])

    @maybe_save(n=1)
    def set_corrected_proofs(self, corrected_proofs: FormalTheoremProofStates) -> None:
        """
        Sets the FormalTheoremProofStates containing corrected_proofs["outputs"] the list of
        FormalTheoremProofState with proofs that have been marked for correction using the errors
        from the previous proof attempt.

        Parameters
        ---------
        corrected_proofs: FormalTheoremProofStates
            FormalTheoremProofStates containing corrected_proofs["outputs"] the list of
            FormalTheoremProofState each of which has been marked for correction using
            the errors from the previous proof attempt.
        """
        # Remove all elements from the queue of proofs to correct
        self._state.proof_correct_queue.clear()

        # Reset proof tree root to modified FormalTheoremProofState
        self._reconstruct_tree(cast(list[TreeNode], corrected_proofs["outputs"]))

        # Place all proofs marked for correction into the queue to be proven
        self._state.proof_prove_queue += corrected_proofs["outputs"]

    def get_proofs_to_parse(self) -> FormalTheoremProofStates:
        """
        Gets FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] the list of
        FormalTheoremProofState that must be parsed to generate an AST, may be an empty list.

        Returns
        -------
        FormalTheoremProofStates
            FormalTheoremProofStates containing FormalTheoremProofStates["inputs"] list of
            FormalTheoremProofState with proofs that must be parsed into an AST, may be
            and empty list.
        """
        return FormalTheoremProofStates(inputs=self._state.proof_ast_queue, outputs=[])

    @maybe_save(n=1)
    def set_parsed_proofs(self, parsed_proofs: FormalTheoremProofStates) -> None:
        """
        Sets FormalTheoremProofStates containing parsed_proofs["outputs"] the list of
        FormalTheoremProofState with proofs with associated ASTs.

        Parameters
        ---------
        parsed_proofs: FormalTheoremProofStates
            FormalTheoremProofStates containing parsed_proofs["outputs"] the list of
            FormalTheoremProofState each of which has a proof associated AST.
        """
        # Remove all elements from the queue of proofs to generate ASTs for
        self._state.proof_ast_queue.clear()

        # Reset proof tree root to modified FormalTheoremProofState
        self._reconstruct_tree(cast(list[TreeNode], parsed_proofs["outputs"]))

        # TODO: Figure out how to deal with parent AST's. Doe we add this AST to ther parent here?
        #       If we do, the grandparent won't have this AST. So do we do so recursively? If we do
        #       when we find a decomposition or proof didn't work, we'll need to to lots of cleanup

    def get_theorems_for_search_query_generation(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing states that need search query generation.

        Returns
        -------
        DecomposedFormalTheoremStates
            States with search_queries=None that need query generation.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_search_queue, outputs=[])

    @maybe_save(n=1)
    def set_theorems_with_search_queries_generated(self, states_with_queries: DecomposedFormalTheoremStates) -> None:
        """
        Sets states with generated search queries and moves them to query queue.

        Parameters
        ----------
        states_with_queries: DecomposedFormalTheoremStates
            States with search_queries populated.
        """
        # Clear the search queue
        self._state.decomposition_search_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], states_with_queries["outputs"]))

        # Move states with queries to query queue (for vector DB lookup)
        self._state.decomposition_query_queue += states_with_queries["outputs"]

    def get_theorems_with_search_queries_for_vectordb(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing states that need vector database queries.

        Returns
        -------
        DecomposedFormalTheoremStates
            States with search_queries populated that need vector DB lookup.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_query_queue, outputs=[])

    @maybe_save(n=1)
    def set_theorems_with_vectordb_results(self, states_with_results: DecomposedFormalTheoremStates) -> None:
        """
        Sets states with vector database search results and moves them to sketch queue.

        Parameters
        ----------
        states_with_results: DecomposedFormalTheoremStates
            States with search_results populated.
        """
        # Clear the query queue
        self._state.decomposition_query_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], states_with_results["outputs"]))

        # Move states with results to sketch queue
        self._state.decomposition_sketch_queue += states_with_results["outputs"]

    def get_theorems_to_sketch(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState whose theorems were too difficult to prove head-on and
        thus must be decomposed into simpler theorems that entail the original theorem.

        Returns
        -------
        DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
            list of DecomposedFormalTheoremState whose theorems were too difficult to prove head-on
            and thus must be decomposed into simpler theorems.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_sketch_queue, outputs=[])

    @maybe_save(n=1)
    def set_sketched_theorems(self, sketched_theorems: DecomposedFormalTheoremStates) -> None:
        """
        Sets the DecomposedFormalTheoremStates containing sketched_theorems["outputs"] the list of
        DecomposedFormalTheoremState whose theorems have been decomposed into simpler theorems.

        Parameters
        ----------
        sketched_theorems: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing sketched_theorems["outputs"] the list of
            DecomposedFormalTheoremState whose theorems have been decomposed into simpler
            theorems.
        """
        # Remove all elements from the queue of theorems to sketch
        self._state.decomposition_sketch_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], sketched_theorems["outputs"]))

        # Partition outputs into parse failures and successful parses
        parse_failure_message = (
            "Malformed LLM response: unable to parse proof sketch from LLM output. "
            "The response did not contain a valid Lean4 code block or the code block could not be extracted."
        )
        parse_failures = [
            st
            for st in sketched_theorems["outputs"]
            if st["proof_sketch"] is None and st["errors"] == parse_failure_message
        ]
        successful_parses = [
            st
            for st in sketched_theorems["outputs"]
            if not (st["proof_sketch"] is None and st["errors"] == parse_failure_message)
        ]

        # Handle parse failures: increment attempts, requeue or handle exhaustion
        for parse_failure in parse_failures:
            parse_failure["self_correction_attempts"] += 1

            # Check if we've exceeded max self-correction attempts
            if parse_failure["self_correction_attempts"] >= DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                # Exceeded max attempts - handle like a failed sketch (backtrack or finish)
                self._handle_failed_sketch(parse_failure)
            else:
                # Still within retry limit - requeue for retry
                self._state.decomposition_sketch_queue.append(parse_failure)

        # Handle successful parses - place all sketched theorems into the queue of sketches to be validated
        self._state.decomposition_validate_queue += successful_parses

    def get_sketches_to_validate(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState containing sketches the syntax of which must be
        validated.

        Returns
        -------
        DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
            list of DecomposedFormalTheoremState containing sketches the syntax of which must
            be validated.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_validate_queue, outputs=[])

    @maybe_save(n=1)
    def set_validated_sketches(self, validated_sketches: DecomposedFormalTheoremStates) -> None:
        """
        Sets DecomposedFormalTheoremStates containing validated_sketches["outputs"] the list of
        DecomposedFormalTheoremState whose decompositions have been syntactically determined to
        be valid or invalid.

        Parameters
        ----------
        validated_sketches: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing validated_sketches["outputs"] the list of
            DecomposedFormalTheoremState whose decompositions have been syntactically
            determined to be valid or invalid.
        """
        # Remove all elements from the queue of decompositions to validate
        self._state.decomposition_validate_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], validated_sketches["outputs"]))

        # Get validated_sketches outputs
        validated_sketches_outputs = validated_sketches["outputs"]

        # Increment the decomposition attempt count
        for validated_sketch in validated_sketches_outputs:
            validated_sketch["self_correction_attempts"] += 1

        # Gather all invalid sketches
        invalid_sketches = [vs for vs in validated_sketches_outputs if (not vs["syntactic"])]

        # Partition invalid sketches into those too difficult to decompose and those to correct
        # Note: We use >= because self_correction_attempts was incremented above (line 930)
        # before this check. When attempts == max, we've exhausted the allowed attempts
        # (e.g., with max=6: after 6 correction attempts, counter reaches 6 and we stop).
        sketches_too_difficult = [
            ivs
            for ivs in invalid_sketches
            if (ivs["self_correction_attempts"] >= DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS)
        ]
        sketches_to_correct = [
            ivs
            for ivs in invalid_sketches
            if (ivs["self_correction_attempts"] < DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS)
        ]

        # Addd sketches to correct to the correction queue
        self._state.decomposition_correct_queue += sketches_to_correct

        # Handle sketches that are too difficult - try backtracking
        for sketch_too_difficult in sketches_too_difficult:
            self._handle_failed_sketch(sketch_too_difficult)

        # Gather all valid sketches and add them to the queue of sketches to parse into an AST
        valid_sketches = [vs for vs in validated_sketches_outputs if vs["syntactic"]]
        self._state.decomposition_ast_queue += valid_sketches

    def _find_backtrackable_ancestor(self, node: DecomposedFormalTheoremState) -> DecomposedFormalTheoremState | None:
        """
        Find the nearest ancestor (closest to the failed node) that has self_correction_attempts
        less than DECOMPOSER_AGENT_MAX_SELF_CORRECTIONS. Returns None if no such ancestor exists.

        Parameters
        ----------
        node : DecomposedFormalTheoremState
            The node from which to start searching upward

        Returns
        -------
        DecomposedFormalTheoremState | None
            The nearest backtrackable ancestor, or None if none exists
        """
        current = node["parent"]
        while current is not None:
            # Check if current is a DecomposedFormalTheoremState (has 'children' attribute)
            if isinstance(current, dict) and "children" in current:
                decomposed_current = cast(DecomposedFormalTheoremState, current)
                if decomposed_current["self_correction_attempts"] < DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                    return decomposed_current
            current = current["parent"] if isinstance(current, dict) else None
        return None

    def _find_backtrackable_grandparent_or_higher(
        self, child: FormalTheoremProofState
    ) -> DecomposedFormalTheoremState | None:
        """
        Find a backtrackable ancestor that is at least a grandparent of the given child.
        This is used when a child exceeds max depth - we need to backtrack at least to the
        grandparent level to avoid the same depth problem if we just re-decompose the parent.

        Parameters
        ----------
        child : FormalTheoremProofState
            The child node that is too deep

        Returns
        -------
        DecomposedFormalTheoremState | None
            A backtrackable ancestor at grandparent level or higher, or None if none exists
        """
        # Get the parent (the DecomposedFormalTheoremState that created this child)
        parent = child["parent"]
        if parent is None:
            return None

        # Get the grandparent (parent's parent)
        grandparent = parent["parent"] if isinstance(parent, dict) else None
        if grandparent is None:
            return None

        # Now search from the grandparent upward for a backtrackable ancestor
        # We use _find_backtrackable_ancestor but we need to ensure we're searching from grandparent
        # Since _find_backtrackable_ancestor starts from node["parent"], we need to create
        # a temporary node structure or search manually
        current = grandparent
        while current is not None:
            # Check if current is a DecomposedFormalTheoremState (has 'children' attribute)
            if isinstance(current, dict) and "children" in current:
                decomposed_current = cast(DecomposedFormalTheoremState, current)
                if decomposed_current["self_correction_attempts"] < DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                    return decomposed_current
            current = current["parent"] if isinstance(current, dict) else None
        return None

    def _collect_all_descendants(self, node: TreeNode) -> list[TreeNode]:
        """
        Recursively collect all descendants of a node in the tree.

        Parameters
        ----------
        node : TreeNode
            The node whose descendants to collect

        Returns
        -------
        list[TreeNode]
            List of all descendant nodes (children, grandchildren, etc.)
        """
        descendants: list[TreeNode] = []
        # Check if this is an internal node with children
        if isinstance(node, dict) and "children" in node:
            internal_node = cast(DecomposedFormalTheoremState, node)
            for child in internal_node["children"].values():
                descendants.append(child)
                # Recursively collect descendants of this child
                descendants.extend(self._collect_all_descendants(child))
        return descendants

    def _remove_proof_node_from_queues(self, proof_node: FormalTheoremProofState) -> None:
        """
        Remove a proof node from all proof queues.

        Parameters
        ----------
        proof_node : FormalTheoremProofState
            The proof node to remove
        """
        if proof_node in self._state.proof_syntax_queue:
            self._state.proof_syntax_queue.remove(proof_node)
        if proof_node in self._state.proof_prove_queue:
            self._state.proof_prove_queue.remove(proof_node)
        if proof_node in self._state.proof_validate_queue:
            self._state.proof_validate_queue.remove(proof_node)
        if proof_node in self._state.proof_correct_queue:
            self._state.proof_correct_queue.remove(proof_node)
        if proof_node in self._state.proof_ast_queue:
            self._state.proof_ast_queue.remove(proof_node)

    def _remove_decomposition_node_from_queues(self, decomp_node: DecomposedFormalTheoremState) -> None:
        """
        Remove a decomposition node from all decomposition queues.

        Parameters
        ----------
        decomp_node : DecomposedFormalTheoremState
            The decomposition node to remove
        """
        if decomp_node in self._state.decomposition_sketch_queue:
            self._state.decomposition_sketch_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_validate_queue:
            self._state.decomposition_validate_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_correct_queue:
            self._state.decomposition_correct_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_backtrack_queue:
            self._state.decomposition_backtrack_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_search_queue:
            self._state.decomposition_search_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_query_queue:
            self._state.decomposition_query_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_ast_queue:
            self._state.decomposition_ast_queue.remove(decomp_node)
        if decomp_node in self._state.decomposition_decompose_queue:
            self._state.decomposition_decompose_queue.remove(decomp_node)

    def _remove_nodes_from_all_queues(self, nodes: list[TreeNode]) -> None:
        """
        Remove the specified nodes from all proof and decomposition queues.

        Parameters
        ----------
        nodes : list[TreeNode]
            List of nodes to remove from all queues
        """
        for node in nodes:
            # Try to remove from proof queues
            if isinstance(node, dict) and "formal_proof" in node:
                self._remove_proof_node_from_queues(cast(FormalTheoremProofState, node))

            # Try to remove from decomposition queues
            if isinstance(node, dict) and "children" in node:
                self._remove_decomposition_node_from_queues(cast(DecomposedFormalTheoremState, node))

    def _prepare_node_for_resketching(self, node: DecomposedFormalTheoremState) -> None:
        """
        Prepare a node for re-sketching by clearing its children, sketch, AST, and errors.
        The decomposition_history and decomposition_attempts are preserved.

        Parameters
        ----------
        node : DecomposedFormalTheoremState
            The node to prepare for re-sketching
        """
        # Backtracking means we are restarting the sketch/decompose loop for this node.
        # Reset any attempt counter and derived artifacts so stale values cannot leak
        # into the next sketching/parsing/validation cycle.
        node["self_correction_attempts"] = 0
        node["llm_lean_output"] = None

        # Clear children (they will be removed from tree separately)
        node["children"] = {}
        # Defensive: clear sketch "result" fields
        node["proof_sketch"] = None
        node["syntactic"] = False
        node["errors"] = None
        node["ast"] = None
        # Clear search queries and results to force regeneration on backtrack
        node["search_queries"] = None
        node["search_results"] = None

    def _handle_failed_sketch(self, failed_sketch: DecomposedFormalTheoremState) -> None:
        """
        Handle a sketch that has exceeded max decomposition attempts by attempting to backtrack
        to the nearest ancestor that can be re-sketched. If no such ancestor exists, sets
        is_finished to True.

        Parameters
        ----------
        failed_sketch : DecomposedFormalTheoremState
            The sketch that has failed and exceeded max attempts
        """
        # Try to find a backtrackable ancestor
        backtrack_target = self._find_backtrackable_ancestor(failed_sketch)

        if backtrack_target is None:
            # No backtrackable ancestor found - we've exhausted all options
            self._state.is_finished = True
            self._state.reason = "Proof failed: Unable to decompose theorem - all decomposition attempts exhausted."
            return

        # We found an ancestor to backtrack to - perform the backtracking
        # 1. Collect all descendants of the backtrack target (to be removed)
        descendants = self._collect_all_descendants(cast(TreeNode, backtrack_target))

        # 2. Remove all descendants from all queues
        self._remove_nodes_from_all_queues(descendants)

        # 3. Remove the backtrack target itself from all queues (it might be in query_queue, sketch_queue, etc.)
        self._remove_decomposition_node_from_queues(backtrack_target)

        # 4. Prepare the backtrack target for re-sketching
        self._prepare_node_for_resketching(backtrack_target)

        # 5. Queue the backtrack target for re-sketching
        self._state.decomposition_backtrack_queue.append(backtrack_target)

    def get_sketches_to_correct(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState containing sketches determined to be syntactically
        invalid, may be an empty list.

        Returns
        -------
        DecomposedFormalTheoremStates
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_correct_queue, outputs=[])

    def get_sketches_to_backtrack(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState that need to be re-sketched due to failed children,
        may be an empty list.

        Returns
        -------
        DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
            list of DecomposedFormalTheoremState that need backtrack re-sketching.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_backtrack_queue, outputs=[])

    @maybe_save(n=1)
    def set_corrected_sketches(self, corrected_sketches: DecomposedFormalTheoremStates) -> None:
        """
        Sets DecomposedFormalTheoremStates containing corrected_sketches["outputs"] the list of
        DecomposedFormalTheoremState with sketchesthat have been marked for correction using the
        errors from the previous proof attempt.

        Parameters
        ----------
        corrected_sketches: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing corrected_sketches["outputs"] the list of
            DecomposedFormalTheoremState with sketchesthat have been marked for correction using
            the errors from the previous proof attempt.
        """
        # Remove all elements from the queue of sketches to correct
        self._state.decomposition_correct_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], corrected_sketches["outputs"]))

        # Place all sketches marked for correction into the queue to be sketched
        self._state.decomposition_sketch_queue += corrected_sketches["outputs"]

    @maybe_save(n=1)
    def set_backtracked_sketches(self, backtracked_sketches: DecomposedFormalTheoremStates) -> None:
        """
        Sets DecomposedFormalTheoremStates containing backtracked_sketches["outputs"] the list of
        DecomposedFormalTheoremState that have been re-sketched due to failed children attempts.

        Parameters
        ----------
        backtracked_sketches: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing backtracked_sketches["outputs"] the list of
            DecomposedFormalTheoremState that have been re-sketched due to failed children.
        """
        # Remove all elements from the queue of sketches to backtrack
        self._state.decomposition_backtrack_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], backtracked_sketches["outputs"]))

        # Place all backtracked sketches into the search queue to regenerate queries
        # (search_queries was cleared in _prepare_node_for_resketching)
        self._state.decomposition_search_queue += backtracked_sketches["outputs"]

    def get_sketches_to_parse(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState that must be parsed to generate an AST, may be an
        empty list.

        Returns
        -------
        DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
            list of DecomposedFormalTheoremState that must be parsed to generate an AST, may be
            an empty list.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_ast_queue, outputs=[])

    @maybe_save(n=1)
    def set_parsed_sketches(self, parsed_sketches: DecomposedFormalTheoremStates) -> None:
        """
        Sets DecomposedFormalTheoremStates containing parsed_sketches["outputs"] the list of
        DecomposedFormalTheoremState with sketches with associated ASTs.

        Parameters
        ----------
        parsed_sketches: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing parsed_sketches["outputs"] The list of
            DecomposedFormalTheoremState each of which has a sketch associated AST.
        """
        # Remove all elements from the queue of elements to parse
        self._state.decomposition_ast_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], parsed_sketches["outputs"]))

        # TODO: Figure out how to deal with parent AST's. Doe we add this AST to ther parent here?
        #       If we do, the grandparent won't have this AST. So do we do so recursively? If we do
        #       when we find a decomposition or proof didn't work, we'll need to to lots of cleanup

        # Add parsed_sketches to the queue of sketches to decompose into entailing FormalTheoremProofState's
        self._state.decomposition_decompose_queue += parsed_sketches["outputs"]

    def get_sketches_to_decompose(self) -> DecomposedFormalTheoremStates:
        """
        Gets DecomposedFormalTheoremStates containing DecomposedFormalTheoremStates["inputs"] the
        list of DecomposedFormalTheoremState ready to be decomposed into dependant
        FormalTheoremProofState's that entail their parent DecomposedFormalTheoremState.

        Returns
        -------
        DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containiing DecomposedFormalTheoremStates["inputs"] the
            list of DecomposedFormalTheoremState ready to be decomposed into dependant
            FormalTheoremProofState's that entail their parent DecomposedFormalTheoremState.
        """
        return DecomposedFormalTheoremStates(inputs=self._state.decomposition_decompose_queue, outputs=[])

    @maybe_save(n=1)
    def set_decomposed_sketches(self, decomposed_sketches: DecomposedFormalTheoremStates) -> None:
        """
        Sets DecomposedFormalTheoremStates containing decomposed_sketches["outputs"] the list of
        DecomposedFormalTheoremState that have been decomposed into dependant
        FormalTheoremProofState's that entail their parent DecomposedFormalTheoremState.

        Parameters
        ----------
        decomposed_sketches: DecomposedFormalTheoremStates
            DecomposedFormalTheoremStates containing decomposed_sketches["outputs"] the list of
            DecomposedFormalTheoremState that have been decomposed into dependant
            FormalTheoremProofState's that entail their parent DecomposedFormalTheoremState.
        """
        # Remove all elements from the queue of elements to decompose
        self._state.decomposition_decompose_queue.clear()

        # Reset proof tree root to modified DecomposedFormalTheoremState
        self._reconstruct_tree(cast(list[TreeNode], decomposed_sketches["outputs"]))

        # Gather all children FormalTheoremProofState's that need to be proven
        all_children = [
            cast(FormalTheoremProofState, dt) for ds in decomposed_sketches["outputs"] for dt in ds["children"].values()
        ]

        # Identify children that are too deep
        too_deep_children = [child for child in all_children if child["depth"] >= PROVER_AGENT_MAX_DEPTH]

        # Handle too-deep children by attempting to backtrack to grandparent or higher
        if too_deep_children:
            # Track which backtrack targets we've already processed (to avoid duplicates)
            # Use id() since DecomposedFormalTheoremState is a dict and not hashable
            processed_backtrack_target_ids: set[int] = set()
            has_backtrackable_ancestor = False

            for too_deep_child in too_deep_children:
                # Find a backtrackable ancestor at grandparent level or higher
                backtrack_target = self._find_backtrackable_grandparent_or_higher(too_deep_child)

                if backtrack_target is not None:
                    has_backtrackable_ancestor = True

                    # Only process each backtrack target once
                    backtrack_target_id = id(backtrack_target)
                    if backtrack_target_id not in processed_backtrack_target_ids:
                        processed_backtrack_target_ids.add(backtrack_target_id)

                        # Collect all descendants of the backtrack target (to be removed)
                        descendants = self._collect_all_descendants(cast(TreeNode, backtrack_target))

                        # Remove all descendants from all queues
                        self._remove_nodes_from_all_queues(descendants)

                        # Remove the backtrack target itself from all queues (it might be in query_queue, sketch_queue, etc.)
                        self._remove_decomposition_node_from_queues(backtrack_target)

                        # Prepare the backtrack target for re-sketching
                        self._prepare_node_for_resketching(backtrack_target)

                        # Queue the backtrack target for re-sketching
                        self._state.decomposition_backtrack_queue.append(backtrack_target)

            # Only finish if no backtrackable ancestors were found
            if not has_backtrackable_ancestor:
                self._state.is_finished = True
                self._state.reason = (
                    "Proof failed: Maximum proof tree depth exceeded and no backtrackable ancestors found."
                )
            else:
                # Queue children that are NOT too deep (too-deep ones will be recreated after backtracking)
                # Use id() for comparison to avoid recursion issues with dict comparison
                too_deep_child_ids = {id(child) for child in too_deep_children}
                not_too_deep_children = [child for child in all_children if id(child) not in too_deep_child_ids]
                self._state.proof_prove_queue += not_too_deep_children
        else:
            # No too-deep children, queue all children normally
            self._state.proof_prove_queue += all_children

    def _reconstruct_node_proof_ast_based(  # noqa: C901
        self,
        node: TreeNode,
        *,
        kimina_client: KiminaClient,
        server_timeout: int = 60,
    ) -> str:
        """
        Recursively reconstruct proof for a node using AST-based methods.

        For leaf nodes: returns proof text directly.
        For decomposed nodes: uses AST to find holes and replace them with child proofs.

        Parameters
        ----------
        node : TreeNode
            The node to reconstruct (FormalTheoremProofState or DecomposedFormalTheoremState)
        kimina_client
            Required client for syntax validation and semantic validation
        server_timeout : int
            Timeout for Kimina requests

        Returns
        -------
        str
            Complete proof text (without preamble)
        """
        from typing import cast

        from goedels_poetry.agents.state import (
            DecomposedFormalTheoremState,
            FormalTheoremProofState,
        )
        from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
        from goedels_poetry.agents.util.kimina_server import (
            parse_kimina_ast_code_response,
            parse_kimina_check_response,
        )
        from goedels_poetry.parsers.ast import AST
        # ProofReconstructionError is defined at module level in goedels_poetry/state.py (same file as this method)

        # Leaf node: return proof text directly
        # Note: formal_proof contains only tactics (proof body after := by), not full theorem declaration
        if isinstance(node, dict) and "formal_proof" in node and "children" not in node:
            proof_state = cast(FormalTheoremProofState, node)
            if proof_state["formal_proof"] is None:
                raise ProofReconstructionError(  # noqa: TRY003
                    "Leaf node has formal_proof is None. "
                    "This violates the assumption that all FormalTheoremProofState instances are proven."
                )

            proof_text = str(proof_state["formal_proof"])

            # For root nodes (parent is None), wrap tactics with theorem signature if needed
            # Root nodes need full theorem declaration for reconstruct_complete_proof() output
            # Non-root nodes return tactics as-is (will be extracted for holes)
            if proof_state["parent"] is None:
                # Root node: ensure output includes theorem header
                # Note: Uses instance methods _strip_decl_assignment(), _skip_leading_trivia(),
                # _indent_proof_body(), and constant PROOF_BODY_INDENT_SPACES from GoedelsPoetryStateManager
                theorem_decl_full = str(proof_state["formal_theorem"]).strip()
                theorem_sig = self._strip_decl_assignment(theorem_decl_full).strip()

                # Validate theorem signature is not empty
                if not theorem_sig:
                    raise ProofReconstructionError(f"Invalid theorem declaration for root node: {theorem_decl_full}")  # noqa: TRY003

                # Skip leading empty lines and single-line comments to avoid redundant wrapping
                leading_skipped = self._skip_leading_trivia(proof_text)

                # Use normalized string comparison to handle multiline theorem signatures
                # This handles cases where theorem signature and proof_text have different formatting
                # (e.g., single-line vs multiline signatures)
                normalized_sig = " ".join(theorem_sig.split())
                normalized_leading = " ".join(leading_skipped.split())
                if normalized_leading.startswith(normalized_sig):
                    # Already has theorem signature, return as-is
                    return proof_text

                # Otherwise treat stored proof as tactics and wrap with theorem signature
                indent = " " * PROOF_BODY_INDENT_SPACES
                indented_body = self._indent_proof_body(proof_text, indent)
                return f"{theorem_sig} := by\n{indented_body}"

            # Non-root nodes: return tactics as-is (will be extracted by _extract_proof_body_ast_guided() for holes)
            return proof_text

        # Internal node (DecomposedFormalTheoremState): reconstruct by replacing holes
        if isinstance(node, dict) and "children" in node:
            decomposed_state = cast(DecomposedFormalTheoremState, node)

            if decomposed_state["proof_sketch"] is None:
                raise ProofReconstructionError(  # noqa: TRY003
                    "Decomposed node has proof_sketch is None. "
                    "This violates the assumption that decomposed nodes have syntactic sketches."
                )

            if decomposed_state["ast"] is None:
                raise ProofReconstructionError(  # noqa: TRY003
                    "Decomposed node has ast is None. This violates the assumption that decomposed nodes have ASTs."
                )

            # Get preamble once at the start (used for validation and reconstruction)
            # Note: This preamble is used consistently throughout reconstruction to ensure coordinate system consistency
            # with the AST (which was created using this same preamble)
            preamble = decomposed_state.get("preamble", DEFAULT_IMPORTS)

            # Normalize sketch: strip to match what combine_preamble_and_body produces
            # This ensures coordinate system consistency with AST (positions are relative to stripped body)
            sketch = str(decomposed_state["proof_sketch"]).strip()
            sketch = sketch if sketch.endswith("\n") else sketch + "\n"

            # Get AST (should already be created from normalized sketch by sketch_parser_agent)
            ast = cast(AST, decomposed_state["ast"])

            # Verify AST was created from normalized sketch
            # This ensures coordinate system consistency - hole positions must align
            ast_source_text = ast.get_source_text()
            ast_body_start = ast.get_body_start()
            # Validate ast_body_start is within bounds (>= 0 and <= len(ast_source_text))
            if ast_source_text and 0 <= ast_body_start <= len(ast_source_text):
                expected_body = ast_source_text[ast_body_start:]
                if sketch != expected_body:
                    raise ProofReconstructionError(  # noqa: TRY003
                        f"Sketch body does not match AST source_text body. This indicates AST was created from different sketch. "
                        f"Expected length {len(expected_body)}, got {len(sketch)}."
                    )

            # Replace holes using AST-based method
            # Pass the preamble that matches the AST (from decomposed_state)
            # This ensures coordinate system consistency - body_start calculations match
            reconstructed = self._replace_holes_using_ast(
                sketch,
                ast,
                list(decomposed_state["children"].values()),
                kimina_client=kimina_client,
                server_timeout=server_timeout,
                preamble=preamble,  # Use the preamble that was used to create the AST
            )

            # Final syntax validation: parse reconstructed proof
            # (incremental validation during hole replacement ensures this should pass)
            reconstructed_with_preamble = combine_preamble_and_body(preamble, reconstructed)

            # Note: combine_preamble_and_body() should already normalize (strip and add trailing newline),
            # but we add this normalization defensively in case the implementation changes.
            normalized_reconstructed = (
                reconstructed_with_preamble
                if reconstructed_with_preamble.endswith("\n")
                else reconstructed_with_preamble + "\n"
            )

            # Import parsing functions (used for validation)
            from goedels_poetry.agents.util.kimina_server import (
                parse_kimina_ast_code_response,
                parse_kimina_check_response,
            )

            ast_response = kimina_client.ast_code(normalized_reconstructed, timeout=server_timeout)
            parsed_ast = parse_kimina_ast_code_response(ast_response)
            if parsed_ast.get("error") is not None:
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Final syntax validation failed after node reconstruction: {parsed_ast['error']}\n"
                    f"Reconstructed proof:\n{reconstructed}"
                )

            # Final semantic validation: type-check reconstructed proof
            # (semantic validation during hole replacement ensures this should pass)
            check_response = kimina_client.check(normalized_reconstructed, timeout=server_timeout)
            parsed_check = parse_kimina_check_response(check_response)
            if not parsed_check.get("complete", False):
                errors = parsed_check.get("errors", [])
                sorries = parsed_check.get("sorries", [])
                error_parts = []
                if errors:
                    error_parts.append("Errors:\n" + "\n".join(err.get("data", str(err)) for err in errors))
                if sorries:
                    error_parts.append(f"Sorries remaining: {len(sorries)}")
                error_msg = "\n\n".join(error_parts) if error_parts else "Unknown semantic validation failure"
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Final semantic validation failed after node reconstruction: {error_msg}\n"
                    f"Reconstructed proof:\n{reconstructed}"
                )

            return reconstructed

        # Unknown node type
        raise ProofReconstructionError("Unable to reconstruct proof for unknown node type.")  # noqa: TRY003

    def _replace_holes_using_ast(  # noqa: C901
        self,
        sketch: str,
        ast: AST,
        children: list[TreeNode],
        *,
        kimina_client: KiminaClient,
        server_timeout: int = 60,
        preamble: str = DEFAULT_IMPORTS,
    ) -> str:
        """
        Replace sorry holes in sketch using AST-determined positions with iterative indentation refinement.

        Required imports:
        - from kimina_client import KiminaClient
        - from goedels_poetry.agents.util.kimina_server import (
            parse_kimina_ast_code_response,
            parse_kimina_check_response,
        )
        - from goedels_poetry.agents.util.common import (
            combine_preamble_and_body,
            DEFAULT_IMPORTS,
            remove_default_imports_from_ast,
        )
        - from goedels_poetry.parsers.ast import AST
        - PROOF_BODY_INDENT_SPACES, _dedent_proof_body, _indent_proof_body from goedels_poetry.state
        - ProofReconstructionError is defined at module level in goedels_poetry/state.py (same file as this method)
        (These can be imported as instance methods/attributes since this is a method of GoedelsPoetryStateManager)

        Uses AST.get_sorry_holes_by_name() for Unicode-safe positions and AST structure
        for indentation context. Tries multiple indentation strategies until syntax validation succeeds.

        Guarantees: Under assumptions (syntactic sketches, proven children), at least one
        indentation strategy will produce syntactically valid code.

        Parameters
        ----------
        sketch : str
            The sketch text (body only, no preamble)
        ast : AST
            The AST for the sketch (must have been created with the same preamble)
        children : list[TreeNode]
            List of child nodes with hole_name, hole_start, hole_end metadata
        kimina_client
            Required client for syntax validation and semantic validation after replacement
        server_timeout : int
            Timeout for syntax validation and semantic validation requests
        preamble : str
            The preamble to use when combining with body for validation.
            Must match the preamble used when creating the AST to ensure coordinate system consistency.

        Returns
        -------
        str
            Sketch with holes replaced by child proofs (guaranteed syntactically and semantically valid)
        """

        from goedels_poetry.agents.util.common import (
            combine_preamble_and_body,
            remove_default_imports_from_ast,
        )
        from goedels_poetry.agents.util.kimina_server import (
            parse_kimina_ast_code_response,
            parse_kimina_check_response,
        )
        from goedels_poetry.parsers.ast import AST
        from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
            extract_outer_scope_variables_ast_based,
            extract_variables_with_origin,
        )
        from goedels_poetry.parsers.util.collection_and_analysis.variable_renaming import (
            rename_conflicting_variables_ast_based,
        )
        # ProofReconstructionError is defined at module level in goedels_poetry/state.py (same file as this method)

        # Sort children by hole_start position to ensure they match textual order of holes
        # This ensures occurrence index matching works correctly (first child → first hole, etc.)
        # Filter children with hole_name first, then sort by hole_start
        children_with_holes = []
        for child in children:
            child_hole_name = child.get("hole_name") if isinstance(child, dict) else None
            if not child_hole_name:
                # Under assumptions, all children should have hole_name (including synthetic names)
                raise ProofReconstructionError(  # noqa: TRY003
                    "Child node is missing hole_name. All children must have hole_name (including synthetic names). "
                    "This violates the assumption that each sorry has a corresponding proven child."
                )

            child_hole_start = child.get("hole_start") if isinstance(child, dict) else None
            if child_hole_start is None:
                # Under assumptions, children should have hole_start metadata
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Child node with hole_name {child_hole_name} is missing hole_start metadata. "
                    f"This violates the assumption that children have complete hole metadata."
                )
            if not isinstance(child_hole_start, int):
                # Under assumptions, hole_start should be an int (character offset)
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Child node with hole_name {child_hole_name} has invalid hole_start type: {type(child_hole_start)}. "
                    f"Expected int (character offset), got {type(child_hole_start)}."
                )

            children_with_holes.append((child_hole_start, child))

        # Sort by hole_start to match textual order (holes appear left-to-right in sketch)
        children_with_holes.sort(key=lambda t: t[0])
        sorted_children = [child for _, child in children_with_holes]

        # Validate that we have valid children after filtering
        if not sorted_children:
            raise ProofReconstructionError(  # noqa: TRY003
                f"After filtering, no valid children remain. All children must have hole_name and hole_start. "
                f"Original children count: {len(children)}"
            )

        # Track occurrence indices for each hole name (which occurrence we're on)
        children_by_name: dict[str, list[TreeNode]] = {}
        for child in sorted_children:
            child_hole_name = child.get("hole_name") if isinstance(child, dict) else None
            if child_hole_name:
                children_by_name.setdefault(child_hole_name, []).append(child)

        occurrence_indices: dict[str, int] = dict.fromkeys(children_by_name.keys(), 0)

        # Validate that sketch (stripped) matches AST source_text body
        # This ensures coordinate system consistency - hole positions must align
        ast_source_text = ast.get_source_text()
        ast_body_start = ast.get_body_start()
        # Normalize sketch first (needed regardless of validation path)
        normalized_sketch = sketch.strip()
        normalized_sketch = normalized_sketch if normalized_sketch.endswith("\n") else normalized_sketch + "\n"

        # Validate ast_body_start is within bounds (>= 0 and <= len(ast_source_text))
        if ast_source_text and 0 <= ast_body_start <= len(ast_source_text):
            expected_body = ast_source_text[ast_body_start:]
            if normalized_sketch != expected_body:
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Sketch body does not match AST source_text body. This indicates AST was created from different sketch. "
                    f"Expected length {len(expected_body)}, got {len(normalized_sketch)}."
                )

        # Validate that sketch has holes if children exist (upfront validation)
        initial_holes = ast.get_sorry_holes_by_name()
        total_initial_holes = sum(len(spans) for spans in initial_holes.values())
        if total_initial_holes == 0 and len(children) > 0:
            raise ProofReconstructionError(  # noqa: TRY003
                f"Sketch has no holes but {len(children)} children provided. "
                f"This violates the assumption that each sorry has a corresponding proven child."
            )
        if total_initial_holes > 0 and len(children) == 0:
            raise ProofReconstructionError(  # noqa: TRY003
                f"Sketch has {total_initial_holes} hole(s) but no children provided. "
                f"This violates the assumption that each sorry has a corresponding proven child."
            )

        # Use the normalized_sketch as result
        # This ensures coordinate system consistency (AST positions are relative to stripped body)
        result = normalized_sketch

        # Phase 3: Extract outer scope variables BEFORE processing children
        # This provides the baseline for conflict detection
        outer_scope_vars = extract_outer_scope_variables_ast_based(
            result,
            ast,
            kimina_client,
            server_timeout,
        )

        # Process children one at a time in sorted order, re-querying holes from AST after each replacement
        for child in sorted_children:
            child_hole_name = child.get("hole_name") if isinstance(child, dict) else None
            # child_hole_name is guaranteed to exist from the filtering above

            # Get fresh hole positions from current AST (positions update after each replacement)
            holes = ast.get_sorry_holes_by_name()

            # Sort holes within each name by start position to ensure textual order
            # This is critical because AST traversal order may not match textual order
            # Note: We modify holes in-place for this iteration only; holes are re-queried
            # at the start of each loop iteration, so modifications don't persist
            for name in holes:
                holes[name] = sorted(holes[name], key=lambda span: span[0])

            hole_spans = holes.get(child_hole_name or "", [])

            if not hole_spans:
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Hole {child_hole_name} not found in AST. This may indicate all holes of this name "
                    f"have been replaced, or the AST is in an inconsistent state."
                )

            # Match child to hole by name + occurrence index (not position)
            # First child with this name (in sorted order) matches first hole with this name, etc.
            # Defensive check: child_hole_name should always be in occurrence_indices after filtering
            if child_hole_name not in occurrence_indices:
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Child with hole_name {child_hole_name} not found in occurrence_indices. "
                    f"This should not happen after filtering - all children should have valid hole_name."
                )
            occurrence_idx = occurrence_indices[child_hole_name]
            if occurrence_idx >= len(hole_spans):
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Occurrence index {occurrence_idx} for hole {child_hole_name} exceeds available "
                    f"holes ({len(hole_spans)}). This indicates a mismatch between children and holes."
                )

            hole_start, hole_end = hole_spans[occurrence_idx]

            # Validate hole position types (defensive check)
            if not isinstance(hole_start, int) or not isinstance(hole_end, int):
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Invalid hole position types for {child_hole_name}: "
                    f"start={type(hole_start)}, end={type(hole_end)}. Expected int, int."
                )

            # Validate hole positions are within bounds of result
            if hole_start < 0 or hole_end > len(result) or hole_start >= hole_end:
                # Debug info for coordinate system mismatch
                ast_source_text = ast.get_source_text()
                ast_body_start = ast.get_body_start()
                child_hole_start = child.get("hole_start") if isinstance(child, dict) else None
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Invalid hole position for {child_hole_name}: start={hole_start}, end={hole_end}, "
                    f"result_length={len(result)}. This indicates coordinate system mismatch. "
                    f"AST body_start={ast_body_start}, AST source_text length={len(ast_source_text) if ast_source_text else None}, "
                    f"child hole_start={child_hole_start}."
                )

            # Phase 4: Analyze proof structure (function types and applications)
            # Note: The analysis["should_preserve_application"] flag indicates when
            # function application structure should be preserved instead of inlining.
            # The actual preservation logic is a TODO - for now, we proceed with
            # normal inlining (with variable renaming from Phase 3).
            # Future enhancement: When should_preserve_application is True, preserve
            # the function application structure in the final proof.
            _ = self._analyze_proof_structure_ast_based(
                child,
                ast,
                child_hole_name or "",
                kimina_client,
                server_timeout,
            )

            # Extract child proof body
            child_proof_body = self._extract_proof_body_ast_guided(
                child, kimina_client=kimina_client, server_timeout=server_timeout
            )

            # Validate proof body is not empty
            if not child_proof_body or not child_proof_body.strip():
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Child proof body for hole {child_hole_name} is empty or whitespace-only. "
                    f"This should not happen under assumptions (proven child proofs)."
                )

            # Phase 3: Rename conflicting variables before inlining
            child_proof_body = rename_conflicting_variables_ast_based(
                child_proof_body,
                outer_scope_vars,
                kimina_client,
                server_timeout,
                child_hole_name or "",
            )

            # Phase 1: Extract indentation context from AST
            # Use result (already normalized at start) since positions are relative to current state
            # Positions from AST are based on normalized text, so we must use normalized text here
            base_indent, is_inline = self._extract_indentation_from_ast(ast, hole_start, result)

            # Validate return values from _extract_indentation_from_ast() (defensive check)
            # This provides an additional safety layer in case the method returns invalid values
            if not isinstance(base_indent, int) or base_indent < 0:
                # Fall back to text-based calculation for base_indent
                # Validate hole_start is within bounds before using it
                if not isinstance(hole_start, int) or hole_start < 0 or hole_start > len(result):
                    raise ProofReconstructionError(  # noqa: TRY003
                        f"Invalid hole_start ({hole_start}) for fallback base_indent calculation. "
                        f"result length: {len(result)}"
                    )
                line_start = result.rfind("\n", 0, hole_start) + 1
                line_prefix = result[line_start:hole_start]
                base_indent = len(line_prefix) - len(line_prefix.lstrip())
                # Only recalculate is_inline if it's also invalid
                if not isinstance(is_inline, bool):
                    is_inline = bool(line_prefix.strip())
                # Ensure base_indent is non-negative after fallback
                base_indent = max(0, base_indent)
            elif not isinstance(is_inline, bool):
                # base_indent is valid but is_inline is invalid - default to non-inline
                is_inline = False

            # Phase 2: Try indentation strategies until syntax validation succeeds
            replacement_text = None
            strategies = self._generate_indentation_strategies(child_proof_body, base_indent, is_inline)

            for _strategy_idx, indent_strategy in enumerate(strategies):
                # FIX: For multiline holes, strip trailing whitespace from prefix to avoid double indentation
                # The prefix already contains the indentation spaces, and indent_strategy adds them again.
                # By stripping trailing whitespace matching base_indent, we prevent double indentation.
                # This is backward-compatible because:
                # - Inline holes (is_inline=True): prefix doesn't end with matching whitespace, so no change
                # - Multiline holes (is_inline=False): prefix ends with whitespace, we strip it before replacement
                prefix = result[:hole_start]
                if not is_inline and base_indent > 0:
                    # For multiline holes, check if prefix ends with whitespace that matches base_indent
                    # Strip only the trailing whitespace characters (spaces), preserving newlines
                    prefix_rstripped = prefix.rstrip()
                    trailing_whitespace_count = len(prefix) - len(prefix_rstripped)
                    if trailing_whitespace_count >= base_indent and len(prefix) >= base_indent:
                        # Prefix has at least base_indent trailing spaces
                        # Strip exactly base_indent spaces to prevent double indentation
                        # Find the position to cut: go back base_indent characters from end, but preserve newlines
                        # Check if the last base_indent chars are all spaces
                        last_chars = prefix[-base_indent:]
                        if last_chars.strip() == "" and last_chars.count("\n") == 0:
                            # Last base_indent chars are spaces (no newlines)
                            # Strip them to prevent double indentation
                            prefix = prefix[:-base_indent]

                test_result = prefix + indent_strategy + result[hole_end:]

                # Syntax validation
                # Normalize test_result: ensure trailing newline before passing to combine_preamble_and_body
                # combine_preamble_and_body will strip leading/trailing whitespace and re-add trailing newline if needed
                # This matches combine_preamble_and_body's behavior (strips input, then adds newline if needed)
                # Note: normalized_test is test_result with trailing newline (if needed)
                # normalized_body (below) is the stripped version used for body_start calculation
                normalized_test = test_result if test_result.endswith("\n") else test_result + "\n"
                test_with_preamble = combine_preamble_and_body(preamble, normalized_test)

                test_response = kimina_client.check(test_with_preamble, timeout=server_timeout)
                test_check = parse_kimina_check_response(test_response)

                ast_response = kimina_client.ast_code(test_with_preamble, timeout=server_timeout)
                parsed_ast = parse_kimina_ast_code_response(ast_response)

                if (
                    parsed_ast.get("error") is None
                    and parsed_ast.get("ast") is not None
                    and bool(test_check.get("pass", False))
                ):
                    # Candidate strategy parses and has no check() errors.
                    #
                    # IMPORTANT: We still need to validate that the updated AST preserves the
                    # remaining sorry holes (total hole count should decrease by exactly 1).
                    # Some indentation strategies can change parsing structure such that remaining
                    # `sorry` tokens are no longer represented as tactic-sorry nodes, which would
                    # break subsequent replacements. Treat such cases as a strategy failure and
                    # try the next strategy.
                    original_result = result
                    original_ast = ast
                    original_outer_scope_vars = dict(outer_scope_vars)

                    # Update AST for subsequent replacements (AST is guaranteed to exist at this point)

                    ast_without_imports = remove_default_imports_from_ast(parsed_ast["ast"], preamble=preamble)

                    # Calculate body_start correctly: combine_preamble_and_body strips preamble and adds "\n\n"
                    # Handle edge cases where preamble or body might be empty
                    # Use the preamble parameter (matches the AST's preamble) to ensure coordinate system consistency
                    normalized_preamble = preamble.strip()
                    # normalized_body is the stripped version of normalized_test (what was passed to combine_preamble_and_body)
                    normalized_body = normalized_test.strip()

                    if not normalized_preamble:
                        body_start = 0
                    elif not normalized_body:
                        # Should not happen (empty body), but handle it
                        body_start = len(test_with_preamble)
                    else:
                        # Both non-empty: preamble + "\n\n" + body
                        body_start = len(normalized_preamble) + 2  # +2 for "\n\n"

                    # Validate body_start is within bounds (defensive check)
                    if body_start < 0 or body_start > len(test_with_preamble):
                        raise ProofReconstructionError(  # noqa: TRY003
                            f"Invalid body_start ({body_start}) calculated. "
                            f"test_with_preamble length: {len(test_with_preamble)}, "
                            f"normalized_preamble length: {len(normalized_preamble)}, "
                            f"normalized_body length: {len(normalized_body)}"
                        )

                    # Extract the stripped body from test_with_preamble to ensure coordinate system consistency
                    # combine_preamble_and_body does: normalized_body = body.strip()
                    # So we need to store the stripped version in result
                    # Handle empty body case explicitly
                    source_text = test_with_preamble
                    if not normalized_body:
                        # Empty body case: combine_preamble_and_body returns just preamble
                        # So body portion is empty (no body)
                        stripped_body = ""
                    else:
                        # Normal case: extract body from test_with_preamble
                        if source_text and body_start <= len(source_text):
                            stripped_body = source_text[body_start:]
                        else:
                            # Fallback: extract from normalized_test (shouldn't happen)
                            # Match combine_preamble_and_body behavior: strip body, then add newline if needed
                            stripped_body = normalized_test.strip()
                            if not stripped_body.endswith("\n"):
                                stripped_body += "\n"

                    # Ensure stripped_body has trailing newline only if it's non-empty
                    # combine_preamble_and_body adds trailing newline to result, but empty body stays empty
                    if stripped_body:
                        result = stripped_body if stripped_body.endswith("\n") else stripped_body + "\n"
                    else:
                        # Empty body case: result should be empty string (no body)
                        result = ""

                    ast = AST(
                        ast_without_imports,
                        sorries=parsed_ast.get("sorries"),
                        source_text=test_with_preamble,
                        body_start=body_start,
                    )

                    # Verify that calculated body_start matches AST's body_start (if available)
                    # This ensures coordinate system consistency between our calculation and the AST
                    ast_body_start = ast.get_body_start()
                    if ast_body_start is not None and ast_body_start != body_start:
                        raise ProofReconstructionError(  # noqa: TRY003
                            f"Calculated body_start ({body_start}) does not match AST's body_start ({ast_body_start}). "
                            f"This indicates a coordinate system mismatch in body_start calculation."
                        )

                    # Verify coordinate system consistency: result should match body portion of source_text
                    # This is the primary verification - result is what's actually used for subsequent replacements
                    ast_source_text = ast.get_source_text()
                    source_text_body = ast_source_text[body_start:] if ast_source_text else ""
                    if result != source_text_body:
                        raise ProofReconstructionError(  # noqa: TRY003
                            f"Result body does not match AST source_text body after replacement. "
                            f"This indicates a coordinate system mismatch. Expected length {len(source_text_body)}, "
                            f"got {len(result)}."
                        )

                    # Verify updated AST correctly reflects the replaced hole
                    # Re-query holes to confirm the replaced hole is gone and positions are updated
                    updated_holes = ast.get_sorry_holes_by_name()

                    # Sort updated holes to ensure textual order
                    for name in updated_holes:
                        updated_holes[name] = sorted(updated_holes[name], key=lambda span: span[0])

                    updated_spans = updated_holes.get(child_hole_name, [])
                    expected_remaining = len(hole_spans) - 1  # One hole should be gone

                    # Verify hole count for this name decreased by 1
                    if len(updated_spans) != expected_remaining:
                        # Strategy produced an AST that doesn't reflect the replacement correctly.
                        # Restore state and try the next indentation strategy.
                        result = original_result
                        ast = original_ast
                        outer_scope_vars = original_outer_scope_vars
                        continue

                    # Phase 3: Update outer_scope_vars with new variables from this replacement
                    # (for subsequent replacements)
                    # Use check() to get ALL new variables, not just have/let
                    child_proof_with_preamble = combine_preamble_and_body(preamble, child_proof_body)
                    child_check_response = kimina_client.check(child_proof_with_preamble, timeout=server_timeout)
                    child_parsed_check = parse_kimina_check_response(child_check_response)

                    # Extract ALL variables from check() response with origin information
                    # Need to create AST for the child proof to determine origins
                    child_ast_response = kimina_client.ast_code(child_proof_with_preamble, timeout=server_timeout)
                    child_parsed_ast = parse_kimina_ast_code_response(child_ast_response)
                    child_ast_for_vars = None
                    if child_parsed_ast.get("ast"):
                        child_ast_without_imports = remove_default_imports_from_ast(
                            child_parsed_ast["ast"], preamble=preamble
                        )
                        child_ast_for_vars = AST(
                            child_ast_without_imports,
                            sorries=child_parsed_ast.get("sorries"),
                            source_text=child_proof_with_preamble,
                            body_start=len(preamble.strip()) + 2 if preamble.strip() else 0,
                        )

                    new_vars = (
                        extract_variables_with_origin(child_parsed_check, child_ast_for_vars)
                        if child_ast_for_vars
                        else []
                    )
                    for var_info in new_vars:
                        # Only add proof body variables to outer scope (not lemma parameters)
                        # Lemma parameters are part of the signature and don't need to be tracked
                        if var_info.get("is_lemma_parameter", False):
                            continue
                        var_name = var_info["name"]
                        declaration_node = var_info.get("declaration_node")
                        declaration_pos = None
                        if declaration_node:
                            info = declaration_node.get("info", {})
                            if isinstance(info, dict):
                                pos = info.get("pos")
                                if isinstance(pos, list) and len(pos) >= 2:
                                    declaration_pos = (pos[0], pos[1])

                        outer_scope_vars[var_name] = {
                            "name": var_name,
                            "type": var_info.get("type"),
                            "hypothesis": var_info["hypothesis"],
                            "declaration_node": declaration_node,
                            "declaration_pos": declaration_pos,
                            "is_lemma_parameter": False,  # These are proof body variables
                            "is_proof_body_variable": True,
                            "source": "check_response",
                        }

                    # Verify total hole count decreased by 1
                    total_holes_before = sum(len(spans) for spans in holes.values())
                    total_holes_after = sum(len(spans) for spans in updated_holes.values())
                    if total_holes_after != total_holes_before - 1:
                        # Strategy changed parsing such that remaining holes are no longer detected.
                        # Restore state and try the next indentation strategy.
                        result = original_result
                        ast = original_ast
                        outer_scope_vars = original_outer_scope_vars
                        continue

                    # All validation passed - commit this strategy.
                    replacement_text = indent_strategy

                    # Increment occurrence index only after successful replacement
                    occurrence_indices[child_hole_name] += 1

                    break

            if replacement_text is None:
                # All indentation strategies failed syntax validation
                # This should not happen under assumptions
                raise ProofReconstructionError(  # noqa: TRY003
                    f"All indentation strategies failed syntax validation for hole {child_hole_name} "
                    f"at position {hole_start}-{hole_end}. This indicates a bug or violated assumption."
                )

        # Verify all holes have been replaced (children list should match all holes)
        # If children is empty but sketch had holes, this would have been caught earlier
        remaining_holes = ast.get_sorry_holes_by_name()
        total_remaining = sum(len(spans) for spans in remaining_holes.values())
        if total_remaining > 0:
            raise ProofReconstructionError(  # noqa: TRY003
                f"After processing all children, {total_remaining} hole(s) remain unreplaced. "
                f"This violates the assumption that each sorry has a corresponding proven child. "
                f"Remaining holes: {list(remaining_holes.keys())}"
            )

        # Verify final result is not empty (should not happen under assumptions)
        if not result or not result.strip():
            raise ProofReconstructionError(  # noqa: TRY003
                "Final reconstructed proof body is empty or whitespace-only. This should not happen under assumptions."
            )

        # REQUIRED: Semantic validation AFTER all holes are replaced
        # (not after each replacement, because other holes may still remain)
        # result is already normalized (has trailing newline if non-empty) from last replacement
        # (empty result would have been caught by check at lines above)
        # Use the preamble parameter (matches the AST's preamble) to ensure coordinate system consistency
        result_with_preamble = combine_preamble_and_body(preamble, result)

        check_response = kimina_client.check(result_with_preamble, timeout=server_timeout)
        parsed_check = parse_kimina_check_response(check_response)
        if not parsed_check.get("complete", False):
            errors = parsed_check.get("errors", [])
            sorries = parsed_check.get("sorries", [])
            error_parts = []
            if errors:
                error_parts.append("Errors:\n" + "\n".join(err.get("data", str(err)) for err in errors))
            if sorries:
                error_parts.append(f"Sorries remaining: {len(sorries)}")
            error_msg = "\n\n".join(error_parts) if error_parts else "Unknown semantic validation failure"
            # Include reconstructed proof in error for debugging
            error_msg += f"\n\nReconstructed proof body (first 500 chars):\n{result[:500]}"
            error_msg += f"\n\nFull proof with preamble (first 800 chars):\n{result_with_preamble[:800]}"
            raise ProofReconstructionError(f"Semantic validation failed after all hole replacements: {error_msg}")  # noqa: TRY003

        return result

    def _extract_indentation_from_ast(self, ast: AST, hole_start: int, sketch: str) -> tuple[int, bool]:
        """
        Extract indentation context from AST structure around hole position.

        Uses AST structure analysis to determine correct indentation from parent nodes
        and token info fields. Falls back to text-based calculation if AST analysis fails.

        Returns (base_indent, is_inline) where:
        - base_indent: Character offset of base indentation level
        - is_inline: True if hole is inline (e.g., "by sorry"), False if standalone
        """
        # Try AST-based extraction first
        source_text = ast.get_source_text()
        body_start = ast.get_body_start()

        if source_text is not None:
            # Convert body-relative hole_start to full-text position
            # (Placeholder for future AST-based extraction - not used yet)
            _full_text_hole_start = body_start + hole_start

            # Find the AST node containing the sorry token at this position
            # Navigate up to find parent container (by block, have statement, etc.)
            # (Placeholder for future AST-based extraction - not used yet)
            _ast_dict = ast.get_ast()

            # Extract indentation from parent node structure
            # Look for parent nodes with 'info' fields containing leading/trailing whitespace
            # The 'by' token's trailing whitespace often contains newline + indentation
            # This indicates the expected indentation for the proof body

            # For now, use text-based fallback, but structure allows for AST enhancement
            # Implementation should:
            # 1. Find sorry token node at hole_start position
            # 2. Navigate to parent 'by' or 'have' node
            # 3. Extract trailing whitespace from 'by' token's info field
            # 4. Parse newline + indentation pattern from trailing whitespace

        # Fallback: text-based calculation
        line_start = sketch.rfind("\n", 0, hole_start) + 1
        line_prefix = sketch[line_start:hole_start]
        base_indent = len(line_prefix) - len(line_prefix.lstrip())
        is_inline = bool(line_prefix.strip())

        # Ensure base_indent is non-negative (defensive check)
        # This ensures the return value is always valid, even if text-based calculation somehow fails
        base_indent = max(0, base_indent)

        # Ensure is_inline is a bool (defensive check)
        if not isinstance(is_inline, bool):
            is_inline = False

        return base_indent, is_inline

    def _generate_indentation_strategies(self, child_proof_body: str, base_indent: int, is_inline: bool) -> list[str]:
        """
        Generate list of indentation strategies to try, in order of preference.

        Under assumptions, at least one strategy must produce syntactically valid code.

        Uses the following from goedels_poetry.state:
        - PROOF_BODY_INDENT_SPACES: constant (currently 2, defined in goedels_poetry/state.py)
        - self._dedent_proof_body(): method (defined in GoedelsPoetryStateManager class in goedels_poetry/state.py)
        - self._indent_proof_body(): method (defined in GoedelsPoetryStateManager class in goedels_poetry/state.py)

        Note: These can remain as instance methods/attributes since this is a method of GoedelsPoetryStateManager.
        Alternatively, they could be moved to a utility module and imported if desired.
        """
        # Ensure base_indent is non-negative (defensive check)
        # This protects against negative values even if validation above failed
        base_indent = max(0, base_indent)

        strategies: list[str] = []

        # Strategy 1: AST-calculated base indentation (most likely to work)
        if is_inline:
            indent = " " * (base_indent + PROOF_BODY_INDENT_SPACES)
            strategies.append("\n" + self._indent_proof_body(child_proof_body, indent))
        else:
            indent = " " * base_indent
            strategies.append(self._indent_proof_body(child_proof_body, indent))

        # Strategy 2: Preserve relative indentation, align base
        dedented = self._dedent_proof_body(child_proof_body)
        if is_inline:
            indent = " " * (base_indent + PROOF_BODY_INDENT_SPACES)
            strategies.append("\n" + self._indent_proof_body(dedented, indent))
        else:
            indent = " " * base_indent
            strategies.append(self._indent_proof_body(dedented, indent))

        # Strategy 3: Try different base levels (±2, ±4)
        for offset in [-4, -2, 2, 4]:
            test_indent = max(0, base_indent + offset)
            if is_inline:
                indent = " " * (test_indent + PROOF_BODY_INDENT_SPACES)
                strategies.append("\n" + self._indent_proof_body(dedented, indent))
            else:
                indent = " " * test_indent
                strategies.append(self._indent_proof_body(dedented, indent))

        # Strategy 4: For inline holes, try keeping inline (no newline)
        if is_inline:
            strategies.append(" " + child_proof_body.strip())

        # Strategy 5: Uniform indentation from base
        if is_inline:
            indent = " " * (base_indent + PROOF_BODY_INDENT_SPACES)
            # Preserve empty lines (don't convert to empty string)
            uniform_body = "\n".join(indent + line if line.strip() else line for line in dedented.split("\n"))
            strategies.append("\n" + uniform_body)

        return strategies

    def _extract_proof_body_ast_guided(
        self,
        child: TreeNode,
        *,
        kimina_client: KiminaClient,
        server_timeout: int = 60,
    ) -> str:
        """
        Extract proof body from child node.

        Uses `formal_proof` directly for leaf nodes (guaranteed to contain complete
        proof body). For decomposed nodes, recursively reconstructs proof.

        Required imports for this method:
        - import re
        - from typing import cast
        - from kimina_client import KiminaClient
        - from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body, remove_default_imports_from_ast
        - from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response
        - from goedels_poetry.agents.state import (
            FormalTheoremProofState,
            DecomposedFormalTheoremState,
            TreeNode,
        )
        - from goedels_poetry.parsers.ast import AST
        - ProofReconstructionError is defined at module level in goedels_poetry/state.py (same file as this method)

        Parameters
        ----------
        child : TreeNode
            The child node (FormalTheoremProofState or DecomposedFormalTheoremState)

        Returns
        -------
        str
            The proof body (tactics) as a string
        """
        from typing import cast

        from goedels_poetry.agents.state import (
            DecomposedFormalTheoremState,
            FormalTheoremProofState,
        )
        from goedels_poetry.agents.util.common import (
            DEFAULT_IMPORTS,
            combine_preamble_and_body,
            remove_default_imports_from_ast,
        )
        from goedels_poetry.agents.util.kimina_server import parse_kimina_ast_code_response
        from goedels_poetry.parsers.ast import AST
        from goedels_poetry.parsers.util.foundation.decl_extraction import (
            extract_proof_body_from_ast,
            extract_signature_from_ast,
        )

        if isinstance(child, dict) and "formal_proof" in child and "children" not in child:
            # Leaf node (FormalTheoremProofState)
            proof_state = cast(FormalTheoremProofState, child)
            if proof_state["formal_proof"] is None:
                # Under assumptions, proven nodes should never have formal_proof is None
                raise ProofReconstructionError(  # noqa: TRY003
                    f"Leaf node with hole_name {proof_state.get('hole_name')} has formal_proof is None. "
                    f"This violates the assumption that all FormalTheoremProofState instances are proven "
                    f"({proof_state.get('proved')} should be True)."
                )

            proof_text = str(proof_state["formal_proof"])

            # Use formal_proof directly - it's already the complete proof body
            # formal_proof is extracted from the proven theorem by _parse_prover_response
            # and contains only the proof body (tactics after ":= by")
            #
            # Verification: _parse_prover_response ALWAYS returns just the proof body
            # (verified via focused Python program - all tests passed)
            #
            # IMPORTANT: Preserve leading indentation.
            # Many tactic bodies rely on alignment (e.g. nested `have ... := by` blocks where a later
            # `apply h` must be aligned with the `have` line to be *outside* the nested `by` block).
            # Stripping leading whitespace from only the first line (while leaving subsequent lines
            # indented) breaks that relative structure and can make all indentation strategies fail.
            return proof_text.rstrip()

        elif isinstance(child, dict) and "children" in child:
            # Internal node (DecomposedFormalTheoremState) - recursively reconstruct
            # Note: decomposed_state is a DecomposedFormalTheoremState which is a valid TreeNode
            decomposed_state = cast(DecomposedFormalTheoremState, child)
            complete_proof = self._reconstruct_node_proof_ast_based(
                decomposed_state,  # type: ignore[arg-type]
                kimina_client=kimina_client,
                server_timeout=server_timeout,
            )

            # Extract tactics after ":=" by from complete proof
            # Note: decomposed_state.get("ast") is the AST of the sketch (with sorry holes), not the reconstructed proof.
            # We need to re-parse complete_proof to get its AST for accurate extraction.
            #
            # Important: For decomposed nodes, `_reconstruct_node_proof_ast_based()` returns the reconstructed
            # sketch *body* (i.e. the full theorem/lemma declaration without the preamble), because
            # `DecomposedFormalTheoremState["proof_sketch"]` stores the complete declaration body.
            #
            # We combine it with the node's preamble so Kimina can parse it and we can structurally
            # extract the tactics after `:= by`.
            #
            # Get preamble from decomposed_state for combining with complete_proof for parsing
            preamble = decomposed_state.get("preamble", DEFAULT_IMPORTS)

            # Normalize complete_proof and combine with preamble for parsing
            normalized_proof = complete_proof if complete_proof.endswith("\n") else complete_proof + "\n"
            proof_with_preamble = combine_preamble_and_body(preamble, normalized_proof)

            # Re-parse complete_proof to get its AST for extraction
            ast_response = kimina_client.ast_code(proof_with_preamble, timeout=server_timeout)
            parsed_ast = parse_kimina_ast_code_response(ast_response)

            proof_body = None  # Initialize to track if AST extraction succeeded

            if parsed_ast.get("error") is None and parsed_ast.get("ast") is not None:
                # Parse succeeded - create AST and use it for extraction
                ast_without_imports = remove_default_imports_from_ast(parsed_ast["ast"], preamble=preamble)

                # Calculate body_start for the parsed AST
                normalized_preamble = preamble.strip()
                normalized_body = normalized_proof.strip()
                if not normalized_preamble:
                    body_start = 0
                elif not normalized_body:
                    body_start = len(proof_with_preamble)
                else:
                    body_start = len(normalized_preamble) + 2  # +2 for "\n\n"

                # Validate body_start is within bounds (defensive check)
                if body_start < 0 or body_start > len(proof_with_preamble):
                    raise ProofReconstructionError(  # noqa: TRY003
                        f"Invalid body_start ({body_start}) calculated for proof extraction. "
                        f"proof_with_preamble length: {len(proof_with_preamble)}, "
                        f"normalized_preamble length: {len(normalized_preamble)}, "
                        f"normalized_body length: {len(normalized_body)}"
                    )

                reconstructed_ast = AST(
                    ast_without_imports,
                    sorries=parsed_ast.get("sorries"),
                    source_text=proof_with_preamble,
                    body_start=body_start,
                )

                # Use robust AST-based extraction from decl_extraction.py
                # IMPORTANT: `extract_proof_body_from_ast` matches against the *signature only*
                # (up to but not including `:=`). Many internal nodes store `formal_theorem` as a
                # full declaration with `:= by ...`, so we compute the signature from the AST we
                # just parsed to avoid brittle string matching.
                target_sig = extract_signature_from_ast(reconstructed_ast)
                if target_sig is None:
                    raise ProofReconstructionError(  # noqa: TRY003
                        "Could not extract a theorem/lemma signature from reconstructed AST for internal node with "
                        f"formal_theorem: {decomposed_state.get('formal_theorem')!r}"
                    )
                proof_body = extract_proof_body_from_ast(reconstructed_ast, target_sig)
                if proof_body is not None and proof_body.strip():
                    # Successfully extracted using AST
                    return proof_body

            # If we reach here, AST extraction failed or returned empty - raise error
            # No regex fallback as per Option B plan to eliminate all regex heuristics.
            raise ProofReconstructionError(  # noqa: TRY003
                f"AST-based extraction failed for reconstructed proof body of internal node with signature: "
                f"'{decomposed_state.get('formal_theorem')}'. "
                f"This should not happen under assumptions (syntactic sketches, proven child proofs)."
            )

        # Should not reach here - all child types should be handled above
        raise ProofReconstructionError("Unable to extract proof body from child node. Unknown child type or structure.")  # noqa: TRY003

    def _analyze_proof_structure_ast_based(  # noqa: C901
        self,
        child: TreeNode,
        parent_ast: AST,
        hole_name: str,
        kimina_client: KiminaClient,
        server_timeout: int,
    ) -> dict:
        """
        Analyze proof structure using AST and check() calls.

        Returns analysis dict with:
        - is_function_type: bool
        - is_pi_type: bool
        - is_function_application: bool
        - should_preserve_application: bool
        - variable_conflicts: list[dict]
        - proof_body: str

        Parameters
        ----------
        child: TreeNode
            The child node (subgoal)
        parent_ast: AST
            The parent sketch AST
        hole_name: str
            The hole name (subgoal name)
        kimina_client: KiminaClient
            Client for AST and check() calls
        server_timeout: int
            Timeout for calls

        Returns
        -------
        dict
            Analysis results
        """
        from typing import cast

        from goedels_poetry.agents.state import (
            DecomposedFormalTheoremState,
            FormalTheoremProofState,
        )
        from goedels_poetry.agents.util.common import (
            DEFAULT_IMPORTS,
            combine_preamble_and_body,
        )
        from goedels_poetry.agents.util.kimina_server import parse_kimina_check_response
        from goedels_poetry.parsers.ast import AST
        from goedels_poetry.parsers.util.collection_and_analysis.application_detection import (
            find_subgoal_usage_in_ast,
            is_app_node,
        )
        from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
            extract_outer_scope_variables_ast_based,
            extract_variables_with_origin,
        )

        # Import type extraction function - use getattr to avoid name mangling
        from goedels_poetry.parsers.util.types_and_binders import type_extraction
        from goedels_poetry.parsers.util.types_and_binders.type_analysis import (
            is_function_type,
            is_pi_or_forall_type,
        )

        _extract_type_ast_func = getattr(type_extraction, "__extract_type_ast")

        # 1. Analyze child type
        child_ast = None
        child_type_ast = None

        if isinstance(child, dict):
            if "ast" in child:
                child_ast = child.get("ast")
            elif "formal_proof" in child and "children" not in child:
                # Leaf node - get AST from formal_proof
                proof_state = cast(FormalTheoremProofState, child)
                child_ast = proof_state.get("ast")
            elif "children" in child:
                # Decomposed node - get AST from decomposed state
                decomposed_state = cast(DecomposedFormalTheoremState, child)
                child_ast = decomposed_state.get("ast")

        if child_ast and isinstance(child_ast, AST):
            # Extract type from child's formal_theorem
            child_ast_node = child_ast.get_ast()
            child_type_ast = _extract_type_ast_func(child_ast_node)

        is_function_type_result = is_function_type(child_type_ast)
        is_pi_type = is_pi_or_forall_type(child_type_ast)

        # 2. Analyze parent usage
        usage_nodes = find_subgoal_usage_in_ast(parent_ast, hole_name)
        # Check if any usage node is an app node, or if its parent is an app node
        is_function_application = False
        for usage_node in usage_nodes:
            if is_app_node(usage_node):
                is_function_application = True
                break
            # Also check if the node is part of an app structure
            # (e.g., the identifier is an argument to an app node)
            # This is a heuristic - in practice, we'd need to check parent context
            # For now, if we found usages and the type is function, assume it might be an application
            # The actual detection will be refined based on AST structure analysis

        # 3. Extract proof body (includes tactics from Phase 1)
        proof_body = self._extract_proof_body_ast_guided(
            child,
            kimina_client=kimina_client,
            server_timeout=server_timeout,
        )

        # 4. Analyze variable conflicts (similar to Phase 3)
        proof_with_preamble = combine_preamble_and_body(DEFAULT_IMPORTS, proof_body)
        check_response = kimina_client.check(proof_with_preamble, timeout=server_timeout)
        parsed_check = parse_kimina_check_response(check_response)

        # Extract ALL variables from check() response with origin information
        # This distinguishes lemma parameters from proof body variables
        proof_variables = []
        if child_ast and isinstance(child_ast, AST):
            # Use check() response to get ALL variables with origin information
            proof_variables = extract_variables_with_origin(parsed_check, child_ast)

        parent_source_text = parent_ast.get_source_text()
        outer_scope_vars = extract_outer_scope_variables_ast_based(
            parent_source_text or "",
            parent_ast,
            kimina_client,
            server_timeout,
        )

        conflicts = []
        outer_scope_names = set(outer_scope_vars.keys())
        for var_info in proof_variables:
            var_name = var_info["name"]

            # CRITICAL: Skip lemma parameters - these are part of the signature
            # and should NOT be renamed, even if they conflict with outer scope
            if var_info.get("is_lemma_parameter", False):
                continue

            # Only consider proof body variables for conflict detection
            if var_name in outer_scope_names and var_info.get("is_proof_body_variable", False):
                from goedels_poetry.parsers.util.collection_and_analysis.variable_extraction import (
                    is_intentional_shadowing,
                )

                var_decl = {
                    "name": var_name,
                    "node": var_info.get("declaration_node"),
                    "hypothesis": var_info["hypothesis"],
                    "is_lemma_parameter": False,
                    "is_proof_body_variable": True,
                }

                if not is_intentional_shadowing(var_decl, parsed_check, outer_scope_vars):
                    conflicts.append(var_decl)

        return {
            "is_function_type": is_function_type_result,
            "is_pi_type": is_pi_type,
            "is_function_application": is_function_application,
            "should_preserve_application": is_pi_type and is_function_application,
            "variable_conflicts": conflicts,
            "proof_body": proof_body,
        }

    def reconstruct_complete_proof(
        self,
        *,
        server_url: str,
        server_max_retries: int = 3,
        server_timeout: int = 60,
    ) -> str:
        """
        Reconstructs the complete Lean4 proof from the proof tree using AST-based methods.

        Uses AST.get_sorry_holes_by_name() for accurate Unicode-safe hole positions
        and requires syntax validation and semantic validation with KiminaClient.

        Parameters
        ----------
        server_url : str
            Required Kimina server URL for syntax validation and semantic validation
        server_max_retries : int
            Max retries for Kimina requests
        server_timeout : int
            Timeout for Kimina requests

        Returns
        -------
        str
            The complete Lean4 proof text with the stored preamble prefix
        """
        from kimina_client import KiminaClient

        from goedels_poetry.agents.util.common import DEFAULT_IMPORTS, combine_preamble_and_body
        from goedels_poetry.agents.util.kimina_server import (
            parse_kimina_ast_code_response,
            parse_kimina_check_response,
        )
        # ProofReconstructionError is defined at module level in goedels_poetry/state.py (same file as this method)

        preamble = self._state._root_preamble or DEFAULT_IMPORTS

        if self._state.formal_theorem_proof is None:
            return combine_preamble_and_body(preamble, "-- No proof available")

        # Create KiminaClient (required for syntax validation and semantic validation)
        kimina_client = KiminaClient(
            api_url=server_url,
            http_timeout=server_timeout,
            n_retries=server_max_retries,
        )

        # Reconstruct using AST-based method (performs incremental syntax validation)
        proof_without_preamble = self._reconstruct_node_proof_ast_based(
            self._state.formal_theorem_proof,
            kimina_client=kimina_client,
            server_timeout=server_timeout,
        )

        complete_proof = combine_preamble_and_body(preamble, proof_without_preamble)

        # REQUIRED: Final syntax validation with ast_code() (should always pass due to incremental syntax validation)
        # Note: combine_preamble_and_body() should already normalize (strip and add trailing newline),
        # but we add this normalization defensively in case the implementation changes.
        normalized_proof = complete_proof if complete_proof.endswith("\n") else complete_proof + "\n"
        ast_response = kimina_client.ast_code(normalized_proof, timeout=server_timeout)
        parsed_ast = parse_kimina_ast_code_response(ast_response)
        if parsed_ast.get("error") is not None:
            raise ProofReconstructionError(  # noqa: TRY003
                f"Final syntax validation failed (this should not happen under assumptions): {parsed_ast['error']}\n"
                f"Reconstructed proof:\n{complete_proof}"
            )

        # REQUIRED: Final semantic validation with check() (should always pass due to incremental validation)
        check_response = kimina_client.check(normalized_proof, timeout=server_timeout)
        parsed_check = parse_kimina_check_response(check_response)

        # Store semantic validation result (should always be True)
        self._state.proof_validation_result = parsed_check.get("complete", False)

        if not parsed_check.get("complete", False):
            # This should not happen under assumptions - raise error with diagnostic info
            errors = parsed_check.get("errors", [])
            sorries = parsed_check.get("sorries", [])
            error_parts = []
            if errors:
                error_parts.append("Errors:\n" + "\n".join(err.get("data", str(err)) for err in errors))
            if sorries:
                error_parts.append(f"Sorries remaining: {len(sorries)}")
            error_msg = "\n\n".join(error_parts) if error_parts else "Unknown semantic validation failure"
            raise ProofReconstructionError(  # noqa: TRY003
                f"Final semantic validation failed (this should not happen under assumptions): {error_msg}\n"
                f"Reconstructed proof:\n{complete_proof}"
            )

        return complete_proof

    def _dedent_proof_body(self, proof_body: str) -> str:
        """
        Dedent a proof body by removing the common leading indentation from non-empty lines.

        This is critical for Lean4 layout-sensitive constructs like `calc`, `match`, `cases`, etc.
        Child proofs frequently arrive already-indented (e.g. copied from inside a lemma or have),
        and re-indenting them naively can push lines too far right, changing parse structure.
        """
        lines = proof_body.split("\n")
        indents: list[int] = []
        for ln in lines:
            if not ln.strip():
                continue
            count = 0
            for ch in ln:
                if ch == " ":
                    count += 1
                else:
                    break
            indents.append(count)
        if not indents:
            return proof_body
        min_indent = min(indents)
        if min_indent <= 0:
            return proof_body
        prefix = " " * min_indent
        dedented: list[str] = []
        for ln in lines:
            if ln.strip():
                dedented.append(ln[min_indent:] if ln.startswith(prefix) else ln.lstrip(" "))
            else:
                dedented.append(ln)
        return "\n".join(dedented)

    def _reconstruct_leaf_node_proof(self, formal_proof_state: FormalTheoremProofState) -> str:
        """
        Reconstruct proof text for a leaf `FormalTheoremProofState`.
        """
        if formal_proof_state["formal_proof"] is not None:
            proof_text = str(formal_proof_state["formal_proof"])
            # If this is the root leaf (no parent), ensure the output includes the theorem header.
            # Avoid regex: if it already starts with the theorem signature, return as-is.
            if formal_proof_state["parent"] is None:
                theorem_decl_full = str(formal_proof_state["formal_theorem"]).strip()
                theorem_sig = self._strip_decl_assignment(theorem_decl_full).strip()

                # Validate theorem signature is not empty
                if not theorem_sig:
                    raise ProofReconstructionError(f"Invalid theorem declaration for root node: {theorem_decl_full}")  # noqa: TRY003

                # Skip leading empty lines and single-line comments to avoid redundant wrapping
                leading_skipped = self._skip_leading_trivia(proof_text)

                # Use normalized string comparison to handle multiline theorem signatures
                # This handles cases where theorem signature and proof_text have different formatting
                # (e.g., single-line vs multiline signatures)
                normalized_sig = " ".join(theorem_sig.split())
                normalized_leading = " ".join(leading_skipped.split())
                if normalized_leading.startswith(normalized_sig):
                    # Already has theorem signature, return as-is
                    return proof_text
                # Otherwise treat stored proof as tactics and wrap once.
                indent = " " * PROOF_BODY_INDENT_SPACES
                indented_body = self._indent_proof_body(proof_text, indent)
                return f"{theorem_sig} := by\n{indented_body}"
            # Non-root leaves are always tactic bodies used for inlining; return as-is.
            return proof_text
        # No proof yet, return the theorem with sorry
        return f"{formal_proof_state['formal_theorem']} := by sorry\n"

    def _skip_leading_trivia(self, text: str) -> str:
        """
        Skip leading empty lines and single-line comments in the given text.

        This removes:
        - Empty lines
        - Line comments starting with '--'
        - Single-line block comments of the form '/- ... -/'
        """
        lines = text.split("\n")
        idx = 0
        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped == "":
                idx += 1
                continue
            if stripped.startswith("--"):
                idx += 1
                continue
            if stripped.startswith("/-") and stripped.endswith("-/"):
                idx += 1
                continue
            break
        return "\n".join(lines[idx:]).lstrip()

    def _strip_decl_assignment(self, formal_decl: str) -> str:
        """
        Strip any ':= ...' suffix from a declaration, returning only the header/signature.
        """
        idx = formal_decl.find(":=")
        return formal_decl[:idx].rstrip() if idx != -1 else formal_decl

    def _calculate_line_indent(self, line: str) -> int:
        """
        Helper to calculate indentation of a line.

        Returns the number of leading whitespace characters.
        """
        return len(line) - len(line.lstrip())

    def _indent_proof_body(self, proof_body: str, indent: str) -> str:
        """
        Indents each line of the proof body.

        For mixed indentation (some lines have 0 indent, others don't),
        preserves relative indentation using minimum non-zero indent as reference.

        Parameters
        ----------
        proof_body : str
            The proof body to indent
        indent : str
            The base indentation string (typically spaces).
            Expected to be non-empty in practice. If empty, relative structure
            is preserved but no base indent is applied.

        Returns
        -------
        str
            The indented proof body with relative structure preserved for mixed cases

        Note
        ----
        The indent parameter is always spaces in practice (Lean 4 convention).
        All call sites create indent as " " * n where n >= 0. If n = 0, indent
        will be empty string, which is handled correctly but rare in practice.
        """
        # Optional: Defensive check for tab characters
        if "\t" in indent:
            # Log warning - Lean 4 uses spaces, tabs may cause misalignment
            # Note: Could use logger.warning() if logging is available
            pass

        lines = proof_body.split("\n")

        # Step 1: Detect mixed indentation
        indents = []
        for ln in lines:
            if not ln.strip():
                continue
            indent_count = self._calculate_line_indent(ln)
            indents.append(indent_count)

        # Step 2: Normal case
        indented_lines = []
        for line in lines:
            if line.strip():
                indented_lines.append(indent + line)
            else:
                indented_lines.append(line)
        return "\n".join(indented_lines)
