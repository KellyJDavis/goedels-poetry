from __future__ import annotations

import os
import pickle
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from typing import cast

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

# Note: FORMALIZER_AGENT_LLM and SEMANTICS_AGENT_LLM are intentionally NOT imported here
# to allow lazy loading when processing formal theorems (which don't need these models)
from goedels_poetry.functools import maybe_save
from goedels_poetry.util.tree import TreeNode

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


class GoedelsPoetryState:
    def __init__(self, formal_theorem: str | None = None, informal_theorem: str | None = None):
        # Check that the proper number of arguments has been provided
        if (formal_theorem is None) and (informal_theorem is None):
            raise ValueError("Either 'formal_theorem' xor 'informal_theorem' must be provided")  # noqa: TRY003
        if (formal_theorem is not None) and (informal_theorem is not None):
            raise ValueError("Only one of 'formal_theorem' or 'informal_theorem' can be provided")  # noqa: TRY003

        # Introduce a bool to indicate if the proof is finished unable to be finished
        self.is_finished: bool = False

        # Introduce a string to hold the reason for finishing
        self.reason: str | None = None

        # Introduce a list of strings to hold the action history
        self.action_history: list[str] = []

        self._root_preamble: str | None = None

        # Initialize state with provided arguments
        self.formal_theorem_proof: TreeNode | None = None
        if formal_theorem is not None:
            preamble, body = split_preamble_and_body(formal_theorem)
            if not preamble.strip():
                raise ValueError(MISSING_FORMAL_PREAMBLE_MSG)

            preamble = ensure_mandatory_preamble(preamble)
            self._root_preamble = preamble
            initial_formal_state = FormalTheoremProofState(
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

        # Place formalized_informal_theorem on the queue to be syntactically validated
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

        # Place attempted proofs in the queue of proofs to be validated
        self._state.proof_validate_queue += proven_theorems["outputs"]

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

        Parameters
        ---------
        validated_proofs: FormalTheoremProofStates
            FormalTheoremProofStates containing validated_proofs["outputs"] the list of
            FormalTheoremProofState each of which has its proof been validated or invalided.
        """
        # Remove all elements from the queue of proofs to validate
        self._state.proof_validate_queue.clear()

        # Get validated_proofs outputs
        validated_proofs_outputs = validated_proofs["outputs"]

        # Increment the proof attempt count for all validated proofs
        for validated_proof in validated_proofs_outputs:
            validated_proof["self_correction_attempts"] += 1

        # Gather all unsuccessful proofs
        unsuccessful_proofs = [vp for vp in validated_proofs_outputs if (not vp["proved"])]

        proofs_too_difficult = []
        proofs_to_correct = []

        for up in unsuccessful_proofs:
            # Note: We use >= because self_correction_attempts was incremented above (line 653)
            # before this check. When attempts == max, we've exhausted the allowed attempts
            # (e.g., with max=2: 0->1 allows correction 1, 1->2 allows correction 2, 2->3 exhausts).
            if up["self_correction_attempts"] >= PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS:
                up["pass_attempts"] += 1
                if up["pass_attempts"] < PROVER_AGENT_MAX_PASS:
                    # Restart self-correction loop: reset state, requeue for correction
                    self._reset_self_correction_state(up)
                    proofs_to_correct.append(up)
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

        # Queue all successful proofs to have their ASTs generated
        successful_proofs = [vp for vp in validated_proofs_outputs if vp["proved"]]
        self._state.proof_ast_queue += successful_proofs

    def _reset_self_correction_state(self, proof: FormalTheoremProofState) -> None:
        """
        Resets the self-correction state for a proof so that a new self-correction pass starts cleanly.
        """
        proof["self_correction_attempts"] = 0
        proof["errors"] = None
        proof["proof_history"] = []
        # reset additional state as needed

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
                parent=proof_too_difficult["parent"],
                children=[],
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
            )
            self._state.decomposition_search_queue.append(formal_theorem_to_decompose)

            # Remove proof_too_difficult from the proof tree
            if proof_too_difficult["parent"] is not None:
                cast(DecomposedFormalTheoremState, proof_too_difficult["parent"])["children"].remove(
                    cast(TreeNode, proof_too_difficult)
                )
                proof_too_difficult["parent"] = None

            # Check to see if formal_theorem_to_decompose is the root theorem
            if formal_theorem_to_decompose["parent"] is None:
                # If so, set the root to formal_theorem_to_decompose
                self._state.formal_theorem_proof = cast(TreeNode, formal_theorem_to_decompose)
            else:
                # If not, add formal_theorem_to_decompose as its parent's child
                cast(DecomposedFormalTheoremState, formal_theorem_to_decompose["parent"])["children"].append(
                    cast(TreeNode, formal_theorem_to_decompose)
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

        # Place all sketched theorems into the queue of sketches to be validated
        self._state.decomposition_validate_queue += sketched_theorems["outputs"]

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
            for child in internal_node["children"]:
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
        # Clear children (they will be removed from tree separately)
        node["children"] = []
        # Clear sketch-related fields
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

        # Gather all children FormalTheoremProofState's that need to be proven
        all_children = [
            cast(FormalTheoremProofState, dt) for ds in decomposed_sketches["outputs"] for dt in ds["children"]
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

    def reconstruct_complete_proof(self) -> str:
        """
        Reconstructs the complete Lean4 proof from the proof tree.

        Returns
        -------
        str
            The complete Lean4 proof text with the stored preamble prefix
        """
        preamble = self._state._root_preamble or DEFAULT_IMPORTS

        if self._state.formal_theorem_proof is None:
            return combine_preamble_and_body(preamble, "-- No proof available")

        proof_without_preamble = self._reconstruct_node_proof(self._state.formal_theorem_proof)
        return combine_preamble_and_body(preamble, proof_without_preamble)

    def _reconstruct_node_proof(self, node: TreeNode) -> str:
        """
        Recursively reconstructs the proof for a given node in the proof tree.

        Parameters
        ----------
        node : TreeNode
            The node to reconstruct proof for

        Returns
        -------
        str
            The proof text for this node and all its children (without preamble)
        """
        # Check if this is a FormalTheoremProofState (leaf node)
        if isinstance(node, dict) and "formal_proof" in node and "children" not in node:
            formal_proof_state = cast(FormalTheoremProofState, node)
            if formal_proof_state["formal_proof"] is not None:
                proof_text = str(formal_proof_state["formal_proof"])
                # If this is the root leaf (no parent), ensure the output includes the theorem header.
                # Avoid regex: if it already starts with the theorem signature, return as-is.
                if formal_proof_state["parent"] is None:
                    theorem_decl_full = str(formal_proof_state["formal_theorem"]).strip()
                    theorem_sig = self._strip_decl_assignment(theorem_decl_full)
                    # Skip leading empty lines and single-line comments to avoid redundant wrapping
                    leading_skipped = self._skip_leading_trivia(proof_text)
                    if leading_skipped.startswith(theorem_sig):
                        return proof_text
                    # Otherwise treat stored proof as tactics and wrap once.
                    indent = " " * PROOF_BODY_INDENT_SPACES
                    indented_body = self._indent_proof_body(proof_text, indent)
                    return f"{theorem_sig} := by\n{indented_body}"
                # Non-root leaves are always tactic bodies used for inlining; return as-is.
                return proof_text
            else:
                # No proof yet, return the theorem with sorry
                return f"{formal_proof_state['formal_theorem']} := by sorry\n"

        # Check if this is a DecomposedFormalTheoremState (internal node)
        if isinstance(node, dict) and "children" in node:
            decomposed_state = cast(DecomposedFormalTheoremState, node)

            if decomposed_state["proof_sketch"] is None:
                # Fallback if no sketch
                return f"{decomposed_state['formal_theorem']} := by sorry\n"

            # Start with the parent's proof sketch
            sketch = str(decomposed_state["proof_sketch"])

            # For each child, inline its proof into the parent's sketch
            for child in decomposed_state["children"]:
                child_proof_body = self._extract_proof_body(child)
                sketch = self._inline_child_proof(sketch, child, child_proof_body)

            return sketch

        # Fallback for unexpected node types
        return "-- Unable to reconstruct proof for this node\n"

    def _extract_proof_body(self, child: TreeNode) -> str:
        """
        Extracts the proof body (tactics after 'by') from a child node.

        Parameters
        ----------
        child : TreeNode
            The child node to extract proof body from

        Returns
        -------
        str
            The proof body (tactic sequence)
        """
        if isinstance(child, dict) and "formal_proof" in child and "children" not in child:
            # This is a FormalTheoremProofState (leaf)
            formal_proof_state = cast(FormalTheoremProofState, child)
            if formal_proof_state["formal_proof"] is not None:
                proof = str(formal_proof_state["formal_proof"])
                # Extract just the tactics after "by"
                return self._extract_tactics_after_by(proof)
            return "sorry"
        elif isinstance(child, dict) and "children" in child:
            # This is a DecomposedFormalTheoremState (internal)
            # Recursively reconstruct this child first
            child_complete = self._reconstruct_node_proof(child)
            return self._extract_tactics_after_by(child_complete)
        return "sorry"

    def _extract_tactics_after_by(self, proof: str) -> str:
        """
        Extracts the tactic sequence after 'by' from a proof.

        Parameters
        ----------
        proof : str
            The complete proof text

        Returns
        -------
        str
            The tactic sequence (indented appropriately)
        """
        # Match ':=' followed by any whitespace (including newlines), then 'by'
        # This handles all variations: ':= by', ':=by', ':=  by', ':=\nby', etc.
        match = re.search(r":=\s*by", proof)

        if match is None:
            # Can't find ':= by' pattern, return the whole proof
            return proof.strip()

        # Extract everything after 'by'
        tactics = proof[match.end() :].strip()
        return tactics

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

    def _inline_child_proof(self, parent_sketch: str, child: TreeNode, child_proof_body: str) -> str:
        """
        Inlines a child's proof body into the parent's sketch by replacing the corresponding sorry.

        Important: The child's formal_theorem may differ from what appears in the parent sketch
        because AST.get_named_subgoal_code() adds earlier dependencies as explicit parameters.
        For example:
          - Parent sketch has: "have sum_not_3 : ... := by sorry"
          - Child formal_theorem: "lemma sum_not_3 (cube_mod9 : ...) : ... := by ..."

        We handle this by:
          1. Extracting just the name from the child (e.g., "sum_not_3")
          2. Searching for that name in the parent sketch (which has the original signature)
          3. Replacing the sorry in the parent's original have statement

        Parameters
        ----------
        parent_sketch : str
            The parent's proof sketch with sorry placeholders
        child : TreeNode
            The child node whose proof we're inlining
        child_proof_body : str
            The proof body to inline

        Returns
        -------
        str
            The parent sketch with the child proof inlined
        """
        # Get the child's theorem name from formal_theorem
        child_formal_theorem = ""
        if isinstance(child, dict) and "formal_theorem" in child:
            child_formal_theorem = str(child["formal_theorem"])

        # Extract the have/lemma name from the child's theorem
        # This strips away any added parameters to get just the name
        have_name = self._extract_have_name(child_formal_theorem)

        if not have_name:
            # Can't identify the have name, might be the main body
            # Try to replace a standalone sorry (not part of a have statement)
            return self._replace_main_body_sorry(parent_sketch, child_proof_body)

        # Check if this name actually appears as a "have" statement in the parent sketch
        # Pattern: "have <name> : ..."
        have_pattern = rf"have\s+{re.escape(have_name)}\s*:"
        if re.search(have_pattern, parent_sketch):
            # Find the "have <name> : ... := by sorry" pattern in the parent sketch
            # Note: The parent sketch still has the original signature without extra parameters
            return self._replace_sorry_for_have(parent_sketch, have_name, child_proof_body)
        else:
            # This name doesn't appear as a "have" statement, so it's the main body
            return self._replace_main_body_sorry(parent_sketch, child_proof_body)

    def _extract_have_name(self, formal_theorem: str) -> str:
        """
        Extracts the name from a have/lemma declaration.

        Note: The child's formal_theorem may have dependencies added as explicit parameters
        by AST.get_named_subgoal_code(), e.g.:
          lemma sum_not_3 (cube_mod9 : ...) : result_type := ...
        We need to extract just the name "sum_not_3", not including the parameters.

        Parameters
        ----------
        formal_theorem : str
            The formal theorem text

        Returns
        -------
        str
            The name of the have/lemma
        """
        # Lean identifiers allow a wide range of unicode characters (e.g., h₁ or names
        # that use Greek letters), so we capture the entire token that appears immediately
        # after the keyword and stop before any whitespace, ':' or '('. This mirrors how
        # the parent sketch spells the have/lemma names, which is what we need for
        # pattern replacement.
        pattern = r"\b(?:lemma|have|theorem)\s+([^\s:(]+)"
        match = re.search(pattern, formal_theorem)

        if match:
            return match.group(1)

        return ""

    def _replace_sorry_for_have(self, parent_sketch: str, have_name: str, child_proof_body: str) -> str:
        """
        Replaces the sorry in a specific have statement with the actual proof body.

        Note: We search by name only in the parent sketch, which contains the original
        "have <name> : ... := by sorry" without any added parameters. The child's
        formal_theorem may have dependencies added as parameters, but we don't use
        that here - we only search in the parent's original text.

        Parameters
        ----------
        parent_sketch : str
            The parent's proof sketch (with original signatures, no added parameters)
        have_name : str
            The name of the have statement to replace
        child_proof_body : str
            The proof body to insert

        Returns
        -------
        str
            The modified sketch with sorry replaced
        """
        # Pattern to match: "have <name> : ... := by sorry" in the parent's original text
        # The parent sketch has the original signature without extra parameters
        pattern = rf"(have\s+{re.escape(have_name)}\s*:.*?:=\s*by\s*)sorry"

        # Find the match to determine indentation
        match = re.search(pattern, parent_sketch, re.DOTALL)
        if not match:
            # Pattern not found, return unchanged
            return parent_sketch

        # Determine the indentation level of the have statement
        start_pos = match.start()
        # Find the start of the line containing this have
        # Note: rfind returns -1 if not found, so +1 makes it 0 (start of string)
        line_start = parent_sketch.rfind("\n", 0, start_pos) + 1
        have_line = parent_sketch[line_start:start_pos]
        base_indent = len(have_line) - len(have_line.lstrip())

        # Add spaces for the indentation of the proof body
        proof_indent = " " * (base_indent + PROOF_BODY_INDENT_SPACES)

        # Indent the child proof body
        indented_proof = self._indent_proof_body(child_proof_body, proof_indent)

        # Replace sorry with the indented proof
        result = re.sub(pattern, rf"\1\n{indented_proof}", parent_sketch, count=1, flags=re.DOTALL)

        return result

    def _is_sorry_part_of_have(self, lines: list[str], sorry_idx: int) -> bool:
        """
        Checks if a sorry at the given line index is part of a have statement.

        Parameters
        ----------
        lines : list[str]
            The lines of the proof sketch
        sorry_idx : int
            The index of the sorry line to check

        Returns
        -------
        bool
            True if the sorry is part of a have statement, False otherwise
        """
        # Look back up to 5 lines to see if we're in a have statement
        lookback_limit = min(5, sorry_idx)
        sorry_indent = len(lines[sorry_idx]) - len(lines[sorry_idx].lstrip())

        for j in range(1, lookback_limit + 1):
            prev_idx = sorry_idx - j
            prev_line = lines[prev_idx]

            # Pattern: "have <name> : <type> := <whitespace/newline> by <whitespace/newline> sorry"
            # Note: Lean4 identifiers can include apostrophes
            if re.search(r"\bhave\s+[\w']+\s*:", prev_line):
                # Found a have statement - now check if our sorry is part of its proof body
                have_indent = len(prev_line) - len(prev_line.lstrip())

                # Check lines between the have and our sorry for ":= by" pattern
                between_text = "\n".join(lines[prev_idx:sorry_idx])

                # If there's a ":= by" pattern in the lines from have to (but not including) sorry,
                # and our sorry is indented MORE than the have line, it's part of the have
                if re.search(r":=\s*by", between_text, re.MULTILINE | re.DOTALL) and sorry_indent > have_indent:
                    return True

            # If we hit an empty line, stop looking back
            if not prev_line.strip():
                break

        return False

    def _replace_main_body_sorry(self, parent_sketch: str, child_proof_body: str) -> str:
        """
        Replaces a standalone sorry (not part of a have statement) with the actual proof body.
        This handles the main proof body after all have statements.

        This method handles both single-line and multiline patterns:
        - Single-line: "have h : Type := by sorry" (skip this)
        - Multiline: "have h : Type :=\n  by sorry" (skip this)
        - Main body: standalone "sorry" after all have statements (replace this)

        Parameters
        ----------
        parent_sketch : str
            The parent's proof sketch
        child_proof_body : str
            The proof body to insert

        Returns
        -------
        str
            The modified sketch with the main body sorry replaced
        """
        # Find a standalone sorry that's not part of a have statement
        # We need to find the last sorry in the sketch that's at the end after all have statements

        # Split into lines to find standalone sorry
        lines = parent_sketch.split("\n")

        # Find the last line that contains just "sorry" (possibly with indentation)
        # that is NOT part of a have statement (checking previous lines for multiline patterns)
        last_sorry_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()

            # Check if this line is just "sorry"
            if stripped == "sorry":
                # Check if this is part of a ":= by sorry" pattern (single-line)
                if re.search(r":=\s*by", lines[i]):
                    continue

                # Check if this is part of a multiline ":= ... by ... sorry" pattern
                if not self._is_sorry_part_of_have(lines, i):
                    # This is a standalone sorry (main body)
                    last_sorry_idx = i
                    break

        if last_sorry_idx == -1:
            # No standalone sorry found
            return parent_sketch

        # Get the indentation of the sorry line
        sorry_line = lines[last_sorry_idx]
        base_indent = len(sorry_line) - len(sorry_line.lstrip())
        indent_str = " " * base_indent

        # Indent the child proof body to match the sorry's indentation
        indented_proof = self._indent_proof_body(child_proof_body, indent_str)

        # Replace the sorry line with the indented proof
        lines[last_sorry_idx] = indented_proof

        return "\n".join(lines)

    def _indent_proof_body(self, proof_body: str, indent: str) -> str:
        """
        Indents each line of the proof body.

        Parameters
        ----------
        proof_body : str
            The proof body to indent
        indent : str
            The indentation string to add

        Returns
        -------
        str
            The indented proof body
        """
        lines = proof_body.split("\n")
        indented_lines = []
        for line in lines:
            if line.strip():  # Only indent non-empty lines
                indented_lines.append(indent + line)
            else:
                indented_lines.append(line)
        return "\n".join(indented_lines)
