from __future__ import annotations

import dataclasses
import logging
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
from goedels_poetry.agents.util.debug import is_debug_enabled
from goedels_poetry.config.llm import (
    DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
    FORMALIZER_AGENT_MAX_RETRIES,
    PROVER_AGENT_MAX_DEPTH,
    PROVER_AGENT_MAX_PASS,
    PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
)

# Note: All LLM instances are imported from goedels_poetry.config.llm
from goedels_poetry.functools import maybe_save
from goedels_poetry.util.tree import TreeNode

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
                hole_name=None,
                hole_start=None,
                hole_end=None,
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
                # Preserve the parent's hole metadata so reconstruction can remain offset-based,
                # even after converting a leaf proof into a decomposed (internal) node.
                hole_name=proof_too_difficult.get("hole_name"),
                hole_start=proof_too_difficult.get("hole_start"),
                hole_end=proof_too_difficult.get("hole_end"),
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

        # If a final proof override was selected (e.g., via Kimina-guided reconstruction),
        # prefer it so downstream writers don't recompute a failing variant.
        final_complete_proof = cast(str | None, getattr(self._state, "final_complete_proof", None))
        if final_complete_proof is not None:
            return final_complete_proof

        if self._state.formal_theorem_proof is None:
            return combine_preamble_and_body(preamble, "-- No proof available")

        proof_without_preamble = self._reconstruct_node_proof(self._state.formal_theorem_proof)
        return combine_preamble_and_body(preamble, proof_without_preamble)

    @dataclasses.dataclass(frozen=True)
    class ReconstructionVariant:
        """
        A parameterization of reconstruction normalization steps.

        These toggles are intentionally coarse-grained: Kimina-guided selection tries a bounded
        set of variants and selects the first that Kimina marks complete.
        """

        variant_id: str
        dedent_common: bool = True
        snap_indent_levels: bool = True
        rewrite_trailing_apply: bool = True
        fix_dangling_closing_tactics_after_comments: bool = True
        snap_self_reference_closers: bool = True
        dedent_last_closer_out_of_inner_have_by: bool = True

    def _variant_key(self, v: ReconstructionVariant) -> tuple:
        return (
            v.dedent_common,
            v.snap_indent_levels,
            v.rewrite_trailing_apply,
            v.fix_dangling_closing_tactics_after_comments,
            v.snap_self_reference_closers,
            v.dedent_last_closer_out_of_inner_have_by,
        )

    def _get_reconstruction_variants(self, max_candidates: int) -> list[ReconstructionVariant]:
        """
        Deterministically generate a bounded list of reconstruction variants to try.

        The exact set is intentionally internal (not configurable) to avoid exposing
        implementation details in config.
        """
        max_candidates = max(1, int(max_candidates))

        baseline = self.ReconstructionVariant("baseline")
        seed_variants: list[GoedelsPoetryStateManager.ReconstructionVariant] = [
            baseline,
            # Directly addresses the partial.log failure mode.
            dataclasses.replace(
                baseline, variant_id="no_comment_fixer", fix_dangling_closing_tactics_after_comments=False
            ),
            dataclasses.replace(baseline, variant_id="no_indent_snapping", snap_indent_levels=False),
            dataclasses.replace(
                baseline,
                variant_id="no_offset_extras",
                fix_dangling_closing_tactics_after_comments=False,
                snap_self_reference_closers=False,
                dedent_last_closer_out_of_inner_have_by=False,
            ),
            dataclasses.replace(
                baseline,
                variant_id="dedent_only",
                snap_indent_levels=False,
                rewrite_trailing_apply=False,
                fix_dangling_closing_tactics_after_comments=False,
                snap_self_reference_closers=False,
                dedent_last_closer_out_of_inner_have_by=False,
            ),
            dataclasses.replace(
                baseline,
                variant_id="minimal",
                dedent_common=False,
                snap_indent_levels=False,
                rewrite_trailing_apply=False,
                fix_dangling_closing_tactics_after_comments=False,
                snap_self_reference_closers=False,
                dedent_last_closer_out_of_inner_have_by=False,
            ),
        ]

        variants: list[GoedelsPoetryStateManager.ReconstructionVariant] = []
        seen: set[tuple] = set()

        def add(v: GoedelsPoetryStateManager.ReconstructionVariant) -> None:
            key = self._variant_key(v)
            if key in seen:
                return
            seen.add(key)
            variants.append(v)

        for v in seed_variants:
            add(v)
            if len(variants) >= max_candidates:
                return variants[:max_candidates]

        # If the user allows more candidates than the seed set, expand deterministically by
        # toggling individual flags off (starting from baseline).
        toggles = [
            ("no_dedent_common", {"dedent_common": False}),
            ("no_rewrite_trailing_apply", {"rewrite_trailing_apply": False}),
            ("no_snap_self_reference_closers", {"snap_self_reference_closers": False}),
            ("no_dedent_last_closer", {"dedent_last_closer_out_of_inner_have_by": False}),
        ]
        for name, changes in toggles:
            add(dataclasses.replace(baseline, variant_id=name, **changes))
            if len(variants) >= max_candidates:
                break

        return variants[:max_candidates]

    def reconstruct_complete_proof_kimina_guided(
        self, *, server_url: str, server_max_retries: int, max_candidates: int
    ) -> tuple[str, bool, str]:
        """
        Attempt to find a reconstruction variant that Kimina marks complete.

        This is a bounded search over whole-file reconstructions and is intended to run only
        after a run reports "Proof completed successfully." but final verification fails.
        """
        preamble = self._state._root_preamble or DEFAULT_IMPORTS
        if self._state.formal_theorem_proof is None:
            proof = combine_preamble_and_body(preamble, "-- No proof available")
            return proof, False, "No proof available"

        # Lazy import to avoid importing `kimina_client` (and its transitive dependencies)
        # during test collection on Python < 3.12 where some environments may have incompatible
        # versions. This function is only invoked in "success-but-final-verification-failed" cases.
        from goedels_poetry.agents.proof_checker_agent import check_complete_proof

        variants = self._get_reconstruction_variants(max_candidates=max_candidates)
        last_err = ""
        for idx, variant in enumerate(variants, start=1):
            proof_without_preamble = self._reconstruct_node_proof(self._state.formal_theorem_proof, variant=variant)
            candidate = combine_preamble_and_body(preamble, proof_without_preamble)
            ok, err = check_complete_proof(candidate, server_url=server_url, server_max_retries=server_max_retries)
            last_err = err

            if is_debug_enabled():
                logger.debug(
                    "Kimina-guided reconstruction attempt %d/%d (%s): %s",
                    idx,
                    len(variants),
                    variant.variant_id,
                    "passed" if ok else "failed",
                )
                if not ok and err:
                    logger.debug("Kimina-guided reconstruction errors (%s):\n%s", variant.variant_id, err)

            if ok:
                self._state.reconstruction_attempts = idx
                self._state.reconstruction_strategy_used = variant.variant_id
                self._state.final_complete_proof = candidate
                return candidate, True, ""

        # Record attempts even on failure.
        self._state.reconstruction_attempts = len(variants)
        self._state.reconstruction_strategy_used = None
        return (
            combine_preamble_and_body(preamble, self._reconstruct_node_proof(self._state.formal_theorem_proof)),
            False,
            last_err,
        )

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
        # No proof yet, return the theorem with sorry
        return f"{formal_proof_state['formal_theorem']} := by sorry\n"

    def _apply_offset_replacements(  # noqa: C901
        self, sketch: str, children: list[TreeNode], *, variant: ReconstructionVariant | None = None
    ) -> str:
        """
        Apply offset-based replacements for children that have hole metadata.
        """
        replacements: list[tuple[int, int, str]] = []
        missing: list[TreeNode] = []

        for child in children:
            child_proof_body = self._extract_proof_body(child, variant=variant)

            if isinstance(child, dict) and "hole_start" in child and "hole_end" in child:
                hole_start = cast(int | None, child.get("hole_start"))
                hole_end = cast(int | None, child.get("hole_end"))
                if (
                    isinstance(hole_start, int)
                    and isinstance(hole_end, int)
                    and 0 <= hole_start < hole_end <= len(sketch)
                ):
                    # Determine indentation prefix on the line containing the hole.
                    line_start = sketch.rfind("\n", 0, hole_start) + 1
                    line_prefix = sketch[line_start:hole_start]
                    # Leading whitespace at the start of this line (used for inline `by sorry` holes).
                    line = sketch[
                        line_start : sketch.find("\n", line_start) if "\n" in sketch[line_start:] else len(sketch)
                    ]
                    leading_ws = line[: len(line) - len(line.lstrip(" \t"))]

                    normalized_body = self._normalize_child_proof_body(
                        child_proof_body, offset_insertion=True, variant=variant
                    )
                    body_lines = normalized_body.split("\n")
                    if not body_lines:
                        body_lines = ["sorry"]

                    # Two forms of `sorry` holes exist in sketches:
                    # 1) Standalone line: `    sorry` (hole indentation is pure whitespace)
                    # 2) Inline: `have h : T := by sorry` (hole indentation includes non-whitespace)
                    #
                    # For (1), we can replace the token in-place: the line already contains the correct
                    # indentation prefix before the token. We only need to indent subsequent lines.
                    #
                    # For (2), replacing `sorry` with a multi-line proof must insert a newline and indent
                    # the *entire* proof body under the surrounding `by`.
                    inline_hole = bool(line_prefix.strip())
                    rebuilt_lines: list[str] = []
                    if inline_hole:
                        proof_indent = leading_ws + (" " * PROOF_BODY_INDENT_SPACES)
                        rebuilt_lines.append("")  # turn `by sorry` into `by\n<proof>`
                        for ln in body_lines:
                            rebuilt_lines.append(f"{proof_indent}{ln}" if ln.strip() else ln)
                    else:
                        indent_prefix = line_prefix
                        for i, ln in enumerate(body_lines):
                            if i == 0:
                                rebuilt_lines.append(ln)
                            else:
                                rebuilt_lines.append(f"{indent_prefix}{ln}" if ln.strip() else ln)

                    replacement_text = "\n".join(rebuilt_lines)
                    replacements.append((hole_start, hole_end, replacement_text))
                    continue

            missing.append(child)

        if replacements:
            replacements.sort(key=lambda t: t[0], reverse=True)
            for start, end, rep in replacements:
                sketch = sketch[:start] + rep + sketch[end:]

        if missing:
            logger.warning(
                "Reconstruction skipped %d child(ren) missing valid hole offsets; "
                "their `sorry` placeholders will remain in the parent sketch.",
                len(missing),
            )
        return sketch

    def _reconstruct_decomposed_node_proof(
        self, decomposed_state: DecomposedFormalTheoremState, *, variant: ReconstructionVariant | None = None
    ) -> str:
        """
        Reconstruct proof text for an internal `DecomposedFormalTheoremState` by filling holes.
        """
        if decomposed_state["proof_sketch"] is None:
            return f"{decomposed_state['formal_theorem']} := by sorry\n"

        sketch = str(decomposed_state["proof_sketch"])

        # Fill holes using absolute offsets recorded during decomposition (hole_start/hole_end).
        # The legacy name/regex-based reconstruction path has been removed.
        return self._apply_offset_replacements(sketch, decomposed_state["children"], variant=variant)

    def _reconstruct_node_proof(self, node: TreeNode, *, variant: ReconstructionVariant | None = None) -> str:
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
        # Leaf node
        if isinstance(node, dict) and "formal_proof" in node and "children" not in node:
            return self._reconstruct_leaf_node_proof(cast(FormalTheoremProofState, node))

        # Internal node
        if isinstance(node, dict) and "children" in node:
            return self._reconstruct_decomposed_node_proof(cast(DecomposedFormalTheoremState, node), variant=variant)

        # Fallback for unexpected node types
        return "-- Unable to reconstruct proof for this node\n"

    def _extract_proof_body(self, child: TreeNode, *, variant: ReconstructionVariant | None = None) -> str:
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
            child_complete = self._reconstruct_node_proof(child, variant=variant)
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
        # Check if this looks like a full lemma/theorem statement.
        #
        # IMPORTANT: Do NOT treat a leading `have` as a top-level declaration here.
        # In many prover outputs (and inlining scenarios), the "proof" we receive is already a
        # tactic script that starts with `have ... := by ...` and ends with `exact ...`.
        # Stripping tactics after the first `:= by` would remove the binder and leave dangling
        # references like `exact h_main` (this was observed in partial.log).
        starts_with_decl = re.search(r"^\s*(lemma|theorem)\s+", proof, re.MULTILINE)

        if starts_with_decl:
            # This is a full lemma/theorem statement, find the first := by and extract from there
            match = re.search(r":=\s*by", proof)
            if match is None:
                # Has declaration but no := by, return sorry
                logger.warning(
                    "_extract_tactics_after_by received a lemma/theorem statement without ':= by'. "
                    "Returning 'sorry' as fallback."
                )
                return "sorry"
            # Extract everything after the first := by
            tactics = proof[match.end() :].strip()

            # Check if tactics contain another lemma/theorem (nested)
            if re.search(r"^\s*(lemma|theorem)\s+", tactics, re.MULTILINE):
                # Nested lemma, extract from it
                inner_match = re.search(r":=\s*by", tactics)
                if inner_match:
                    tactics = tactics[inner_match.end() :].strip()
                else:
                    logger.error("Could not extract tactics from nested lemma/theorem. Returning 'sorry'.")
                    return "sorry"

            return tactics

        # Not a full declaration, check if it has := by pattern (might be tactics with nested := by)
        match = re.search(r":=\s*by", proof)
        if match is None:
            # No := by pattern, return the whole proof (pure tactics)
            return proof.strip()

        # Has := by but doesn't start with declaration - this is tactics that contain := by
        # Return as-is (it's already just tactics)
        return proof.strip()

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

    def _snap_proof_indentation_levels(self, text: str) -> str:
        """
        Normalize indentation transitions using a simple indent stack.

        When indentation decreases, only allow dedenting to a previously-seen indentation
        level; otherwise, snap to the nearest enclosing (previous) indentation level.
        This avoids producing intermediate indentation levels like 2 when the script
        only used 0 and 4, which can break Lean's layout-sensitive parsing.
        """
        lines = text.split("\n")
        normalized_lines: list[str] = []
        indent_stack: list[int] = [0]

        for raw in lines:
            if not raw.strip():
                normalized_lines.append(raw)
                continue

            indent = len(raw) - len(raw.lstrip(" "))
            content = raw.lstrip(" ")

            current = indent_stack[-1]
            if indent > current:
                indent_stack.append(indent)
                normalized_lines.append(raw)
                continue

            if indent == current:
                normalized_lines.append(raw)
                continue

            while len(indent_stack) > 1 and indent < indent_stack[-1]:
                indent_stack.pop()

            snapped = indent_stack[-1]
            if indent != snapped:
                normalized_lines.append((" " * snapped) + content)
            else:
                normalized_lines.append(raw)

        return "\n".join(normalized_lines)

    def _fix_dangling_closing_tactics_after_comments(self, text: str) -> str:
        """
        Fix a common LLM formatting issue in tactic scripts:

          -- some comment
            exact h

        where the closing tactic is over-indented relative to the surrounding block. This can
        break Lean's layout-sensitive parsing after reconstruction.

        This method is intentionally conservative and is only applied for offset-based insertion
        paths where we know the exact hole indentation.
        """
        # Minimal "safe" set of closing tactics to snap when they appear after a comment and are
        # over-indented. These are common one-line goal-closing tactics in prover outputs.
        #
        # Keep this intentionally small to avoid accidentally changing semantics of genuinely
        # nested tactic blocks.
        closing_tactic_re = re.compile(
            r"^(exact|apply|simpa|simp|assumption|trivial|rfl|decide|aesop|omega|linarith|nlinarith|ring_nf|norm_num)\b"
        )
        lines = text.split("\n")
        changed = False
        prev_nonempty: str | None = None
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            stripped = ln.lstrip(" ")
            indent = len(ln) - len(stripped)
            if (
                prev_nonempty is not None
                and prev_nonempty.strip().startswith("--")
                and closing_tactic_re.match(stripped)
                and indent > 0
            ):
                # Snap to column 0 within the child proof body; the caller will indent it to the hole.
                lines[i] = stripped
                changed = True
            prev_nonempty = ln

        if changed:
            logger.debug("Reconstruction normalized an over-indented closing tactic following a comment.")
        return "\n".join(lines)

    def _snap_self_reference_closers_to_have_indent(self, text: str) -> str:
        """
        Another common LLM formatting error during inlining:

          have h_main : P := by
            ...
            exact h_main

        If `exact h_main` is indented under the `have` line, Lean treats it as part of the inner
        `by` block (where `h_main` is not yet in scope), leading to unsolved goals / layout errors.

        We conservatively snap any self-referencing closing tactics (`exact/apply/simpa using`) back
        to the indentation level of the corresponding `have` declaration.
        """
        have_re = re.compile(r"^(\s*)have\s+([^\s:(]+)\s*:")
        close_re_tpl = r"^\s*(?:exact|apply)\s+{name}\b|^\s*simpa\s+using\s+{name}\b"

        lines = text.split("\n")
        have_indents: dict[str, int] = {}
        for ln in lines:
            m = have_re.match(ln)
            if m:
                have_indents[m.group(2)] = len(m.group(1))

        if not have_indents:
            return text

        changed = False
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            stripped = ln.lstrip(" ")
            indent = len(ln) - len(stripped)
            for name, have_indent in have_indents.items():
                if indent <= have_indent:
                    continue
                close_re = re.compile(close_re_tpl.format(name=re.escape(name)))
                if close_re.match(stripped):
                    lines[i] = (" " * have_indent) + stripped
                    changed = True
                    break

        if changed:
            logger.debug("Reconstruction snapped a self-referencing closer back to its `have` indentation.")
        return "\n".join(lines)

    def _dedent_last_closer_out_of_inner_have_by(self, text: str) -> str:
        """
        Handle a common layout pitfall in nested `have ... := by` blocks.

        LLMs often emit a child proof of the form:

          have h_main : P := by
            simpa using h₁
            rfl

        where the final line (`rfl` here) is intended to close the *outer* goal, but is indented as
        if it were still inside the inner `by` block. This can produce "no goals to be solved" or
        leave the outer goal unsolved after inlining.

        We conservatively dedent the *last non-empty line* if:
        - it is indented under a `have ... := by` line, and
        - it matches a small goal-closing tactic set, and
        - it immediately follows another goal-closing tactic line at the same inner indentation.

        This is only applied for offset-based insertion paths.
        """
        lines = text.split("\n")
        last_nonempty = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_nonempty = i
                break
        if last_nonempty == -1:
            return text

        have_by_re = re.compile(r"^(\s*)have\s+[^\s:(]+\s*:.*?:=(?:\s|--[^\n]*|/-.*?-/)*by\b")
        closer_re = re.compile(r"^(exact|apply|simpa|simp|assumption|trivial|rfl|decide)\b")

        current_have_indent: int | None = None
        inner_indent: int | None = None
        prev_nonempty_idx: int | None = None
        prev_nonempty_indent: int | None = None

        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            stripped = ln.lstrip(" ")
            indent = len(ln) - len(stripped)

            m_have = have_by_re.match(ln)
            if m_have:
                current_have_indent = len(m_have.group(1))
                inner_indent = current_have_indent + PROOF_BODY_INDENT_SPACES

            # If we've dedented back out, drop the current have context.
            if current_have_indent is not None and indent <= current_have_indent and not m_have:
                current_have_indent = None
                inner_indent = None

            if (
                i == last_nonempty
                and current_have_indent is not None
                and inner_indent is not None
                and indent == inner_indent
                and closer_re.match(stripped)
                and prev_nonempty_idx is not None
                and prev_nonempty_indent == inner_indent
                and closer_re.match(lines[prev_nonempty_idx].lstrip(" "))
            ):
                lines[i] = (" " * current_have_indent) + stripped
                logger.debug("Reconstruction dedented final closing tactic out of inner `have ... := by` block.")
                return "\n".join(lines)

            prev_nonempty_idx = i
            prev_nonempty_indent = indent

        return text

    def _rewrite_trailing_apply_of_have_to_exact(self, text: str) -> str:
        """
        If the last non-empty line is `apply <name>` and the script previously defines
        `have <name> : ...`, rewrite it to `exact <name>`.
        """
        lines = text.split("\n")
        last_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_idx = i
                break
        if last_idx == -1:
            return text

        m = re.match(r"^(\s*)apply\s+([^\s]+)\s*$", lines[last_idx])
        if not m:
            return text

        base_indent, name = m.group(1), m.group(2)
        if not re.search(rf"\bhave\s+{re.escape(name)}\s*:", text):
            return text

        lines[last_idx] = f"{base_indent}exact {name}"
        return "\n".join(lines)

    def _normalize_child_proof_body(
        self,
        proof_body: str,
        *,
        offset_insertion: bool = False,
        variant: ReconstructionVariant | None = None,
    ) -> str:
        """
        Normalize a child proof body to be safe for textual inlining.

        This handles two common failure modes seen in partial logs:
        1) Mis-indentation where a line dedents to an indentation level that never occurred before
           (e.g., top-level 0, nested 4, then a line at 2). Lean is layout-sensitive, and this can
           produce "expected command" errors.
        2) Prover scripts that build `have h_main : goal := by ...` and then end with
           `apply h_main` (which often fails to close the goal). Inlining is more reliable when the
           script ends with `exact h_main`.
        """
        # Default behavior (variant=None) matches historical behavior: apply all normalizations.
        v = variant or self.ReconstructionVariant("default")

        text = proof_body
        if v.dedent_common:
            # First remove any common indentation (helps when the entire proof is shifted right).
            text = self._dedent_proof_body(text)
        if v.snap_indent_levels:
            # Then snap indentation transitions to valid previously-seen indentation levels.
            text = self._snap_proof_indentation_levels(text)
        if v.rewrite_trailing_apply:
            # Finally, rewrite trailing `apply <haveName>` into `exact <haveName>` when applicable.
            text = self._rewrite_trailing_apply_of_have_to_exact(text)

        # Additional normalization is only used for offset-based insertion (AST-guided) paths.
        if offset_insertion:
            if v.fix_dangling_closing_tactics_after_comments:
                text = self._fix_dangling_closing_tactics_after_comments(text)
            if v.snap_self_reference_closers:
                text = self._snap_self_reference_closers_to_have_indent(text)
            if v.dedent_last_closer_out_of_inner_have_by:
                text = self._dedent_last_closer_out_of_inner_have_by(text)
        return text

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
