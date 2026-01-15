from __future__ import annotations

import dataclasses
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

# Note: All LLM instances are imported from goedels_poetry.config.llm
from goedels_poetry.functools import maybe_save
from goedels_poetry.parsers.ast import AST
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

        # Introduce a bool | None to hold the final proof validation result
        # True = validation passed, False = validation failed, None = validation not run or exception occurred
        self.proof_validation_result: bool | None = None

        # Kimina-guided reconstruction metadata (persisted in checkpoints)
        self.reconstruction_attempts: int = 0
        self.reconstruction_strategy_used: str | None = None
        self.reconstruction_partial: bool = False
        self.reconstruction_failure_reason: str | None = None
        self.reconstruction_audit: dict[str, object] | None = None

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
        if not hasattr(self, "reconstruction_partial"):
            self.reconstruction_partial = False
        if not hasattr(self, "reconstruction_failure_reason"):
            self.reconstruction_failure_reason = None
        if not hasattr(self, "reconstruction_audit"):
            self.reconstruction_audit = None
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

    @staticmethod
    def load(filepath: str) -> GoedelsPoetryState:
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
    def set_parsed_proofs(self) -> None:
        """
        Clears the queue of proofs that needed AST parsing.
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

        modes = [
            self._reconstruction_mode_strict(),
            self._reconstruction_mode_permissive_a(),
            self._reconstruction_mode_permissive_b(),
        ]
        strict_result = self._reconstruct_with_mode(self._state.formal_theorem_proof, modes[0])
        chosen = strict_result
        if (
            (strict_result.failure_reason is not None or strict_result.partial)
            and strict_result.failure_reason != "ambiguity"
            and strict_result.permissive_ok
        ):
            for mode in modes[1:]:
                perm_result = self._reconstruct_with_mode(
                    self._state.formal_theorem_proof,
                    mode,
                    strict_result,
                )
                if perm_result.failure_reason is None and not perm_result.partial:
                    chosen = perm_result
                    break

        if chosen.mode_id == "strict":
            self._state.reconstruction_attempts = 1
        elif chosen.mode_id == "permissive_a":
            self._state.reconstruction_attempts = 2
        else:
            self._state.reconstruction_attempts = 3
        self._state.reconstruction_strategy_used = chosen.mode_id
        self._state.reconstruction_partial = chosen.partial
        self._state.reconstruction_failure_reason = chosen.failure_reason
        self._state.reconstruction_audit = chosen.audit

        proof_without_preamble = chosen.proof
        complete = combine_preamble_and_body(preamble, proof_without_preamble)
        self._state.final_complete_proof = complete
        return complete

    @dataclasses.dataclass(frozen=True)
    class ReconstructionMode:
        mode_id: str
        min_confidence: int
        allow_widen_span: bool
        allow_margin_normalization: bool
        allow_non_uniform_margin: bool

    @dataclasses.dataclass
    class ReconstructionContext:
        mode_id: str
        partial: bool = False
        failure_reason: str | None = None
        unresolved_holes: int = 0
        ambiguous: bool = False
        non_unique_holes: bool = False
        layout_sensitive: bool = False
        non_uniform_margin: bool = False
        boundary_mismatch: bool = False
        low_confidence: bool = False
        audit: list[dict[str, object]] = dataclasses.field(default_factory=list)

        def mark_ambiguity(self) -> None:
            self.ambiguous = True
            self.partial = True
            self.failure_reason = "ambiguity"

        def mark_failure(self, reason: str) -> None:
            if self.failure_reason is None:
                self.failure_reason = reason

        def permissive_ok(self) -> bool:
            if self.ambiguous:
                return False
            if self.non_unique_holes:
                return False
            if self.boundary_mismatch:
                return False
            if self.layout_sensitive:
                return False
            return self.unresolved_holes <= 1

    @dataclasses.dataclass(frozen=True)
    class ReconstructionResult:
        proof: str
        partial: bool
        failure_reason: str | None
        mode_id: str
        audit: dict[str, object]
        permissive_ok: bool

    def _reconstruction_mode_strict(self) -> ReconstructionMode:
        return self.ReconstructionMode(
            mode_id="strict",
            min_confidence=3,
            allow_widen_span=False,
            allow_margin_normalization=True,
            allow_non_uniform_margin=False,
        )

    def _reconstruction_mode_permissive_a(self) -> ReconstructionMode:
        return self.ReconstructionMode(
            mode_id="permissive_a",
            min_confidence=2,
            allow_widen_span=True,
            allow_margin_normalization=True,
            allow_non_uniform_margin=False,
        )

    def _reconstruction_mode_permissive_b(self) -> ReconstructionMode:
        return self.ReconstructionMode(
            mode_id="permissive_b",
            min_confidence=1,
            allow_widen_span=True,
            allow_margin_normalization=True,
            allow_non_uniform_margin=True,
        )

    def _reconstruct_with_mode(
        self,
        node: TreeNode,
        mode: ReconstructionMode,
        strict_result: ReconstructionResult | None = None,
    ) -> ReconstructionResult:
        context = self.ReconstructionContext(mode_id=mode.mode_id)
        proof = self._reconstruct_node_proof_ast(node, mode=mode, context=context)
        audit = {
            "mode_id": mode.mode_id,
            "partial": context.partial,
            "failure_reason": context.failure_reason,
            "unresolved_holes": context.unresolved_holes,
            "ambiguous": context.ambiguous,
            "non_unique_holes": context.non_unique_holes,
            "layout_sensitive": context.layout_sensitive,
            "non_uniform_margin": context.non_uniform_margin,
            "boundary_mismatch": context.boundary_mismatch,
            "low_confidence": context.low_confidence,
            "audits": context.audit,
        }
        permissive_ok = context.permissive_ok()
        if strict_result is not None and strict_result.failure_reason == "ambiguity":
            permissive_ok = False
        return self.ReconstructionResult(
            proof=proof,
            partial=context.partial,
            failure_reason=context.failure_reason,
            mode_id=mode.mode_id,
            audit=audit,
            permissive_ok=permissive_ok,
        )

    def _reconstruct_node_proof_ast(
        self,
        node: TreeNode,
        *,
        mode: ReconstructionMode,
        context: ReconstructionContext,
    ) -> str:
        if isinstance(node, dict) and "formal_proof" in node and "children" not in node:
            return self._reconstruct_leaf_node_proof_ast(cast(FormalTheoremProofState, node), mode, context)

        if isinstance(node, dict) and "children" in node:
            return self._reconstruct_decomposed_node_proof_ast(cast(DecomposedFormalTheoremState, node), mode, context)

        context.mark_failure("unexpected_node")
        return "-- Unable to reconstruct proof for this node\n"

    def _reconstruct_decomposed_node_proof(self, decomposed_state: DecomposedFormalTheoremState) -> str:
        """
        Reconstruct proof text for an internal `DecomposedFormalTheoremState` using AST-guided logic.
        """
        mode = self._reconstruction_mode_strict()
        context = self.ReconstructionContext(mode_id=mode.mode_id)
        return self._reconstruct_decomposed_node_proof_ast(decomposed_state, mode=mode, context=context)

    def _reconstruct_leaf_node_proof_ast(
        self,
        formal_proof_state: FormalTheoremProofState,
        mode: ReconstructionMode,
        context: ReconstructionContext,
    ) -> str:
        ast = formal_proof_state.get("ast")
        if ast is None:
            context.mark_failure("missing_ast")
            context.partial = True
            return f"{formal_proof_state['formal_theorem']} := by sorry\n"

        proof_body = self._extract_proof_body_from_ast(ast, mode, context)
        if proof_body is None:
            context.partial = True
            return f"{formal_proof_state['formal_theorem']} := by sorry\n"

        if formal_proof_state["parent"] is None:
            theorem_decl_full = str(formal_proof_state["formal_theorem"]).strip()
            theorem_sig = self._strip_decl_assignment(theorem_decl_full)
            return f"{theorem_sig} := by{proof_body}"

        return proof_body

    def _reconstruct_decomposed_node_proof_ast(  # noqa: C901
        self,
        decomposed_state: DecomposedFormalTheoremState,
        mode: ReconstructionMode,
        context: ReconstructionContext,
    ) -> str:
        sketch = decomposed_state.get("proof_sketch")
        ast = decomposed_state.get("ast")
        if sketch is None or ast is None:
            context.mark_failure("missing_ast")
            context.partial = True
            return f"{decomposed_state['formal_theorem']} := by sorry\n"

        sketch_text = str(sketch)
        holes_by_name = ast.get_sorry_holes_by_name()

        children = list(decomposed_state.get("children", []))

        def _child_sort_key(ch: TreeNode) -> tuple[int, int]:
            if isinstance(ch, dict):
                return (int(ch.get("hole_start") or 0), int(ch.get("hole_end") or 0))
            return (0, 0)

        children.sort(key=_child_sort_key)

        replacements: list[tuple[int, int, str]] = []
        for child in children:
            if not isinstance(child, dict):
                context.mark_failure("invalid_child")
                context.unresolved_holes += 1
                context.partial = True
                continue
            hole_name = child.get("hole_name")
            hole_start = None
            hole_end = None
            spans = []
            if isinstance(hole_name, str):
                spans = holes_by_name.get(hole_name, [])
            if len(spans) == 1:
                hole_start, hole_end = spans[0]
            elif child.get("hole_start") is not None and child.get("hole_end") is not None:
                hole_start = cast(int, child.get("hole_start"))
                hole_end = cast(int, child.get("hole_end"))
            else:
                fallback_spans = self._find_standalone_sorry_spans(sketch_text)
                if len(fallback_spans) == 1:
                    hole_start, hole_end = fallback_spans[0]

            if hole_start is None or hole_end is None:
                context.unresolved_holes += 1
                context.partial = True
                continue
            if len(spans) > 1 and hole_start is None:
                context.non_unique_holes = True
                context.mark_ambiguity()
                context.unresolved_holes += 1
                continue
            if not (0 <= hole_start < hole_end <= len(sketch_text)):
                context.unresolved_holes += 1
                context.partial = True
                continue

            proof_body = self._extract_proof_body_ast(child, mode, context)
            if proof_body is None:
                context.unresolved_holes += 1
                context.partial = True
                continue

            replacement = self._format_body_for_hole(
                sketch_text, hole_start, hole_end, proof_body, mode, context, child
            )
            if replacement is None:
                context.unresolved_holes += 1
                context.partial = True
                continue
            replacements.append((hole_start, hole_end, replacement))

        if replacements:
            replacements.sort(key=lambda t: t[0], reverse=True)
            for start, end, rep in replacements:
                sketch_text = sketch_text[:start] + rep + sketch_text[end:]

        return sketch_text

    def _extract_proof_body_ast(
        self,
        child: TreeNode,
        mode: ReconstructionMode,
        context: ReconstructionContext,
    ) -> str | None:
        if isinstance(child, dict) and "formal_proof" in child and "children" not in child:
            ast = child.get("ast")
            if ast is None:
                context.mark_failure("missing_ast")
                return None
            return self._extract_proof_body_from_ast(ast, mode, context)
        if isinstance(child, dict) and "children" in child:
            reconstructed = self._reconstruct_decomposed_node_proof_ast(
                cast(DecomposedFormalTheoremState, child),
                mode,
                context,
            )
            ast = child.get("ast")
            if ast is None:
                context.mark_failure("missing_ast")
                return None
            return self._extract_proof_body_from_ast(ast, mode, context, override_text=reconstructed)
        return None

    def _extract_proof_body_from_ast(  # noqa: C901
        self,
        ast: AST,
        mode: ReconstructionMode,
        context: ReconstructionContext,
        *,
        override_text: str | None = None,
    ) -> str | None:
        source = override_text or ast.get_source_text()
        if source is None:
            context.mark_failure("missing_source")
            return None

        by_info = self._find_by_tactic_info(ast.get_ast(), ast)
        if by_info is None:
            context.mark_failure("missing_by")
            return None
        by_token, tactic_node = by_info
        tokens = ast.get_tokens(tactic_node)
        if not tokens:
            context.mark_failure("missing_tokens")
            return None
        non_synth = [tok for tok in tokens if not tok.synthetic]
        span_tokens = non_synth if non_synth else tokens
        span_start = min(tok.start for tok in span_tokens)
        span_end = max(tok.end for tok in span_tokens)
        coverage = sum(tok.end - tok.start for tok in span_tokens)
        span_len = max(1, span_end - span_start)
        coverage_ratio = coverage / span_len
        contiguous = all(tok.start >= by_token.end and tok.end <= span_end for tok in span_tokens)

        boundary_ok = True
        if override_text is None:
            boundary_ok = self._token_boundaries_match(source, span_tokens)
            if not boundary_ok:
                context.boundary_mismatch = True
        confidence = 0
        if by_token.immediate:
            confidence += 1
        if non_synth:
            confidence += 1
        if contiguous:
            confidence += 1
        if coverage_ratio >= 0.15:
            confidence += 1

        context.audit.append({
            "coverage_ratio": coverage_ratio,
            "contiguous": contiguous,
            "boundary_ok": boundary_ok,
            "confidence": confidence,
        })

        if confidence < mode.min_confidence:
            context.low_confidence = True
            if not mode.allow_widen_span:
                context.mark_failure("low_confidence")
                return None

        if not contiguous:
            context.mark_failure("non_contiguous")
            return None

        start_c = ast.byte_to_char_index(by_token.end)
        end_c = len(source) if override_text is not None else ast.byte_to_char_index(span_end)
        return source[start_c:end_c]

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

    @dataclasses.dataclass(frozen=True)
    class _ByToken:
        start: int
        end: int
        immediate: bool

    def _find_by_tactic_info(  # noqa: C901
        self,
        node: dict[str, object] | list[object],
        ast: AST,
    ) -> tuple[_ByToken, dict[str, object]] | None:
        queue: list[dict[str, object]] = []
        if isinstance(node, dict):
            queue.append(node)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    queue.append(item)
        while queue:
            current = queue.pop(0)
            kind = current.get("kind")
            if kind == "Lean.Parser.Term.byTactic":
                args = current.get("args")
                if isinstance(args, list):
                    for idx, arg in enumerate(args):
                        if isinstance(arg, dict) and arg.get("val") == "by":
                            info = arg.get("info")
                            if isinstance(info, dict):
                                pos = info.get("pos")
                                if isinstance(pos, list) and len(pos) == 2:
                                    try:
                                        start_b_int = int(pos[0])
                                        end_b_int = int(pos[1])
                                    except (TypeError, ValueError):
                                        continue
                                    tactic_node = None
                                    for j in range(idx + 1, len(args)):
                                        if isinstance(args[j], dict):
                                            tactic_node = args[j]
                                            break
                                    if tactic_node is not None:
                                        return self._ByToken(start_b_int, end_b_int, True), tactic_node
            for val in current.values():
                if isinstance(val, dict):
                    queue.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            queue.append(item)
        return None

    def _token_boundaries_match(self, source: str, tokens: list[AST.Token]) -> bool:
        if not tokens:
            return False
        source_bytes = source.encode("utf-8")
        for tok in tokens:
            if tok.start < 0 or tok.end > len(source_bytes):
                return False
            if source_bytes[tok.start : tok.end] != tok.val.encode("utf-8"):
                return False
        return True

    def _find_standalone_sorry_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        idx = 0
        while True:
            idx = text.find("sorry", idx)
            if idx == -1:
                break
            before = text[idx - 1] if idx > 0 else " "
            after_idx = idx + len("sorry")
            after = text[after_idx] if after_idx < len(text) else " "
            if before.isspace() and after.isspace():
                spans.append((idx, after_idx))
            idx = after_idx
        return spans

    def _has_layout_sensitive_constructs(self, node: object) -> bool:
        if isinstance(node, dict):
            kind = node.get("kind")
            if isinstance(kind, str) and any(key in kind for key in ("calc", "match", "cases")):
                return True
            return any(self._has_layout_sensitive_constructs(val) for val in node.values())
        if isinstance(node, list):
            return any(self._has_layout_sensitive_constructs(item) for item in node)
        return False

    def _format_body_for_hole(  # noqa: C901
        self,
        sketch: str,
        hole_start: int,
        hole_end: int,
        proof_body: str,
        mode: ReconstructionMode,
        context: ReconstructionContext,
        child: TreeNode,
    ) -> str | None:
        line_start = sketch.rfind("\n", 0, hole_start) + 1
        line_prefix = sketch[line_start:hole_start]
        line_end = sketch.find("\n", line_start)
        if line_end == -1:
            line_end = len(sketch)
        line = sketch[line_start:line_end]
        leading_ws = line[: len(line) - len(line.lstrip(" \t"))]
        inline_hole = bool(line_prefix.strip())

        body = proof_body
        if body.startswith("\n"):
            body = body[1:]
        lines = body.split("\n")
        nonempty = [ln for ln in lines if ln.strip()]
        if not nonempty:
            return None

        ast_obj = child.get("ast") if isinstance(child, dict) else None
        if isinstance(ast_obj, AST):
            node = ast_obj.get_ast()
            layout_sensitive = self._has_layout_sensitive_constructs(cast(dict[str, object] | list[object], node))
        else:
            layout_sensitive = False
        if layout_sensitive:
            context.layout_sensitive = True

        indents = [len(ln) - len(ln.lstrip(" \t")) for ln in nonempty]
        min_indent = min(indents)
        uniform_margin = all(indent == min_indent for indent in indents)
        if not uniform_margin:
            context.non_uniform_margin = True

        if layout_sensitive:
            if leading_ws and nonempty[0].startswith(leading_ws):
                normalized_lines = lines
            else:
                context.mark_failure("layout_sensitive_mismatch")
                return None
        else:
            normalized_lines = lines
            if mode.allow_margin_normalization and (uniform_margin or mode.allow_non_uniform_margin):
                normalized_lines = []
                for ln in lines:
                    if ln.strip():
                        normalized_lines.append(ln[min_indent:] if ln.startswith(" " * min_indent) else ln)
                    else:
                        normalized_lines.append(ln)
            elif not uniform_margin and not mode.allow_non_uniform_margin:
                normalized_lines = lines

        rebuilt_lines: list[str] = []
        if inline_hole:
            proof_indent = leading_ws + (" " * PROOF_BODY_INDENT_SPACES)
            rebuilt_lines.append("")
            for ln in normalized_lines:
                rebuilt_lines.append(f"{proof_indent}{ln}" if ln.strip() else ln)
        else:
            for i, ln in enumerate(normalized_lines):
                if i == 0:
                    rebuilt_lines.append(ln)
                else:
                    rebuilt_lines.append(f"{line_prefix}{ln}" if ln.strip() else ln)
        return "\n".join(rebuilt_lines)

    def _strip_decl_assignment(self, formal_decl: str) -> str:
        """
        Strip any ':= ...' suffix from a declaration, returning only the header/signature.
        """
        idx = formal_decl.find(":=")
        return formal_decl[:idx].rstrip() if idx != -1 else formal_decl
