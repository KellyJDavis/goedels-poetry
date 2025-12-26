import traceback
from typing import ClassVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from goedels_poetry.agents.formal_theorem_syntax_agent import FormalTheoremSyntaxAgentFactory
from goedels_poetry.agents.formalizer_agent import FormalizerAgentFactory
from goedels_poetry.agents.informal_theorem_semantics_agent import InformalTheoremSemanticsAgentFactory
from goedels_poetry.agents.informal_theorem_syntax_agent import InformalTheoremSyntaxAgentFactory
from goedels_poetry.agents.proof_checker_agent import ProofCheckerAgentFactory, check_complete_proof
from goedels_poetry.agents.proof_corrector_agent import ProofCorrectorAgentFactory
from goedels_poetry.agents.proof_parser_agent import ProofParserAgentFactory
from goedels_poetry.agents.proof_sketcher_agent import ProofSketcherAgentFactory
from goedels_poetry.agents.prover_agent import ProverAgentFactory
from goedels_poetry.agents.search_query_agent import SearchQueryAgentFactory
from goedels_poetry.agents.sketch_backtrack_agent import SketchBacktrackAgentFactory
from goedels_poetry.agents.sketch_checker_agent import SketchCheckerAgentFactory
from goedels_poetry.agents.sketch_corrector_agent import SketchCorrectorAgentFactory
from goedels_poetry.agents.sketch_decomposition_agent import SketchDecompositionAgentFactory
from goedels_poetry.agents.sketch_parser_agent import SketchParserAgentFactory
from goedels_poetry.agents.state import DecomposedFormalTheoremStates, FormalTheoremProofStates, InformalTheoremState
from goedels_poetry.agents.supervisor_agent import SupervisorAgentFactory
from goedels_poetry.agents.vector_db_agent import VectorDBAgentFactory
from goedels_poetry.config.config import PROOF_RECONSTRUCTION
from goedels_poetry.config.kimina_server import KIMINA_LEAN_SERVER
from goedels_poetry.config.lean_explore_server import LEAN_EXPLORE_SERVER
from goedels_poetry.config.llm import (
    DECOMPOSER_AGENT_LLM,
    FORMALIZER_AGENT_LLM,
    FORMALIZER_AGENT_MAX_RETRIES,
    PROVER_AGENT_LLM,
    PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
    SEARCH_QUERY_AGENT_LLM,
    SEMANTICS_AGENT_LLM,
)
from goedels_poetry.state import GoedelsPoetryStateManager


class GoedelsPoetryConfig:
    """
    Configuration for the GoedelsPoetry system.

    Attributes
    ----------
    formalizer_agent_llm : BaseChatModel
        The language model that formalizes informal statements.
    prover_agent_llm : BaseChatModel
        The language model that proves formal statements.
    prover_agent_max_retries : int
        The the max number of retries for the prover agent
    semantics_agent_llm : BaseChatModel
        The language model that judges if a formalization retains the informal statement's semantics.
    decomposer_agent_llm : BaseChatModel
        The language model that decomposes a formal statement into formal statement that entail the
        formal statement
    kimina_lean_server_url: str
        The url of the REPL Kimina Lean server
    kimina_lean_server_max_retries: int
        The max number of retries for the Kimina Lean server

    """

    def __init__(
        self,
        formalizer_agent_llm: BaseChatModel = FORMALIZER_AGENT_LLM,
        formalizer_agent_max_retries: int = FORMALIZER_AGENT_MAX_RETRIES,
        prover_agent_llm: BaseChatModel = PROVER_AGENT_LLM,
        prover_agent_max_retries: int = PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS,
        semantics_agent_llm: BaseChatModel = SEMANTICS_AGENT_LLM,
        decomposer_agent_llm: BaseChatModel = DECOMPOSER_AGENT_LLM,
        kimina_lean_server_url: str = KIMINA_LEAN_SERVER["url"],
        kimina_lean_server_max_retries: int = KIMINA_LEAN_SERVER["max_retries"],
        proof_reconstruction_max_candidates: int = PROOF_RECONSTRUCTION["max_candidates"],
    ):
        self.formalizer_agent_llm = formalizer_agent_llm
        self.formalizer_agent_max_retries = formalizer_agent_max_retries
        self.prover_agent_llm = prover_agent_llm
        self.prover_agent_max_retries = prover_agent_max_retries
        self.semantics_agent_llm = semantics_agent_llm
        self.decomposer_agent_llm = decomposer_agent_llm
        self.kimina_lean_server_url = kimina_lean_server_url
        self.kimina_lean_server_max_retries = kimina_lean_server_max_retries
        self.proof_reconstruction_max_candidates = proof_reconstruction_max_candidates


class GoedelsPoetryFramework:
    """
    The framework that takes a GoedelsPoetryStateManager setups the agents, and organizes the
    multi-agent system. The framework will be controlled by a supervisor agent.
    """

    # Mapping from method names to user-friendly phase names
    _PHASE_NAMES: ClassVar[dict[str, str]] = {
        "formalize_informal_theorem": "Formalizing theorem",
        "check_informal_theorem_syntax": "Checking formalization syntax",
        "check_informal_theorem_semantics": "Checking formalization semantics",
        "check_theorems_syntax": "Checking theorem syntax",
        "prove_theorems": "Proving theorems",
        "check_theorems_proofs": "Validating proofs",
        "request_proofs_corrections": "Requesting proof corrections",
        "parse_proofs": "Parsing proofs",
        "sketch_proofs": "Sketching proofs",
        "check_proof_sketches_syntax": "Checking proof sketch syntax",
        "request_proof_sketches_corrections": "Requesting sketch corrections",
        "request_proof_sketches_backtrack": "Backtracking proof sketches",
        "parse_proof_sketches": "Parsing proof sketches",
        "decompose_proof_sketches": "Decomposing proof sketches",
        "generate_search_queries": "Generating search queries",
        "query_vectordb": "Querying vector database for relevant theorems",
    }

    def __init__(
        self,
        config: GoedelsPoetryConfig,
        state_manager: GoedelsPoetryStateManager,
        console: Console | None = None,
    ):
        self._config = config
        self._state_manager = state_manager
        self._console = console if console is not None else Console()

    def run(self) -> None:
        """
        Runs the GoedelsPoetry system until it has proven the informal theorem or until it has
        failed to prove that theorem.
        """
        supervisor_agent = SupervisorAgentFactory.create_agent(state_manager=self._state_manager)

        while not self._state_manager.is_finished:
            action = supervisor_agent.get_action()
            self._state_manager.add_action(action)

            # Show progress indicator for the current phase
            phase_name = self._PHASE_NAMES.get(action, action)
            with self._console.status(f"[bold blue]{phase_name}..."):
                getattr(self, action)()

        # Ensure finish() is called if it wasn't the last action
        # This handles cases where is_finished was set inside an action method
        if self._state_manager._state.action_history and self._state_manager._state.action_history[-1] != "finish":
            self._state_manager.add_action("finish")
            self.finish()

    def formalize_informal_theorem(self) -> None:
        """
        Formalizes the pending informal theorem
        """
        # Create formalizer agent
        formalizer_agent = FormalizerAgentFactory.create_agent(llm=self._config.formalizer_agent_llm)

        # Get informal theorem state and formalize informal theorem state
        informal_theorem_state = self._state_manager.get_informal_theorem_to_formalize()
        informal_theorem_state = cast(InformalTheoremState, formalizer_agent.invoke(informal_theorem_state))
        self._state_manager.set_formalized_informal_theorem(informal_theorem_state)

    def check_informal_theorem_syntax(self) -> None:
        """
        Checks syntax of the pending informal theorem's formalization
        """
        # Create informal theorem syntax agent
        syntax_agent = InformalTheoremSyntaxAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get informal theorem state and syntax check informal theorem's formalization
        informal_theorem_state = self._state_manager.get_informal_theorem_to_validate()
        informal_theorem_state = cast(InformalTheoremState, syntax_agent.invoke(informal_theorem_state))
        self._state_manager.set_validated_informal_theorem(informal_theorem_state)

    def check_informal_theorem_semantics(self) -> None:
        """
        Checks semantics of the pending informal theorem's formalization
        """
        # Create semantics agent
        semantics_agent = InformalTheoremSemanticsAgentFactory.create_agent(llm=self._config.semantics_agent_llm)

        # Get informal theorem state and check semantics of formalization
        informal_theorem_state = self._state_manager.get_informal_theorem_to_check_semantics_of()
        informal_theorem_state = cast(InformalTheoremState, semantics_agent.invoke(informal_theorem_state))
        self._state_manager.set_semantically_checked_informal_theorem(informal_theorem_state)

    def check_theorems_syntax(self) -> None:
        """
        Checks syntax of the pending formal theorems'
        """
        # Create formal theorem syntax agent
        syntax_agent = FormalTheoremSyntaxAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get formal theorem states and syntax check formal theorems
        formal_theorem_states = self._state_manager.get_theorems_to_validate()
        formal_theorem_states = cast(FormalTheoremProofStates, syntax_agent.invoke(formal_theorem_states))
        self._state_manager.set_validated_theorems(formal_theorem_states)

    def prove_theorems(self) -> None:
        """
        Proves the pending formal theorems
        """
        # Create prover agent
        prover_agent = ProverAgentFactory.create_agent(llm=self._config.prover_agent_llm)

        # Get formal theorems to prove and prove them
        formal_theorem_states = self._state_manager.get_theorems_to_prove()
        formal_theorem_states = cast(FormalTheoremProofStates, prover_agent.invoke(formal_theorem_states))
        self._state_manager.set_proven_theorems(formal_theorem_states)

    def check_theorems_proofs(self) -> None:
        """
        Checks validity of the pending formal theorems' proofs
        """
        # Create proof checker agent
        proof_checker_agent = ProofCheckerAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get formal theorem states and check their proofs' validity
        formal_theorem_states = self._state_manager.get_proofs_to_validate()
        formal_theorem_states = cast(FormalTheoremProofStates, proof_checker_agent.invoke(formal_theorem_states))
        self._state_manager.set_validated_proofs(formal_theorem_states)

    def request_proofs_corrections(self) -> None:
        """
        Requests corrections for the the pending invalid formal theorems' proofs
        """
        # Create corrector agent
        corrector_agent = ProofCorrectorAgentFactory.create_agent()

        # Get formal theorem states with invalid proofs and request corrections
        formal_theorem_states = self._state_manager.get_proofs_to_correct()
        formal_theorem_states = cast(FormalTheoremProofStates, corrector_agent.invoke(formal_theorem_states))
        self._state_manager.set_corrected_proofs(formal_theorem_states)

    def parse_proofs(self) -> None:
        """
        Parses validated formal proofs into abstract syntax trees (AST)
        """
        # Create proof parser agent
        proof_parser_agent = ProofParserAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get formal theorem states and parse their proofs into ASTs
        formal_theorem_states = self._state_manager.get_proofs_to_parse()
        formal_theorem_states = cast(FormalTheoremProofStates, proof_parser_agent.invoke(formal_theorem_states))
        self._state_manager.set_parsed_proofs(formal_theorem_states)

    def generate_search_queries(self) -> None:
        """
        Generates search queries for theorems pending decomposition.
        """
        # Create search query agent
        search_query_agent = SearchQueryAgentFactory.create_agent(llm=SEARCH_QUERY_AGENT_LLM)

        # Get states needing query generation
        states = self._state_manager.get_theorems_for_search_query_generation()
        states = cast(DecomposedFormalTheoremStates, search_query_agent.invoke(states))

        # Store states with generated queries
        self._state_manager.set_theorems_with_search_queries_generated(states)

    def query_vectordb(self) -> None:
        """
        Queries the vector database for relevant theorems using generated search queries.
        """
        # Create vector DB agent
        vector_db_agent = VectorDBAgentFactory.create_agent(
            server_url=LEAN_EXPLORE_SERVER["url"], package_filters=LEAN_EXPLORE_SERVER["package_filters"]
        )

        # Get states with search queries that need vector DB lookup
        states = self._state_manager.get_theorems_with_search_queries_for_vectordb()
        states = cast(DecomposedFormalTheoremStates, vector_db_agent.invoke(states))

        # Store states with vector DB results
        self._state_manager.set_theorems_with_vectordb_results(states)

    def sketch_proofs(self) -> None:
        """
        Sketched a proof of pending formal theorems' that proved too difficult to prove directly
        """
        # Create proof sketch agent
        proof_sketch_agent = ProofSketcherAgentFactory.create_agent(llm=self._config.decomposer_agent_llm)

        # Get decomposed formal theorem states and sketch their proofs
        decomposed_states = self._state_manager.get_theorems_to_sketch()
        decomposed_states = cast(DecomposedFormalTheoremStates, proof_sketch_agent.invoke(decomposed_states))
        self._state_manager.set_sketched_theorems(decomposed_states)

    def check_proof_sketches_syntax(self) -> None:
        """
        Checks the syntax of the pending proof sketches
        """
        # Create sketch checker agent
        sketch_checker_agent = SketchCheckerAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get decomposed formal theorem states and check their sketches' syntax
        decomposed_states = self._state_manager.get_sketches_to_validate()
        decomposed_states = cast(DecomposedFormalTheoremStates, sketch_checker_agent.invoke(decomposed_states))
        self._state_manager.set_validated_sketches(decomposed_states)

    def request_proof_sketches_corrections(self) -> None:
        """
        Requests corrections for the the pending invalid proof sketches
        """
        # Create corrector agent
        corrector_agent = SketchCorrectorAgentFactory.create_agent()

        # Get decomposed formal theorem states with invalid sketches and request corrections
        decomposed_states = self._state_manager.get_sketches_to_correct()
        decomposed_states = cast(DecomposedFormalTheoremStates, corrector_agent.invoke(decomposed_states))
        self._state_manager.set_corrected_sketches(decomposed_states)

    def request_proof_sketches_backtrack(self) -> None:
        """
        Requests re-sketching for proof sketches whose children failed to prove. This is a
        different type of correction than syntax corrections - we're asking for a completely
        different decomposition approach because the previous one didn't work out.
        """
        # Create backtrack agent with specialized prompts
        backtrack_agent = SketchBacktrackAgentFactory.create_agent()

        # Get decomposed formal theorem states that need backtracking and request new sketches
        decomposed_states = self._state_manager.get_sketches_to_backtrack()
        decomposed_states = cast(DecomposedFormalTheoremStates, backtrack_agent.invoke(decomposed_states))
        self._state_manager.set_backtracked_sketches(decomposed_states)

    def parse_proof_sketches(self) -> None:
        """
        Parses validated sketch into abstract syntax trees (AST)
        """
        # Create sketch parser agent
        sketch_parser_agent = SketchParserAgentFactory.create_agent(
            server_url=KIMINA_LEAN_SERVER["url"], server_max_retries=KIMINA_LEAN_SERVER["max_retries"]
        )

        # Get decomposed formal theorem states and parse their sketches into ASTs
        decomposed_states = self._state_manager.get_sketches_to_parse()
        decomposed_states = cast(DecomposedFormalTheoremStates, sketch_parser_agent.invoke(decomposed_states))
        self._state_manager.set_parsed_sketches(decomposed_states)

    def decompose_proof_sketches(self) -> None:
        """
        Extracts unproven child theorems from decomposed formal theorem states and for each such
        extracted theorm introduces a formal theorem child of the parent decomposed formal theorem.
        """
        # Create decomposition agent
        decomposition_agent = SketchDecompositionAgentFactory.create_agent()

        # Get decomposed formal theorem states with sketches and decompose each into formal theorems
        decomposed_states = self._state_manager.get_sketches_to_decompose()
        decomposed_states = cast(DecomposedFormalTheoremStates, decomposition_agent.invoke(decomposed_states))
        self._state_manager.set_decomposed_sketches(decomposed_states)

    def finish(self) -> None:
        """
        Finishes the proof process.
        """
        # Print the reason for finishing
        reason = self._state_manager.reason if self._state_manager.reason else "Unknown reason"
        self._console.print(f"\n{'=' * 80}")
        self._console.print(f"Proof process completed: {reason}")
        self._console.print(f"{'=' * 80}\n")

        # If successful, print the complete proof
        if self._state_manager.reason == "Proof completed successfully.":
            try:
                complete_proof = self._state_manager.reconstruct_complete_proof()

                # Perform final verification of the complete proof
                self._console.print("\n[bold blue]Performing final proof verification...[/bold blue]")
                is_valid, error_msg = check_complete_proof(
                    complete_proof,
                    server_url=self._config.kimina_lean_server_url,
                    server_max_retries=self._config.kimina_lean_server_max_retries,
                )

                # If final verification fails, try Kimina-guided reconstruction variants.
                if not is_valid:
                    self._console.print(
                        "[bold yellow]⚠ Final verification failed. Trying Kimina-guided reconstruction variants...[/bold yellow]"
                    )
                    guided_proof, guided_ok, guided_err = self._state_manager.reconstruct_complete_proof_kimina_guided(
                        server_url=self._config.kimina_lean_server_url,
                        server_max_retries=self._config.kimina_lean_server_max_retries,
                        max_candidates=self._config.proof_reconstruction_max_candidates,
                    )
                    if guided_ok:
                        complete_proof = guided_proof
                        is_valid = True
                        error_msg = ""
                        self._console.print(
                            "[bold green]✓ Kimina-guided reconstruction succeeded: Proof is valid[/bold green]"
                        )
                    else:
                        # Keep the original failing proof for display, but show the last guided error.
                        if guided_err:
                            error_msg = guided_err

                # Store validation result in state (used by CLI for .proof vs .failed-proof)
                self._state_manager._state.proof_validation_result = is_valid
                # Store the final proof so writers don't recompute a failing variant.
                self._state_manager._state.final_complete_proof = complete_proof

                if is_valid:
                    self._console.print("[bold green]✓ Final verification passed: Proof is valid[/bold green]")
                else:
                    self._console.print("[bold yellow]⚠ Final verification failed: Proof may be invalid[/bold yellow]")
                    if error_msg:
                        self._console.print(f"[yellow]Verification errors:[/yellow]\n{error_msg}")

                self._console.print("\nComplete Lean4 Proof:")
                self._console.print("-" * 80)
                self._console.print(complete_proof, markup=False)
                self._console.print("-" * 80)
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                self._console.print(f"Error reconstructing proof: {e}")
                self._console.print(traceback.format_exc())
            except Exception as e:
                # If final verification fails, still print the proof but warn the user
                self._console.print(f"[bold yellow]Warning:[/bold yellow] Final verification encountered an error: {e}")
                # Try to still print the proof if it was successfully reconstructed
                try:
                    complete_proof = self._state_manager.reconstruct_complete_proof()
                    self._console.print("\nComplete Lean4 Proof:")
                    self._console.print("-" * 80)
                    self._console.print(complete_proof, markup=False)
                    self._console.print("-" * 80)
                except Exception:
                    self._console.print("Could not reconstruct proof for display.")
