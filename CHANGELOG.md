# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.6] - 2025-12-30

### Added
- Timestamps in debug logging output: all debug log entries now include date and time information (formatted as YYYY-MM-DD HH:MM:SS) when `GOEDELS_POETRY_DEBUG` environment variable is enabled. Timestamps are displayed in dimmed style within log titles and messages for all logging functions (`log_llm_prompt`, `log_llm_response`, `log_kimina_response`, `log_vectordb_response`, and `log_debug_message`), providing temporal context for debugging without overwhelming the main content.

## [1.2.5] - 2025-12-29

### Fixed
- Fixed type extraction for `set_with_hypothesis` bindings from earlier sorries: improved type collection from multiple sorries by merging types from all sorries and using exact key matching instead of substring matching to prevent false positives (e.g., "h1" matching "h10")
- Fixed value extraction failures for `let` and `set` bindings: enhanced error detection, added comprehensive fallback strategies (AST extraction → type extraction → goal context types), and improved error messages with context about what went wrong
- Fixed missing types for `set_with_hypothesis` bindings: now constructs equality types directly from the set statement AST when goal context is unavailable (e.g., for `set S := Finset.range 10000 with hS`, constructs type `S = Finset.range 10000`)
- Fixed type determination warnings for general bindings (`have`, `obtain`, `choose`, `generalize`, `match`, `suffices`): reduced frequency of Prop fallbacks through improved type determination with binding-type-specific handling

### Changed
- Improved type determination for general bindings: refactored and consolidated type determination logic, removed redundant `goal_var_types` checks, and added binding-type-specific handling that matches the characteristics of each binding type
- Enhanced goal context parsing: added support for assignment syntax (`name : type := value`) and improved handling of multiple variables with same type declaration
- Improved fallback chain for `set_with_hypothesis` bindings: primary uses goal context types, fallback constructs type from AST, final fallback uses Prop only when all methods fail

### Added
- New helper function `__construct_set_with_hypothesis_type` for constructing equality type AST for `set_with_hypothesis` bindings when goal context is unavailable
- New helper function `__determine_general_binding_type` for binding-type-specific type determination with appropriate fallback strategies
- Comprehensive test coverage: added 14+ unit tests for type determination, 8 unit tests for `set_with_hypothesis` type construction, 15+ edge case tests for value extraction, and multiple integration tests verifying improvements work in full context

## [1.2.4] - 2025-12-29

### Fixed
- Fixed sorry type lookup to check target then enclosed context instead of just local context
- Replaced substring matching with exact key matching when identifying target-specific sorries to prevent false positives.

## [1.2.3] - 2025-12-29

### Fixed
- Fixed inconsistent escape sequence handling for informal theorems: added escape sequence normalization (e.g., converting literal \n to actual newline) when using `--informal-theorem` command-line argument, ensuring consistent behavior with formal theorems and directory-based informal theorems.

## [1.2.2] - 2025-12-28

### Fixed
- Fixed proof reset routing: when proofs reach `max_self_correction_attempts` and `pass_attempts < max_pass`, they are now correctly routed to the `proof_prove_queue` instead of the `proof_correct_queue`. This ensures reset proofs use the initial prompt (via prover agent) rather than the correction prompt, and eliminates misleading log messages showing "Round -1" and error "None".

### Changed
- Documentation styling: decreased h2 code font size to 0.6em in documentation stylesheets for improved readability.

### Tests
- Added comprehensive test suite (`test_proof_reset_routing.py`) with 5 tests verifying correct routing for reset proofs, correction proofs, max_pass proofs, successful proofs, and mixed scenarios.

## [1.2.1] - 2026-01-10

### Fixed
- Fixed incorrect handling of literal escape sequences (e.g., \n as two characters: backslash followed by 'n') for formal theorem files or command-line arguments by converting literal escape sequences to actual characters.

## [1.2.0] - 2026-01-10

### Added
- Kimina-guided proof assembly fallback: when a run finishes with "Proof completed successfully." but final verification fails, the system now attempts a bounded search over reconstruction normalization variants and selects the first whole-file proof that Kimina marks complete
- `[PROOF_RECONSTRUCTION]` configuration section with `max_candidates` parameter (default 64) to control the maximum number of reconstruction variants to try during Kimina-guided assembly
- `reconstruction_attempts` and `reconstruction_strategy_used` fields in state persistence to track reconstruction attempts across checkpoints
- `final_complete_proof` field in state to persist successful reconstruction results, preventing recomputation of failing variants
- Unit test coverage for deterministic and capped variant generation
- Kimina-backed integration test for guided reconstruction that fails under baseline reconstruction but passes with guided selection

### Changed
- Increased default `max_candidates` from 12 to 64 to improve reconstruction success rates
- Detailed attempt logging for Kimina-guided reconstruction now gated behind `GOEDELS_POETRY_DEBUG` environment variable for consistent debug output
- Final verification failure handling: when guided reconstruction succeeds after initial verification failure, the run is now treated as successful with `proof_validation_result=True` and `.proof` output is written

### Fixed
- Fixed anonymous-have decomposition to preserve enclosing binders: Kimina's placeholder have-id `"[anonymous]"` (and `have _`) are now treated as truly anonymous, with decomposition assigning stable synthetic names (`gp_anon_have__...`) instead of emitting `lemma [anonymous] : ...` with missing context
- Fixed lemma-in-lemma decomposition to preserve enclosing binders: normalized declaration `kind` strings in AST utilities so unqualified `lemma`/`theorem`/`def` (as emitted by Kimina) are treated equivalently to fully-qualified parser kinds
- Improved `_extract_decl_id_name` robustness by searching for `declId` within the subtree, working with `declModifiers`/`group` wrapper AST shapes
- Enhanced binder extraction to prefer extracting binders from `Lean.Parser.Command.declSig` to reliably recover parameters/hypotheses for unqualified declaration nodes
- Fixed CI failures on Python 3.11 by implementing lazy import of `check_complete_proof` inside Kimina-guided reconstruction, avoiding `kimina_client` import crashes during test collection
- Broadened Kimina integration-test import guards to skip on any import exception, improving robustness across different Python environments
- Fixed enclosing-theorem lookup to thread anonymous-have map through recursion so anonymous subgoals inherit enclosing parameters/hypotheses reliably across AST shapes

### Removed
- Removed legacy regex proof reconstruction path: dropped the transitional, regex/name-based proof reconstruction mechanism (v1.1.5-era) and made the v1.1.6 AST-guided, offset-based reconstruction the only supported mechanism
- Deleted all legacy reconstruction helpers and fallbacks in `goedels_poetry/state.py` including `_replace_*`, `_extract_have_name`, anonymous-have regex matching, main-body regex scanning, and related fallback code

### Tests
- Added regression test ensuring `"[anonymous]"` is converted to `gp_anon_have__...` and the generated lemma carries enclosing binders
- Added regression tests covering "lemma-in-lemma" decomposition including qualified/unqualified lemma nodes, anonymous `have`, and `<main body>` extraction
- Added backward-compatible unpickling defaults for older checkpoints to ensure state persistence works across versions

## [1.1.6] - 2025-12-25

### Changed
- Proof reconstruction now uses AST-guided, offset-based replacement strategy instead of brittle regex-based matching, enabling deterministic splicing of proofs into exact `sorry` placeholder locations and avoiding failures from formatting/comments/layout variations.
- AST now carries `source_text` (exact Kimina-parsed text) and `body_start` (offset where the body begins) to enable translation of token positions from the AST into body-relative offsets used by stored `proof_sketch` strings.
- Reconstruction now prefers offset-based splicing (collecting start/end/replacement tuples and applying replacements right-to-left) with fallback to legacy name-based inlining only when hole metadata is missing.

### Fixed
- Fixed reconstruction correctness issues: convert Kimina/ast_export UTF-8 byte offsets to Python string indices when mapping `sorry` spans to avoid misalignment errors.
- Added conservative, offset-only normalization for nested `have ... := by` blocks to avoid layout/scope errors ("unsolved goals" / "no goals to be solved") after inlining.
- Narrowed closing-tactic indentation fix to a minimal, high-confidence set of one-line goal-closing tactics (e.g., `exact`, `apply`, `simp/simpa`, `assumption`, `rfl`, `decide`, `aesop`, `linarith`, `nlinarith`, `ring_nf`, `norm_num`) to prevent Lean layout-sensitive parsing failures from over-indented closing tactics.

### Added
- New utility to extract `tacticSorry` occurrences from the Kimina/ast_export syntax tree and map them to surrounding subgoal context (named `have`, synthetic anonymous-have, or `<main body>`), supporting multiple holes per name.
- `FormalTheoremProofState` now includes `hole_name`, `hole_start`, and `hole_end` fields to track target subgoal/hole metadata in parent sketches, populated when subgoals are created from a sketch.
- Kimina-backed, runtime-generated reconstruction integration suite (`tests/test_reconstruction_kimina_generated.py`) with configurable corpus generation via `RECONSTRUCTION_TEST_CASES` (default 600) and `RECONSTRUCTION_TEST_SEED` (default 0) environment variables, validating reconstruction by running Kimina `check` on reconstructed code.

### Tests
- Added comprehensive regression test suite for AST-guided proof reconstruction (`tests/test_reconstruction_ast_guided.py`) covering offset-based replacement, hole metadata extraction, and edge cases.
- Generated integration suite passes on default corpus (600 test cases) against running Kimina server, ensuring reconstruction robustness across diverse proof patterns.

## [1.1.5] - 2025-12-24

### Fixed
- Proof reconstruction now handles anonymous have statements (`have : ... := by sorry`) by assigning stable synthetic names (`gp_anon_have__<decl>__<idx>`) to enable proper subgoal extraction and proof inlining.
- AST traversal, lookup, and rewriting now resolve synthetic subgoal names end-to-end, allowing anonymous have statements to be treated as named subgoals throughout the proof pipeline.

### Tests
- Added comprehensive unit tests for anonymous have statement handling including synthetic name generation, code extraction, determinism across multiple anonymous haves, and proof reconstruction with anonymous have subgoals.

## [1.1.4] - 2025-12-23

### Fixed
- Proof reconstruction now correctly handles comments between `:=` and `by` and between `by` and `sorry` in have statements, preventing `sorry` statements from remaining in final proofs when child proofs are inlined.
- Proof reconstruction improved to treat proofs starting with `have` as tactic scripts (don't strip `:= by`), preventing dangling `exact h_*` statements.
- Proof reconstruction now dedents child proof bodies before re-indenting to preserve Lean layout requirements (e.g., in `calc` statements).
- Proof reconstruction robustified: normalized child proof bodies by snapping dedents to previously-seen indentation levels to avoid Lean layout/"expected command" failures.
- Proof reconstruction now rewrites trailing `apply <haveName>` to `exact <haveName>` when the proof defines `have <haveName> : …`, ensuring goals close reliably.
- Fixed `_extract_tactics_after_by()` to correctly distinguish between full lemma/theorem statements and pure tactics that may contain `:= by` in the middle (e.g., in calc statements).

### Tests
- Added comprehensive regression test coverage for proof reconstruction edge cases including comments around `sorry`, multiple comment lines, and misindented `apply h_main` scenarios.
- Added tests for `_extract_tactics_after_by()` edge cases covering nested lemma statements and tactics containing `:= by` patterns.

## [1.1.3] - 2025-12-23

### Fixed
- Proof reconstruction now tolerates comments between `:=` and `by` and between `by` and `sorry`, ensuring placeholder sorries are removed before submission.
- AST parser now extracts all theorem binders even when `byTactic`/`tacticSeq` nodes appear in type expressions before `:=`, preventing missing hypotheses in subgoal extraction.

### Tests
- Added regression coverage for combining theorem stubs that include comments around `sorry` to prevent reintroducing incomplete proofs.
- Added extensive AST parser regression tests covering binder extraction edge cases (e.g., `byTactic` in types, multiple/nested binder lists, no-binder theorems).

## [1.1.2] - 2025-12-19

### Documentation
- README proof flow clarified: verification is called out separately, vector search happens before decomposition, and sketches explicitly note use of retrieved lemmas.
- Configuration examples now include provider/url/max_tokens defaults for all agents and highlight optional provider-specific tuning flags.

## [1.1.1] - 2025-12-18

### Changed
- Default OpenAI model bumped to `gpt-5.2-2025-12-11` across all agents and configuration defaults (sample config, README quick start, and configuration guide)

### Removed
- Dropped the `openai-model-list` helper file that is no longer needed

## [1.1.0] - 2025-12-18

### Added
- Quick Start: OpenAI instructions covering all agents using OpenAI-compatible endpoints
- BibTeX citations updated to include the Gödel's Poetry arXiv entry and a LeanExplore reference

### Changed
- Unified LLM provider handling: all agents now accept `provider` values of `ollama`, `vllm`, `lmstudio`, or `openai`, with provider-specific defaults and automatic API key handling
- Decomposer agent configuration now aligns with other agents (`provider`, `url`, `max_tokens`, optional `num_ctx`) instead of using OpenAI-only `max_completion_tokens`
- Removed `api_key` requirements from configuration sections; OpenAI now relies on `OPENAI_API_KEY` and other providers derive keys automatically
- Formalizer prompt refined to encourage explicit thinking before generating Lean statements

### Documentation
- Clarified provider/endpoints section to reflect unified provider support for every agent
- Added LeanExplore citation and refreshed instructions to set `OPENAI_API_KEY` when any agent uses OpenAI

## [1.0.0] - 2025-12-15

### Changed
- Version bump to 1.0.0: First stable release marking production readiness and API stability
- Updated version in `pyproject.toml` and `goedels_poetry/__init__.py` to reflect stable release status

## [0.0.14] - 2025-12-14

### Documentation
- Updated documentation to accurately reflect current codebase state: fixed discrepancies and added missing information across all documentation files
- Updated README.md batch processing description to mention both `.proof` and `.failed-proof` file outputs, matching actual CLI behavior
- Added `max_remote_retries` parameter documentation for all LLM agents (FORMALIZER, PROVER, SEMANTICS, SEARCH_QUERY, DECOMPOSER) in both README.md and CONFIGURATION.md, including in simplified config examples
- Clarified distinction between `max_retries` (formalization attempts, FORMALIZER only) and `max_remote_retries` (network/API retries, all LLM agents) in CONFIGURATION.md
- Updated Makefile test-integration instructions to use `kimina-ast-server` command instead of deprecated `python -m server`, matching README.md
- Enhanced CONFIGURATION.md with complete LM Studio parameter documentation and ensured all provider descriptions are accurate

## [0.0.13] - 2025-12-14

### Added
- Parse failure handling with requeueing and attempt tracking: implemented robust error handling for `LLMParsingError` exceptions across all agents (formalizer, semantics, proof sketcher, prover) with centralized attempt tracking and automatic requeueing in the state manager
- `max_remote_retries` configuration option to LLM settings (`FORMALIZER_AGENT_LLM`, `PROVER_AGENT_LLM`, `SEMANTICS_AGENT_LLM`, `SEARCH_QUERY_AGENT_LLM`, `DECOMPOSER_AGENT_LLM`) for controlling maximum remote API retry attempts
- Proof file extensions based on validation result: proofs are now written to `.proof` files for valid proofs and `.failed-proof` files for invalid proofs, validation exceptions, or non-successful completions
- `proof_validation_result` field in `GoedelsPoetryState` to track final validation status from the Kimina server
- Comprehensive test coverage for parse failure handling functionality (519 lines of new tests)
- `parse_semantic_check_response` function moved to `common.py` for better code organization and Python 3.11 compatibility

### Changed
- Refactored LLM initialization from lazy loading to eager loading, improving performance and error detection
- Updated type hinting in CLI module to use `TYPE_CHECKING` for conditional imports of `GoedelsPoetryStateManager`, improving type checker compatibility without affecting runtime performance
- Enhanced CLI proof file handling logic with refactored `_write_proof_result()` helper function for better maintainability and clarity
- Moved `parse_semantic_check_response` from `kimina_server.py` to `common.py` to avoid kimina_client import dependencies that caused Python 3.11 compatibility issues

### Fixed
- Fixed CI failures on Python 3.11 by avoiding problematic state module import that triggered Pydantic validation errors about using `typing.TypedDict` instead of `typing_extensions.TypedDict`
- Fixed import chain issues in test files by using `patch.dict(sys.modules, ...)` to inject mock modules before imports occur
- Fixed Python 3.11 compatibility issues by eliminating kimina_client import dependencies in semantics agent

### Documentation
- Updated README.md and CONFIGURATION.md to reflect LM Studio integration changes, including updated model references to new GGUF versions and detailed setup instructions for LM Studio

## [0.0.12] - 2025-12-08

### Changed
- Updated system architecture diagram to reflect the current architecture.

## [0.0.11] - 2025-12-08

### Added
- LM Studio provider support in the LLM configuration, enabling LM Studio to be used alongside existing Ollama and vLLM providers.

### Changed
- Unified Ollama and vLLM handling by migrating from `ChatOllama` to `ChatOpenAI` for consistent provider integration.
- Updated type hints across the codebase to improve clarity and compatibility.

### Removed
- Support for Google Generative AI decomposer agent: removed Google-specific configuration options (`google_model`, `google_max_output_tokens`, `google_max_self_correction_attempts`) and provider selection logic. The decomposer agent now exclusively uses OpenAI.
- Automatic Ollama model downloading; users now manage model availability explicitly.

### Fixed
- Addressed make check failures to keep automated checks green.

### Documentation
- Expanded README Quick Start instructions for Ollama, vLLM, and LM Studio, clarifying prerequisites and environment configuration.
- Added an architecture diagram to the README to improve the visual overview of the system.

## [0.0.10] - 2025-12-02

### Added
- Vector database querying phase: introduces a new phase that queries the Lean Explore vector database to retrieve relevant theorems and lemmas after search query generation and before proof sketching
- VectorDBAgent with factory pattern matching existing agent patterns, using asyncio.run() to wrap async client.search() calls
- APISearchResponseTypedDict TypedDict for type-safe handling of vector database search results
- search_results field in DecomposedFormalTheoremState to store vector database query results
- decomposition_query_queue in GoedelsPoetryState to manage states awaiting vector database queries
- LEAN_EXPLORE_SERVER configuration section in config.ini with url and package_filters options
- get_theorems_with_search_queries_for_vectordb() and set_theorems_with_vectordb_results() methods in GoedelsPoetryStateManager
- Comprehensive test coverage (15 new tests) for vector database querying functionality
- Search query generation phase before theorem decomposition: introduces a new phase that generates search queries for vector database retrieval before theorems are decomposed
- SearchQueryAgent with factory pattern matching existing agent patterns
- Two new prompt templates (search-query-initial.md and search-query-backtrack.md) using `<search>` tags for structured parsing
- Template-based backtrack detection that replaces brittle keyword matching with exact prompt template matching
- SEARCH_QUERY_AGENT_LLM configuration section in config.ini
- Comprehensive test coverage (19 new tests) for search query generation functionality
- Theorem hints feature: proof sketcher and backtrack agents now display relevant theorems and lemmas from vector database results to guide proof decomposition
- Prompt logging for LLM agents: added debug logging via `log_llm_prompt()` and `log_llm_response()` functions that output formatted prompts and responses when `GOEDELS_POETRY_DEBUG` environment variable is enabled
- Debug logging for vector database responses when `GOEDELS_POETRY_DEBUG` is enabled
- Expanded proof composition test suite with nested decomposition scenarios

### Changed
- Queue flow updated: decomposition_search_queue → decomposition_query_queue → decomposition_sketch_queue (initial flow)
- Backtracked states now route through search query generation queue to allow intelligent query regeneration based on failure context
- Backtracking now properly removes nodes from decomposition_query_queue when backtracking occurs
- LLM prompt handling: headers are now folded into bodies during preamble splitting, improving consistency and extending splitter test coverage

### Fixed
- Fixed off-by-one errors in self-correction attempt limits that could cause agents to attempt one more correction than configured

## [0.0.9] - 2025-11-23

### Documentation
- Updated Kimina Lean Server installation instructions to use PyPI package (`kimina-ast-server`) as the recommended primary method
- Replaced detailed source installation steps with explicit PyPI commands (`kimina-ast-server setup`, `kimina-ast-server run`)
- Updated server startup command from `python -m server` to `kimina-ast-server`
- Updated API endpoint references from `/verify` to `/api/check` to match the PyPI package API
- Simplified integration tests setup section to use PyPI installation method
- Streamlined documentation to provide a clear "golden path" for installation without requiring users to reference multiple external documentation sources

## [0.0.8] - 2025-11-23

### Fixed
- Fixed proof reconstruction for unicode have names: broadened `_extract_have_name` to correctly capture Lean identifiers with unicode characters (e.g., h₁ or names using Greek letters) when stitching child proofs back into parent sketches, with regression tests to prevent future issues.

## [0.0.7] - 2025-11-22

### Fixed
- Hardened `prover_agent` Lean code block parsing by always taking the last block (even when the closing fence is missing) and covering multi-block responses with regression tests to prevent truncated proofs.

### Documentation
- Publishing workflow now explicitly runs `uv lock` to sync `uv.lock` with `pyproject.toml` and reminds maintainers to include both files when committing a release bump.

## [0.0.6] - 2025-11-21

### Added
- Final proof verification for complete assembled proofs: added `check_complete_proof()` function that verifies proofs assembled from multiple subgoals using the Kimina Lean server before they are printed or written to files
- User-friendly progress indicators: added animated progress indicators using Rich's console.status() that display the current framework phase (e.g., "Formalizing theorem", "Proving theorems", "Validating proofs") during execution
- Phase name mapping: added `_PHASE_NAMES` dictionary mapping all 14 framework phase methods to user-friendly descriptions

### Changed
- Disabled tqdm progress bars: set `TQDM_DISABLE=1` at CLI startup to suppress "Batches" messages during LangGraph batch processing operations, providing cleaner terminal output

## [0.0.5] - 2025-11-20

### Changed
- Standardized configuration parameter naming: renamed `max_self_corrections` to `max_self_correction_attempts` for consistency across prover and decomposer agents
- Updated default Google model from `gemini-2.5-flash` to `gemini-2.5-pro` for improved performance
- Enhanced configuration documentation with clearer parameter descriptions and examples
- Improved documentation badges and links across all documentation files

### Fixed
- Fixed Kimina Lean Server repository URL in README to point to correct repository

### Documentation
- Updated README.md with improved configuration parameter documentation
- Enhanced CONFIGURATION.md with detailed parameter descriptions for all agents
- Improved CONTRIBUTING.md with clearer formatting and testing instructions
- Updated PUBLISHING.md with version 0.0.5 examples
- Enhanced docs/index.md with better badges, codecov integration, and improved description

## [0.0.4] - 2025-11-20

### Added
- Support for additional Lean 4 constructs in AST subgoal extraction: `set`, `suffices`, `choose`, `generalize`, `match`, `let`, and `obtain` statements
- Comprehensive test coverage for new AST parsing features including edge cases
- New theorem datasets:
  - compfiles v4.15 problems
  - minif2f v4.9 problems
  - MOBench v4.9 problems
  - PutnamBench theorem formalizations
- README documentation for compfiles problems
- Backtracking on max depth instead of terminating, improving proof search strategies

### Fixed
- Fixed theorem/proof parsing and reconstruction errors
- Fixed let/set bindings being incorrectly converted to equality hypotheses in subgoals
- Fixed set/let dependencies being incorrectly converted to equality hypotheses in subgoals
- Fixed missing hypothesis from 'set ... with h' statements in subgoal decomposition
- Removed `sorry` from proof reconstruction output
- Ensured final proofs include root theorem statement
- Fixed Python 3.9 unsupported operand type compatibility issue
- Fixed type issues in preamble handling
- Fixed bracket notation in docstrings causing mkdocs cross-reference errors
- Fixed let and set binding value/type extraction from AST

### Changed
- Increased `max_pass` to Goedel-Prover-V2's recommended value of 32
- Decreased `max_self_correction_attempts` to Goedel-Prover-V2's recommended value of 2
- Normalized Lean preamble handling and enforced headers for formal theorems
- Refactored preamble code for improved maintainability
- Improved AST parsing robustness and maintainability
- Enhanced binding name verification for match, choose, obtain, and generalize type extraction

## [0.0.3] - 2025-11-01

### Fixed
- Fixed bug where proofs containing `sorry` were incorrectly marked as successful. The proof checker now uses the `complete` field from Kimina server responses instead of the `pass` field to properly detect proofs with sorries.

### Added
- Support for Google Generative AI as an alternative to OpenAI for the decomposer agent
- Automatic provider selection based on available API keys (OpenAI takes priority)
- Provider-specific configuration parameters for OpenAI and Google models
- Backward compatibility with existing OpenAI-only configurations

### Changed
- Updated decomposer agent configuration to support multiple providers
- Enhanced configuration documentation with Google Generative AI setup instructions
- Updated default Google model from `gemini-2.0-flash-exp` to `gemini-2.5-flash` for improved performance and capabilities

## [0.0.2] - 2025-01-21
- Fixed printout of final proof

## [0.0.1] - 2025-01-17

### Added
- Initial release of Gödel's Poetry
- Multi-agent architecture for automated theorem proving
- Support for both informal and formal theorem inputs
- Integration with Kimina Lean Server for proof verification
- Command-line interface (`goedels_poetry`) for proving theorems
- Batch processing support for multiple theorems
- Proof sketching and recursive decomposition for complex theorems
- Configuration via environment variables and config.ini
- Fine-tuned models: Goedel-Prover-V2 and Goedel-Formalizer-V2
- Integration with GPT-5 and Qwen3 for advanced reasoning
- Comprehensive test suite including integration tests
- Documentation with examples and configuration guide

### Dependencies
- Python 3.9+ support
- LangGraph for multi-agent orchestration
- LangChain for LLM integration
- Kimina AST Client for Lean 4 verification
- Typer for CLI
- Rich for beautiful terminal output

[1.2.6]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.6
[1.2.5]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.5
[1.2.4]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.4
[1.2.3]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.3
[1.2.2]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.2
[1.2.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.1
[1.2.0]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.2.0
[1.1.6]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.6
[1.1.5]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.5
[1.1.4]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.4
[1.1.3]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.3
[1.1.2]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.2
[1.1.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.1
[1.1.0]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.1.0
[1.0.0]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v1.0.0
[0.0.14]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.14
[0.0.13]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.13
[0.0.12]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.12
[0.0.11]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.11
[0.0.10]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.10
[0.0.9]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.9
[0.0.8]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.8
[0.0.6]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.6
[0.0.5]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.5
[0.0.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.1
