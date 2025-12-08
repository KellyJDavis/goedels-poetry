# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.11]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.11
[0.0.10]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.10
[0.0.9]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.9
[0.0.8]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.8
[0.0.6]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.6
[0.0.5]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.5
[0.0.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.1
