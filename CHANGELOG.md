# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.5]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.5
[0.0.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.1
