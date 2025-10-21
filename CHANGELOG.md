# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.1]: https://github.com/KellyJDavis/goedels-poetry/releases/tag/v0.0.1
