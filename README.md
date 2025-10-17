# Gödel's Poetry

[![Release](https://img.shields.io/github/v/release/KellyJDavis/goedels-poetry)](https://github.com/KellyJDavis/goedels-poetry/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/KellyJDavis/goedels-poetry/main.yml?branch=main)](https://github.com/KellyJDavis/goedels-poetry/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/KellyJDavis/goedels-poetry/branch/main/graph/badge.svg)](https://codecov.io/gh/KellyJDavis/goedels-poetry)
[![Commit activity](https://img.shields.io/github/commit-activity/m/KellyJDavis/goedels-poetry)](https://github.com/KellyJDavis/goedels-poetry/graphs/commit-activity)
[![License](https://img.shields.io/github/license/KellyJDavis/goedels-poetry)](https://github.com/KellyJDavis/goedels-poetry/blob/main/LICENSE)

> *A recursive, reflective POETRY algorithm variant using Goedel-Prover-V2*

**Gödel's Poetry** is an advanced automated theorem proving system that combines Large Language Models (LLMs) with formal verification in Lean 4. The system takes mathematical theorems—either in informal natural language or formal Lean syntax—and automatically generates verified proofs through a sophisticated multi-agent architecture.

- **Github repository**: <https://github.com/KellyJDavis/goedels-poetry/>
- **Documentation**: <https://KellyJDavis.github.io/goedels-poetry/>

---

## Table of Contents

- [What Does Gödel's Poetry Do?](#what-does-gödels-poetry-do)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Kimina Lean Server](#running-the-kimina-lean-server)
  - [Setting Up Your OpenAI API Key](#setting-up-your-openai-api-key)
  - [Using the Command Line Tool](#using-the-command-line-tool)
- [Examples](#examples)
- [How It Works](#how-it-works)
- [Developer Guide](#developer-guide)
  - [Development Setup](#development-setup)
  - [Testing](#testing)
  - [Makefile Targets](#makefile-targets)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
- [License](#license)

---

## What Does Gödel's Poetry Do?

Gödel's Poetry is an AI-powered theorem proving system that bridges the gap between informal mathematical reasoning and formal verification. The system:

1. **Accepts theorems in multiple formats**:
   - Informal natural language (e.g., "Prove that the square root of 2 is irrational")
   - Formal Lean 4 syntax (e.g., `theorem sqrt_two_irrational : Irrational (√2) := by sorry`)

2. **Automatically generates verified proofs** through a multi-agent workflow:
   - **Formalization**: Converts informal statements into formal Lean 4 theorems
   - **Semantic Checking**: Validates that formalizations preserve the original meaning
   - **Proof Generation**: Creates proofs using specialized LLMs trained on Lean 4
   - **Proof Sketching**: Decomposes difficult theorems into manageable subgoals
   - **Verification**: Validates all proofs using the Lean 4 proof assistant
   - **Recursive Refinement**: Iteratively improves proofs until they are complete and verified

3. **Leverages state-of-the-art technology**:
   - Custom fine-tuned models (Goedel-Prover-V2, Goedel-Formalizer-V2)
   - Integration with frontier LLMs (GPT-5, Qwen3)
   - The [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server) for high-performance Lean 4 verification
   - LangGraph for orchestrating complex multi-agent workflows

The system is designed for researchers, mathematicians, and AI practitioners interested in automated theorem proving, formal verification, and the intersection of natural and formal languages.

---

## Quick Start

### Prerequisites

Before installing Gödel's Poetry, ensure you have:

- **Python 3.9 or higher** (tested on Python 3.9-3.13)
- **pip** (comes with Python)
- **Lean 4** for the Kimina server (installation covered below)

For development:
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package installer (optional, but recommended for development)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Git** for cloning the repository

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
# Install using pip
pip install goedels-poetry

# Verify installation
goedels_poetry --help
```

#### Option 2: Install from Source (For Development)

```bash
# Clone the repository
git clone https://github.com/KellyJDavis/goedels-poetry.git
cd goedels-poetry

# Install with uv (recommended) or pip
uv sync

# The command line tool is now available via:
uv run goedels_poetry --help
```

### Running the Kimina Lean Server

The Kimina Lean Server is **required** for Gödel's Poetry to verify Lean 4 proofs. It provides high-performance parallel proof checking.

#### Setup Steps:

1. **Clone the Kimina Lean Server** (separate repository):
   ```bash
   git clone https://github.com/KellyJDavis/kimina-lean-server.git
   cd kimina-lean-server
   ```

2. **Run the setup script** (installs Lean 4, mathlib4, and dependencies):
   ```bash
   bash setup.sh
   ```
   This will:
   - Install Elan (the Lean version manager)
   - Install Lean 4 (default version v4.15.0)
   - Clone and build the Lean REPL
   - Clone and build the AST export tool
   - Clone and build mathlib4 (Lean's math library)

   ⚠️ **Note**: This process can take 15-30 minutes depending on your system.

3. **Install server dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install .
   prisma generate
   ```

4. **Start the server**:
   ```bash
   python -m server
   ```

   The server will start on `http://0.0.0.0:8000` by default.

5. **Verify the server is running** (in a new terminal):
   ```bash
   curl --request POST \
     --url http://localhost:8000/verify \
     --header 'Content-Type: application/json' \
     --data '{
       "codes": [{"custom_id": "test", "proof": "#check Nat"}],
       "infotree_type": "original"
     }' | jq
   ```

#### Alternative: Docker (Production)

For production deployments, you can use Docker:

```bash
cd kimina-lean-server
docker compose up
```

See the [Kimina Server README](https://github.com/KellyJDavis/kimina-lean-server/blob/main/README.md) for more deployment options.

### Setting Up Your OpenAI API Key

Gödel's Poetry uses OpenAI's GPT models for certain reasoning tasks. You'll need an API key:

1. **Get an API key** from [OpenAI's platform](https://platform.openai.com/api-keys)

2. **Set the environment variable**:

   **On Linux/macOS**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   **On Windows (Command Prompt)**:
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```

   **On Windows (PowerShell)**:
   ```powershell
   $env:OPENAI_API_KEY='your-api-key-here'
   ```

3. **Make it permanent** (optional):

   Add the export command to your shell configuration file:
   - Bash: `~/.bashrc` or `~/.bash_profile`
   - Zsh: `~/.zshrc`
   - Fish: `~/.config/fish/config.fish`

### Using the Command Line Tool

Once installed, you can use the `goedels_poetry` command to prove theorems:

#### Prove a Single Formal Theorem

```bash
goedels_poetry --formal-theorem "theorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

#### Prove a Single Informal Theorem

```bash
goedels_poetry --informal-theorem "Prove that the sum of two even numbers is even"
```

#### Batch Process Multiple Theorems

Process all `.lean` files in a directory:
```bash
goedels_poetry --formal-theorems ./my-theorems/
```

Process all `.txt` files containing informal theorems:
```bash
goedels_poetry --informal-theorems ./informal-theorems/
```

For batch processing, the tool will:
- Read each theorem from its file
- Attempt to generate and verify a proof
- Save results to `.proof` files alongside the originals

#### Get Help

```bash
goedels_poetry --help
```

#### Enable Debug Mode

To see detailed LLM and Kimina server responses during execution, set the `GOEDELS_POETRY_DEBUG` environment variable:

**On Linux/macOS**:
```bash
export GOEDELS_POETRY_DEBUG=1
goedels_poetry --formal-theorem "theorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**On Windows (Command Prompt)**:
```cmd
set GOEDELS_POETRY_DEBUG=1
goedels_poetry --formal-theorem "theorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**On Windows (PowerShell)**:
```powershell
$env:GOEDELS_POETRY_DEBUG=1
goedels_poetry --formal-theorem "theorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

When debug mode is enabled, all responses from:
- **FORMALIZER_AGENT_LLM** - Formalization responses
- **PROVER_AGENT_LLM** - Proof generation responses
- **SEMANTICS_AGENT_LLM** - Semantic checking responses
- **DECOMPOSER_AGENT_LLM** - Proof sketching/decomposition responses
- **KIMINA_SERVER** - Lean 4 verification and AST parsing responses

will be printed to the console with rich formatting for easy debugging and inspection.

---

## Examples

### Example 1: Simple Arithmetic

```bash
goedels_poetry --formal-theorem \
  "theorem add_comm_example : 3 + 5 = 5 + 3 := by sorry"
```

### Example 2: Informal Theorem

```bash
goedels_poetry --informal-theorem \
  "Prove that for any natural numbers a and b, a + b = b + a"
```

### Example 3: Batch Processing

Create a directory with theorem files:
```bash
mkdir theorems
echo "theorem test1 : 2 + 2 = 4 := by sorry" > theorems/test1.lean
echo "theorem test2 : 5 * 5 = 25 := by sorry" > theorems/test2.lean

goedels_poetry --formal-theorems ./theorems/
```

Results will be saved as `test1.proof` and `test2.proof`.

---

## How It Works

Gödel's Poetry uses a sophisticated multi-agent architecture coordinated by a supervisor agent. The workflow adapts based on the input:

### For Informal Theorems:

1. **Formalizer Agent** - Converts natural language to Lean 4 syntax
2. **Syntax Checker Agent** - Validates the formal theorem syntax
3. **Semantics Agent** - Ensures the formalization preserves meaning
4. **Prover Agent** - Generates the proof
5. **Proof Checker Agent** - Verifies the proof in Lean 4
6. **Parser Agent** - Extracts the AST structure

### For Complex Theorems (Recursive Decomposition):

When direct proving fails, the system activates **proof sketching**:

1. **Proof Sketcher Agent** - Creates a high-level proof outline
2. **Sketch Checker Agent** - Validates the sketch syntax
3. **Decomposition Agent** - Extracts sub-theorems from the sketch
4. **Recursive Proving** - Each sub-theorem is proved independently
5. **Proof Reconstruction** - Combines verified sub-proofs into the final proof

### Key Features:

- **Automatic Correction**: Agents iteratively fix syntax and logical errors
- **Backtracking**: When a decomposition approach fails, the system tries alternatives
- **State Management**: Complete provenance tracking for reproducibility
- **Parallel Processing**: Batch theorem proving with efficient resource usage

---

## Developer Guide

### Development Setup

1. **Clone and install with development dependencies**:
   ```bash
   git clone https://github.com/KellyJDavis/goedels-poetry.git
   cd goedels-poetry
   make install
   ```

   This will:
   - Create a virtual environment using `uv`
   - Install all dependencies
   - Set up pre-commit hooks for code quality

2. **Activate the environment** (if needed):
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

### Testing

The project includes comprehensive unit and integration tests.

#### Unit Tests Only (Fast)

```bash
make test
```

This runs all tests except those requiring Lean installation.

#### Integration Tests (Requires Lean Server)

Integration tests verify the Kimina Lean Server integration. **These tests require a running Kimina Lean server.**

**First-time setup:**

```bash
# Install integration test dependencies
uv sync

# Clone the Kimina Lean Server (if not already cloned)
cd .. && git clone https://github.com/KellyJDavis/kimina-lean-server.git
cd kimina-lean-server

# Install Lean and build dependencies (takes 15-30 minutes)
bash setup.sh

# Install server dependencies
pip install -r requirements.txt
pip install .
prisma generate
```

**Run integration tests:**

```bash
# Terminal 1: Start the Kimina server
cd ../kimina-lean-server
python -m server

# Terminal 2: Run the tests
cd ../goedels-poetry
make test-integration
```

The tests will automatically connect to `http://localhost:8000`. To use a different URL:

```bash
export KIMINA_SERVER_URL=http://localhost:9000
make test-integration
```

**Note**: Integration tests require Python 3.10+ and a running Lean server with proper REPL configuration.

#### All Tests

```bash
make test-all
```

This runs both unit and integration tests sequentially.

### Makefile Targets

The repository provides several convenient Make targets:

| Target | Description |
|--------|-------------|
| `make install` | Install the virtual environment and pre-commit hooks |
| `make check` | Run all code quality checks (linting, type checking, dependency audit) |
| `make test` | Run unit tests with coverage (excludes integration tests) |
| `make test-integration` | Run integration tests (requires Lean installation) |
| `make test-all` | Run all tests (unit + integration) |
| `make build` | Build wheel distribution package |
| `make clean-build` | Remove build artifacts |
| `make publish` | Publish to PyPI (requires credentials) |
| `make docs` | Build and serve documentation locally |
| `make docs-test` | Test documentation build without serving |
| `make help` | Display all available targets with descriptions |

#### Code Quality Tools

The `make check` target runs:
- **uv lock** - Ensures lock file consistency
- **pre-commit** - Runs linting and formatting (Ruff)
- **mypy** - Static type checking
- **deptry** - Checks for obsolete dependencies

### Configuration

#### Default Configuration Parameters

Configuration is stored in `goedels_poetry/data/config.ini`:

```ini
[FORMALIZER_AGENT_LLM]
model = kdavis/goedel-formalizer-v2:32b
num_ctx = 40960
max_retries = 10

[PROVER_AGENT_LLM]
model = kdavis/Goedel-Prover-V2:32b
num_ctx = 40960
max_retries = 10
max_depth = 20

[SEMANTICS_AGENT_LLM]
model = qwen3:30b
num_ctx = 262144

[DECOMPOSER_AGENT_LLM]
model = gpt-5-2025-08-07
max_completion_tokens = 50000
max_remote_retries = 5
max_retries = 3

[KIMINA_LEAN_SERVER]
url = http://0.0.0.0:8000
max_retries = 5
```

#### Configuration Parameters Explained

**Formalizer Agent**:
- `model`: The LLM used to convert informal theorems to Lean 4
- `num_ctx`: Context window size (tokens)
- `max_retries`: Maximum attempts to formalize a theorem

**Prover Agent**:
- `model`: The LLM used to generate proofs
- `num_ctx`: Context window size (tokens)
- `max_retries`: Maximum proof generation attempts
- `max_depth`: Maximum recursion depth for proof decomposition

**Semantics Agent**:
- `model`: The LLM used to validate semantic equivalence
- `num_ctx`: Context window size (tokens)

**Decomposer Agent**:
- `model`: The LLM used for proof sketching and decomposition
- `max_completion_tokens`: Maximum tokens in generated response
- `max_remote_retries`: Retry attempts for API calls
- `max_retries`: Retry attempts for decomposition

**Kimina Lean Server**:
- `url`: Server endpoint for Lean verification
- `max_retries`: Maximum retry attempts for server requests

#### Overriding Configuration with Environment Variables

The **recommended** way to customize configuration is using environment variables. This approach doesn't require modifying files and works great for different environments (development, testing, production):

**Format**: `SECTION__OPTION` (double underscore separator, uppercase)

**Examples**:

```bash
# Use a different prover model
export PROVER_AGENT_LLM__MODEL="custom-model:latest"

# Change the Kimina server URL
export KIMINA_LEAN_SERVER__URL="http://localhost:9000"

# Use a smaller context window for faster testing
export PROVER_AGENT_LLM__NUM_CTX="8192"

# Run with custom configuration
goedels_poetry --formal-theorem "theorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**Multiple overrides**:
```bash
export PROVER_AGENT_LLM__MODEL="kdavis/Goedel-Prover-V2:70b"
export DECOMPOSER_AGENT_LLM__MODEL="gpt-5-pro"
export KIMINA_LEAN_SERVER__MAX_RETRIES="10"
goedels_poetry --formal-theorem "..."
```

**Environment variables are optional** - if not set, the system uses values from `config.ini`.

For more details and advanced configuration options, see [CONFIGURATION.md](./CONFIGURATION.md).

#### Alternative: Modifying config.ini Directly

If you prefer, you can still modify the configuration file directly:

```bash
# Find the installation path
uv run python -c "import goedels_poetry; print(goedels_poetry.__file__)"

# Edit the config.ini in the installation directory
# Typically: .venv/lib/python3.x/site-packages/goedels_poetry/data/config.ini
```

**Note**: Direct file changes persist until you reinstall or update the package, while environment variables are more flexible and don't require reinstallation.

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

**Quick contribution workflow**:

1. Fork the repository
2. Clone your fork: `git clone git@github.com:YOUR_NAME/goedels-poetry.git`
3. Install development environment: `make install`
4. Create a feature branch: `git checkout -b feature-name`
5. Make your changes and add tests
6. Run quality checks: `make check`
7. Run tests: `make test`
8. Commit with descriptive messages
9. Push and create a pull request

**Code quality requirements**:
- All tests must pass (`make test`)
- Code must pass linting and type checking (`make check`)
- New features should include tests and documentation
- Follow the existing code style and conventions

### Project Structure

```
goedels-poetry/
├── goedels_poetry/           # Main package
│   ├── agents/               # Multi-agent system components
│   │   ├── formalizer_agent.py
│   │   ├── prover_agent.py
│   │   ├── proof_checker_agent.py
│   │   ├── sketch_*.py       # Proof sketching agents
│   │   └── ...
│   ├── config/               # Configuration management
│   ├── data/                 # Prompts and config files
│   │   ├── config.ini
│   │   └── prompts/
│   ├── parsers/              # AST parsing utilities
│   ├── cli.py                # Command-line interface
│   ├── framework.py          # Core orchestration logic
│   └── state.py              # State management
├── tests/                    # Test suite
├── Makefile                  # Development automation
├── pyproject.toml            # Package configuration
├── CHANGELOG.md              # Version history
└── README.md                 # This file
```

**Note**: The [Kimina Lean Server](https://github.com/KellyJDavis/kimina-lean-server) is a separate repository that must be installed and run independently.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

- **Kimina Lean Server**: Built on [Project Numina's](https://projectnumina.ai/) excellent Lean verification server
- **Lean 4**: The formal verification system that powers proof checking
- **LangChain & LangGraph**: Frameworks for LLM orchestration
- **Mathlib4**: Comprehensive mathematics library for Lean

---

## Citation

If you use Gödel's Poetry in your research, please cite:

```bibtex
@software{goedels_poetry,
  author = {Davis, Kelly J.},
  title = {Gödel's Poetry: Recursive Automated Theorem Proving},
  year = {2025},
  url = {https://github.com/KellyJDavis/goedels-poetry}
}
```

For the Kimina Lean Server:

```bibtex
@misc{santos2025kiminaleanservertechnical,
  title={Kimina Lean Server: Technical Report},
  author={Marco Dos Santos and Haiming Wang and Hugues de Saxcé and Ran Wang and Mantas Baksys and Mert Unsal and Junqi Liu and Zhengying Liu and Jia Li},
  year={2025},
  eprint={2504.21230},
  archivePrefix={arXiv},
  primaryClass={cs.LO},
  url={https://arxiv.org/abs/2504.21230}
}
```

---

## Support

- **Issues**: Report bugs or request features at [GitHub Issues](https://github.com/KellyJDavis/goedels-poetry/issues)
- **Discussions**: Ask questions at [GitHub Discussions](https://github.com/KellyJDavis/goedels-poetry/discussions)
- **Documentation**: Visit the [official docs](https://KellyJDavis.github.io/goedels-poetry/)

---

**Ready to prove some theorems?** 🚀

```bash
goedels_poetry --informal-theorem "Prove that the sum of the first n natural numbers equals n(n+1)/2"
```
