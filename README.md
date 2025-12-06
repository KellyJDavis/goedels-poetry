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
  - [Setting Up Your API Keys](#setting-up-your-api-keys)
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
   - **Search Query Generation**: Generates queries for retrieving relevant theorems from knowledge bases
   - **Vector Database Querying**: Queries a Lean Explore vector database to retrieve relevant theorems and lemmas
   - **Proof Sketching**: Decomposes difficult theorems into manageable subgoals
   - **Verification**: Validates all proofs using the Lean 4 proof assistant
   - **Recursive Refinement**: Iteratively improves proofs until they are complete and verified

3. **Leverages state-of-the-art technology**:
   - Custom fine-tuned models (Goedel-Prover-V2, Goedel-Formalizer-V2)
   - Integration with frontier LLMs (GPT-5, Qwen3)
   - The [Kimina Lean Server](https://github.com/KellyJDavis/kimina-lean-server) for high-performance Lean 4 verification
   - LangGraph for orchestrating complex multi-agent workflows

The system is designed for researchers, mathematicians, and AI practitioners interested in automated theorem proving, formal verification, and the intersection of natural and formal languages.

---

## Quick Start

### Prerequisites

Before installing Gödel's Poetry, ensure you have:

- **Python 3.10 or higher** (tested on Python 3.10-3.13)
- **pip** (comes with Python)
- **Lean 4** for the Kimina server (installation covered below)
- **Lean Explore server** for vector database queries (installation covered below)

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

#### Option 1: Install from PyPI (Recommended)

The easiest way to install and run the Kimina Lean Server is via PyPI:

1. **Install the server package**:
   ```bash
   pip install kimina-ast-server
   ```

2. **Set up the Lean workspace** (installs Lean 4, mathlib4, and dependencies):

   Run the setup command to automatically install and configure everything:

   ```bash
   # Setup in current directory
   kimina-ast-server setup

   # Or setup in a specific directory
   kimina-ast-server setup --workspace ~/lean-workspace

   # Setup and save configuration to .env file
   kimina-ast-server setup --workspace ~/lean-workspace --save-config
   ```

   This will automatically:
   - Install Elan (the Lean version manager)
   - Install Lean 4 (default version v4.15.0)
   - Clone and build the Lean REPL
   - Clone and build the AST export tool
   - Clone and build mathlib4 (Lean's math library)

   ⚠️ **Note**: This process can take 15-30 minutes depending on your system, primarily due to building mathlib4.

3. **Start the server**:
   ```bash
   # If you ran setup in current directory
   kimina-ast-server

   # Or if workspace is elsewhere, set the environment variable first
   export LEAN_SERVER_WORKSPACE=~/lean-workspace
   kimina-ast-server

   # You can also use the explicit run command
   kimina-ast-server run
   ```

   The server will start on `http://0.0.0.0:8000` by default.

4. **Verify the server is running** (in a new terminal):
   ```bash
   curl --request POST \
     --url http://localhost:8000/api/check \
     --header 'Content-Type: application/json' \
     --data '{
       "codes": [{"custom_id": "test", "proof": "#check Nat"}],
       "infotree_type": "original"
     }' | jq
   ```

   You can also visit `http://localhost:8000/docs` for interactive API documentation.

#### Install from Source

If you prefer to install from source, please refer to the [Kimina Lean Server repository](https://github.com/KellyJDavis/kimina-lean-server) for detailed installation instructions.

#### Alternative: Docker (Production)

For production deployments, you can use Docker. See the [Kimina Server README](https://github.com/KellyJDavis/kimina-lean-server/blob/main/README.md) for Docker deployment options.

### Running the Lean Explore Server

The Lean Explore Server is **required** for Gödel's Poetry to query the vector database for relevant theorems and lemmas. It provides semantic search capabilities across Lean 4 declarations.

#### Option 1: Install from PyPI (Recommended)

The easiest way to install and run the Lean Explore Server is via PyPI:

1. **Install the server package**:
   ```bash
   pip install lean-xplore
   ```

2. **Download local data**:
   ```bash
   leanexplore data fetch
   ```

   This downloads the database, embeddings, and search index needed for local searches. This step is required for the local backend.

   ⚠️ **Note**: The local backend must be used (not the remote API) because the remote API uses a Mathlib version incompatible with that required by Gödel's Poetry.

3. **Start the HTTP server**:
   ```bash
   leanexplore http serve --backend local
   ```

   The server will start on `http://localhost:8001/api/v1` by default.

4. **Verify the server is running** (in a new terminal):
   ```bash
   curl --request POST \
     --url http://localhost:8001/api/v1/search \
     --header 'Content-Type: application/json' \
     --data '{"query": "natural numbers", "package_filters": ["Mathlib"]}' | jq
   ```

   You can also visit `http://localhost:8001/docs` for interactive API documentation (if available).

#### Install from Source

If you prefer to install from source, please refer to the [Lean Explore repository](https://github.com/KellyJDavis/lean-explore) for detailed installation instructions.

For more information, see the [Lean Explore documentation](https://kellyjdavis.github.io/lean-explore/).

### Setting Up Your API Keys

Gödel's Poetry supports both OpenAI and Google Generative AI for certain reasoning tasks. You can use either provider:

#### Option 1: OpenAI (Default)

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

#### Option 2: Google Generative AI

1. **Get an API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Set the environment variable**:

   **On Linux/macOS**:
   ```bash
   export GOOGLE_API_KEY='your-api-key-here'
   ```

   **On Windows (Command Prompt)**:
   ```cmd
   set GOOGLE_API_KEY=your-api-key-here
   ```

   **On Windows (PowerShell)**:
   ```powershell
   $env:GOOGLE_API_KEY='your-api-key-here'
   ```

#### Provider Selection

The system automatically selects the provider based on available API keys:
- If both keys are set, **OpenAI takes priority**
- If only one key is set, that provider is used
- If no keys are set, the system falls back to OpenAI with a warning

3. **Make it permanent** (optional):

   Add the export command to your shell configuration file:
   - Bash: `~/.bashrc` or `~/.bash_profile`
   - Zsh: `~/.zshrc`
   - Fish: `~/.config/fish/config.fish`

### Using the Command Line Tool

Once installed, you can use the `goedels_poetry` command to prove theorems:

#### Prove a Single Formal Theorem

```bash
goedels_poetry --formal-theorem "import Mathlib\n\nopen BigOperators\n\ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

Formal theorems supplied on the command line (or via files) must include their full Lean preamble—imports, options, namespaces, and any comments required to state the theorem. Gödel's Poetry no longer prepends the default header for user-supplied formal problems. (The default header is still added automatically when an informal theorem is formalized by the system.)

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
goedels_poetry --formal-theorem "import Mathlib\n\nopen BigOperators\n\ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**On Windows (Command Prompt)**:
```cmd
set GOEDELS_POETRY_DEBUG=1
goedels_poetry --formal-theorem "import Mathlib\n\nopen BigOperators\n\ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**On Windows (PowerShell)**:
```powershell
$env:GOEDELS_POETRY_DEBUG=1
goedels_poetry --formal-theorem "import Mathlib`n`nopen BigOperators`n`ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

When debug mode is enabled, all responses from:
- **FORMALIZER_AGENT_LLM** - Formalization responses
- **PROVER_AGENT_LLM** - Proof generation responses
- **SEMANTICS_AGENT_LLM** - Semantic checking responses
- **SEARCH_QUERY_AGENT_LLM** - Search query generation responses
- **DECOMPOSER_AGENT_LLM** - Proof sketching/decomposition responses
- **KIMINA_SERVER** - Lean 4 verification and AST parsing responses
- **LEAN_EXPLORE_SERVER** - Vector database search responses

will be printed to the console with rich formatting for easy debugging and inspection.

---

## Examples

### Example 1: Simple Arithmetic

```bash
goedels_poetry --formal-theorem \
  "import Mathlib\n\nopen BigOperators\n\ntheorem add_comm_example : 3 + 5 = 5 + 3 := by sorry"
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
cat <<'EOF' > theorems/test1.lean
import Mathlib

open BigOperators

theorem test1 : 2 + 2 = 4 := by sorry
EOF

cat <<'EOF' > theorems/test2.lean
import Mathlib

open BigOperators

theorem test2 : 5 * 5 = 25 := by sorry
EOF

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

1. **Search Query Agent** - Generates search queries for retrieving relevant theorems
2. **Vector DB Agent** - Queries the Lean Explore vector database to find relevant theorems and lemmas
3. **Proof Sketcher Agent** - Creates a high-level proof outline using retrieved theorems
4. **Sketch Checker Agent** - Validates the sketch syntax
5. **Decomposition Agent** - Extracts sub-theorems from the sketch
6. **Recursive Proving** - Each sub-theorem is proved independently
7. **Proof Reconstruction** - Combines verified sub-proofs into the final proof

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

You can set up the Kimina server using the PyPI package (recommended):

```bash
# Install integration test dependencies
uv sync

# Install Kimina server from PyPI
pip install kimina-ast-server

# Set up Lean workspace (installs Lean 4, mathlib4, and dependencies - takes 15-30 minutes)
kimina-ast-server setup

# Or setup in a specific directory
kimina-ast-server setup --workspace ~/lean-workspace
```

If you prefer to install from source, please refer to the [Kimina Lean Server repository](https://github.com/KellyJDavis/kimina-lean-server) for detailed installation instructions.

**Run integration tests:**

```bash
# Terminal 1: Start the Kimina server
kimina-ast-server

# Or if workspace is in a different location
export LEAN_SERVER_WORKSPACE=~/lean-workspace
kimina-ast-server

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
model = Goedel-LM/Goedel-Formalizer-V2-32B
url = http://localhost:8002/v1
max_retries = 5
timeout_seconds = 10800

[PROVER_AGENT_LLM]
model = Goedel-LM/Goedel-Prover-V2-32B
url = http://localhost:8003/v1
max_retries = 5
timeout_seconds = 10800
max_self_correction_attempts = 2
max_depth = 20
max_pass = 32

[SEMANTICS_AGENT_LLM]
model = Qwen/Qwen3-30B-A3B-Instruct-2507
url = http://localhost:8004/v1
max_retries = 5
timeout_seconds = 10800

[SEARCH_QUERY_AGENT_LLM]
model = Qwen/Qwen3-30B-A3B-Instruct-2507
url = http://localhost:8004/v1
max_retries = 5
timeout_seconds = 10800

[DECOMPOSER_AGENT_LLM]
# Provider selection (openai, google, auto)
provider = auto

# OpenAI-specific settings
openai_model = gpt-5-2025-08-07
openai_max_completion_tokens = 50000
openai_max_remote_retries = 5
openai_max_self_correction_attempts = 6

# Google-specific settings
google_model = gemini-2.5-pro
google_max_output_tokens = 50000
google_max_self_correction_attempts = 6

[KIMINA_LEAN_SERVER]
url = http://0.0.0.0:8000
max_retries = 5

[LEAN_EXPLORE_SERVER]
url = http://localhost:8001/api/v1
package_filters = Mathlib,Batteries,Std,Init,Lean

```

#### Configuration Parameters Explained

**Formalizer Agent**:
- `model`: Hugging Face model served by vLLM
- `max_retries`: Maximum attempts to formalize a theorem
- `timeout_seconds`: Request timeout

**Prover Agent**:
- `model`: Hugging Face model served by vLLM
- `max_retries`: Maximum attempts for a request
- `timeout_seconds`: Request timeout
- `max_self_correction_attempts`: Maximum proof generation self-correction attempts
- `max_depth`: Maximum recursion depth for proof decomposition
- `max_pass`: Maximum number of proof attempts before triggering decomposition

**Semantics/Search Agents**:
- `model`: Hugging Face model served by vLLM
- `max_retries`: Maximum attempts for a request
- `timeout_seconds`: Request timeout

**Decomposer Agent**:
- `provider`: Provider selection (`openai`, `google`, or `auto`)
- `openai_model`: The OpenAI model used for proof sketching (when OpenAI is selected)
- `openai_max_completion_tokens`: Maximum tokens in OpenAI-generated response
- `openai_max_remote_retries`: Retry attempts for OpenAI API calls
- `openai_max_self_correction_attempts`: Maximum decomposition self-correction attempts for OpenAI
- `google_model`: The Google model used for proof sketching (when Google is selected)
- `google_max_output_tokens`: Maximum tokens in Google-generated response
- `google_max_self_correction_attempts`: Maximum decomposition self-correction attempts for Google

**Kimina Lean Server**:
- `url`: Server endpoint for Lean verification
- `max_retries`: Maximum retry attempts for server requests

**Lean Explore Server**:
- `url`: Server endpoint for the Lean Explore vector database
- `package_filters`: Comma-separated list of package names to filter search results (e.g., "Mathlib,Batteries,Std,Init,Lean")

#### Overriding Configuration with Environment Variables

The **recommended** way to customize configuration is using environment variables. This approach doesn't require modifying files and works great for different environments (development, testing, production):

**Format**: `SECTION__OPTION` (double underscore separator, uppercase)

**Examples**:

```bash
# Use a different prover model
export PROVER_AGENT_LLM__MODEL="custom-model:latest"

# Change the Kimina server URL
export KIMINA_LEAN_SERVER__URL="http://localhost:9000"

# Change the Lean Explore server URL
export LEAN_EXPLORE_SERVER__URL="http://localhost:8002/api/v1"

# Change package filters for vector database searches
export LEAN_EXPLORE_SERVER__PACKAGE_FILTERS="Mathlib,Batteries"

# Adjust prover retries/timeouts for testing
export PROVER_AGENT_LLM__MAX_RETRIES="2"
export PROVER_AGENT_LLM__TIMEOUT_SECONDS="120"

# Run with custom configuration
goedels_poetry --formal-theorem "import Mathlib\n\nopen BigOperators\n\ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**Multiple overrides**:
```bash
export PROVER_AGENT_LLM__MODEL="Goedel-LM/Goedel-Prover-V2-32B"
export PROVER_AGENT_LLM__MAX_SELF_CORRECTION_ATTEMPTS="3"
export PROVER_AGENT_LLM__MAX_PASS="64"
export DECOMPOSER_AGENT_LLM__OPENAI_MODEL="gpt-5-pro"
export KIMINA_LEAN_SERVER__MAX_RETRIES="10"
# Provide the full preamble plus theorem body when invoking formal problems
goedels_poetry --formal-theorem "import Mathlib\n\nopen BigOperators\n\ntheorem theorem_54_43 : 1 + 1 = 2 := by sorry"
```

**Using Google Generative AI**:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export DECOMPOSER_AGENT_LLM__GOOGLE_MODEL="gemini-2.5-pro"
export DECOMPOSER_AGENT_LLM__GOOGLE_MAX_OUTPUT_TOKENS="100000"
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
│   │   ├── search_query_agent.py
│   │   ├── vector_db_agent.py
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
