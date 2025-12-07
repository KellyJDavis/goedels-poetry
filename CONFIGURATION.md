# Configuration

## Overview

The `goedels-poetry` configuration system uses a standard INI file (`goedels_poetry/data/config.ini`) with support for environment variable overrides.

## Configuration File

The default configuration is stored in `goedels_poetry/data/config.ini`:

```ini
[FORMALIZER_AGENT_LLM]
model = kdavis/goedel-formalizer-v2:32b
provider = ollama
url = http://localhost:11434/v1
api_key = ollama
max_tokens = 50000
num_ctx = 40960
max_retries = 10

[PROVER_AGENT_LLM]
model = kdavis/Goedel-Prover-V2:32b
provider = ollama
url = http://localhost:11434/v1
api_key = ollama
max_tokens = 50000
num_ctx = 40960
max_self_correction_attempts = 2
max_depth = 20
max_pass = 32

[SEMANTICS_AGENT_LLM]
model = qwen3:30b
provider = ollama
url = http://localhost:11434/v1
api_key = ollama
max_tokens = 50000
num_ctx = 262144

[SEARCH_QUERY_AGENT_LLM]
model = qwen3:30b
provider = ollama
url = http://localhost:11434/v1
api_key = ollama
max_tokens = 50000
num_ctx = 262144

[DECOMPOSER_AGENT_LLM]
model = gpt-5-2025-08-07
max_completion_tokens = 50000
max_remote_retries = 5
max_self_correction_attempts = 6

[KIMINA_LEAN_SERVER]
url = http://0.0.0.0:8000
max_retries = 5

[LEAN_EXPLORE_SERVER]
url = http://localhost:8001/api/v1
package_filters = Mathlib,Batteries,Std,Init,Lean
```

## Configuration Parameters Explained

### Lean Explore Server

The Lean Explore Server provides vector database search capabilities for retrieving relevant theorems and lemmas:

- **`url`**: The base URL of the Lean Explore server API endpoint (default: `http://localhost:8001/api/v1`)
- **`package_filters`**: Comma-separated list of package names to filter search results. Only theorems from these packages will be returned. Default: `Mathlib,Batteries,Std,Init,Lean`

The vector database agent queries this server after search queries are generated and before proof sketching, allowing the proof sketcher to use relevant theorems found in the database.

### LLM Agent Configuration

Gödel's Poetry uses OpenAI-compatible APIs (via `ChatOpenAI`) to connect to LLM providers. The system supports both Ollama and vLLM through their OpenAI-compatible endpoints.

#### Required Models

Gödel's Poetry requires several models to be available on your configured provider:

- **`kdavis/goedel-formalizer-v2:32b`** - Used by the formalizer agent (FORMALIZER_AGENT_LLM)
- **`kdavis/Goedel-Prover-V2:32b`** - Used by the prover agent (PROVER_AGENT_LLM)
- **`qwen3:30b`** - Used by the semantics agent (SEMANTICS_AGENT_LLM) and search query agent (SEARCH_QUERY_AGENT_LLM)

#### Configuration Parameters

Each LLM agent section supports the following parameters:

- **`model`**: The model name/identifier (required)
- **`provider`**: The provider type - `"ollama"` or `"vllm"` (required)
- **`url`**: The base URL for the OpenAI-compatible API endpoint (default: `http://localhost:11434/v1` for Ollama)
- **`api_key`**: API key for authentication (default: `"ollama"` for Ollama, which ignores this value)
- **`max_tokens`**: Maximum tokens in generated response (default: `50000`)
- **`num_ctx`**: Context window size (Ollama-specific, passed via `extra_body`)
- **`max_retries`**: Retry attempts for API calls (FORMALIZER_AGENT_LLM only)

#### Optional vLLM-Specific Parameters

The following parameters are supported for vLLM and are ignored by Ollama:

- **`use_beam_search`**: Enable beam search decoding (boolean, default: not set)
- **`best_of`**: Number of completions to generate server-side and return the best (integer, default: not set)
- **`top_k`**: Limit the number of highest probability vocabulary tokens to consider (integer, default: not set)
- **`repetition_penalty`**: Penalty for repeated tokens to reduce repetition (float, default: not set)
- **`length_penalty`**: Control the length of the output (float, default: not set)

These parameters are passed via `extra_body` and will be ignored by providers that don't support them.

#### Setting Up Ollama

**Prerequisites:**
- [Ollama](https://ollama.com/download) must be installed and running

**Download the models:**
```bash
ollama pull kdavis/goedel-formalizer-v2:32b
ollama pull kdavis/Goedel-Prover-V2:32b
ollama pull qwen3:30b
```

⚠️ **Important**: These models must be downloaded before using Gödel's Poetry. The system will not automatically download them.

**Default Configuration:**
The default configuration uses Ollama with the OpenAI-compatible endpoint at `http://localhost:11434/v1`. Ollama exposes this endpoint automatically when running.

#### Setting Up vLLM

To use vLLM instead of Ollama, configure the agent sections with:

```ini
[FORMALIZER_AGENT_LLM]
provider = vllm
url = http://localhost:8000/v1
api_key = dummy-key
model = Goedel-LM/Goedel-Formalizer-V2-32B
max_tokens = 50000
# Optional vLLM-specific parameters
use_beam_search = false
best_of = 1

[PROVER_AGENT_LLM]
provider = vllm
url = http://localhost:8000/v1
api_key = dummy-key
model = Goedel-LM/Goedel-Prover-V2-32B
max_tokens = 50000

[SEMANTICS_AGENT_LLM]
provider = vllm
url = http://localhost:8000/v1
api_key = dummy-key
model = Qwen/Qwen3-30B-A3B-Instruct-2507
max_tokens = 50000

[SEARCH_QUERY_AGENT_LLM]
provider = vllm
url = http://localhost:8000/v1
api_key = dummy-key
model = Qwen/Qwen3-30B-A3B-Instruct-2507
max_tokens = 50000
```

Ensure your vLLM server is running and accessible at the configured URL.

### Decomposer Agent

The decomposer agent uses OpenAI for proof sketching. Configuration parameters:

- **`model`**: The OpenAI model used for proof sketching (default: `gpt-5-2025-08-07`)
- **`max_completion_tokens`**: Maximum tokens in generated response (default: `50000`)
- **`max_remote_retries`**: Retry attempts for API calls (default: `5`)
- **`max_self_correction_attempts`**: Maximum decomposition self-correction attempts (default: `6`)

**API Key Setup:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Environment Variable Overrides

You can override any configuration value using environment variables. The format is:

```
SECTION__OPTION=value
```

### Examples

Override the prover model:
```bash
export PROVER_AGENT_LLM__MODEL="custom-model:latest"
```

Override the Kimina server URL:
```bash
export KIMINA_LEAN_SERVER__URL="http://localhost:9000"
```

Override the Lean Explore server URL:
```bash
export LEAN_EXPLORE_SERVER__URL="http://localhost:8002/api/v1"
```

Override package filters for vector database searches:
```bash
export LEAN_EXPLORE_SERVER__PACKAGE_FILTERS="Mathlib,Batteries"
```

Override multiple values:
```bash
export PROVER_AGENT_LLM__MODEL="custom-model"
export PROVER_AGENT_LLM__NUM_CTX="8192"
export KIMINA_LEAN_SERVER__URL="http://custom-server:8888"
export LEAN_EXPLORE_SERVER__URL="http://custom-vector-db:8001/api/v1"
export LEAN_EXPLORE_SERVER__PACKAGE_FILTERS="Mathlib"
```

### How It Works

1. **Environment variables are optional** - If not set, values from `config.ini` are used
2. **Environment variables take precedence** - When set, they override `config.ini` values
3. **Standard naming convention** - Use uppercase with double underscore (`__`) separator
4. **No code changes needed** - The existing code continues to work without modification

### Use Cases

**Development Environment:**
```bash
# Use a smaller model for faster testing
export PROVER_AGENT_LLM__MODEL="llama2:7b"
export PROVER_AGENT_LLM__NUM_CTX="4096"
```

**CI/CD Pipeline:**
```bash
# Use different server in CI
export KIMINA_LEAN_SERVER__URL="http://ci-server:8000"
export KIMINA_LEAN_SERVER__MAX_RETRIES="10"
```

**Production Deployment:**
```bash
# Use production-grade models
export PROVER_AGENT_LLM__MODEL="kdavis/Goedel-Prover-V2:70b"
export PROVER_AGENT_LLM__MAX_SELF_CORRECTION_ATTEMPTS="3"
export PROVER_AGENT_LLM__MAX_PASS="64"
export DECOMPOSER_AGENT_LLM__MODEL="gpt-5-pro"
```

## Implementation Details

The configuration system is implemented in `goedels_poetry/config/config.py` using a wrapper around Python's standard `ConfigParser`. The wrapper:

1. Checks for environment variables first (format: `SECTION__OPTION`)
2. Falls back to `config.ini` if environment variable is not set
3. Uses fallback values if neither environment variable nor config file has the value

This design provides flexibility without adding external dependencies.
