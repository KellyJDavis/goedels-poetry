# Configuration

## Overview

The `goedels-poetry` configuration system uses a standard INI file (`goedels_poetry/data/config.ini`) with support for environment variable overrides.

## Configuration File

The default configuration is stored in `goedels_poetry/data/config.ini`:

```ini
[FORMALIZER_AGENT_LLM]
model = kdavis/goedel-formalizer-v2:32b
num_ctx = 40960
max_retries = 10

[PROVER_AGENT_LLM]
model = kdavis/Goedel-Prover-V2:32b
num_ctx = 40960
max_self_correction_attempts = 2
max_depth = 20
max_pass = 32

[SEMANTICS_AGENT_LLM]
model = qwen3:30b
num_ctx = 262144

[SEARCH_QUERY_AGENT_LLM]
model = qwen3:30b
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
