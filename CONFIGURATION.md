# Configuration

## Overview

The `goedels-poetry` configuration system uses a standard INI file (`goedels_poetry/data/config.ini`) with support for environment variable overrides.

## Configuration File

The default configuration is stored in `goedels_poetry/data/config.ini`:

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

## Configuration Parameters Explained

### Lean Explore Server

The Lean Explore Server provides vector database search capabilities for retrieving relevant theorems and lemmas:

- **`url`**: The base URL of the Lean Explore server API endpoint (default: `http://localhost:8001/api/v1`)
- **`package_filters`**: Comma-separated list of package names to filter search results. Only theorems from these packages will be returned. Default: `Mathlib,Batteries,Std,Init,Lean`

The vector database agent queries this server after search queries are generated and before proof sketching, allowing the proof sketcher to use relevant theorems found in the database.

## Decomposer Agent Provider Selection

The decomposer agent supports both OpenAI and Google Generative AI providers. The system automatically selects the provider based on available API keys:

### Provider Priority Order

1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Google Generative AI** (if `GOOGLE_API_KEY` is set and no OpenAI key)
3. **Fallback to OpenAI** (with warning if no keys are found)

### API Key Setup

**For OpenAI:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**For Google Generative AI:**
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

**Both providers available:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
# OpenAI will be selected (higher priority)
```

### Provider-Specific Configuration

The decomposer agent uses different configuration parameters depending on the selected provider:

– **OpenAI**: Uses `openai_model`, `openai_max_completion_tokens`, `openai_max_remote_retries`, `openai_max_self_correction_attempts`
– **Google**: Uses `google_model`, `google_max_output_tokens`, `google_max_self_correction_attempts`

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

Override the prover retries/timeouts:
```bash
export PROVER_AGENT_LLM__MAX_RETRIES="2"
export PROVER_AGENT_LLM__TIMEOUT_SECONDS="120"
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
export PROVER_AGENT_LLM__MAX_RETRIES="9"
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
export PROVER_AGENT_LLM__MAX_RETRIES="2"
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
export PROVER_AGENT_LLM__MODEL="Goedel-LM/Goedel-Prover-V2-32B"
export PROVER_AGENT_LLM__MAX_SELF_CORRECTION_ATTEMPTS="3"
export PROVER_AGENT_LLM__MAX_PASS="64"
export DECOMPOSER_AGENT_LLM__OPENAI_MODEL="gpt-5-pro"
```

**Using Google Generative AI:**
```bash
# Use Google's Gemini model for decomposer
export GOOGLE_API_KEY="your-google-api-key"
export DECOMPOSER_AGENT_LLM__GOOGLE_MODEL="gemini-2.5-pro"
export DECOMPOSER_AGENT_LLM__GOOGLE_MAX_OUTPUT_TOKENS="100000"
```

## Implementation Details

The configuration system is implemented in `goedels_poetry/config/config.py` using a wrapper around Python's standard `ConfigParser`. The wrapper:

1. Checks for environment variables first (format: `SECTION__OPTION`)
2. Falls back to `config.ini` if environment variable is not set
3. Uses fallback values if neither environment variable nor config file has the value

This design provides flexibility without adding external dependencies.
