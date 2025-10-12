import warnings

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama import ResponseError, chat, pull
from rich.console import Console
from tqdm import tqdm

from goedels_poetry.config.config import parsed_config

# Create Console for outputs
console = Console()


def _download_llm(llm: str) -> None:
    """
    Method which ensures the specified LLM is downloaded. This code if based off of that provided
    # by Ollama https://github.com/ollama/ollama-python/blob/main/examples/pull.py

    Parameters
    ----------
    llm: str
        The LLM to ensure download of
    """
    # Inform user of progress
    console.print(f"Starting download of {llm}")

    # Track progress for each layer
    bars: dict = {}
    current_digest: str = ""
    for progress in pull(llm, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            console.print(progress.get("status"))
            continue

        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True)

        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest


def _download_llms(llms: list[str]) -> None:
    """
    Method which ensures the specified LLMs are downloaded.

    Parameters
    ----------
    llms: list[str]
        The LLMs to download
    """
    # Download the LLMs one at a time
    for llm in llms:
        try:
            # Check to see if it's already downloaded
            chat(llm)
        except ResponseError as e:
            # If it isn't downloaded, download it
            if e.status_code == 404:
                console.print(f"Starting download of {llm}")
                _download_llm(llm)
        except ConnectionError:
            # Ollama is not running (e.g., in CI/test environments)
            # Warn the user but allow import to succeed for testing
            warnings.warn(
                "Could not connect to Ollama. LLM functionality will not work until "
                "Ollama is running. Download and start Ollama from https://ollama.com/download",
                UserWarning,
                stacklevel=2,
            )
            break  # Only warn once, not for each LLM


# Create LLMS (with error handling for environments without Ollama)
def _create_llm_safe(**kwargs):  # type: ignore[no-untyped-def]
    """Create a ChatOllama instance, catching connection errors in test/CI environments."""
    try:
        return ChatOllama(**kwargs)
    except ConnectionError:
        # In test/CI environments without Ollama, create with validation disabled
        # Note: A warning was already issued by _download_llms() above
        kwargs["validate_model_on_init"] = False
        return ChatOllama(**kwargs)


def _create_openai_llm_safe(**kwargs):  # type: ignore[no-untyped-def]
    """Create a ChatOpenAI instance, with error handling for missing API keys in test/CI environments."""
    import os

    try:
        return ChatOpenAI(**kwargs)
    except Exception:
        # In test/CI environments without OPENAI_API_KEY, create with a dummy key
        # This allows module imports to succeed for testing non-OpenAI functionality
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            warnings.warn(
                "OPENAI_API_KEY not set. OpenAI LLM functionality will not work until "
                "the API key is configured. Set the OPENAI_API_KEY environment variable.",
                UserWarning,
                stacklevel=2,
            )
            # Set a dummy key to allow module import
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
            return ChatOpenAI(**kwargs)
        else:
            # Re-raise if it's a different error
            raise


# ============================================================================
# Lazy-loaded LLMs (for informal theorem processing only)
# ============================================================================
# These LLMs are only needed when processing informal theorems. By lazy-loading
# them, we avoid downloading/initializing large Ollama models during startup
# when processing formal theorems.

_FORMALIZER_AGENT_LLM = None  # Cache for lazy-loaded formalizer LLM
_SEMANTICS_AGENT_LLM = None  # Cache for lazy-loaded semantics LLM


def get_formalizer_agent_llm():  # type: ignore[no-untyped-def]
    """
    Lazy-load and return the FORMALIZER_AGENT_LLM.

    Only downloads and creates the LLM on first access, which speeds up
    startup when processing formal theorems that don't need formalization.

    Returns
    -------
    ChatOllama
        The formalizer agent LLM instance
    """
    global _FORMALIZER_AGENT_LLM
    if _FORMALIZER_AGENT_LLM is None:
        model = parsed_config.get(
            section="FORMALIZER_AGENT_LLM", option="model", fallback="kdavis/goedel-formalizer-v2:32b"
        )
        # Download the model if needed
        _download_llms([model])
        # Create the LLM instance
        _FORMALIZER_AGENT_LLM = _create_llm_safe(
            model=model,
            validate_model_on_init=True,
            num_predict=50000,
            num_ctx=parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="num_ctx", fallback=40960),
        )
    return _FORMALIZER_AGENT_LLM


def get_semantics_agent_llm():  # type: ignore[no-untyped-def]
    """
    Lazy-load and return the SEMANTICS_AGENT_LLM.

    Only downloads and creates the LLM on first access, which speeds up
    startup when processing formal theorems that don't need semantic checking.

    Returns
    -------
    ChatOllama
        The semantics agent LLM instance
    """
    global _SEMANTICS_AGENT_LLM
    if _SEMANTICS_AGENT_LLM is None:
        model = parsed_config.get(section="SEMANTICS_AGENT_LLM", option="model", fallback="qwen3:30b")
        # Download the model if needed
        _download_llms([model])
        # Create the LLM instance
        _SEMANTICS_AGENT_LLM = _create_llm_safe(
            model=model,
            validate_model_on_init=True,
            num_predict=50000,
            num_ctx=parsed_config.getint(section="SEMANTICS_AGENT_LLM", option="num_ctx", fallback=262144),
        )
    return _SEMANTICS_AGENT_LLM


# ============================================================================
# Eagerly-loaded LLMs (needed for all theorem processing)
# ============================================================================
# These LLMs are used for both formal and informal theorems, so we load them
# immediately at module import time.

# Download prover model if needed
_PROVER_MODEL = parsed_config.get(section="PROVER_AGENT_LLM", option="model", fallback="kdavis/Goedel-Prover-V2:32b")
_download_llms([_PROVER_MODEL])

# Create prover LLM
PROVER_AGENT_LLM = _create_llm_safe(
    model=_PROVER_MODEL,
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="PROVER_AGENT_LLM", option="num_ctx", fallback=40960),
)

DECOMPOSER_AGENT_LLM = _create_openai_llm_safe(
    model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="model", fallback="gpt-5-2025-08-07"),
    max_completion_tokens=parsed_config.getint(
        section="DECOMPOSER_AGENT_LLM", option="max_completion_tokens", fallback=50000
    ),
    max_retries=parsed_config.getint(section="DECOMPOSER_AGENT_LLM", option="max_remote_retries", fallback=5),
)

# Create LLM configurations
PROVER_AGENT_MAX_RETRIES = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_retries", fallback=10)
PROVER_AGENT_MAX_DEPTH = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_depth", fallback=20)
DECOMPOSER_AGENT_MAX_RETRIES = parsed_config.getint(section="DECOMPOSER_AGENT_LLM", option="max_retries", fallback=3)
FORMALIZER_AGENT_MAX_RETRIES = parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_retries", fallback=10)
