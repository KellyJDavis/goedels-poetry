import warnings

from langchain_google_genai import ChatGoogleGenerativeAI
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


def _create_decomposer_llm_safe(**kwargs):  # type: ignore[no-untyped-def]
    """
    Create a decomposer LLM instance, automatically selecting between OpenAI and Google providers.

    Priority order:
    1. If OPENAI_API_KEY is set -> use ChatOpenAI
    2. Else if GOOGLE_API_KEY is set -> use ChatGoogleGenerativeAI
    3. Else -> fall back to OpenAI with warning

    Parameters
    ----------
    **kwargs
        Configuration parameters for the LLM

    Returns
    -------
    BaseChatModel
        The appropriate LLM instance based on available API keys
    """
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Determine provider based on available API keys
    if openai_key and openai_key.strip() and openai_key != "dummy-key-for-testing":
        provider = "openai"
    elif google_key and google_key.strip():
        provider = "google"
    else:
        # Fall back to OpenAI with warning
        provider = "openai"
        warnings.warn(
            "No valid API key found. Falling back to OpenAI. Please set either "
            "OPENAI_API_KEY or GOOGLE_API_KEY environment variable.",
            UserWarning,
            stacklevel=2,
        )

    # Create LLM based on selected provider
    if provider == "openai":
        try:
            return ChatOpenAI(**kwargs)
        except Exception:
            # In test/CI environments without OPENAI_API_KEY, create with a dummy key
            if not openai_key or openai_key == "dummy-key-for-testing":
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
    else:  # provider == "google"
        # Convert OpenAI parameters to Google parameters
        google_kwargs = kwargs.copy()
        if "max_completion_tokens" in google_kwargs:
            google_kwargs["max_output_tokens"] = google_kwargs.pop("max_completion_tokens")

        try:
            return ChatGoogleGenerativeAI(**google_kwargs)
        except Exception:
            # In test/CI environments without GOOGLE_API_KEY, create with a dummy key
            if not google_key:
                warnings.warn(
                    "GOOGLE_API_KEY not set. Google LLM functionality will not work until "
                    "the API key is configured. Set the GOOGLE_API_KEY environment variable.",
                    UserWarning,
                    stacklevel=2,
                )
                # Set a dummy key to allow module import
                os.environ["GOOGLE_API_KEY"] = "dummy-key-for-testing"
                return ChatGoogleGenerativeAI(**google_kwargs)
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


# Create decomposer LLM with automatic provider selection
def _create_decomposer_llm():  # type: ignore[no-untyped-def]
    """Create decomposer LLM with automatic provider selection and appropriate configuration."""
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Determine provider based on available API keys
    if openai_key and openai_key.strip() and openai_key != "dummy-key-for-testing":
        provider = "openai"
    elif google_key and google_key.strip():
        provider = "google"
    else:
        # Fall back to OpenAI with warning
        provider = "openai"
        warnings.warn(
            "No valid API key found. Falling back to OpenAI. Please set either "
            "OPENAI_API_KEY or GOOGLE_API_KEY environment variable.",
            UserWarning,
            stacklevel=2,
        )

    # Create LLM based on selected provider
    if provider == "openai":
        return _create_decomposer_llm_safe(
            model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="openai_model", fallback="gpt-5-2025-08-07"),
            max_completion_tokens=parsed_config.getint(
                section="DECOMPOSER_AGENT_LLM", option="openai_max_completion_tokens", fallback=50000
            ),
            max_retries=parsed_config.getint(
                section="DECOMPOSER_AGENT_LLM", option="openai_max_remote_retries", fallback=5
            ),
        )
    else:  # provider == "google"
        return _create_decomposer_llm_safe(
            model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="google_model", fallback="gemini-2.5-flash"),
            max_completion_tokens=parsed_config.getint(
                section="DECOMPOSER_AGENT_LLM", option="google_max_output_tokens", fallback=50000
            ),
            max_retries=parsed_config.getint(
                section="DECOMPOSER_AGENT_LLM", option="google_max_self_correction_attempts", fallback=6
            ),
        )


DECOMPOSER_AGENT_LLM = _create_decomposer_llm()

# Create LLM configurations
PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS = parsed_config.getint(
    section="PROVER_AGENT_LLM", option="max_self_correction_attempts", fallback=3
)
PROVER_AGENT_MAX_DEPTH = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_depth", fallback=20)
PROVER_AGENT_MAX_PASS = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_pass", fallback=4)
# For DECOMPOSER_AGENT (openai and google), use the new key names
DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS = parsed_config.getint(
    section="DECOMPOSER_AGENT_LLM", option="openai_max_self_correction_attempts", fallback=6
)
FORMALIZER_AGENT_MAX_RETRIES = parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_retries", fallback=10)
