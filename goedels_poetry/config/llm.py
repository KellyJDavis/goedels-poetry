import os
import warnings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from goedels_poetry.config.config import parsed_config


class VLLMClientError(Exception):
    """Raised when vLLM client initialization fails."""

    def __init__(self, section: str, cause: Exception) -> None:
        super().__init__(f"Failed to initialize vLLM client for section [{section}]")
        self.section = section
        self.__cause__ = cause


def _get_vllm_api_key() -> str | None:
    """Return VLLM_API_KEY if set and non-empty."""
    key = os.getenv("VLLM_API_KEY", "").strip()
    return key or None


def _create_vllm_client(*, model: str, base_url: str, max_tokens: int, section: str) -> ChatOpenAI:
    """Create a ChatOpenAI client pointing to a vLLM server."""
    api_key = _get_vllm_api_key()
    kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "max_retries": parsed_config.getint(section=section, option="max_retries", fallback=5),
        "timeout": parsed_config.getint(section=section, option="timeout_seconds", fallback=10800),
    }
    if api_key:
        kwargs["api_key"] = api_key
    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        raise VLLMClientError(section, exc) from exc


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
# them, we avoid constructing remote clients during startup when processing
# formal theorems.

_FORMALIZER_AGENT_LLM = None  # Cache for lazy-loaded formalizer LLM
_SEMANTICS_AGENT_LLM = None  # Cache for lazy-loaded semantics LLM
_SEARCH_QUERY_AGENT_LLM = None  # Cache for lazy-loaded search query LLM
_PROVER_AGENT_LLM = None  # Cache for lazy-loaded prover LLM


def _get_server_url(section: str, *, fallback: str) -> str:
    """Read a server url from config/env with fallback."""
    return parsed_config.get(section=section, option="url", fallback=fallback)


def get_formalizer_agent_llm():  # type: ignore[no-untyped-def]
    """Lazy-load and return the FORMALIZER_AGENT_LLM (vLLM via ChatOpenAI)."""
    global _FORMALIZER_AGENT_LLM
    if _FORMALIZER_AGENT_LLM is None:
        model = parsed_config.get(
            section="FORMALIZER_AGENT_LLM", option="model", fallback="Goedel-LM/Goedel-Formalizer-V2-32B"
        )
        base_url = _get_server_url(section="FORMALIZER_AGENT_LLM", fallback="http://localhost:8002/v1")
        _FORMALIZER_AGENT_LLM = _create_vllm_client(
            model=model,
            base_url=base_url,
            max_tokens=parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_tokens", fallback=30000),
            section="FORMALIZER_AGENT_LLM",
        )
    return _FORMALIZER_AGENT_LLM


def get_semantics_agent_llm():  # type: ignore[no-untyped-def]
    """Lazy-load and return the SEMANTICS_AGENT_LLM (vLLM via ChatOpenAI)."""
    global _SEMANTICS_AGENT_LLM
    if _SEMANTICS_AGENT_LLM is None:
        model = parsed_config.get(
            section="SEMANTICS_AGENT_LLM", option="model", fallback="Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
        base_url = _get_server_url(section="SEMANTICS_AGENT_LLM", fallback="http://localhost:8004/v1")
        _SEMANTICS_AGENT_LLM = _create_vllm_client(
            model=model,
            base_url=base_url,
            max_tokens=parsed_config.getint(section="SEMANTICS_AGENT_LLM", option="max_tokens", fallback=240000),
            section="SEMANTICS_AGENT_LLM",
        )
    return _SEMANTICS_AGENT_LLM


def get_search_query_agent_llm():  # type: ignore[no-untyped-def]
    """Lazy-load and return the SEARCH_QUERY_AGENT_LLM (vLLM via ChatOpenAI)."""
    global _SEARCH_QUERY_AGENT_LLM
    if _SEARCH_QUERY_AGENT_LLM is None:
        model = parsed_config.get(
            section="SEARCH_QUERY_AGENT_LLM", option="model", fallback="Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
        base_url = _get_server_url(section="SEARCH_QUERY_AGENT_LLM", fallback="http://localhost:8004/v1")
        _SEARCH_QUERY_AGENT_LLM = _create_vllm_client(
            model=model,
            base_url=base_url,
            max_tokens=parsed_config.getint(section="SEARCH_QUERY_AGENT_LLM", option="max_tokens", fallback=240000),
            section="SEARCH_QUERY_AGENT_LLM",
        )
    return _SEARCH_QUERY_AGENT_LLM


def get_prover_agent_llm():  # type: ignore[no-untyped-def]
    """Lazy-load and return the PROVER_AGENT_LLM (vLLM via ChatOpenAI)."""
    global _PROVER_AGENT_LLM
    if _PROVER_AGENT_LLM is None:
        model = parsed_config.get(section="PROVER_AGENT_LLM", option="model", fallback="Goedel-LM/Goedel-Prover-V2-32B")
        base_url = _get_server_url(section="PROVER_AGENT_LLM", fallback="http://localhost:8003/v1")
        _PROVER_AGENT_LLM = _create_vllm_client(
            model=model,
            base_url=base_url,
            max_tokens=parsed_config.getint(section="PROVER_AGENT_LLM", option="max_tokens", fallback=30000),
            section="PROVER_AGENT_LLM",
        )
    return _PROVER_AGENT_LLM


# Create decomposer LLM with automatic provider selection
def _create_decomposer_llm():  # type: ignore[no-untyped-def]
    """Create decomposer LLM with automatic provider selection and appropriate configuration."""
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
            model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="google_model", fallback="gemini-2.5-pro"),
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
    section="PROVER_AGENT_LLM", option="max_self_correction_attempts", fallback=2
)
PROVER_AGENT_MAX_DEPTH = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_depth", fallback=20)
PROVER_AGENT_MAX_PASS = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_pass", fallback=32)
# For DECOMPOSER_AGENT (openai and google), use the new key names
DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS = parsed_config.getint(
    section="DECOMPOSER_AGENT_LLM", option="openai_max_self_correction_attempts", fallback=6
)
FORMALIZER_AGENT_MAX_RETRIES = parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_retries", fallback=5)
