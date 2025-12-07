import warnings

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from goedels_poetry.config.config import parsed_config


# Create LLMS (with error handling for environments without Ollama)
def _create_llm_safe(**kwargs):  # type: ignore[no-untyped-def]
    """Create a ChatOllama instance, catching connection errors in test/CI environments."""
    try:
        return ChatOllama(**kwargs)
    except ConnectionError:
        # In test/CI environments without Ollama, create with validation disabled
        kwargs["validate_model_on_init"] = False
        return ChatOllama(**kwargs)


def _create_decomposer_llm_safe(**kwargs):  # type: ignore[no-untyped-def]
    """
    Create a decomposer LLM instance using OpenAI.

    Parameters
    ----------
    **kwargs
        Configuration parameters for the LLM

    Returns
    -------
    BaseChatModel
        The OpenAI LLM instance
    """
    import os

    openai_key = os.getenv("OPENAI_API_KEY")

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


# ============================================================================
# Lazy-loaded LLMs (for informal theorem processing only)
# ============================================================================
# These LLMs are only needed when processing informal theorems. By lazy-loading
# them, we avoid initializing large Ollama models during startup when processing
# formal theorems.

_FORMALIZER_AGENT_LLM = None  # Cache for lazy-loaded formalizer LLM
_SEMANTICS_AGENT_LLM = None  # Cache for lazy-loaded semantics LLM
_SEARCH_QUERY_AGENT_LLM = None  # Cache for lazy-loaded search query LLM


def get_formalizer_agent_llm():  # type: ignore[no-untyped-def]
    """
    Lazy-load and return the FORMALIZER_AGENT_LLM.

    Only creates the LLM on first access, which speeds up startup when processing
    formal theorems that don't need formalization.

    Note: The required Ollama model must be downloaded beforehand using:
    `ollama pull kdavis/goedel-formalizer-v2:32b`

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

    Only creates the LLM on first access, which speeds up startup when processing
    formal theorems that don't need semantic checking.

    Note: The required Ollama model must be downloaded beforehand using:
    `ollama pull qwen3:30b`

    Returns
    -------
    ChatOllama
        The semantics agent LLM instance
    """
    global _SEMANTICS_AGENT_LLM
    if _SEMANTICS_AGENT_LLM is None:
        model = parsed_config.get(section="SEMANTICS_AGENT_LLM", option="model", fallback="qwen3:30b")
        # Create the LLM instance
        _SEMANTICS_AGENT_LLM = _create_llm_safe(
            model=model,
            validate_model_on_init=True,
            num_predict=50000,
            num_ctx=parsed_config.getint(section="SEMANTICS_AGENT_LLM", option="num_ctx", fallback=262144),
        )
    return _SEMANTICS_AGENT_LLM


def get_search_query_agent_llm():  # type: ignore[no-untyped-def]
    """
    Lazy-load and return the SEARCH_QUERY_AGENT_LLM.

    Only creates the LLM on first access, which speeds up startup when processing
    theorems that don't need search query generation.

    Note: The required Ollama model must be downloaded beforehand using:
    `ollama pull qwen3:30b`

    Returns
    -------
    ChatOllama
        The search query agent LLM instance
    """
    global _SEARCH_QUERY_AGENT_LLM
    if _SEARCH_QUERY_AGENT_LLM is None:
        model = parsed_config.get(section="SEARCH_QUERY_AGENT_LLM", option="model", fallback="qwen3:30b")
        # Create the LLM instance
        _SEARCH_QUERY_AGENT_LLM = _create_llm_safe(
            model=model,
            validate_model_on_init=True,
            num_predict=50000,
            num_ctx=parsed_config.getint(section="SEARCH_QUERY_AGENT_LLM", option="num_ctx", fallback=262144),
        )
    return _SEARCH_QUERY_AGENT_LLM


# ============================================================================
# Eagerly-loaded LLMs (needed for all theorem processing)
# ============================================================================
# These LLMs are used for both formal and informal theorems, so we load them
# immediately at module import time.

# Note: The required Ollama model must be downloaded beforehand using:
# `ollama pull kdavis/Goedel-Prover-V2:32b`
_PROVER_MODEL = parsed_config.get(section="PROVER_AGENT_LLM", option="model", fallback="kdavis/Goedel-Prover-V2:32b")

# Create prover LLM
PROVER_AGENT_LLM = _create_llm_safe(
    model=_PROVER_MODEL,
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="PROVER_AGENT_LLM", option="num_ctx", fallback=40960),
)


# Create decomposer LLM
def _create_decomposer_llm():  # type: ignore[no-untyped-def]
    """Create decomposer LLM with OpenAI configuration."""
    return _create_decomposer_llm_safe(
        model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="model", fallback="gpt-5-2025-08-07"),
        max_completion_tokens=parsed_config.getint(
            section="DECOMPOSER_AGENT_LLM", option="max_completion_tokens", fallback=50000
        ),
        max_retries=parsed_config.getint(section="DECOMPOSER_AGENT_LLM", option="max_remote_retries", fallback=5),
    )


DECOMPOSER_AGENT_LLM = _create_decomposer_llm()

# Create LLM configurations
PROVER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS = parsed_config.getint(
    section="PROVER_AGENT_LLM", option="max_self_correction_attempts", fallback=2
)
PROVER_AGENT_MAX_DEPTH = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_depth", fallback=20)
PROVER_AGENT_MAX_PASS = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_pass", fallback=32)
DECOMPOSER_AGENT_MAX_SELF_CORRECTION_ATTEMPTS = parsed_config.getint(
    section="DECOMPOSER_AGENT_LLM", option="max_self_correction_attempts", fallback=6
)
FORMALIZER_AGENT_MAX_RETRIES = parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_retries", fallback=10)
