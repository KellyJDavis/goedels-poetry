import contextlib
import warnings
from configparser import NoOptionError, NoSectionError
from typing import Any

from langchain_openai import ChatOpenAI

from goedels_poetry.config.config import parsed_config


def _build_extra_body(section: str, provider: str) -> dict[str, Any]:
    """
    Build extra_body dictionary with provider-specific parameters.

    Parameters
    ----------
    section : str
        Configuration section name (e.g., "FORMALIZER_AGENT_LLM")
    provider : str
        Provider type ("ollama", "vllm", or "lmstudio")

    Returns
    -------
    dict[str, Any]
        Dictionary of parameters to include in extra_body
    """
    extra_body: dict[str, Any] = {}

    # Ollama-specific parameters
    if provider == "ollama":
        try:
            num_ctx = parsed_config.getint(section=section, option="num_ctx")
            extra_body["num_ctx"] = num_ctx
        except (NoOptionError, NoSectionError):
            pass  # num_ctx is optional

    # LM Studio-specific parameters
    if provider == "lmstudio":
        try:
            ttl = parsed_config.getint(section=section, option="ttl")
            extra_body["ttl"] = ttl
        except (NoOptionError, NoSectionError):
            pass  # ttl is optional

    # vLLM-specific parameters (always include, Ollama will ignore them)
    try:
        use_beam_search = parsed_config.get(section=section, option="use_beam_search")
        extra_body["use_beam_search"] = use_beam_search.lower() in ("true", "1", "yes")
    except (NoOptionError, NoSectionError):
        pass  # use_beam_search is optional

    try:
        best_of = parsed_config.getint(section=section, option="best_of")
        extra_body["best_of"] = best_of
    except (NoOptionError, NoSectionError):
        pass  # best_of is optional

    try:
        top_k = parsed_config.getint(section=section, option="top_k")
        extra_body["top_k"] = top_k
    except (NoOptionError, NoSectionError):
        pass  # top_k is optional

    try:
        repetition_penalty_str = parsed_config.get(section=section, option="repetition_penalty")
        with contextlib.suppress(ValueError):
            extra_body["repetition_penalty"] = float(repetition_penalty_str)
    except (NoOptionError, NoSectionError):
        pass  # repetition_penalty is optional

    try:
        length_penalty_str = parsed_config.get(section=section, option="length_penalty")
        with contextlib.suppress(ValueError):
            extra_body["length_penalty"] = float(length_penalty_str)
    except (NoOptionError, NoSectionError):
        pass  # length_penalty is optional

    return extra_body


def _create_llm_safe(section: str, **kwargs):  # type: ignore[no-untyped-def]
    """
    Create a ChatOpenAI instance configured for Ollama, vLLM, or LM Studio.

    Parameters
    ----------
    section : str
        Configuration section name (e.g., "FORMALIZER_AGENT_LLM")
    **kwargs
        Additional parameters to pass to ChatOpenAI (will override config values)

    Returns
    -------
    ChatOpenAI
        The ChatOpenAI instance configured for the specified provider

    Raises
    ------
    ConnectionError
        If connection to the LLM server fails
    """
    # Read provider-specific configuration
    provider_raw = parsed_config.get(section=section, option="provider", fallback="ollama").lower()
    provider = provider_raw.replace("-", "").replace("_", "")
    if provider not in ("ollama", "vllm", "lmstudio"):
        msg = (
            f"Invalid provider '{provider_raw}' in section '{section}'. "
            "Must be 'ollama', 'vllm', or 'lmstudio' (LM Studio)."
        )
        raise ValueError(msg)

    default_url_by_provider = {
        "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1",
        "lmstudio": "http://localhost:1234/v1",
    }
    url = parsed_config.get(section=section, option="url", fallback=default_url_by_provider[provider])

    default_api_key_by_provider = {"ollama": "ollama", "vllm": "dummy-key", "lmstudio": "lm-studio"}
    api_key = parsed_config.get(section=section, option="api_key", fallback=default_api_key_by_provider[provider])
    model = kwargs.pop("model", None)
    if model is None:
        try:
            model = parsed_config.get(section=section, option="model")
        except (NoOptionError, NoSectionError) as err:
            msg = f"Model not specified in section '{section}' and not provided as argument."
            raise ValueError(msg) from err

    max_completion_tokens = kwargs.pop(
        "max_completion_tokens",
        parsed_config.getint(section=section, option="max_tokens", fallback=50000),
    )

    # Build extra_body with provider-specific parameters
    extra_body = _build_extra_body(section, provider)

    # Create ChatOpenAI instance
    return ChatOpenAI(
        base_url=url,
        api_key=api_key,  # type: ignore[arg-type]
        model=model,
        max_completion_tokens=max_completion_tokens,
        extra_body=extra_body if extra_body else None,
        **kwargs,
    )


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
# LLM Instances
# ============================================================================
# All LLM instances are created at module import time. These are ChatOpenAI
# clients that connect to remote servers (Ollama, vLLM, or LM Studio), so
# creation is lightweight and does not require loading models into memory.

# Note: The required models must be available on the configured provider
# (Ollama, vLLM, or LM Studio). For Ollama, download the models beforehand using:
# - `ollama pull kdavis/goedel-formalizer-v2:32b` (for formalizer)
# - `ollama pull kdavis/Goedel-Prover-V2:32b` (for prover)
# - `ollama pull qwen3:30b` (for semantics and search query)

# Create formalizer LLM
FORMALIZER_AGENT_LLM = _create_llm_safe(
    section="FORMALIZER_AGENT_LLM",
    max_retries=parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_remote_retries", fallback=5),
)

# Create prover LLM
PROVER_AGENT_LLM = _create_llm_safe(
    section="PROVER_AGENT_LLM",
    max_retries=parsed_config.getint(section="PROVER_AGENT_LLM", option="max_remote_retries", fallback=5),
)

# Create semantics LLM
SEMANTICS_AGENT_LLM = _create_llm_safe(
    section="SEMANTICS_AGENT_LLM",
    max_retries=parsed_config.getint(section="SEMANTICS_AGENT_LLM", option="max_remote_retries", fallback=5),
)

# Create search query LLM
SEARCH_QUERY_AGENT_LLM = _create_llm_safe(
    section="SEARCH_QUERY_AGENT_LLM",
    max_retries=parsed_config.getint(section="SEARCH_QUERY_AGENT_LLM", option="max_remote_retries", fallback=5),
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
