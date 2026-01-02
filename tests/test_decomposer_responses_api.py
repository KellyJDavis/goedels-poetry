"""Tests for decomposer agent Responses API configuration."""

from unittest.mock import patch

from langchain_openai import ChatOpenAI

from goedels_poetry.config.config import parsed_config
from goedels_poetry.config.llm import DECOMPOSER_AGENT_LLM, _create_llm_safe


def test_decomposer_agent_llm_instance_exists():
    """Test that DECOMPOSER_AGENT_LLM instance exists and is a ChatOpenAI."""
    assert DECOMPOSER_AGENT_LLM is not None
    assert isinstance(DECOMPOSER_AGENT_LLM, ChatOpenAI)


def test_decomposer_agent_llm_has_responses_api_when_openai():
    """Test that DECOMPOSER_AGENT_LLM has use_responses_api=True when provider is OpenAI."""
    # Check provider from config
    provider = parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="provider", fallback="openai").lower()

    if provider == "openai":
        # If provider is OpenAI, verify Responses API is enabled
        assert hasattr(DECOMPOSER_AGENT_LLM, "use_responses_api")
        assert DECOMPOSER_AGENT_LLM.use_responses_api is True

        # Verify store: false is in extra_body
        assert hasattr(DECOMPOSER_AGENT_LLM, "extra_body")
        assert DECOMPOSER_AGENT_LLM.extra_body is not None
        assert "store" in DECOMPOSER_AGENT_LLM.extra_body
        assert DECOMPOSER_AGENT_LLM.extra_body["store"] is False


@patch("goedels_poetry.config.llm.parsed_config")
def test_decomposer_agent_llm_configured_with_responses_api(mock_config):
    """Test that _create_llm_safe configures Responses API for DECOMPOSER_AGENT_LLM with OpenAI."""
    # Mock config to return OpenAI provider
    mock_config.get.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "provider"): "openai",
        ("DECOMPOSER_AGENT_LLM", "model"): "gpt-4",
        ("DECOMPOSER_AGENT_LLM", "url"): "https://api.openai.com/v1",
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    mock_config.getint.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    # Create LLM instance
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        llm = _create_llm_safe(
            section="DECOMPOSER_AGENT_LLM",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    # Verify Responses API is configured
    assert hasattr(llm, "use_responses_api")
    assert llm.use_responses_api is True

    # Verify store: false is in extra_body
    assert hasattr(llm, "extra_body")
    assert llm.extra_body is not None
    assert "store" in llm.extra_body
    assert llm.extra_body["store"] is False


@patch("goedels_poetry.config.llm.parsed_config")
def test_decomposer_agent_llm_no_responses_api_for_non_openai(mock_config):
    """Test that DECOMPOSER_AGENT_LLM does not use Responses API when provider is not OpenAI."""
    # Mock config to return Ollama provider
    mock_config.get.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "provider"): "ollama",
        ("DECOMPOSER_AGENT_LLM", "model"): "test-model",
        ("DECOMPOSER_AGENT_LLM", "url"): "http://localhost:11434/v1",
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    mock_config.getint.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    # Create LLM instance
    llm = _create_llm_safe(
        section="DECOMPOSER_AGENT_LLM",
        model="test-model",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Verify Responses API is not enabled
    if hasattr(llm, "use_responses_api"):
        assert llm.use_responses_api is not True

    # Verify store is not in extra_body
    if hasattr(llm, "extra_body") and llm.extra_body:
        assert "store" not in llm.extra_body


@patch("goedels_poetry.config.llm.parsed_config")
def test_other_agents_no_responses_api(mock_config):
    """Test that other agents do not use Responses API even with OpenAI."""
    # Mock config to return OpenAI provider for FORMALIZER
    mock_config.get.side_effect = lambda section, option, fallback=None: {
        ("FORMALIZER_AGENT_LLM", "provider"): "openai",
        ("FORMALIZER_AGENT_LLM", "model"): "gpt-4",
        ("FORMALIZER_AGENT_LLM", "url"): "https://api.openai.com/v1",
        ("FORMALIZER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    mock_config.getint.side_effect = lambda section, option, fallback=None: {
        ("FORMALIZER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    # Create LLM instance for a different agent
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        llm = _create_llm_safe(
            section="FORMALIZER_AGENT_LLM",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )

    # Verify Responses API is not enabled for non-decomposer agents
    if hasattr(llm, "use_responses_api"):
        assert llm.use_responses_api is not True

    # Verify store is not in extra_body
    if hasattr(llm, "extra_body") and llm.extra_body:
        assert "store" not in llm.extra_body


@patch("goedels_poetry.config.llm.parsed_config")
def test_decomposer_responses_api_configuration_complete(mock_config):
    """Test that DECOMPOSER_AGENT_LLM has both use_responses_api and store: false configured."""
    # Mock config to return OpenAI provider
    mock_config.get.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "provider"): "openai",
        ("DECOMPOSER_AGENT_LLM", "model"): "gpt-4",
        ("DECOMPOSER_AGENT_LLM", "url"): "https://api.openai.com/v1",
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    mock_config.getint.side_effect = lambda section, option, fallback=None: {
        ("DECOMPOSER_AGENT_LLM", "max_tokens"): 50000,
    }.get((section, option), fallback)

    # Create LLM instance
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        llm = _create_llm_safe(
            section="DECOMPOSER_AGENT_LLM",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            max_retries=3,  # Additional kwargs should not break configuration
        )

    # Verify both configurations are present
    assert hasattr(llm, "use_responses_api")
    assert llm.use_responses_api is True

    assert hasattr(llm, "extra_body")
    assert llm.extra_body is not None
    assert "store" in llm.extra_body
    assert llm.extra_body["store"] is False
