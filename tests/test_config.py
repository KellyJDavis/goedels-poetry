"""Tests for configuration management with environment variable overrides."""

from goedels_poetry.config.config import ConfigParserWrapper


def test_config_get_from_ini():
    """Test that config values are read from config.ini."""
    config = ConfigParserWrapper()

    # These values should come from config.ini
    model = config.get(section="PROVER_AGENT_LLM", option="model")
    assert model == "kdavis/Goedel-Prover-V2:32b"

    url = config.get(section="KIMINA_LEAN_SERVER", option="url")
    assert url == "http://0.0.0.0:8000"


def test_config_getint_from_ini():
    """Test that integer config values are read from config.ini."""
    config = ConfigParserWrapper()

    # These values should come from config.ini
    num_ctx = config.getint(section="PROVER_AGENT_LLM", option="num_ctx")
    assert num_ctx == 40960

    max_retries = config.getint(section="KIMINA_LEAN_SERVER", option="max_retries")
    assert max_retries == 5


def test_config_env_override_string(monkeypatch):
    """Test that environment variables override config.ini values for strings."""
    config = ConfigParserWrapper()

    # Set environment variable
    monkeypatch.setenv("PROVER_AGENT_LLM__MODEL", "custom-model:latest")

    # Value should come from environment variable
    model = config.get(section="PROVER_AGENT_LLM", option="model")
    assert model == "custom-model:latest"


def test_config_env_override_int(monkeypatch):
    """Test that environment variables override config.ini values for integers."""
    config = ConfigParserWrapper()

    # Set environment variable
    monkeypatch.setenv("PROVER_AGENT_LLM__NUM_CTX", "8192")

    # Value should come from environment variable
    num_ctx = config.getint(section="PROVER_AGENT_LLM", option="num_ctx")
    assert num_ctx == 8192


def test_config_env_override_uppercase_required(monkeypatch):
    """Test that environment variables must be uppercase (standard convention)."""
    config = ConfigParserWrapper()

    # Set environment variable in uppercase (standard convention)
    monkeypatch.setenv("KIMINA_LEAN_SERVER__URL", "http://localhost:9000")

    # Value should come from environment variable
    url = config.get(section="KIMINA_LEAN_SERVER", option="url")
    assert url == "http://localhost:9000"


def test_config_fallback_when_missing():
    """Test that fallback values are used when key is missing."""
    config = ConfigParserWrapper()

    # Try to get a non-existent key with fallback
    value = config.get(section="NONEXISTENT_SECTION", option="nonexistent_option", fallback="default_value")
    assert value == "default_value"

    int_value = config.getint(section="NONEXISTENT_SECTION", option="nonexistent_option", fallback=42)
    assert int_value == 42


def test_config_env_takes_precedence_over_fallback(monkeypatch):
    """Test that environment variables take precedence over fallback values."""
    config = ConfigParserWrapper()

    # Set environment variable
    monkeypatch.setenv("NONEXISTENT_SECTION__CUSTOM_OPTION", "env_value")

    # Even with a fallback, env var should win
    value = config.get(section="NONEXISTENT_SECTION", option="custom_option", fallback="fallback_value")
    assert value == "env_value"


def test_config_multiple_env_overrides(monkeypatch):
    """Test that multiple environment variables can override different config values."""
    config = ConfigParserWrapper()

    # Set multiple environment variables
    monkeypatch.setenv("PROVER_AGENT_LLM__MODEL", "override-model")
    monkeypatch.setenv("PROVER_AGENT_LLM__NUM_CTX", "16384")
    monkeypatch.setenv("KIMINA_LEAN_SERVER__URL", "http://custom:8888")

    # All should be overridden
    assert config.get(section="PROVER_AGENT_LLM", option="model") == "override-model"
    assert config.getint(section="PROVER_AGENT_LLM", option="num_ctx") == 16384
    assert config.get(section="KIMINA_LEAN_SERVER", option="url") == "http://custom:8888"
