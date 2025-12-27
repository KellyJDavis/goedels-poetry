import os
from configparser import ConfigParser
from pathlib import Path
from typing import cast, overload

from typing_extensions import Required, TypedDict

# Sentinel value for "no fallback provided" (similar to ConfigParser's _UNSET)
_UNSET = object()


class ConfigParserWrapper:
    """
    Wrapper around ConfigParser that adds environment variable override support.

    Environment variables override config.ini values using the format: SECTION__OPTION
    For example: PROVER_AGENT_LLM__MODEL overrides [PROVER_AGENT_LLM] model

    This implementation follows the python-decouple pattern but uses the standard library,
    providing:
    - Standard INI file format with sections (no code changes needed)
    - Environment variable overrides (optional, not required)
    - Zero external dependencies
    """

    def __init__(self) -> None:
        """Initialize the config wrapper with the default config.ini file."""
        config_path = Path(__file__).parent.parent / "data" / "config.ini"
        self._config_parser = ConfigParser()
        self._config_parser.read(str(config_path))

    @overload
    def get(self, section: str, option: str) -> str: ...

    @overload
    def get(self, section: str, option: str, fallback: str) -> str: ...

    def get(self, section: str, option: str, fallback: str | object = _UNSET) -> str:
        """
        Get a configuration value as a string.

        Checks environment variable SECTION__OPTION first, then falls back to config.ini.

        Parameters
        ----------
        section : str
            The configuration section (e.g., "PROVER_AGENT_LLM")
        option : str
            The configuration option (e.g., "model")
        fallback : str, optional
            Default value if not found in env or config file

        Returns
        -------
        str
            The configuration value
        """
        # Check environment variable first (format: SECTION__OPTION, case-insensitive)
        env_key = f"{section}__{option}".upper()
        if env_key in os.environ:
            return os.environ[env_key]

        # Fall back to config.ini
        if fallback is _UNSET:
            return self._config_parser.get(section=section, option=option)
        else:
            # Type narrowing: if fallback is not _UNSET, it must be str (ensured by overloads)
            return self._config_parser.get(section=section, option=option, fallback=cast(str, fallback))

    @overload
    def getint(self, section: str, option: str) -> int: ...

    @overload
    def getint(self, section: str, option: str, fallback: int) -> int: ...

    def getint(self, section: str, option: str, fallback: int | object = _UNSET) -> int:
        """
        Get a configuration value as an integer.

        Checks environment variable SECTION__OPTION first, then falls back to config.ini.

        Parameters
        ----------
        section : str
            The configuration section (e.g., "PROVER_AGENT_LLM")
        option : str
            The configuration option (e.g., "num_ctx")
        fallback : int, optional
            Default value if not found in env or config file

        Returns
        -------
        int
            The configuration value as an integer
        """
        # Check environment variable first (format: SECTION__OPTION, case-insensitive)
        env_key = f"{section}__{option}".upper()
        if env_key in os.environ:
            return int(os.environ[env_key])

        # Fall back to config.ini
        if fallback is _UNSET:
            return self._config_parser.getint(section=section, option=option)
        else:
            # Type narrowing: if fallback is not _UNSET, it must be int (ensured by overloads)
            return self._config_parser.getint(section=section, option=option, fallback=cast(int, fallback))


# Global config instance
parsed_config = ConfigParserWrapper()


class ProofReconstructionConfig(TypedDict):
    """
    Configuration for proof reconstruction.
    """

    max_candidates: Required[int]


PROOF_RECONSTRUCTION = ProofReconstructionConfig(
    max_candidates=parsed_config.getint(section="PROOF_RECONSTRUCTION", option="max_candidates", fallback=64),
)
