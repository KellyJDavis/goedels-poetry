from typing_extensions import Required, TypedDict

from goedels_poetry.config.config import parsed_config


class KiminaLeanServerConfig(TypedDict):
    """
    Utility class to hold Kimina Lean Server configuration

    Attributes
    ----------
    url : Required[str]
        The URL of the Kimina Lean Server.
    max_retries : Required[int]
        The maximum number of retries for requests to the Kimina Lean Server.
    """

    url: Required[str]
    max_retries: Required[int]


# Gather Kimina Lean Server configuration
KIMINA_LEAN_SERVER = KiminaLeanServerConfig(
    url=parsed_config.get(section="KIMINA_LEAN_SERVER", option="url", fallback="http://0.0.0.0:8000"),
    max_retries=parsed_config.getint(section="KIMINA_LEAN_SERVER", option="max_retries", fallback=5),
)
