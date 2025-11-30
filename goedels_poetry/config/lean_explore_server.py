from typing_extensions import Required, TypedDict

from goedels_poetry.config.config import parsed_config


class LeanExploreServerConfig(TypedDict):
    """
    Utility class to hold Lean Explore Server configuration

    Attributes
    ----------
    url : Required[str]
        The URL of the Lean Explore Server.
    package_filters : Required[list[str]]
        List of package names to filter search results (e.g., ["Mathlib", "Batteries", "Std", "Init", "Lean"]).
    """

    url: Required[str]
    package_filters: Required[list[str]]


def _parse_package_filters(package_filters_str: str) -> list[str]:
    """
    Parse a comma-separated string of package filters into a list.

    Parameters
    ----------
    package_filters_str : str
        Comma-separated string of package names

    Returns
    -------
    list[str]
        List of package names, stripped of whitespace
    """
    if not package_filters_str.strip():
        return []
    return [pkg.strip() for pkg in package_filters_str.split(",") if pkg.strip()]


# Gather Lean Explore Server configuration
LEAN_EXPLORE_SERVER = LeanExploreServerConfig(
    url=parsed_config.get(section="LEAN_EXPLORE_SERVER", option="url", fallback="http://localhost:8001/api/v1"),
    package_filters=_parse_package_filters(
        parsed_config.get(
            section="LEAN_EXPLORE_SERVER",
            option="package_filters",
            fallback="Mathlib,Batteries,Std,Init,Lean",
        )
    ),
)
