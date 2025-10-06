import typer
from rich.console import Console

from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    # informal_statement = "Prove that 3 cannot be written as the sum of two cubes."
    formal_statement = "theorem theorem_af1bdc12cb92 : ¬∃ (a b : ℤ), a^3 + b^3 = 3 := by sorry"  # noqa: RUF001

    config = GoedelsPoetryConfig()
    initial_state = GoedelsPoetryState(formal_theorem=formal_statement)

    state_manager = GoedelsPoetryStateManager(initial_state)

    framework = GoedelsPoetryFramework(config, state_manager)

    framework.run()


if __name__ == "__main__":
    app()
