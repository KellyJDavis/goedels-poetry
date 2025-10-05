import typer
from rich.console import Console

from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

app = typer.Typer()
console = Console()


@app.command()
def main():
    informal_statement = "Prove that 3 cannot be written as the sum of two cubes."

    config = GoedelsPoetryConfig()
    initial_state = GoedelsPoetryState(informal_theorem=informal_statement)

    state_manager = GoedelsPoetryStateManager(initial_state)

    framework = GoedelsPoetryFramework(config, state_manager)

    framework.run()


if __name__ == "__main__":
    app()
