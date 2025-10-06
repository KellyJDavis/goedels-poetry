import typer
from rich.console import Console

from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    # informal_statement = "Prove that 3 cannot be written as the sum of two cubes."
    formal_statement = "theorem u31 (O A C B D : ℂ) (hd₁ : ¬B = D) (hd₂ : ¬C = D) (hd₃ : ¬O = A) (hd₄ : ¬O = B) (hd₅ : ¬O = C) (hd₆ : ¬O = D) : ‖O - A‖ * ‖D - B‖ = ‖A - C‖ * ‖B - O‖ → ((O - A) / (C - A)).im = 0 → ((O - B) / (D - B)).im = 0 → ¬((A - B) / (C - B)).im = 0 → ((O.re - A.re) * (C.re - A.re) - (-O.im + A.im) * (C.im - A.im)).sign = ((O.re - B.re) * (D.re - B.re) - (-O.im + B.im) * (D.im - B.im)).sign → ((A - B) / (C - D)).im = 0 := sorry"  # noqa: RUF001

    config = GoedelsPoetryConfig()
    initial_state = GoedelsPoetryState(formal_theorem=formal_statement)

    state_manager = GoedelsPoetryStateManager(initial_state)

    framework = GoedelsPoetryFramework(config, state_manager)

    framework.run()


if __name__ == "__main__":
    app()
