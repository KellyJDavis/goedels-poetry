import os

os.environ["TQDM_DISABLE"] = "1"

import traceback
from pathlib import Path

import typer
from rich.console import Console

from goedels_poetry.agents.util.common import split_preamble_and_body

app = typer.Typer()
console = Console()


def _has_preamble(code: str) -> bool:
    preamble, _ = split_preamble_and_body(code)
    return bool(preamble.strip())


def _read_theorem_content(theorem_file: Path) -> str | None:
    theorem_content = theorem_file.read_text(encoding="utf-8").strip()
    if not theorem_content:
        console.print(f"[bold yellow]Warning:[/bold yellow] {theorem_file.name} is empty, skipping")
        return None
    return theorem_content


def _handle_missing_header(theorem_file: Path) -> None:
    console.print("[bold red]Error:[/bold red] Formal theorems must include a Lean header (imports/options).")
    output_file = theorem_file.with_suffix(".proof")
    output_file.write_text(
        "Proof failed: Missing Lean header/preamble in supplied formal theorem.",
        encoding="utf-8",
    )


def _handle_processing_error(theorem_file: Path, error: Exception) -> None:
    console.print(f"[bold red]Error processing {theorem_file.name}:[/bold red] {error}")
    console.print(traceback.format_exc())

    output_file = theorem_file.with_suffix(".proof")
    error_message = f"Error during processing: {error}\n\n{traceback.format_exc()}"
    output_file.write_text(error_message, encoding="utf-8")
    console.print(f"[bold yellow]Error details saved to {output_file.name}[/bold yellow]")


def process_single_theorem(
    formal_theorem: str | None = None,
    informal_theorem: str | None = None,
) -> None:
    """
    Process a single theorem (either formal or informal) and output proof to stdout.
    """
    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

    config = GoedelsPoetryConfig()

    if formal_theorem:
        if not _has_preamble(formal_theorem):
            console.print("[bold red]Error:[/bold red] Formal theorems must include a Lean header (imports/options).")
            raise typer.Exit(code=1)
        initial_state = GoedelsPoetryState(formal_theorem=formal_theorem)
        console.print("[bold blue]Processing formal theorem...[/bold blue]")
    else:
        initial_state = GoedelsPoetryState(informal_theorem=informal_theorem)
        console.print("[bold blue]Processing informal theorem...[/bold blue]")

    state_manager = GoedelsPoetryStateManager(initial_state)
    framework = GoedelsPoetryFramework(config, state_manager, console)

    try:
        framework.run()
    except Exception as e:
        console.print(f"[bold red]Error during proof process:[/bold red] {e}")
        console.print(traceback.format_exc())


def process_theorems_from_directory(
    directory: Path,
    file_extension: str,
    is_formal: bool,
) -> None:
    """
    Process all theorem files from a directory and write proofs to .proof files.

    Args:
        directory: Directory containing theorem files
        file_extension: File extension to look for (.lean or .txt)
        is_formal: True for formal theorems, False for informal theorems
    """
    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

    if not directory.exists():
        console.print(f"[bold red]Error:[/bold red] Directory {directory} does not exist")
        raise typer.Exit(code=1)

    if not directory.is_dir():
        console.print(f"[bold red]Error:[/bold red] {directory} is not a directory")
        raise typer.Exit(code=1)

    # Find all theorem files
    theorem_files = list(directory.glob(f"*{file_extension}"))

    if not theorem_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No {file_extension} files found in {directory}")
        return

    console.print(f"[bold blue]Found {len(theorem_files)} theorem file(s) to process[/bold blue]")

    # Process each theorem file
    for theorem_file in theorem_files:
        console.print(f"\n{'=' * 80}")
        console.print(f"[bold cyan]Processing: {theorem_file.name}[/bold cyan]")
        console.print(f"{'=' * 80}")

        try:
            theorem_content = _read_theorem_content(theorem_file)
            if theorem_content is None:
                continue

            if is_formal and not _has_preamble(theorem_content):
                _handle_missing_header(theorem_file)
                continue

            config = GoedelsPoetryConfig()
            initial_state = (
                GoedelsPoetryState(formal_theorem=theorem_content)
                if is_formal
                else GoedelsPoetryState(informal_theorem=theorem_content)
            )

            state_manager = GoedelsPoetryStateManager(initial_state)
            file_console = Console()
            framework = GoedelsPoetryFramework(config, state_manager, file_console)
            framework.run()

            output_file = theorem_file.with_suffix(".proof")
            if state_manager.reason == "Proof completed successfully.":
                try:
                    # Note: Final verification already performed in framework.finish()
                    # No need to verify again here
                    complete_proof = state_manager.reconstruct_complete_proof()
                    output_file.write_text(complete_proof, encoding="utf-8")
                    console.print(f"[bold green]✓ Successfully proved and saved to {output_file.name}[/bold green]")
                except Exception as e:
                    error_message = f"Proof completed but error reconstructing proof: {e}\n{traceback.format_exc()}"
                    output_file.write_text(error_message, encoding="utf-8")
                    console.print(f"[bold yellow]⚠ Proof had errors, details saved to {output_file.name}[/bold yellow]")
            else:
                failure_message = f"Proof failed: {state_manager.reason}"
                output_file.write_text(failure_message, encoding="utf-8")
                console.print(f"[bold red]✗ Failed to prove, details saved to {output_file.name}[/bold red]")

        except Exception as e:
            _handle_processing_error(theorem_file, e)
            continue

    console.print("\n[bold blue]Finished processing all theorem files[/bold blue]")


@app.command()
def main(
    formal_theorem: str | None = typer.Option(
        None,
        "--formal-theorem",
        "-ft",
        help="A single formal theorem to prove (e.g., 'theorem example : 1 + 1 = 2 := by sorry')",
    ),
    informal_theorem: str | None = typer.Option(
        None,
        "--informal-theorem",
        "-ift",
        help="A single informal theorem to prove (e.g., 'Prove that 3 cannot be written as the sum of two cubes.')",
    ),
    formal_theorems: Path | None = typer.Option(
        None,
        "--formal-theorems",
        "-fts",
        help="Directory containing .lean files with formal theorems to prove",
    ),
    informal_theorems: Path | None = typer.Option(
        None,
        "--informal-theorems",
        "-ifts",
        help="Directory containing .txt files with informal theorems to prove",
    ),
) -> None:
    """
    Gödel's Poetry: An automated theorem proving system.

    Provide exactly one of the four options to process theorems.
    """
    # Count how many options were provided
    options_provided = sum([
        formal_theorem is not None,
        informal_theorem is not None,
        formal_theorems is not None,
        informal_theorems is not None,
    ])

    # Ensure exactly one option is provided
    if options_provided == 0:
        console.print("[bold red]Error:[/bold red] You must provide exactly one of the following options:")
        console.print("  --formal-theorem (-ft): A single formal theorem")
        console.print("  --informal-theorem (-ift): A single informal theorem")
        console.print("  --formal-theorems (-fts): Directory of formal theorems")
        console.print("  --informal-theorems (-ifts): Directory of informal theorems")
        raise typer.Exit(code=1)

    if options_provided > 1:
        console.print("[bold red]Error:[/bold red] Only one option can be provided at a time")
        raise typer.Exit(code=1)

    # Process based on which option was provided
    if formal_theorem:
        process_single_theorem(formal_theorem=formal_theorem)
    elif informal_theorem:
        process_single_theorem(informal_theorem=informal_theorem)
    elif formal_theorems:
        process_theorems_from_directory(formal_theorems, ".lean", is_formal=True)
    elif informal_theorems:
        process_theorems_from_directory(informal_theorems, ".txt", is_formal=False)


if __name__ == "__main__":
    app()
