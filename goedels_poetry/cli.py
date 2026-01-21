import os

os.environ["TQDM_DISABLE"] = "1"

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from goedels_poetry.agents.util.common import normalize_escape_sequences, split_preamble_and_body

if TYPE_CHECKING:
    from goedels_poetry.state import GoedelsPoetryStateManager

app = typer.Typer()
console = Console()
GOEDELS_POETRY_DEBUG = bool(os.environ.get("GOEDELS_POETRY_DEBUG"))


def _has_preamble(code: str) -> bool:
    preamble, _ = split_preamble_and_body(code)
    return bool(preamble.strip())


def _read_theorem_content(theorem_file: Path) -> str | None:
    theorem_content = theorem_file.read_text(encoding="utf-8").strip()
    if not theorem_content:
        console.print(f"[bold yellow]Warning:[/bold yellow] {theorem_file.name} is empty, skipping")
        return None
    # Normalize escape sequences (e.g., convert literal \n to actual newline)
    theorem_content = normalize_escape_sequences(theorem_content)
    return theorem_content


def _handle_missing_header(theorem_file: Path) -> None:
    console.print("[bold red]Error:[/bold red] Formal theorems must include a Lean header (imports/options).")
    output_file = theorem_file.with_suffix(".failed-proof")
    output_file.write_text(
        "Proof failed: Missing Lean header/preamble in supplied formal theorem.",
        encoding="utf-8",
    )


def _handle_processing_error(theorem_file: Path, error: Exception) -> None:
    console.print(f"[bold red]Error processing {theorem_file.name}:[/bold red] {error}")
    console.print(traceback.format_exc())

    output_file = theorem_file.with_suffix(".failed-proof")
    error_message = f"Error during processing: {error}\n\n{traceback.format_exc()}"
    output_file.write_text(error_message, encoding="utf-8")
    console.print(f"[bold yellow]Error details saved to {output_file.name}[/bold yellow]")


def _write_proof_result(
    theorem_file: Path,
    state_manager: "GoedelsPoetryStateManager",
    console: Console,
    *,
    server_url: str | None = None,
    server_max_retries: int | None = None,
    server_timeout: int | None = None,
) -> None:
    """
    Write proof result to appropriate file based on completion status and validation result.

    Args:
        theorem_file: The theorem file being processed
        state_manager: The state manager containing proof results
        console: Console for output messages
        server_url: Kimina server URL (required for reconstruction)
        server_max_retries: Max retries for Kimina requests
        server_timeout: Timeout for Kimina requests
    """
    from goedels_poetry.config.kimina_server import KIMINA_LEAN_SERVER
    from goedels_poetry.state import ProofReconstructionError

    # Use provided values or defaults from config
    final_server_url = server_url or KIMINA_LEAN_SERVER["url"]
    final_server_max_retries = (
        server_max_retries if server_max_retries is not None else KIMINA_LEAN_SERVER["max_retries"]
    )
    final_server_timeout = server_timeout if server_timeout is not None else KIMINA_LEAN_SERVER["timeout"]

    # Determine output filename based on completion status and validation result
    if state_manager.reason == "Proof completed successfully.":
        # Check validation result to determine filename
        validation_result = state_manager._state.proof_validation_result
        if validation_result is True:
            output_file = theorem_file.with_suffix(".proof")
        else:
            # validation_result is False or None (validation failed or exception)
            output_file = theorem_file.with_suffix(".failed-proof")

        try:
            # Note: Final verification already performed in framework.finish()
            # No need to verify again here, but reconstruction still needs server config
            complete_proof = state_manager.reconstruct_complete_proof(
                server_url=final_server_url,
                server_max_retries=final_server_max_retries,
                server_timeout=final_server_timeout,
            )
            output_file.write_text(complete_proof, encoding="utf-8")
            if validation_result is True:
                console.print(f"[bold green]✓ Successfully proved and saved to {output_file.name}[/bold green]")
            else:
                console.print(
                    f"[bold yellow]⚠ Proof completed but validation failed, saved to {output_file.name}[/bold yellow]"
                )
        except ProofReconstructionError as e:
            error_message = f"Proof completed but error reconstructing proof: {e}\n{traceback.format_exc()}"
            output_file.write_text(error_message, encoding="utf-8")
            console.print(f"[bold yellow]⚠ Proof had errors, details saved to {output_file.name}[/bold yellow]")
        except Exception as e:
            error_message = f"Proof completed but error reconstructing proof: {e}\n{traceback.format_exc()}"
            output_file.write_text(error_message, encoding="utf-8")
            console.print(f"[bold yellow]⚠ Proof had errors, details saved to {output_file.name}[/bold yellow]")
    else:
        # Non-successful completion - use .failed-proof
        output_file = theorem_file.with_suffix(".failed-proof")
        failure_message = f"Proof failed: {state_manager.reason}"
        output_file.write_text(failure_message, encoding="utf-8")
        console.print(f"[bold red]✗ Failed to prove, details saved to {output_file.name}[/bold red]")


def process_single_theorem(
    formal_theorem: str | None = None,
    informal_theorem: str | None = None,
    *,
    start_with_decomposition: bool = False,
) -> None:
    """
    Process a single theorem (either formal or informal) and output proof to stdout.
    """
    from goedels_poetry.framework import GoedelsPoetryConfig, GoedelsPoetryFramework
    from goedels_poetry.state import GoedelsPoetryState, GoedelsPoetryStateManager

    config = GoedelsPoetryConfig()

    if formal_theorem is not None:
        # Normalize escape sequences (e.g., convert literal \n to actual newline)
        formal_theorem = normalize_escape_sequences(formal_theorem)
        if (not start_with_decomposition) and (not _has_preamble(formal_theorem)):
            console.print("[bold red]Error:[/bold red] Formal theorems must include a Lean header (imports/options).")
            raise typer.Exit(code=1)
        initial_state = GoedelsPoetryState(
            formal_theorem=formal_theorem,
            start_with_decomposition=start_with_decomposition,
        )
        if start_with_decomposition:
            console.print(
                "[bold blue]DEBUG: Decomposing formal theorem immediately (skipping initial syntax checks)...[/bold blue]"
            )
        else:
            console.print("[bold blue]Processing formal theorem...[/bold blue]")
    else:
        if informal_theorem is None:
            console.print("[bold red]Error:[/bold red] You must provide either a formal or informal theorem.")
            raise typer.Exit(code=1)
        # Normalize escape sequences (e.g., convert literal \n to actual newline)
        informal_theorem = normalize_escape_sequences(informal_theorem)
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
    *,
    start_with_decomposition: bool = False,
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
    # Note: Each theorem is processed in its own try-except block to ensure that if one theorem
    # fails with an exception, processing continues with the remaining theorems in the directory.
    for theorem_file in theorem_files:
        console.print(f"\n{'=' * 80}")
        console.print(f"[bold cyan]Processing: {theorem_file.name}[/bold cyan]")
        console.print(f"{'=' * 80}")

        try:
            theorem_content = _read_theorem_content(theorem_file)
            if theorem_content is None:
                continue

            if is_formal and (not start_with_decomposition) and (not _has_preamble(theorem_content)):
                _handle_missing_header(theorem_file)
                continue

            config = GoedelsPoetryConfig()
            initial_state = (
                GoedelsPoetryState(
                    formal_theorem=theorem_content,
                    start_with_decomposition=start_with_decomposition,
                )
                if is_formal
                else GoedelsPoetryState(informal_theorem=theorem_content)
            )

            state_manager = GoedelsPoetryStateManager(initial_state)
            file_console = Console()
            framework = GoedelsPoetryFramework(config, state_manager, file_console)
            framework.run()

            # Write proof result to appropriate file
            _write_proof_result(
                theorem_file,
                state_manager,
                console,
                server_url=config.kimina_lean_server_url,
                server_max_retries=config.kimina_lean_server_max_retries,
                server_timeout=config.kimina_lean_server_timeout,
            )

        except Exception as e:
            # Catch any exception (including ProofReconstructionError) and continue processing
            # remaining theorems. The error is logged and saved to a .failed-proof file.
            _handle_processing_error(theorem_file, e)
            continue

    console.print("\n[bold blue]Finished processing all theorem files[/bold blue]")


def _main_router(  # noqa: C901
    *,
    formal_theorem: str | None,
    informal_theorem: str | None,
    formal_theorems: Path | None,
    informal_theorems: Path | None,
    decompose_formal_theorem: str | None = None,
    decompose_formal_theorems: Path | None = None,
) -> None:
    """
    Shared CLI entrypoint logic for both debug and non-debug command signatures.
    """
    option_bools = [
        formal_theorem is not None,
        informal_theorem is not None,
        formal_theorems is not None,
        informal_theorems is not None,
    ]
    if GOEDELS_POETRY_DEBUG:
        option_bools += [
            decompose_formal_theorem is not None,
            decompose_formal_theorems is not None,
        ]

    options_provided = sum(option_bools)

    if options_provided == 0:
        console.print("[bold red]Error:[/bold red] You must provide exactly one of the following options:")
        console.print("  --formal-theorem (-ft): A single formal theorem")
        console.print("  --informal-theorem (-ift): A single informal theorem")
        console.print("  --formal-theorems (-fts): Directory of formal theorems")
        console.print("  --informal-theorems (-ifts): Directory of informal theorems")
        if GOEDELS_POETRY_DEBUG:
            console.print(
                "  --decompose-formal-theorem (-dfs): DEBUG: Immediately decompose a single formal theorem (skips initial header/preamble requirement and root syntax check)"
            )
            console.print(
                "  --decompose-formal-theorems (-dfts): DEBUG: Immediately decompose all .lean files in a directory (skips initial header/preamble requirement and root syntax check)"
            )
        raise typer.Exit(code=1)

    if options_provided > 1:
        console.print("[bold red]Error:[/bold red] Only one option can be provided at a time")
        raise typer.Exit(code=1)

    # Process based on which option was provided
    if formal_theorem is not None:
        process_single_theorem(formal_theorem=formal_theorem)
    elif informal_theorem is not None:
        process_single_theorem(informal_theorem=informal_theorem)
    elif formal_theorems is not None:
        process_theorems_from_directory(formal_theorems, ".lean", is_formal=True)
    elif informal_theorems is not None:
        process_theorems_from_directory(informal_theorems, ".txt", is_formal=False)
    elif decompose_formal_theorem is not None:
        process_single_theorem(formal_theorem=decompose_formal_theorem, start_with_decomposition=True)
    elif decompose_formal_theorems is not None:
        process_theorems_from_directory(
            decompose_formal_theorems,
            ".lean",
            is_formal=True,
            start_with_decomposition=True,
        )


def _main_debug(
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
    decompose_formal_theorem: str | None = typer.Option(
        None,
        "--decompose-formal-theorem",
        "-dfs",
        help=(
            "DEBUG: A single formal theorem to prove (e.g., 'theorem example : 1 + 1 = 2 := by sorry') "
            "that is immediately decomposed (skips initial header/preamble requirement and root syntax check)"
        ),
    ),
    decompose_formal_theorems: Path | None = typer.Option(
        None,
        "--decompose-formal-theorems",
        "-dfts",
        help=(
            "DEBUG: Directory containing .lean files with formal theorems to prove; "
            "each is immediately decomposed (skips initial header/preamble requirement and root syntax check)"
        ),
    ),
) -> None:
    """
    Gödel's Poetry: An automated theorem proving system.

    Provide exactly one option to process theorems.
    """
    _main_router(
        formal_theorem=formal_theorem,
        informal_theorem=informal_theorem,
        formal_theorems=formal_theorems,
        informal_theorems=informal_theorems,
        decompose_formal_theorem=decompose_formal_theorem,
        decompose_formal_theorems=decompose_formal_theorems,
    )


def _main_nodebug(
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

    Provide exactly one option to process theorems.
    """
    _main_router(
        formal_theorem=formal_theorem,
        informal_theorem=informal_theorem,
        formal_theorems=formal_theorems,
        informal_theorems=informal_theorems,
    )


if GOEDELS_POETRY_DEBUG:
    app.command()(_main_debug)
else:
    app.command()(_main_nodebug)


if __name__ == "__main__":
    app()
