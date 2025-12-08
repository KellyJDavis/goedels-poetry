#!/usr/bin/env python3
"""
lean-from-json.py

A small CLI utility to convert a JSONL dataset (e.g., MiniF2F or MOBench)
into a directory of Lean (.lean) files, one per entry. Uses Typer for the
CLI and Rich for a friendly terminal UI.

Example:
  python lean-from-json.py --json dataset/minif2f.jsonl --output-dir theorems/minif2f
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

app = typer.Typer(add_completion=False, help="Convert JSONL (MiniF2F/MOBench) to Lean files.")
console = Console()
EXPECTED_OBJECT_MSG = "Expected JSON object (dict) per line"


def sanitize_filename(name: str) -> str:
    """
    Convert an arbitrary name into a filesystem-friendly Lean filename stem.
    - Keep alphanumerics and underscores
    - Replace all other characters with underscores
    - Collapse multiple underscores
    - Trim leading/trailing underscores
    """
    # Replace invalid chars with underscore
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Strip leading/trailing underscores
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "entry"
    return cleaned


def ensure_newlines(content: str) -> str:
    """
    Ensure JSON-encoded newlines are converted to actual newlines and
    normalize line endings. JSON parsing usually converts '\\n' to '\n'
    already, but some datasets might carry literal backslash-n sequences.
    """
    # Convert literal backslash-n sequences to real newlines
    # Note: do a conservative pass; if already proper, this is a no-op.
    content = content.replace("\\n", "\n")
    # Normalize Windows CRLF to LF
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # Ensure file ends with a single trailing newline
    if not content.endswith("\n"):
        content = content + "\n"
    return content


def detect_fields(obj: dict) -> tuple[str | None, str | None]:
    """
    Extract (name, lean_code) from a dataset object.
    - Prefer 'name' for filename stem; fallback to 'problem_id'
    - Prefer 'lean4_code' for content; fallback to 'formal_statement'
    """
    name = obj.get("name") or obj.get("problem_id")
    lean_code = obj.get("lean4_code") or obj.get("formal_statement")
    return name, lean_code


def iter_jsonl(path: Path) -> Iterable[tuple[int, str]]:
    """
    Yield (1-based line_number, raw_line) for each non-empty line in the JSONL.
    """
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if line.strip():
                yield idx, line


@app.command("convert")
def convert_command(
    json: Path = typer.Option(  # noqa: B008
        ...,
        "--json",
        "-j",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
        help="Path to the input JSONL file.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to write .lean files into."),  # noqa: B008
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files instead of skipping."),
) -> None:
    """
    Convert a JSONL dataset of the form MiniF2F/MOBench to individual Lean files.
    Each entry must provide 'lean4_code' (preferred) or 'formal_statement', and a
    'name' (preferred) or 'problem_id' used to derive the filename.
    """
    try:
        # Prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preload lines to have a fixed-length iterable for progress display
        lines = list(iter_jsonl(json))
        total = len(lines)
        if total == 0:
            console.print(f"[yellow]No non-empty lines found in[/yellow] [bold]{json}[/bold]. Nothing to do.")
            raise typer.Exit(code=0)

        created_count = 0
        overwritten_count = 0
        skipped_count = 0
        error_count = 0

        console.print(
            f"[bold]Converting[/bold] [cyan]{json}[/cyan] [bold]→[/bold] [magenta]{output_dir}[/magenta] ({total} entries)"
        )

        for line_no, raw in track(lines, description="Processing", total=total):
            try:
                obj = jsonlib_loads_strict(raw)
            except json.JSONDecodeError as e:
                error_count += 1
                console.print(f"[red]JSON parse error[/red] at line {line_no}: {e}")
                continue

            name, lean_code = detect_fields(obj)
            if not name:
                error_count += 1
                console.print(f"[red]Missing 'name'/'problem_id'[/red] at line {line_no}, skipping.")
                continue
            if not lean_code:
                error_count += 1
                console.print(
                    f"[red]Missing 'lean4_code'/'formal_statement'[/red] for '{name}' at line {line_no}, skipping."
                )
                continue

            filename_stem = sanitize_filename(str(name))
            target_path = output_dir / f"{filename_stem}.lean"
            content = ensure_newlines(str(lean_code))

            if target_path.exists() and not overwrite:
                skipped_count += 1
                continue

            if target_path.exists() and overwrite:
                overwritten_count += 1
            else:
                created_count += 1

            target_path.write_text(content, encoding="utf-8")

        # Summary table
        table = Table(title="Conversion Summary", show_lines=False)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Count", justify="right", style="bold")
        table.add_row("Total entries", str(total))
        table.add_row("Created files", str(created_count))
        table.add_row("Overwritten files", str(overwritten_count))
        table.add_row("Skipped (exists)", str(skipped_count))
        table.add_row("Errors", str(error_count), style="red" if error_count else "green")
        console.print()
        console.print(table)

        if error_count > 0:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.[/red]")
        raise typer.Exit(code=130) from None


def jsonlib_loads_strict(raw_line: str) -> dict:
    """
    Parse a single JSON object from a JSONL line with strict error surfacing.
    """
    obj = json.loads(raw_line)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError(EXPECTED_OBJECT_MSG, raw_line, 0)
    return obj


def main() -> None:
    app()


if __name__ == "__main__":
    main()
