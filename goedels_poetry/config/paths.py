import os
from pathlib import Path

# Shared base output directory for Goedels Poetry artifacts.
# Respects the GOEDELS_POETRY_DIR env var; defaults to ~/.goedels_poetry.
OUTPUT_DIR = os.environ.get("GOEDELS_POETRY_DIR", os.path.expanduser("~/.goedels_poetry"))

# Ensure the base directory exists early to avoid scattered mkdir calls.
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
