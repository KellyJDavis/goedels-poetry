"""Verify the bug fix works for all .lean files in theorems/minif2f directory."""

import re
from pathlib import Path

import pytest

from goedels_poetry.agents.prover_agent import _parse_prover_response
from goedels_poetry.agents.util.common import (
    combine_preamble_and_body,
    combine_theorem_with_proof,
    split_preamble_and_body,
)


def get_all_lean_files() -> list[Path]:
    """Get all .lean files in theorems/minif2f directory."""
    minif2f_dir = Path(__file__).parent.parent / "theorems" / "minif2f"
    if not minif2f_dir.exists():
        pytest.skip(f"Directory {minif2f_dir} does not exist")
    return sorted(minif2f_dir.glob("*.lean"))


def simulate_llm_response(theorem_with_proof: str) -> str:
    """Simulate an LLM response that returns the full theorem with proof."""
    # Replace "sorry" with a simple proof
    proof_body = "trivial"  # Simple proof for testing
    theorem_with_proof_body = theorem_with_proof.replace(":= by sorry", f":= by\n  {proof_body}")
    return f"""Here's the proof:

```lean4
{theorem_with_proof_body}
```"""


def extract_theorem_name(theorem_body: str) -> str:
    """Extract theorem/lemma/def name from the body."""
    # Match: theorem/lemma/def <name> ...
    match = re.search(r"(theorem|lemma|def|example)\s+([a-zA-Z0-9_']+)", theorem_body)
    if match:
        return match.group(2)
    return "unknown"


def verify_proof_body_extraction(proof_body: str, theorem_name: str, file_name: str) -> None:
    """Verify that proof body extraction is correct."""
    assert theorem_name not in proof_body or theorem_name == "unknown", (
        f"{file_name}: Proof body should not contain theorem statement '{theorem_name}'"
    )
    assert "import " not in proof_body, f"{file_name}: Proof body should not contain imports"


def verify_theorem_combination(theorem_with_proof: str, theorem_name: str, proof_body: str, file_name: str) -> None:
    """Verify that theorem combination is correct."""
    if theorem_name != "unknown":
        declaration_pattern = rf"(theorem|lemma|def)\s+{re.escape(theorem_name)}\s*[:(]"
        declaration_count = len(re.findall(declaration_pattern, theorem_with_proof))
        assert declaration_count <= 1, (
            f"{file_name}: Should have at most 1 theorem declaration, found {declaration_count}"
        )

    if proof_body.strip():
        assert "sorry" not in theorem_with_proof, (
            f"{file_name}: Theorem should not contain 'sorry' after combining with proof"
        )


def verify_final_code(final_code: str, preamble: str, theorem_name: str, proof_body: str, file_name: str) -> None:
    """Verify that final code is correct."""
    if "import " in preamble:
        assert "import " in final_code, f"{file_name}: Final code should have imports if original had them"

    if theorem_name != "unknown":
        declaration_pattern = rf"(theorem|lemma|def)\s+{re.escape(theorem_name)}\s*[:(]"
        final_declaration_count = len(re.findall(declaration_pattern, final_code))
        assert final_declaration_count == 1, (
            f"{file_name}: Final code should have exactly 1 theorem declaration, found {final_declaration_count}"
        )

    if proof_body.strip():
        assert "sorry" not in final_code, f"{file_name}: Final code should not contain 'sorry'"
        assert proof_body.strip() in final_code or any(
            line.strip() in final_code for line in proof_body.split("\n") if line.strip()
        ), f"{file_name}: Final code should contain the proof body"


@pytest.mark.parametrize("lean_file", get_all_lean_files())
def test_bug_fix_for_file(lean_file: Path) -> None:
    """Test that the bug fix works for each .lean file."""
    # Read the file
    file_content = lean_file.read_text(encoding="utf-8")

    # Skip if file is empty or doesn't have the expected structure
    if not file_content.strip():
        pytest.skip(f"File {lean_file.name} is empty")

    # Split into preamble and body
    try:
        preamble, theorem_body = split_preamble_and_body(file_content)
    except Exception as e:
        pytest.skip(f"Could not parse {lean_file.name}: {e}")

    # Skip if no theorem body found
    if not theorem_body.strip():
        pytest.skip(f"File {lean_file.name} has no theorem body")

    # Skip if theorem doesn't have ":= by sorry" pattern
    if ":= by sorry" not in theorem_body and ":=\n  by\n  sorry" not in theorem_body:
        pytest.skip(f"File {lean_file.name} doesn't have ':= by sorry' pattern")

    # Extract theorem name for checking duplicates
    theorem_name = extract_theorem_name(theorem_body)

    # Simulate LLM response (full theorem with proof)
    full_theorem_with_preamble = combine_preamble_and_body(preamble, theorem_body)
    llm_response = simulate_llm_response(full_theorem_with_preamble)

    # Step 1: Parse LLM response to extract only proof body
    try:
        proof_body = _parse_prover_response(llm_response, preamble)
    except Exception as e:
        pytest.fail(f"Failed to parse LLM response for {lean_file.name}: {e}")

    verify_proof_body_extraction(proof_body, theorem_name, lean_file.name)

    # Step 2: Combine original theorem with proof body
    try:
        theorem_with_proof = combine_theorem_with_proof(theorem_body, proof_body)
    except Exception as e:
        pytest.fail(f"Failed to combine theorem with proof for {lean_file.name}: {e}")

    verify_theorem_combination(theorem_with_proof, theorem_name, proof_body, lean_file.name)

    # Step 3: Combine with preamble (what gets sent to server)
    try:
        final_code = combine_preamble_and_body(preamble, theorem_with_proof)
    except Exception as e:
        pytest.fail(f"Failed to combine preamble with theorem for {lean_file.name}: {e}")

    verify_final_code(final_code, preamble, theorem_name, proof_body, lean_file.name)
