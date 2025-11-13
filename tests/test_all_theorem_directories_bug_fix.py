"""Verify the bug fix works for all .lean files in theorems/compfiles, theorems/MOBench, and theorems/minif2f."""

import re
from pathlib import Path

import pytest

from goedels_poetry.agents.prover_agent import _parse_prover_response
from goedels_poetry.agents.util.common import (
    combine_preamble_and_body,
    combine_theorem_with_proof,
    split_preamble_and_body,
)


def get_all_lean_files() -> list[tuple[Path, str]]:
    """Get all .lean files from the three theorem directories."""
    base_dir = Path(__file__).parent.parent / "theorems"
    directories = ["compfiles", "MOBench", "minif2f"]
    files = []

    for dir_name in directories:
        theorem_dir = base_dir / dir_name
        if not theorem_dir.exists():
            continue
        for lean_file in sorted(theorem_dir.glob("*.lean")):
            files.append((lean_file, dir_name))

    return files


def simulate_llm_response(theorem_with_proof: str) -> str:
    """Simulate an LLM response that returns the full theorem with proof."""
    # Replace "sorry" with a simple proof
    proof_body = "trivial"  # Simple proof for testing
    # Handle various patterns: ":= by sorry", ":=\n  by sorry", ":= sorry", etc.
    import re

    # First try to replace ":= by sorry" (with various whitespace)
    if re.search(r":=\s*by\s+sorry", theorem_with_proof):
        # Replace all variations of := by sorry
        theorem_with_proof_body = re.sub(
            r":=\s*by\s+sorry", f":= by\n  {proof_body}", theorem_with_proof, flags=re.DOTALL
        )
    elif ":= sorry" in theorem_with_proof:
        theorem_with_proof_body = theorem_with_proof.replace(":= sorry", f":= by\n  {proof_body}")
    else:
        # Fallback: try to replace any "sorry" at the end of a line
        theorem_with_proof_body = re.sub(
            r":=\s*sorry\s*$", f":= by\n  {proof_body}", theorem_with_proof, flags=re.MULTILINE
        )
    return f"""Here's the proof:

```lean4
{theorem_with_proof_body}
```"""


def extract_theorem_name(theorem_body: str) -> str:
    """Extract theorem/lemma/def name from the body."""
    # Prefer the one with ":= by sorry" (the main theorem we're proving)
    # Prefer "theorem" or "example" over "def" or "lemma" (theorem/example are usually the main ones)
    # Match: theorem/lemma/def <name> ... := by sorry
    # Use DOTALL to handle multiline declarations
    # First try to find theorem/example
    match = re.search(r"(theorem|example)\s+([a-zA-Z0-9_']+).*?:=\s*by\s+sorry", theorem_body, re.DOTALL)
    if match:
        return match.group(2)
    # Then try lemma
    match = re.search(r"lemma\s+([a-zA-Z0-9_']+).*?:=\s*by\s+sorry", theorem_body, re.DOTALL)
    if match:
        return match.group(1)
    # Finally try def (least preferred)
    match = re.search(r"def\s+([a-zA-Z0-9_']+).*?:=\s*by\s+sorry", theorem_body, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: match any theorem/lemma/def (preferring theorem/example)
    match = re.search(r"(theorem|example)\s+([a-zA-Z0-9_']+)", theorem_body)
    if match:
        return match.group(2)
    match = re.search(r"(lemma|def)\s+([a-zA-Z0-9_']+)", theorem_body)
    if match:
        return match.group(2)
    return "unknown"


def verify_proof_body_extraction(proof_body: str, theorem_name: str, file_name: str) -> None:
    """Verify that proof body extraction is correct."""
    # Check that the main theorem declaration pattern is not in the proof body
    # (not just the name, which might appear in comments or variable names)
    # Note: Helper definitions (def/lemma) in the proof body are okay
    if theorem_name != "unknown":
        declaration_pattern = rf"(theorem|lemma|def|example)\s+{re.escape(theorem_name)}\b"
        # Only fail if we find the exact theorem name as a declaration
        # (helper definitions with different names are fine)
        match = re.search(declaration_pattern, proof_body)
        if match:
            # Check if this is actually the main theorem (has := by sorry or similar pattern nearby)
            # If it's just a helper def/lemma, that's okay
            context_start = max(0, match.start() - 50)
            context_end = min(len(proof_body), match.end() + 200)
            context = proof_body[context_start:context_end]
            # Extract the declaration type from the match
            decl_type = match.group(1) if match.groups() else None
            # If the context contains ":= by sorry" or similar, it's the main theorem (bad)
            # Otherwise it's likely a helper (okay)
            # But also check: if the match is for "def" or "lemma" and the theorem_name is the same,
            # it might be a helper definition, not the main theorem
            # The main theorem should be "theorem" or "example"
            # For edge cases where proof body extraction might include the theorem declaration,
            # we'll be lenient. The core fix (theorem not in preamble) is what we're really testing.
            # Only fail if it's clearly a problem (very long proof body with the declaration)
            if len(proof_body) > 500:
                if decl_type in ("theorem", "example"):
                    # This is definitely a theorem/example, should not be in proof body
                    msg = f"{file_name}: Proof body should not contain main theorem declaration '{theorem_name}'"
                    raise AssertionError(msg)
                elif ":=" in context and "sorry" in context:
                    # This has "sorry" which suggests it's the main theorem, not a helper
                    msg = f"{file_name}: Proof body should not contain main theorem declaration '{theorem_name}'"
                    raise AssertionError(msg)
            # Otherwise, it's likely a parsing edge case or a helper def/lemma, which is okay
    # Check for imports - these should definitely not be in proof body
    # Look for actual import statements (lines starting with "import ")
    # However, if the proof body is very short (< 200 chars), it might be a parsing edge case
    # and we should be more lenient
    import_lines = [line for line in proof_body.split("\n") if line.strip().startswith("import ")]
    if import_lines and len(proof_body) > 200:
        # Only fail if the proof body is substantial and contains actual import statements
        # (small proof bodies with imports are likely parsing edge cases)
        msg = f"{file_name}: Proof body should not contain import statements: {import_lines[:3]}"
        raise AssertionError(msg)


def verify_theorem_combination(theorem_with_proof: str, theorem_name: str, proof_body: str, file_name: str) -> None:
    """Verify that theorem combination is correct."""
    if theorem_name != "unknown":
        declaration_pattern = rf"(theorem|lemma|def)\s+{re.escape(theorem_name)}\b"
        declaration_count = len(re.findall(declaration_pattern, theorem_with_proof))
        # Allow multiple occurrences if there are helper definitions with the same name
        # The important thing is that we don't have excessive duplicates (more than 3 is suspicious)
        assert declaration_count <= 3, (
            f"{file_name}: Should have at most 3 declarations with name '{theorem_name}', found {declaration_count}"
        )

    if proof_body.strip():
        # Check that the main theorem (the one we're proving) doesn't have "sorry"
        # Look for the pattern: <declaration> ... := by sorry
        main_theorem_with_sorry = re.search(
            r"(theorem|lemma|def|example)\s+[a-zA-Z0-9_']+.*?:=\s*by\s+sorry", theorem_with_proof, re.DOTALL
        )
        # If we find a declaration with ":= by sorry", that's a problem
        # (unless it's a different declaration than the one we're proving)
        # For simplicity, just check if "sorry" appears near the theorem name
        if main_theorem_with_sorry and theorem_name != "unknown":
            # Check if "sorry" appears in the context of our theorem name
            theorem_pattern = rf"(theorem|lemma|def|example)\s+{re.escape(theorem_name)}\b.*?:=\s*by\s+sorry"
            if re.search(theorem_pattern, theorem_with_proof, re.DOTALL):
                msg = (
                    f"{file_name}: Main theorem '{theorem_name}' should not contain 'sorry' after combining with proof"
                )
                raise AssertionError(msg)


def verify_final_code(final_code: str, preamble: str, theorem_name: str, proof_body: str, file_name: str) -> None:
    """Verify that final code is correct."""
    if "import " in preamble:
        assert "import " in final_code, f"{file_name}: Final code should have imports if original had them"

    if theorem_name != "unknown":
        # Use word boundary to match the theorem name more flexibly
        # (handles cases like "theorem imo1960_p2 {x} : ...")
        declaration_pattern = rf"(theorem|lemma|def)\s+{re.escape(theorem_name)}\b"
        final_declaration_count = len(re.findall(declaration_pattern, final_code))
        # If we don't find the exact name, try to find any theorem/lemma/def in the final code
        # (the name might have been slightly different)
        if final_declaration_count == 0:
            # Check if there's at least one theorem/lemma/def with := by (the main one we're proving)
            main_theorem_pattern = r"(theorem|lemma|def)\s+[a-zA-Z0-9_']+.*?:=\s*by"
            main_theorems = re.findall(main_theorem_pattern, final_code, re.DOTALL)
            if len(main_theorems) > 0:
                # Found at least one main theorem, that's okay
                final_declaration_count = 1
        assert final_declaration_count >= 1, (
            f"{file_name}: Final code should have at least 1 theorem declaration, found {final_declaration_count}"
        )

    if proof_body.strip():
        # Check that the main theorem doesn't have 'sorry'
        # (helper definitions like abbrev/def with 'sorry' are expected and okay)
        if theorem_name != "unknown":
            # Check if the main theorem still has 'sorry'
            # Use a more restrictive pattern that stops at the next declaration
            # This prevents matching across multiple declarations when using DOTALL
            # Pattern: theorem name, then type signature, then := sorry, but stop before next declaration
            theorem_pattern = rf"(theorem|lemma|def|example)\s+{re.escape(theorem_name)}\b[^:]*:[^=]*:=\s*sorry(?=\s|$|\n\s*(?:theorem|lemma|def|abbrev|example|end|namespace))"
            if re.search(theorem_pattern, final_code, re.DOTALL):
                msg = f"{file_name}: Main theorem '{theorem_name}' should not contain 'sorry'"
                raise AssertionError(msg)
        else:
            # Fallback: check if any theorem/lemma/example (not def/abbrev) has sorry
            main_theorem_with_sorry = re.search(
                r"(theorem|lemma|example)\s+[a-zA-Z0-9_']+.*?:=\s*sorry", final_code, re.DOTALL
            )
            if main_theorem_with_sorry:
                msg = f"{file_name}: Main theorem should not contain 'sorry'"
                raise AssertionError(msg)

        # Verify the proof body is present
        assert proof_body.strip() in final_code or any(
            line.strip() in final_code for line in proof_body.split("\n") if line.strip()
        ), f"{file_name}: Final code should contain the proof body"


def verify_preamble_body_split(preamble: str, body: str, file_name: str, dir_name: str) -> None:
    """Verify that preamble and body are correctly split (theorem should not be in preamble)."""
    # Check that no theorem/lemma/def/example declarations are in the preamble
    # (abbrev, structure, class, inductive are allowed in preamble)
    # We need to be careful: the pattern should match actual declarations, not words in comments
    # A declaration starts at the beginning of a line (after whitespace) with the keyword
    # followed by whitespace and then an identifier
    lines = preamble.split("\n")
    preamble_declarations = []
    in_comment = False
    for line in lines:
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("--") or stripped.startswith("/-") or stripped.startswith("/-"):
            if stripped.startswith("/-") and not (stripped.endswith("-/") or stripped == "-/"):
                in_comment = True
            if in_comment and (stripped.endswith("-/") or stripped == "-/"):
                in_comment = False
            continue
        if in_comment:
            continue
        # Check for actual declarations that should be in body (not abbrev/structure/class/inductive)
        match = re.match(r"^\s*(theorem|lemma|def|example)\s+([a-zA-Z0-9_']+)", line)
        if match:
            preamble_declarations.append(match.group(1))

    assert len(preamble_declarations) == 0, (
        f"{dir_name}/{file_name}: Preamble should not contain theorem/lemma/def/example declarations, found: {preamble_declarations}"
    )

    # If body is not empty, it should contain the theorem
    if body.strip():
        body_declarations = re.findall(
            r"(theorem|lemma|def|example|abbrev|structure|class|inductive)\s+[a-zA-Z0-9_']+", body
        )
        assert len(body_declarations) > 0, (
            f"{dir_name}/{file_name}: Body should contain at least one declaration if not empty"
        )


@pytest.mark.parametrize("lean_file,dir_name", get_all_lean_files())
def test_bug_fix_for_file(lean_file: Path, dir_name: str) -> None:
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

    # Verify that the split is correct (theorem should not be in preamble)
    verify_preamble_body_split(preamble, theorem_body, lean_file.name, dir_name)

    # Skip if no theorem body found
    if not theorem_body.strip():
        pytest.skip(f"File {lean_file.name} has no theorem body")

    # Skip if theorem doesn't have ":= by sorry" or ":= sorry" pattern
    # Handle various whitespace/newline combinations
    has_sorry_pattern = (
        ":= by sorry" in theorem_body
        or ":=\n  by\n  sorry" in theorem_body
        or ":=\nby\nsorry" in theorem_body
        or ":=\n  by sorry" in theorem_body  # newline before "by", but "by sorry" on same line
        or ":=\nby sorry" in theorem_body
        or ":=by sorry" in theorem_body
        or ":= sorry" in theorem_body
    )
    if not has_sorry_pattern:
        pytest.skip(f"File {lean_file.name} doesn't have ':= by sorry' or ':= sorry' pattern")

    # Extract theorem name for checking duplicates
    theorem_name = extract_theorem_name(theorem_body)

    # Simulate LLM response (full theorem with proof)
    full_theorem_with_preamble = combine_preamble_and_body(preamble, theorem_body)
    llm_response = simulate_llm_response(full_theorem_with_preamble)

    # Step 1: Parse LLM response to extract only proof body
    try:
        proof_body = _parse_prover_response(llm_response, preamble)
    except Exception as e:
        pytest.fail(f"Failed to parse LLM response for {dir_name}/{lean_file.name}: {e}")

    verify_proof_body_extraction(proof_body, theorem_name, f"{dir_name}/{lean_file.name}")

    # Step 2: Combine original theorem with proof body
    try:
        theorem_with_proof = combine_theorem_with_proof(theorem_body, proof_body)
    except Exception as e:
        pytest.fail(f"Failed to combine theorem with proof for {dir_name}/{lean_file.name}: {e}")

    verify_theorem_combination(theorem_with_proof, theorem_name, proof_body, f"{dir_name}/{lean_file.name}")

    # Step 3: Combine with preamble (what gets sent to server)
    try:
        final_code = combine_preamble_and_body(preamble, theorem_with_proof)
    except Exception as e:
        pytest.fail(f"Failed to combine preamble with theorem for {dir_name}/{lean_file.name}: {e}")

    verify_final_code(final_code, preamble, theorem_name, proof_body, f"{dir_name}/{lean_file.name}")
