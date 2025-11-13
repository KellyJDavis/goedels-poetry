"""Test to verify the original bug fix: preventing duplicate theorems and missing imports.

This test simulates the exact scenario from the original bug report:
- LLM returns full theorem statement with proof
- Code should extract only proof body
- Combine with original theorem statement
- Result should have: preamble (with imports) + single theorem (no duplicates)
"""

from goedels_poetry.agents.prover_agent import _parse_prover_response
from goedels_poetry.agents.util.common import (
    DEFAULT_IMPORTS,
    combine_preamble_and_body,
    combine_theorem_with_proof,
)


def test_original_bug_scenario() -> None:
    """Test the exact scenario from the original bug report."""
    # Original theorem file content (what's in imo_1959_p1.lean)
    original_preamble = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""
    original_theorem = """/-- Prove that the fraction $\frac{21n+4}{14n+3}$ is irreducible for every natural number $n$.-/
theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry"""  # noqa: RUF001

    # Simulate LLM response (what the LLM returns)
    llm_response = f"""Here's the proof:

```lean4
{original_preamble}{
        original_theorem.replace(
            "sorry",
            '''have h₁ : 3 * (14 * n + 3) = 2 * (21 * n + 4) + 1 := by
  ring_nf
  omega
exact Nat.gcd_eq_one_of_dvd_one (Nat.gcd_dvd_left _ _) (Nat.gcd_dvd_right _ _)''',
        )
    }
```"""

    # Step 1: Parse LLM response to extract only proof body
    proof_body = _parse_prover_response(llm_response, original_preamble)

    # Verify: proof_body should NOT contain the theorem statement
    assert "theorem imo_1959_p1" not in proof_body, "Proof body should not contain theorem statement"
    assert "import Mathlib" not in proof_body, "Proof body should not contain imports"
    assert "have h₁" in proof_body, "Proof body should contain the proof tactics"

    # Step 2: Combine original theorem with proof body
    theorem_with_proof = combine_theorem_with_proof(original_theorem, proof_body)

    # Verify: Should have single theorem declaration (no duplicate)
    theorem_count = theorem_with_proof.count("theorem imo_1959_p1")
    assert theorem_count == 1, f"Should have exactly 1 theorem declaration, found {theorem_count}"

    # Verify: Should NOT have "sorry" anymore
    assert "sorry" not in theorem_with_proof, "Theorem should not contain 'sorry' after combining with proof"

    # Verify: Should have the proof body
    assert "have h₁" in theorem_with_proof, "Theorem should contain the proof body"

    # Step 3: Combine with preamble
    final_code = combine_preamble_and_body(original_preamble, theorem_with_proof)

    # Verify: Should have imports in preamble
    assert "import Mathlib" in final_code, "Final code should have imports"
    assert "import Aesop" in final_code, "Final code should have all imports"

    # Verify: Should have exactly ONE theorem declaration (no duplicates)
    final_theorem_count = final_code.count("theorem imo_1959_p1")
    assert final_theorem_count == 1, (
        f"Final code should have exactly 1 theorem declaration, found {final_theorem_count}"
    )

    # Verify: Should NOT have "sorry"
    assert "sorry" not in final_code, "Final code should not contain 'sorry'"

    # Verify: Should have the proof
    assert "have h₁" in final_code, "Final code should contain the proof"

    # Verify: Structure is correct - preamble, then theorem with proof
    preamble_end = final_code.find("theorem imo_1959_p1")
    assert preamble_end > 0, "Theorem should come after preamble"
    assert "import Mathlib" in final_code[:preamble_end], "Imports should be in preamble section"


def test_no_duplicate_theorem_declaration() -> None:
    """Test that we don't create duplicate theorem declarations."""
    preamble = DEFAULT_IMPORTS
    original_theorem = "theorem test : True := by sorry"

    # Simulate LLM returning full theorem with proof
    llm_response = f"""```lean4
{original_theorem.replace("sorry", "trivial")}
```"""

    # Extract proof body
    proof_body = _parse_prover_response(llm_response, preamble)

    # Combine
    theorem_with_proof = combine_theorem_with_proof(original_theorem, proof_body)
    final_code = combine_preamble_and_body(preamble, theorem_with_proof)

    # Should have exactly one theorem declaration
    count = final_code.count("theorem test")
    assert count == 1, f"Should have exactly 1 theorem, found {count}"


def test_preamble_preserved_with_imports() -> None:
    """Test that preamble with imports is preserved correctly."""
    custom_preamble = """import Mathlib
import Aesop
import OtherLib

set_option maxHeartbeats 0

open BigOperators
"""
    original_theorem = "theorem test : True := by sorry"

    llm_response = """```lean4
theorem test : True := by
  trivial
```"""

    proof_body = _parse_prover_response(llm_response, custom_preamble)
    theorem_with_proof = combine_theorem_with_proof(original_theorem, proof_body)
    final_code = combine_preamble_and_body(custom_preamble, theorem_with_proof)

    # All imports should be present
    assert "import Mathlib" in final_code
    assert "import Aesop" in final_code
    assert "import OtherLib" in final_code
    assert "set_option maxHeartbeats 0" in final_code
    assert "open BigOperators" in final_code

    # Should have exactly one theorem
    assert final_code.count("theorem test") == 1
