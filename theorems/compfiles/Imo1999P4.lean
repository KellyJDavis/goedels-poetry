import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1999, Problem 4

Determine all pairs of positive integers (x,p) such that p is
a prime and xᵖ⁻¹ is a divisor of (p-1)ˣ + 1.
-/

namespace Imo1999P4

/- determine -/ abbrev SolutionSet : Set (ℕ × ℕ) := sorry

theorem imo1999_p4 (x p : ℕ) :
    ⟨x,p⟩ ∈ SolutionSet ↔
    0 < x ∧ p.Prime ∧ x^(p - 1) ∣ (p - 1)^x + 1 := sorry


end Imo1999P4
