
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1999, Problem 4

Determine all pairs of positive integers (x,p) such that p is
a prime and xᵖ⁻¹ is a divisor of (p-1)ˣ + 1.
-/
abbrev SolutionSet : Set (ℕ × ℕ) :=
  {⟨x, p⟩ | (x = 1 ∧ Nat.Prime p) ∨ (x = 2 ∧ p = 2) ∨ (x = 3 ∧ p = 3)}

theorem imo1999_p4 (x p : ℕ) :
    ⟨x,p⟩ ∈ SolutionSet ↔
    0 < x ∧ p.Prime ∧ x^(p - 1) ∣ (p - 1)^x + 1 := by sorry
