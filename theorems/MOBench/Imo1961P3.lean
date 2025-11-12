
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1961, Problem 3

Solve the equation

  cosⁿ x - sinⁿ x = 1,

where n is a given positive integer.
-/
abbrev solutionSet (n : ℕ+) : Set ℝ :=
{ x | (∃ k : ℤ, k * Real.pi = x) ∧ Even n.val ∨ (∃ k : ℤ, k * (2 * Real.pi) = x) ∧ Odd n.val ∨
      (∃ k : ℤ, -(Real.pi / 2) + k * (2 * Real.pi) = x) ∧ Odd n.val }

theorem imo1961_p3 {n : ℕ} {x : ℝ} (npos : 0 < n) :
    x ∈ solutionSet ⟨n, npos⟩ ↔
    (cos x) ^ n - (sin x) ^ n = 1 := by sorry
