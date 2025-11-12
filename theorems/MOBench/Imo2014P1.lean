
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2014, Problem 1

Let a₀ < a₁ < a₂ < ... an infinite sequence of positive integers.
Prove that there exists a unique integer n ≥ 1 such that

  aₙ < (a₀ + a₁ + ... + aₙ)/n ≤ aₙ₊₁.
-/
theorem imo2014_p1 (a : ℕ → ℤ) (apos : ∀ i, 0 < a i) (ha : ∀ i, a i < a (i + 1)) :
    ∃! n : ℕ, 0 < n ∧
              n * a n < (∑ i ∈ Finset.range (n + 1), a i) ∧
              (∑ i ∈ Finset.range (n + 1), a i) ≤ n * a (n + 1) := by sorry
