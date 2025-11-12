
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2021, Problem 1

Let `n≥100` be an integer. Ivan writes the numbers `n, n+1,..., 2n` each on different cards.
He then shuffles these `n+1` cards, and divides them into two piles. Prove that at least one
of the piles contains two cards such that the sum of their numbers is a perfect square.
-/
theorem imo2021_p1 :
    ∀ n : ℕ, 100 ≤ n → ∀ (A) (_ : A ⊆ Finset.Icc n (2 * n)),
    (∃ (a : _) (_ : a ∈ A) (b : _) (_ : b ∈ A), a ≠ b ∧ ∃ k : ℕ, a + b = k ^ 2) ∨
    ∃ (a : _) (_ : a ∈ Finset.Icc n (2 * n) \ A) (b : _) (_ : b ∈ Finset.Icc n (2 * n) \ A),
      a ≠ b ∧ ∃ k : ℕ, a + b = k ^ 2 := by sorry
