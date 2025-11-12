
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Intertional Mathematical Olympiad 2005, Problem 4

Determine all positive integers relatively prime to all the terms of the infinite sequence
`a n = 2 ^ n + 3 ^ n + 6 ^ n - 1`, for `n ≥ 1`.
-/
def a (n : ℕ) : ℤ := 2 ^ n + 3 ^ n + 6 ^ n - 1

abbrev SolutionSet : Set ℕ+ := { 1 }

theorem imo2005_p4 {k : ℕ} (hk : 0 < k) :
    (∀ n : ℕ, 1 ≤ n → IsCoprime (a n) k) ↔ ⟨k, hk⟩ ∈ SolutionSet := by sorry
