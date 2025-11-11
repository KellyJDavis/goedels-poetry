import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the smallest positive integer, other than $1$, that is both a perfect cube and a perfect fourth power? Show that it is 4096.-/
theorem mathd_numbertheory_296 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : ∃ x, x ^ 3 = n) (h₂ : ∃ t, t ^ 4 = n) :
    4096 ≤ n := by sorry
