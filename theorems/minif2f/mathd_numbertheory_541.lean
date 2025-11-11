import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The product of two positive whole numbers is 2005. If neither number is 1, what is the sum of the two numbers? Show that it is 406.-/
theorem mathd_numbertheory_541 (m n : ℕ) (h₀ : 1 < m) (h₁ : 1 < n) (h₂ : m * n = 2005) :
    m + n = 406 := by sorry
