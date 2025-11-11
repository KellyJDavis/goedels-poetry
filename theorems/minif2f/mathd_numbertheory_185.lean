import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- When a number is divided by 5, the remainder is 3. What is the remainder when twice the number is divided by 5? Show that it is 1.-/
theorem mathd_numbertheory_185 (n : ℕ) (h₀ : n % 5 = 3) : 2 * n % 5 = 1 := by sorry
