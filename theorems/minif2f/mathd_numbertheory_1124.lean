import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The four-digit number $\underline{374n}$ is divisible by 18. Find the units digit $n$. Show that it is 4.-/
theorem mathd_numbertheory_1124 (n : ℕ) (h₀ : n ≤ 9) (h₁ : 18 ∣ 374 * 10 + n) : n = 4 := by sorry
