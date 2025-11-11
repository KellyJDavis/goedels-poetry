import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What integer $n$ satisfies $0\le n<{101}$ and $$123456\equiv n\pmod {101}~?$$ Show that it is 34.-/
theorem mathd_numbertheory_320 (n : ℕ) (h₀ : n < 101) (h₁ : 101 ∣ 123456 - n) : n = 34 := by sorry
