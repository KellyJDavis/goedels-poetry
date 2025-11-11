import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any natural number $n$, 12 divides $4^{n+1} + 20$.-/
theorem induction_12dvd4expnp1p20 (n : ℕ) : 12 ∣ 4 ^ (n + 1) + 20 := by sorry
