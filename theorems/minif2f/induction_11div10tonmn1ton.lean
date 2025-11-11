import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any natural number $n$, we have $11\mid 10^n-(-1)^n$.-/
theorem induction_11div10tonmn1ton (n : ℕ) : 11 ∣ 10 ^ n - (-1 : ℤ) ^ n := by sorry
