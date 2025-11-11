import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find the units digit of $16^{17} \times 17^{18} \times 18^{19}$. Show that it is 8.-/
theorem mathd_numbertheory_212 : 16 ^ 17 * 17 ^ 18 * 18 ^ 19 % 10 = 8 := by sorry
