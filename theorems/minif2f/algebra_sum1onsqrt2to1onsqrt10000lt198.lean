import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that $\sum_{k=2}^{10000} \frac{1}{\sqrt{k}} < 198$.-/
theorem algebra_sum1onsqrt2to1onsqrt10000lt198 :
    (∑ k in Finset.Icc (2 : ℕ) 10000, 1 / Real.sqrt k) < 198 := by sorry
