import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any real numbers $a$ and $b$, $\frac{|a+b|}{1+|a+b|}\leq \frac{|a|}{1+|a|}+\frac{|b|}{1+|b|}$.-/
theorem algebra_absapbon1pabsapbleqsumabsaon1pabsa (a b : ℝ) :
    abs (a + b) / (1 + abs (a + b)) ≤ abs a / (1 + abs a) + abs b / (1 + abs b) := by sorry
