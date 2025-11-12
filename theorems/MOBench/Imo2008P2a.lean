
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2008, Problem 2
(a) Prove that
          ```
          x^2 / (x-1)^2 + y^2 / (y-1)^2 + z^2 / (z-1)^2 ≥ 1
          ```
for all real numbers `x`,`y`, `z`, each different from 1, and satisfying `xyz = 1`.
-/
theorem imo2008_p2a (x y z : ℝ) (h : x * y * z = 1) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
    x ^ 2 / (x - 1) ^ 2 + y ^ 2 / (y - 1) ^ 2 + z ^ 2 / (z - 1) ^ 2 ≥ 1 := by sorry
