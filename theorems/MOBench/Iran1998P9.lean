
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Iranian Mathematical Olympiad 1998, Problem 9

Let x,y,z > 1 and 1/x + 1/y + 1/z = 2. Prove that

  √(x + y + z) ≥ √(x - 1) + √(y - 1) + √(z - 1).

-/
theorem iran1998_p9
    (x y z : ℝ)
    (hx : 1 < x)
    (hy : 1 < y)
    (hz : 1 < z)
    (h : 1/x + 1/y + 1/z = 2) :
    √(x - 1) + √(y - 1) + √(z - 1) ≤ √(x + y + z) := by sorry
