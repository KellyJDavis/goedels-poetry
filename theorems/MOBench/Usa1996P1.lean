
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1996, Problem 1

Prove that the average of the numbers n⬝sin(n π / 180)
for n ∈ {2,4,6,…,180} is 1/tan(π/180).
-/
theorem usa1996_p1 :
    (1 / (90:ℝ)) * ∑ n ∈ Finset.range 90, (2 * (n+1)) * Real.sin ((2 * (n+1)) * Real.pi / 180)
    = 1 / Real.tan (Real.pi / 180) := by sorry
