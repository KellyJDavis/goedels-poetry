
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1974, Problem 2

Prove that if a, b, and c are positive real numbers, then
a^a * b^b * c^c ≥ (abc)^((a+b+c)/3)
-/
theorem usa1974_p2 :
    ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by sorry
