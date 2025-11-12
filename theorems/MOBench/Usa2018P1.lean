
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2018, Problem 1

Given that a,b,c are positive real numbers such that

  a + b + c = 4 ∛(abc)

prove that 2(ab + bc + ca) + 4min(a²,b²,c²) ≥ a² + b² + c²
-/
theorem usa2018_p1 (a b c : ℝ) :
    a > 0 → b > 0 → c > 0 → a + b + c = 4 * (a * b * c) ^ ((1 : ℝ) / 3) →
    2 * (a * b + b * c + c * a) +
     4 * (min (min (a * a) (b * b)) (c * c)) ≥ a^2 + b^2 + c^2 := by sorry
