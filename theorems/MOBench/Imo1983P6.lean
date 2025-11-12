
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1983, Problem 6

Suppose that a,b,c are the side lengths of a triangle. Prove that

   a²b(a - b) + b²c(b - c) + c²a(c - a) ≥ 0.

Determine when equality occurs.
-/
abbrev EqualityCondition (a b c : ℝ) : Prop := a = b ∧ a = c

theorem imo1983_p6 (T : Affine.Triangle ℝ (EuclideanSpace ℝ (Fin 2))) :
    let a := dist (T.points 1) (T.points 2)
    let b := dist (T.points 0) (T.points 2)
    let c := dist (T.points 0) (T.points 1)
    0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ∧
    (0 = a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ↔
     EqualityCondition a b c) := by sorry
