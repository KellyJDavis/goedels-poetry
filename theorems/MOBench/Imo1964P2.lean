
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1964, Problem 2

Suppose that a,b,c are the side lengths of a triangle. Prove that

   a²(b + c - a) + b²(c + a - b) + c²(a + b - c) ≤ 3abc.
-/
theorem imo1964_p2
    (T : Affine.Triangle ℝ (EuclideanSpace ℝ (Fin 2)))
    (a b c : ℝ)
    (ha : a = dist (T.points 1) (T.points 2))
    (hb : b = dist (T.points 2) (T.points 0))
    (hc : c = dist (T.points 0) (T.points 1)) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤
    3 * a * b * c := by sorry
