import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1983, Problem 6

Suppose that a,b,c are the side lengths of a triangle. Prove that

   a²b(a - b) + b²c(b - c) + c²a(c - a) ≥ 0.

Determine when equality occurs.
-/

namespace Imo1983P6

/- determine -/ abbrev EqualityCondition (a b c : ℝ) : Prop := sorry

theorem imo1983_p6
    (T : Affine.Triangle ℝ (EuclideanSpace ℝ (Fin 2)))
    (a b c : ℝ)
    (ha : a = dist (T.points 1) (T.points 2))
    (hb : b = dist (T.points 2) (T.points 0))
    (hc : c = dist (T.points 0) (T.points 1)) :
    0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)
    ∧ (0 = a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ↔
       EqualityCondition a b c) := sorry


end Imo1983P6
