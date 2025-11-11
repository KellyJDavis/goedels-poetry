import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle

/-!
# International Mathematical Olympiad 1971, Problem 5

Prove that for every natural number m there exists a finite set S of
points in the plane with the following property:
For every point s in S, there are exactly m points which are at a unit
distance from s.
-/

namespace Imo1971P5

open scoped EuclideanGeometry

abbrev Pt := EuclideanSpace ℝ (Fin 2)

theorem imo1971_p5 (m : ℕ) :
    ∃ S : Set Pt, S.Finite ∧ ∀ s ∈ S, Nat.card {t | dist s t = 1} = m := sorry


end Imo1971P5
